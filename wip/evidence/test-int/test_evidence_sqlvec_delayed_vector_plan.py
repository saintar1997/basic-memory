"""SQLite regressions: a delayed vector plan never overwrites newer note state.

Production runs many ``SQLiteSearchRepository`` instances against one database.
FastAPI dependencies build a repository per request, every accepted write
schedules its own background vector sync (``LocalEntityVectorSyncScheduler``),
the file watcher owns another instance, and each MCP client starts its own
server process. The per-instance ``asyncio.Lock`` behind
``_prepare_entity_write_scope`` therefore orders nothing between them.

Each test pauses one vector sync after it has read and planned, lets a second
repository publish newer state, then resumes the first. The resumed sync must
leave the newest state in place: the newest manifest, no vectors for a deleted
note, and no vectors for a note that opted out of embeddings.

The two-process test runs this module as a script for the delayed side, so the
pause crosses a process boundary that no in-process lock can bridge.
"""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, override

import pytest
import pytest_asyncio
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory import db
from basic_memory.config import BasicMemoryConfig, DatabaseBackend
from basic_memory.models import Entity
from basic_memory.repository import EntityRepository
from basic_memory.repository.semantic_chunking import VectorChunkRecord
from basic_memory.repository.semantic_vector_sync import PendingEmbeddingJob, VectorChunkState
from basic_memory.repository.sqlite_search_repository import SQLiteSearchRepository
from basic_memory.schemas.search import SearchRetrievalMode
from basic_memory.services.file_service import FileService
from basic_memory.services.search_service import SearchService

type LatestChange = Literal["update", "delete", "opt_out"]
type PausePoint = Literal["after_read", "before_write"]
type EntryPoint = Literal["repository", "search_service"]

PAUSE_TIMEOUT_SECONDS = 30

ORIGINAL = "Quartz repair notes stay searchable during refresh."
OLDER = "Quartz older update awaiting embedding."
NEWEST = "Quartz newest update must remain indexed."


class StubEmbeddingProvider:
    """Deterministic unit vectors, so every chunk is a perfect semantic match."""

    model_name = "delayed-vector-plan-stub"
    dimensions = 4

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def runtime_log_attrs(self) -> dict[str, object]:
        return {}


class EmbeddingPause(StubEmbeddingProvider):
    """Hold one sync between its committed prepare and its vector write."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.resume = asyncio.Event()

    @override
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.started.set()
        await asyncio.wait_for(self.resume.wait(), timeout=PAUSE_TIMEOUT_SECONDS)
        return await super().embed_documents(texts)


class PausedBeforeWrite(SQLiteSearchRepository):
    """Stop after the shared read planned a mutation, before the write transaction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.plan_ready = asyncio.Event()
        self.resume = asyncio.Event()

    @asynccontextmanager
    @override
    async def _prepare_entity_write_scope(self):
        self.plan_ready.set()
        await asyncio.wait_for(self.resume.wait(), timeout=PAUSE_TIMEOUT_SECONDS)
        async with super()._prepare_entity_write_scope():
            yield


class PausedAfterRead(SQLiteSearchRepository):
    """Stop inside the shared read pass, right after the manifest snapshot."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.plan_ready = asyncio.Event()
        self.resume = asyncio.Event()

    @override
    async def _fetch_prepare_window_existing_rows(
        self,
        session: AsyncSession,
        entity_ids: list[int],
    ) -> dict[int, list[VectorChunkState]]:
        rows = await super()._fetch_prepare_window_existing_rows(session, entity_ids)
        # Only the first read pauses; a re-read under the write lock must not.
        if not self.plan_ready.is_set():
            self.plan_ready.set()
            await asyncio.wait_for(self.resume.wait(), timeout=PAUSE_TIMEOUT_SECONDS)
        return rows


class PausedInsideWrite(SQLiteSearchRepository):
    """Stop inside the prepare write transaction, just before its first manifest upsert."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.in_write = asyncio.Event()
        self.resume = asyncio.Event()

    @override
    async def _upsert_scheduled_chunk_records(
        self,
        session: AsyncSession,
        *,
        entity_id: int,
        scheduled_records: list[VectorChunkRecord],
        existing_by_key: dict[str, VectorChunkState],
        entity_fingerprint: str,
        embedding_model: str,
    ) -> list[PendingEmbeddingJob]:
        self.in_write.set()
        await asyncio.wait_for(self.resume.wait(), timeout=PAUSE_TIMEOUT_SECONDS)
        return await super()._upsert_scheduled_chunk_records(
            session,
            entity_id=entity_id,
            scheduled_records=scheduled_records,
            existing_by_key=existing_by_key,
            entity_fingerprint=entity_fingerprint,
            embedding_model=embedding_model,
        )


def _try_begin_write(database: str) -> str | None:
    """Try to take SQLite's write lock without waiting; return SQLite's refusal, if any."""
    connection = sqlite3.connect(database, timeout=0, isolation_level=None)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("ROLLBACK")
        return None
    except sqlite3.OperationalError as exc:
        return str(exc)
    finally:
        connection.close()


def _require_sqlite_vec() -> None:
    probe = sqlite3.connect(":memory:")
    try:
        if not hasattr(probe, "enable_load_extension"):
            pytest.skip("Python build cannot load SQLite extensions (sqlite-vec)")
    finally:
        probe.close()
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        pytest.skip("sqlite-vec is required for SQLite vector-sync regressions")


class NoteHarness:
    """One real note plus the services every scenario shares."""

    def __init__(
        self,
        *,
        session_maker: async_sessionmaker[AsyncSession],
        project_id: int,
        project_path: Path,
        config: BasicMemoryConfig,
        entity: Entity,
    ) -> None:
        self.session_maker = session_maker
        self.project_id = project_id
        self.config = config
        self.entity = entity
        self.file_service = FileService(project_path)
        # The "current" writer: refreshes full-text rows and publishes vectors.
        self.repository = self.build_repository(SQLiteSearchRepository)
        self.service = self.search_service(self.repository)

    def build_repository[RepositoryT: SQLiteSearchRepository](
        self,
        repository_type: type[RepositoryT],
        *,
        embedding_provider: StubEmbeddingProvider | None = None,
    ) -> RepositoryT:
        return repository_type(
            self.session_maker,
            self.project_id,
            app_config=self.config,
            embedding_provider=embedding_provider or StubEmbeddingProvider(),
        )

    def search_service(self, repository: SQLiteSearchRepository) -> SearchService:
        return SearchService(
            repository,
            EntityRepository(project_id=self.project_id),
            self.file_service,
            self.session_maker,
        )

    async def refresh(self, content: str) -> None:
        await self.service.index_entity(self.entity, content=content)

    async def apply(self, change: LatestChange) -> None:
        """Make the newest change land through the production service paths."""
        match change:
            case "update":
                await self.refresh(NEWEST)
                await self.service.sync_entity_vectors(self.entity.id)
            case "delete":
                # EntityService.delete_entity order: search cleanup, then the row.
                await self.service.handle_delete(self.entity)
                async with db.scoped_session(self.session_maker) as session:
                    await EntityRepository(project_id=self.project_id).delete(
                        session, self.entity.id
                    )
            case "opt_out":
                async with db.scoped_session(self.session_maker) as session:
                    await session.execute(
                        update(Entity)
                        .where(Entity.id == self.entity.id)
                        .values(entity_metadata={"embed": False})
                    )
                await self.service.sync_entity_vectors(self.entity.id)

    async def manifest(self) -> list[tuple[str, str, str, str]]:
        rows = await self.repository.get_entity_chunk_manifest(self.entity.id)
        return [
            (row.chunk_key, row.chunk_text, row.source_hash, row.embedding_status) for row in rows
        ]

    async def vector_row_count(self) -> int:
        async with db.scoped_session(self.session_maker) as session:
            await self.repository._ensure_sqlite_vec_loaded(session)
            count = await session.scalar(text("SELECT COUNT(*) FROM search_vector_embeddings"))
        return int(count or 0)

    async def semantic_hits(self) -> list[int]:
        rows = await self.repository.search(
            search_text="quartz",
            retrieval_mode=SearchRetrievalMode.VECTOR,
            min_similarity=0.0,
        )
        return [row.entity_id for row in rows if row.entity_id is not None]

    async def keyword_hits(self) -> list[int]:
        rows = await self.repository.search(search_text="quartz")
        return [row.entity_id for row in rows if row.entity_id is not None]


@pytest_asyncio.fixture
async def note(engine_factory, test_project, config_home) -> NoteHarness:
    """A markdown note whose search projection and vectors start in sync."""
    engine, session_maker = engine_factory
    if engine.dialect.name != "sqlite":
        pytest.skip("SQLite vector-prepare locking; PostgreSQL has its own coverage")
    _require_sqlite_vec()

    config = BasicMemoryConfig(
        env="test",
        database_backend=DatabaseBackend.SQLITE,
        semantic_search_enabled=True,
    )
    now = datetime.now(timezone.utc)
    entity = Entity(
        project_id=test_project.id,
        title="Quartz repair note",
        note_type="note",
        content_type="text/markdown",
        permalink="quartz-repair",
        file_path="quartz-repair.md",
        observations=[],
        outgoing_relations=[],
        incoming_relations=[],
        created_at=now,
        updated_at=now,
    )
    async with db.scoped_session(session_maker) as session:
        session.add(entity)
        await session.flush()

    harness = NoteHarness(
        session_maker=session_maker,
        project_id=test_project.id,
        project_path=Path(config_home),
        config=config,
        entity=entity,
    )
    # Database initialization creates vector storage once, as at startup.
    await harness.service.init_search_index()
    return harness


async def _settle(task: asyncio.Task[None]) -> BaseException | None:
    """Wait for the delayed sync and return what it raised, if anything."""
    outcome = (await asyncio.gather(task, return_exceptions=True))[0]
    return outcome if isinstance(outcome, BaseException) else None


def _pause_type(pause: PausePoint) -> type[PausedBeforeWrite] | type[PausedAfterRead]:
    return PausedAfterRead if pause == "after_read" else PausedBeforeWrite


def _delayed_sync(
    note: NoteHarness, delayed: SQLiteSearchRepository, entry_point: EntryPoint
) -> asyncio.Task[None]:
    if entry_point == "repository":
        return asyncio.create_task(delayed.sync_entity_vectors(note.entity.id))
    return asyncio.create_task(note.search_service(delayed).sync_entity_vectors(note.entity.id))


async def _assert_latest_state(
    note: NoteHarness,
    change: LatestChange,
    current_manifest: Sequence[tuple[str, str, str, str]],
    delayed_error: BaseException | str | None,
) -> None:
    # State first, so a spurious failure on top of a correct state reads as such.
    assert await note.manifest() == current_manifest, (
        "the delayed vector sync rewrote the manifest published after its read"
    )
    match change:
        case "update":
            assert all(NEWEST in chunk_text for _, chunk_text, _, _ in current_manifest)
            assert await note.vector_row_count() == len(current_manifest)
            assert await note.semantic_hits() == [note.entity.id]
        case "delete":
            assert current_manifest == []
            assert await note.vector_row_count() == 0, "vectors outlived the deleted note"
        case "opt_out":
            assert current_manifest == []
            assert await note.vector_row_count() == 0, "vectors came back after an opt-out"
            assert await note.semantic_hits() == []
            # Opting out of embeddings keeps the note in keyword search.
            assert await note.keyword_hits() == [note.entity.id]
    assert delayed_error is None, f"delayed vector sync failed: {delayed_error!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["repository", "search_service"])
@pytest.mark.parametrize("pause", ["after_read", "before_write"])
@pytest.mark.parametrize("change", ["update", "delete", "opt_out"])
async def test_delayed_vector_plan_preserves_latest_state(
    note: NoteHarness,
    change: LatestChange,
    pause: PausePoint,
    entry_point: EntryPoint,
) -> None:
    """An older plan resumed after a newer change leaves the newer state intact."""
    await note.refresh(ORIGINAL)
    await note.service.sync_entity_vectors(note.entity.id)
    await note.refresh(OLDER)

    delayed = note.build_repository(_pause_type(pause))
    pending = _delayed_sync(note, delayed, entry_point)
    try:
        await asyncio.wait_for(delayed.plan_ready.wait(), timeout=PAUSE_TIMEOUT_SECONDS)
        await note.apply(change)
        current_manifest = await note.manifest()
    finally:
        delayed.resume.set()
        delayed_error = await _settle(pending)

    await _assert_latest_state(note, change, current_manifest, delayed_error)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["delete", "opt_out"])
async def test_delayed_first_embedding_respects_latest_eligibility(
    note: NoteHarness,
    change: LatestChange,
) -> None:
    """A first embedding planned before a delete or opt-out must not create vectors."""
    # No vectors exist yet, so the delayed plan inserts new manifest rows.
    await note.refresh(OLDER)

    delayed = note.build_repository(PausedBeforeWrite)
    pending = _delayed_sync(note, delayed, "search_service")
    try:
        await asyncio.wait_for(delayed.plan_ready.wait(), timeout=PAUSE_TIMEOUT_SECONDS)
        await note.apply(change)
        current_manifest = await note.manifest()
    finally:
        delayed.resume.set()
        delayed_error = await _settle(pending)

    await _assert_latest_state(note, change, current_manifest, delayed_error)


@pytest.mark.asyncio
async def test_opt_out_during_embedding_is_not_a_vanished_manifest_error(
    note: NoteHarness,
) -> None:
    """An opt-out that lands while vectors are computed retires that work quietly."""
    await note.refresh(ORIGINAL)
    await note.service.sync_entity_vectors(note.entity.id)
    await note.refresh(OLDER)

    embedding = EmbeddingPause()
    delayed = note.build_repository(SQLiteSearchRepository, embedding_provider=embedding)
    pending = _delayed_sync(note, delayed, "search_service")
    try:
        # The delayed sync committed its pending manifest before embedding.
        await asyncio.wait_for(embedding.started.wait(), timeout=PAUSE_TIMEOUT_SECONDS)
        await note.apply("opt_out")
        current_manifest = await note.manifest()
    finally:
        embedding.resume.set()
        delayed_error = await _settle(pending)

    await _assert_latest_state(note, "opt_out", current_manifest, delayed_error)


@pytest.mark.asyncio
async def test_sqlite_prepare_write_holds_the_database_write_lock(
    note: NoteHarness,
    engine_factory,
) -> None:
    """A prepare transaction's re-read and the mutations it plans commit as one unit."""
    engine, _session_maker = engine_factory
    database = engine.url.database
    assert database is not None
    await note.refresh(ORIGINAL)
    await note.service.sync_entity_vectors(note.entity.id)
    await note.refresh(OLDER)

    delayed = note.build_repository(PausedInsideWrite)
    pending = asyncio.create_task(delayed.sync_entity_vectors(note.entity.id))
    try:
        await asyncio.wait_for(delayed.in_write.wait(), timeout=PAUSE_TIMEOUT_SECONDS)
        # Readers are never blocked: the note stays searchable during the write.
        assert await note.keyword_hits() == [note.entity.id]
        # Every other writer is: another request, the watcher, or another process.
        writer_refusal = await asyncio.to_thread(_try_begin_write, database)
    finally:
        delayed.resume.set()
        delayed_error = await _settle(pending)

    assert writer_refusal == "database is locked", (
        "another writer could commit between this prepare's planning and its mutations"
    )
    assert delayed_error is None, f"delayed vector sync failed: {delayed_error!r}"
    assert [chunk_text for _, chunk_text, _, _ in await note.manifest()] == [
        f"Quartz repair note\n\nquartz-repair\n\n{OLDER}"
    ]


def _sectioned(*headings: str) -> str:
    """One sized heading section per vector chunk (see test_sqlite_vector_search_repository)."""
    body = " ".join(["vector sync section body content."] * 15)
    return "\n\n".join(f"## {heading}\n{body}" for heading in headings)


@pytest.mark.asyncio
async def test_delayed_stale_chunk_cleanup_keeps_a_restored_chunk(note: NoteHarness) -> None:
    """A cleanup planned from an intermediate edit must not delete current chunks."""
    await note.refresh(_sectioned("Alpha", "Beta"))
    await note.service.sync_entity_vectors(note.entity.id)
    published_manifest = await note.manifest()
    assert len(published_manifest) == 2

    # An intermediate edit drops the Beta section; the delayed sync plans its cleanup.
    await note.refresh(_sectioned("Alpha"))
    delayed = note.build_repository(PausedBeforeWrite)
    pending = _delayed_sync(note, delayed, "search_service")
    try:
        await asyncio.wait_for(delayed.plan_ready.wait(), timeout=PAUSE_TIMEOUT_SECONDS)
        # The edit is undone; the current sync finds every chunk already embedded.
        await note.refresh(_sectioned("Alpha", "Beta"))
        await note.service.sync_entity_vectors(note.entity.id)
        current_manifest = await note.manifest()
    finally:
        delayed.resume.set()
        delayed_error = await _settle(pending)

    assert delayed_error is None, f"delayed vector sync failed: {delayed_error!r}"
    assert current_manifest == published_manifest
    assert await note.manifest() == current_manifest, (
        "a delayed cleanup deleted a chunk the current note still contains"
    )
    assert await note.vector_row_count() == 2


# --- Two processes, one SQLite file ---
#
# The delayed side runs this module as a script. It plans the vector sync, prints
# PLAN_READY, and blocks on stdin; the test applies the newest change from its own
# process, then sends one line to let the delayed side write.

PLAN_READY = "PLAN_READY"
CHILD_TIMEOUT_SECONDS = 60


class StdinGatedRepository(SQLiteSearchRepository):
    """Child-process repository that waits for the parent before its write."""

    @asynccontextmanager
    @override
    async def _prepare_entity_write_scope(self):
        print(PLAN_READY, flush=True)
        await asyncio.to_thread(sys.stdin.readline)
        async with super()._prepare_entity_write_scope():
            yield


async def _run_delayed_child(db_path: Path, project_id: int, entity_id: int) -> int:
    """Run one delayed SearchService vector sync against the parent's database."""
    config = BasicMemoryConfig(
        env="test",
        database_backend=DatabaseBackend.SQLITE,
        semantic_search_enabled=True,
    )
    async with db.engine_session_factory(db_path, db.DatabaseType.FILESYSTEM, config) as (
        _engine,
        session_maker,
    ):
        repository = StdinGatedRepository(
            session_maker,
            project_id,
            app_config=config,
            embedding_provider=StubEmbeddingProvider(),
        )
        service = SearchService(
            repository,
            EntityRepository(project_id=project_id),
            FileService(db_path.parent),
            session_maker,
        )
        try:
            await service.sync_entity_vectors(entity_id)
        except Exception as exc:  # reported to the parent, which owns the assertion
            print(f"FAILED {exc!r}", flush=True)
            return 1
    print("DONE", flush=True)
    return 0


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["update", "delete", "opt_out"])
async def test_delayed_vector_plan_in_another_process_preserves_latest_state(
    note: NoteHarness,
    engine_factory,
    tmp_path: Path,
    change: LatestChange,
) -> None:
    """The guarantee must hold for a writer in another process, as with several MCP clients."""
    engine, _session_maker = engine_factory
    database = engine.url.database
    assert database is not None
    await note.refresh(ORIGINAL)
    await note.service.sync_entity_vectors(note.entity.id)
    await note.refresh(OLDER)

    child_log = tmp_path / "delayed-child.log"
    with child_log.open("wb") as child_stderr:
        # Constraint: Windows runs Basic Memory on the selector event loop, which has no
        #   asyncio subprocess support, so the child is driven through blocking pipes on
        #   worker threads. -P keeps test-int/ off the child's sys.path.
        child = subprocess.Popen(
            [sys.executable, "-P", __file__, database, str(note.project_id), str(note.entity.id)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=child_stderr,
        )
        assert child.stdin is not None and child.stdout is not None
        try:
            ready = await asyncio.wait_for(
                asyncio.to_thread(child.stdout.readline), CHILD_TIMEOUT_SECONDS
            )
            assert ready.decode().strip() == PLAN_READY, child_log.read_text()[-4000:]

            await note.apply(change)
            current_manifest = await note.manifest()

            child.stdin.write(b"resume\n")
            child.stdin.flush()
            outcome = await asyncio.wait_for(
                asyncio.to_thread(child.stdout.readline), CHILD_TIMEOUT_SECONDS
            )
            await asyncio.to_thread(child.wait, CHILD_TIMEOUT_SECONDS)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdin.close()
            child.stdout.close()

    child_outcome = outcome.decode().strip()
    delayed_error = None if child_outcome == "DONE" else child_outcome
    await _assert_latest_state(note, change, current_manifest, delayed_error)


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(_run_delayed_child(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])))
    )
