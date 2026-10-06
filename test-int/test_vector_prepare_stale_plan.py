"""A delayed vector plan never overwrites newer note state on SQLite.

Many writers share one SQLite database: FastAPI builds a search repository per request,
every accepted write schedules its own background vector sync, the file watcher owns
another repository, and each MCP client runs its own server process. The asyncio lock
behind ``_prepare_entity_write_scope`` belongs to one repository instance, so it orders
none of them.

A vector sync reads the note's full-text rows and vector manifest, plans, and only then
opens its write transaction. Each test here pauses one sync between that read and its
write, lets the note move on, and resumes it. The resumed sync must leave the newest
state in place: the newest manifest, no vectors for a deleted note, and none for a note
that opted out of embeddings. Before the fix it rewrote the manifest to older text, put
vectors back after an opt-out or a delete, or failed with "Vector manifest rows
disappeared before write".
"""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, override

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
type ManifestRow = tuple[str, str, str]

PAUSE_TIMEOUT_SECONDS = 30
CHILD_TIMEOUT_SECONDS = 60

ORIGINAL = "Quartz repair notes stay searchable during refresh."
OLDER = "Quartz older update awaiting embedding."
NEWEST = "Quartz newest update must remain indexed."


class StubEmbeddingProvider:
    """One unit vector for every text, so any vector query matches every chunk."""

    model_name = "stale-plan-stub"
    dimensions = 4

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def runtime_log_attrs(self) -> dict[str, object]:
        return {}


class PausedEmbedding(StubEmbeddingProvider):
    """Hold a sync after its prepare committed, before its vectors are written."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.resume = asyncio.Event()

    @override
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.started.set()
        await asyncio.wait_for(self.resume.wait(), timeout=PAUSE_TIMEOUT_SECONDS)
        return await super().embed_documents(texts)


class PausedBeforeWrite(SQLiteSearchRepository):
    """Hold a sync after its read planned a mutation, before its write transaction."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.planned = asyncio.Event()
        self.resume = asyncio.Event()

    @asynccontextmanager
    @override
    async def _prepare_entity_write_scope(self):
        self.planned.set()
        await asyncio.wait_for(self.resume.wait(), timeout=PAUSE_TIMEOUT_SECONDS)
        async with super()._prepare_entity_write_scope():
            yield


class PausedInsideWrite(SQLiteSearchRepository):
    """Hold a sync inside its write transaction, just before its first chunk upsert."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.writing = asyncio.Event()
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
        self.writing.set()
        await asyncio.wait_for(self.resume.wait(), timeout=PAUSE_TIMEOUT_SECONDS)
        return await super()._upsert_scheduled_chunk_records(
            session,
            entity_id=entity_id,
            scheduled_records=scheduled_records,
            existing_by_key=existing_by_key,
            entity_fingerprint=entity_fingerprint,
            embedding_model=embedding_model,
        )


@dataclass
class Note:
    """One indexed note and the "current" writer that keeps moving it on."""

    session_maker: async_sessionmaker[AsyncSession]
    project_id: int
    project_path: Path
    config: BasicMemoryConfig
    entity: Entity

    def repository(
        self, embedding_provider: StubEmbeddingProvider | None = None
    ) -> SQLiteSearchRepository:
        """A fresh repository instance, as each request or process builds its own."""
        return self.paused(SQLiteSearchRepository, embedding_provider)

    def paused[RepositoryT: SQLiteSearchRepository](
        self,
        repository_type: type[RepositoryT],
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
            FileService(self.project_path),
            self.session_maker,
        )

    @property
    def current(self) -> SearchService:
        return self.search_service(self.repository())

    async def refresh(self, content: str) -> None:
        await self.current.index_entity(self.entity, content=content)

    async def sync_vectors(self) -> None:
        await self.current.sync_entity_vectors(self.entity.id)

    async def apply(self, change: LatestChange) -> None:
        """Land the newest change through the production paths."""
        match change:
            case "update":
                await self.refresh(NEWEST)
                await self.sync_vectors()
            case "delete":
                # EntityService.delete_entity order: search cleanup, then the entity row.
                await self.current.handle_delete(self.entity)
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
                await self.sync_vectors()

    async def manifest(self) -> list[ManifestRow]:
        rows = await self.repository().get_entity_chunk_manifest(self.entity.id)
        return [(row.chunk_key, row.chunk_text, row.embedding_status) for row in rows]

    async def stored_vectors(self) -> int:
        repository = self.repository()
        async with db.scoped_session(self.session_maker) as session:
            await repository._ensure_sqlite_vec_loaded(session)
            count = await session.scalar(text("SELECT COUNT(*) FROM search_vector_embeddings"))
        return int(count or 0)

    async def hits(self, mode: SearchRetrievalMode) -> list[int]:
        rows = await self.repository().search(
            search_text="quartz", retrieval_mode=mode, min_similarity=0.0
        )
        return [row.entity_id for row in rows if row.entity_id is not None]


@pytest_asyncio.fixture
async def note(engine_factory, test_project, config_home) -> Note:
    """A Markdown note, indexed for full-text search, with semantic search enabled."""
    engine, session_maker = engine_factory
    if engine.dialect.name != "sqlite":
        pytest.skip("SQLite takes the database write lock; PostgreSQL prepares differently")

    now = datetime.now(timezone.utc)
    entity = Entity(
        project_id=test_project.id,
        title="Quartz repair note",
        note_type="note",
        content_type="text/markdown",
        permalink="quartz-repair",
        file_path="quartz-repair.md",
        # Loaded up front: indexing reads them after this session has closed.
        observations=[],
        outgoing_relations=[],
        created_at=now,
        updated_at=now,
    )
    async with db.scoped_session(session_maker) as session:
        session.add(entity)

    note = Note(
        session_maker=session_maker,
        project_id=test_project.id,
        project_path=Path(config_home),
        config=BasicMemoryConfig(
            env="test", database_backend=DatabaseBackend.SQLITE, semantic_search_enabled=True
        ),
        entity=entity,
    )
    # Database initialization creates vector storage, as at startup.
    await note.current.init_search_index()
    return note


async def assert_latest_state(
    note: Note, change: LatestChange, latest_manifest: list[ManifestRow]
) -> None:
    """The resumed sync left exactly the state the newest change produced."""
    assert await note.manifest() == latest_manifest, "a delayed plan rewrote the manifest"
    match change:
        case "update":
            assert [chunk_text for _, chunk_text, _ in latest_manifest] == [
                f"Quartz repair note\n\nquartz-repair\n\n{NEWEST}"
            ]
            assert await note.stored_vectors() == 1
            assert await note.hits(SearchRetrievalMode.VECTOR) == [note.entity.id]
        case "delete":
            assert latest_manifest == []
            assert await note.stored_vectors() == 0, "vectors outlived the deleted note"
        case "opt_out":
            assert latest_manifest == []
            assert await note.stored_vectors() == 0, "vectors came back after the opt-out"
            # Opting out of embeddings keeps the note in keyword search.
            assert await note.hits(SearchRetrievalMode.FTS) == [note.entity.id]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["search_service", "repository"])
@pytest.mark.parametrize("change", ["update", "delete", "opt_out"])
async def test_delayed_vector_plan_keeps_the_latest_state(
    note: Note, change: LatestChange, entry_point: str
) -> None:
    await note.refresh(ORIGINAL)
    await note.sync_vectors()
    await note.refresh(OLDER)

    delayed = note.paused(PausedBeforeWrite)
    if entry_point == "search_service":
        sync = note.search_service(delayed).sync_entity_vectors(note.entity.id)
    else:
        sync = delayed.sync_entity_vectors(note.entity.id)
    pending = asyncio.create_task(sync)
    await asyncio.wait_for(delayed.planned.wait(), timeout=PAUSE_TIMEOUT_SECONDS)

    await note.apply(change)
    latest_manifest = await note.manifest()
    delayed.resume.set()
    await pending

    await assert_latest_state(note, change, latest_manifest)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["delete", "opt_out"])
async def test_delayed_first_embedding_respects_a_later_delete_or_opt_out(
    note: Note, change: LatestChange
) -> None:
    # No manifest exists yet, so the delayed plan inserts rows instead of updating them.
    await note.refresh(OLDER)

    delayed = note.paused(PausedBeforeWrite)
    pending = asyncio.create_task(note.search_service(delayed).sync_entity_vectors(note.entity.id))
    await asyncio.wait_for(delayed.planned.wait(), timeout=PAUSE_TIMEOUT_SECONDS)

    await note.apply(change)
    delayed.resume.set()
    await pending

    await assert_latest_state(note, change, [])


@pytest.mark.asyncio
async def test_opt_out_while_vectors_are_computed_retires_the_work(note: Note) -> None:
    """The opt-out removes the rows the sync is writing; that is not a lost manifest."""
    await note.refresh(ORIGINAL)
    await note.sync_vectors()
    await note.refresh(OLDER)

    embedding = PausedEmbedding()
    delayed = note.repository(embedding)
    pending = asyncio.create_task(note.search_service(delayed).sync_entity_vectors(note.entity.id))
    await asyncio.wait_for(embedding.started.wait(), timeout=PAUSE_TIMEOUT_SECONDS)

    await note.apply("opt_out")
    embedding.resume.set()
    # Before the fix this raised "Vector manifest rows disappeared before write".
    await pending

    await assert_latest_state(note, "opt_out", [])


def sectioned(*headings: str) -> str:
    """One heading section per vector chunk (see test_sqlite_vector_search_repository)."""
    body = " ".join(["vector sync section body content."] * 15)
    return "\n\n".join(f"## {heading}\n{body}" for heading in headings)


@pytest.mark.asyncio
async def test_delayed_cleanup_keeps_a_chunk_the_note_has_again(note: Note) -> None:
    await note.refresh(sectioned("Alpha", "Beta"))
    await note.sync_vectors()
    published = await note.manifest()
    assert len(published) == 2

    # An intermediate edit drops the Beta section, and the delayed sync plans its cleanup.
    await note.refresh(sectioned("Alpha"))
    delayed = note.paused(PausedBeforeWrite)
    pending = asyncio.create_task(note.search_service(delayed).sync_entity_vectors(note.entity.id))
    await asyncio.wait_for(delayed.planned.wait(), timeout=PAUSE_TIMEOUT_SECONDS)

    # The edit is undone; Beta's chunk is current again with its original text.
    await note.refresh(sectioned("Alpha", "Beta"))
    await note.sync_vectors()
    delayed.resume.set()
    await pending

    assert await note.manifest() == published, "the delayed cleanup deleted a current chunk"
    assert await note.stored_vectors() == 2


def try_to_begin_writing(database: str) -> str | None:
    """Ask SQLite for its write lock without waiting; return the refusal, if any."""
    connection = sqlite3.connect(database, timeout=0, isolation_level=None)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("ROLLBACK")
        return None
    except sqlite3.OperationalError as exc:
        return str(exc)
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_prepare_write_holds_the_sqlite_write_lock_from_its_first_read(
    note: Note, engine_factory
) -> None:
    engine, _ = engine_factory
    assert engine.url.database is not None
    await note.refresh(ORIGINAL)
    await note.sync_vectors()
    await note.refresh(OLDER)

    delayed = note.paused(PausedInsideWrite)
    pending = asyncio.create_task(delayed.sync_entity_vectors(note.entity.id))
    await asyncio.wait_for(delayed.writing.wait(), timeout=PAUSE_TIMEOUT_SECONDS)

    # Readers are never blocked: the note stays searchable while the plan is applied.
    assert await note.hits(SearchRetrievalMode.FTS) == [note.entity.id]
    # Every other writer is, so nothing can commit between the re-read and the upsert.
    refusal = await asyncio.to_thread(try_to_begin_writing, engine.url.database)
    delayed.resume.set()
    await pending

    assert refusal == "database is locked"


# --- Two processes, one SQLite file ---
#
# The delayed side runs this module as a script: it plans a vector sync, prints PLANNED,
# and blocks on stdin until the test, in its own process, has landed the newest change.

PLANNED = "PLANNED"


class StdinGatedRepository(SQLiteSearchRepository):
    """Child-process repository that waits for the parent between its read and write."""

    @asynccontextmanager
    @override
    async def _prepare_entity_write_scope(self):
        print(PLANNED, flush=True)
        await asyncio.to_thread(sys.stdin.readline)
        async with super()._prepare_entity_write_scope():
            yield


async def run_delayed_child(db_path: Path, project_id: int, entity_id: int) -> None:
    config = BasicMemoryConfig(
        env="test", database_backend=DatabaseBackend.SQLITE, semantic_search_enabled=True
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
        await service.sync_entity_vectors(entity_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["update", "delete", "opt_out"])
async def test_delayed_vector_plan_in_another_process_keeps_the_latest_state(
    note: Note, engine_factory, tmp_path: Path, change: LatestChange
) -> None:
    """The guarantee holds across processes, as with several MCP clients on one project."""
    engine, _ = engine_factory
    assert engine.url.database is not None
    await note.refresh(ORIGINAL)
    await note.sync_vectors()
    await note.refresh(OLDER)

    child_stderr = tmp_path / "delayed-child.log"
    with child_stderr.open("wb") as stderr:
        # Constraint: Windows runs Basic Memory on the selector event loop, which has no
        #   asyncio subprocess support, so blocking pipes are driven from worker threads.
        #   -P keeps test-int/ off the child's sys.path.
        child = subprocess.Popen(
            [
                sys.executable,
                "-P",
                __file__,
                engine.url.database,
                str(note.project_id),
                str(note.entity.id),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        assert child.stdin is not None and child.stdout is not None
        try:
            planned = await asyncio.wait_for(
                asyncio.to_thread(child.stdout.readline), CHILD_TIMEOUT_SECONDS
            )
            assert planned.decode().strip() == PLANNED, child_stderr.read_text()[-4000:]

            await note.apply(change)
            latest_manifest = await note.manifest()

            child.stdin.write(b"resume\n")
            child.stdin.flush()
            returncode = await asyncio.wait_for(
                asyncio.to_thread(child.wait), CHILD_TIMEOUT_SECONDS
            )
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdin.close()
            child.stdout.close()

    assert returncode == 0, child_stderr.read_text()[-4000:]
    await assert_latest_state(note, change, latest_manifest)


if __name__ == "__main__":
    asyncio.run(run_delayed_child(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])))
