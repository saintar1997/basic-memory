"""A note stays searchable while another worker refreshes its search projection.

Several writers share one Basic Memory database: MCP clients, the file watcher, and the
CLI. Refreshing a note replaces its whole search projection -- the old rows are deleted
and the replacement rows written -- and since #1623 both happen in one transaction.

The October 2026 incident report (several MCP clients writing one SQLite project on
0.23.2) described the failure that shape prevents: the delete used to commit on its own,
a concurrent vector sync read the resulting empty full-text interval as an entity
deletion, and the note was left with full-text rows but zero vector chunks. Nothing
re-embeds a note whose source rows did not change, so it stayed out of semantic search
until its next edit or a full reindex.

These tests hold one refresh open *after its delete and before its replacement rows* and
record what every other reader sees in that window, on file-backed SQLite (WAL) and on
PostgreSQL:

- full-text search through ``SearchService`` and the search repository still finds it;
- a vector sync plans from the last committed projection, so it skips the unchanged note
  instead of planning a deletion;
- vector search still returns the note, and its manifest and stored embeddings are the
  ones it had before the refresh started.

The last test runs the refreshing writer in a second OS process, the deployment shape of
the incident.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, override

import pytest
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory import db
from basic_memory.config import BasicMemoryConfig, DatabaseBackend, ProjectEntry
from basic_memory.markdown import EntityParser
from basic_memory.markdown.markdown_processor import MarkdownProcessor
from basic_memory.models import Entity, Observation
from basic_memory.repository import EntityRepository
from basic_memory.repository.postgres_search_repository import PostgresSearchRepository
from basic_memory.repository.search_index_row import SearchIndexRow
from basic_memory.repository.search_repository_base import SearchRepositoryBase
from basic_memory.repository.semantic_vector_sync import PreparedEntityVectorSync
from basic_memory.repository.sqlite_search_repository import SQLiteSearchRepository
from basic_memory.schemas.search import SearchQuery, SearchRetrievalMode
from basic_memory.services.file_service import FileService
from basic_memory.services.search_service import SearchService

ORIGINAL_BODY = "Quartz repair notes stay searchable while another worker refreshes them."
REFRESHED_BODY = "Quartz repair notes, refreshed by a lapidary."
# Every reader below searches for a title word, so the query matches before, during,
# and after the refresh; the refreshed body adds a word only the new rows contain.
TITLE_TERM = "quartz"
REFRESHED_TERM = "lapidary"
# Generous bounds for a loaded 4-CPU runner; SQLite's own busy timeout is 10 seconds.
STEP_TIMEOUT_SECONDS = 30

type SyncEntryPoint = Literal["service", "service_batch", "repository"]


class ConstantEmbeddingProvider:
    """Embed every text to one unit vector, so any vector query matches every chunk."""

    model_name = "evidence-refresh-constant"
    dimensions = 4

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def runtime_log_attrs(self) -> dict[str, object]:
        return {}


@dataclass(frozen=True, slots=True)
class SearchWorker:
    """One worker's view of a project: its own repository instance and service."""

    service: SearchService
    repository: SearchRepositoryBase


@dataclass(frozen=True, slots=True)
class ProjectionView:
    """Everything a reader can observe about one note's derived search state."""

    found_by_service_fts: bool
    found_by_repository_fts: bool
    found_by_vector_search: bool
    manifest: tuple[tuple[str, str, str], ...]
    stored_embedding_keys: frozenset[str]


@dataclass(frozen=True, slots=True)
class VectorPlanView:
    """What one vector sync read and decided for the note."""

    source_rows_read: int
    skipped_unchanged_note: bool
    planned_entity_vector_deletion: bool


@dataclass(slots=True)
class VectorSyncProbe:
    """Holds a vector sync after its prepare window so the test can act inside it.

    The sync reads, plans, and applies its prepare-window mutations, then waits for
    ``release`` before its remaining writes (embedding flushes, the deferral marker).
    On SQLite those writes would otherwise queue behind the paused refresh's write lock
    and spend SQLite's 10-second busy timeout while the test is still observing.
    """

    window_prepared: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    source_rows_read: list[int] = field(default_factory=list)
    prepared: list[PreparedEntityVectorSync | BaseException] = field(default_factory=list)

    def plan_for(self, entity_id: int) -> VectorPlanView:
        assert len(self.prepared) == 1, self.prepared
        prepared = self.prepared[0]
        if isinstance(prepared, BaseException):
            raise prepared
        assert prepared.entity_id == entity_id
        return VectorPlanView(
            source_rows_read=sum(self.source_rows_read),
            skipped_unchanged_note=prepared.entity_skipped,
            planned_entity_vector_deletion=prepared.delete_entity_vectors,
        )


# --- Workers ---


def semantic_config(app_config: BasicMemoryConfig) -> BasicMemoryConfig:
    """The test project's config with semantic search on and no similarity floor."""
    return app_config.model_copy(
        update={"semantic_search_enabled": True, "semantic_min_similarity": 0.0}
    )


def build_worker(
    session_maker: async_sessionmaker[AsyncSession],
    *,
    project_id: int,
    project_path: Path,
    config: BasicMemoryConfig,
) -> SearchWorker:
    """Build an independent worker, as each MCP process or watcher builds its own."""
    repository: SearchRepositoryBase
    if config.database_backend == DatabaseBackend.POSTGRES:
        repository = PostgresSearchRepository(
            session_maker,
            project_id,
            app_config=config,
            embedding_provider=ConstantEmbeddingProvider(),
        )
    else:
        repository = SQLiteSearchRepository(
            session_maker,
            project_id,
            app_config=config,
            embedding_provider=ConstantEmbeddingProvider(),
        )
    file_service = FileService(project_path, MarkdownProcessor(EntityParser(project_path)))
    service = SearchService(
        repository,
        EntityRepository(project_id=project_id),
        file_service,
        session_maker,
    )
    return SearchWorker(service=service, repository=repository)


async def load_entity(
    session_maker: async_sessionmaker[AsyncSession],
    *,
    project_id: int,
    entity_id: int,
) -> Entity:
    """Load the note with the observations and relations its projection is built from."""
    async with db.scoped_session(session_maker) as session:
        entity = await EntityRepository(project_id=project_id).find_by_id(session, entity_id)
    assert entity is not None
    return entity


async def seed_embedded_note(
    session_maker: async_sessionmaker[AsyncSession],
    worker: SearchWorker,
    *,
    project_id: int,
) -> Entity:
    """Create a note with one observation, index it, and embed it."""
    now = datetime.now(timezone.utc)
    async with db.scoped_session(session_maker) as session:
        note = Entity(
            project_id=project_id,
            title="Quartz repair note",
            note_type="note",
            permalink="notes/quartz-repair",
            file_path="notes/quartz-repair.md",
            content_type="text/markdown",
            created_at=now,
            updated_at=now,
        )
        session.add(note)
        await session.flush()
        session.add(
            Observation(
                project_id=project_id,
                entity_id=note.id,
                category="fact",
                content="Quartz repairs need a steady hand",
            )
        )
        note_id = note.id

    entity = await load_entity(session_maker, project_id=project_id, entity_id=note_id)
    await worker.service.index_entity_data(entity, content=ORIGINAL_BODY)
    await worker.service.sync_entity_vectors(entity.id)
    return entity


# --- Observation ---


def hit_entity_ids(rows: list[SearchIndexRow]) -> set[int | None]:
    return {row.entity_id for row in rows}


async def observe_projection(reader: SearchWorker, entity_id: int) -> ProjectionView:
    """Read the note the way search clients and the readiness inspector do."""
    service_hits = await reader.service.search(SearchQuery(text=TITLE_TERM))
    repository_hits = await reader.repository.search(search_text=TITLE_TERM)
    vector_hits = await reader.repository.search(
        search_text=TITLE_TERM,
        retrieval_mode=SearchRetrievalMode.VECTOR,
        min_similarity=0.0,
    )
    manifest = await reader.repository.get_entity_chunk_manifest(entity_id)
    stored_embedding_keys = await reader.repository.get_entity_physical_chunk_keys(entity_id)
    return ProjectionView(
        found_by_service_fts=entity_id in hit_entity_ids(service_hits),
        found_by_repository_fts=entity_id in hit_entity_ids(repository_hits),
        found_by_vector_search=entity_id in hit_entity_ids(vector_hits),
        manifest=tuple((row.chunk_key, row.source_hash, row.embedding_status) for row in manifest),
        stored_embedding_keys=frozenset(stored_embedding_keys or ()),
    )


async def entity_row_count(session: AsyncSession, *, project_id: int, entity_id: int) -> int:
    result = await session.execute(
        text(
            "SELECT COUNT(*) FROM search_index "
            "WHERE project_id = :project_id AND entity_id = :entity_id"
        ),
        {"project_id": project_id, "entity_id": entity_id},
    )
    return int(result.scalar_one())


def probe_vector_sync(monkeypatch: pytest.MonkeyPatch, worker: SearchWorker) -> VectorSyncProbe:
    """Record what the worker's vector sync reads and plans, and hold it there."""
    probe = VectorSyncProbe()
    repository = worker.repository
    fetch_source_rows = repository._fetch_prepare_window_source_rows
    prepare_window = repository._prepare_entity_vector_jobs_window

    async def recording_fetch(session: AsyncSession, entity_ids: list[int]) -> dict[int, list[Any]]:
        rows = await fetch_source_rows(session, entity_ids)
        probe.source_rows_read.extend(len(entity_rows) for entity_rows in rows.values())
        return rows

    async def held_window(
        entity_ids: list[int],
    ) -> list[PreparedEntityVectorSync | BaseException]:
        prepared = await prepare_window(entity_ids)
        probe.prepared.extend(prepared)
        probe.window_prepared.set()
        await asyncio.wait_for(probe.release.wait(), timeout=STEP_TIMEOUT_SECONDS)
        return prepared

    monkeypatch.setattr(repository, "_fetch_prepare_window_source_rows", recording_fetch)
    monkeypatch.setattr(repository, "_prepare_entity_vector_jobs_window", held_window)
    return probe


async def run_vector_sync(
    worker: SearchWorker, entity_id: int, entry_point: SyncEntryPoint
) -> None:
    """Run one of the production vector-sync entry points for the note."""
    match entry_point:
        case "service":
            await worker.service.sync_entity_vectors(entity_id)
        case "service_batch":
            result = await worker.service.sync_entity_vectors_batch([entity_id])
            assert result.entities_failed == 0, result.sample_errors
        case "repository":
            await worker.repository.sync_entity_vectors(entity_id)


# --- Single process, two workers ---

type ReplacementWrite = Literal["bulk_index_items", "index_item"]


@dataclass(frozen=True, slots=True)
class RefreshWindow:
    """What another worker saw while a refresh sat between its delete and replacement."""

    rows_visible_to_refresh: int
    during: ProjectionView
    plan: VectorPlanView


async def start_workers(
    session_maker: async_sessionmaker[AsyncSession],
    *,
    project_id: int,
    project_path: Path,
    config: BasicMemoryConfig,
    count: int,
) -> list[SearchWorker]:
    workers = [
        build_worker(session_maker, project_id=project_id, project_path=project_path, config=config)
        for _ in range(count)
    ]
    # Workers initialize at startup, as production does; nothing below creates storage.
    for worker in workers:
        await worker.service.init_search_index()
    return workers


async def seed_embedded_file(
    session_maker: async_sessionmaker[AsyncSession],
    worker: SearchWorker,
    *,
    project_id: int,
) -> Entity:
    """Create a non-Markdown file entity as the regular-file indexer does, and embed it."""
    now = datetime.now(timezone.utc)
    async with db.scoped_session(session_maker) as session:
        scan = Entity(
            project_id=project_id,
            title="Quartz repair scan",
            note_type="file",
            permalink=None,
            file_path="files/quartz-repair-scan.pdf",
            content_type="application/pdf",
            created_at=now,
            updated_at=now,
        )
        session.add(scan)
        await session.flush()
        scan_id = scan.id

    entity = await load_entity(session_maker, project_id=project_id, entity_id=scan_id)
    assert not entity.is_markdown
    await worker.service.index_entity_data(entity)
    await worker.service.sync_entity_vectors(entity.id)
    return entity


async def observe_refresh_window(
    monkeypatch: pytest.MonkeyPatch,
    *,
    writer: SearchWorker,
    reader: SearchWorker,
    note: Entity,
    content: str | None,
    replacement_write: ReplacementWrite,
    entry_point: SyncEntryPoint,
) -> RefreshWindow:
    """Hold the writer's refresh after its delete while the reader syncs and searches."""
    refresh_paused = asyncio.Event()
    resume_refresh = asyncio.Event()
    rows_visible_to_refresh: list[int] = []
    # Markdown notes write their replacement through bulk_index_items, files through
    # index_item; both run inside index_entity_data's transaction, after its delete.
    write_replacement = (
        writer.repository.bulk_index_items
        if replacement_write == "bulk_index_items"
        else writer.repository.index_item
    )

    async def pause_before_replacement(
        replacement: Any, session: AsyncSession | None = None
    ) -> None:
        assert session is not None
        # The writer's own transaction no longer sees the old rows: the window is open.
        rows_visible_to_refresh.append(
            await entity_row_count(session, project_id=note.project_id, entity_id=note.id)
        )
        refresh_paused.set()
        await asyncio.wait_for(resume_refresh.wait(), timeout=STEP_TIMEOUT_SECONDS)
        await write_replacement(replacement, session)

    monkeypatch.setattr(writer.repository, replacement_write, pause_before_replacement)
    probe = probe_vector_sync(monkeypatch, reader)

    refresh = asyncio.create_task(writer.service.index_entity_data(note, content=content))
    sync: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(refresh_paused.wait(), timeout=STEP_TIMEOUT_SECONDS)

        # --- Read and sync while the replacement is not yet written ---
        # Trigger: the incident was a sync that read this window as a deletion.
        # Why: what a sync reads and plans happens in its prepare window; SQLite admits
        #   one writer, so its later writes must wait for the refresh to commit anyway.
        # Outcome: observe the plan and the projection at the end of the prepare window,
        #   then let the refresh commit and the sync finish its writes.
        sync = asyncio.create_task(run_vector_sync(reader, note.id, entry_point))
        await asyncio.wait_for(probe.window_prepared.wait(), timeout=STEP_TIMEOUT_SECONDS)
        during = await observe_projection(reader, note.id)
        plan = probe.plan_for(note.id)
    finally:
        resume_refresh.set()
        probe.release.set()
        await asyncio.wait_for(refresh, timeout=STEP_TIMEOUT_SECONDS)
        if sync is not None:
            await asyncio.wait_for(sync, timeout=STEP_TIMEOUT_SECONDS)

    [rows_visible] = rows_visible_to_refresh
    return RefreshWindow(rows_visible_to_refresh=rows_visible, during=during, plan=plan)


def unchanged_note_window(before: ProjectionView, *, source_rows: int) -> RefreshWindow:
    """The only correct observation: the last committed projection, synced as unchanged."""
    return RefreshWindow(
        rows_visible_to_refresh=0,
        during=before,
        plan=VectorPlanView(
            source_rows_read=source_rows,
            skipped_unchanged_note=True,
            planned_entity_vector_deletion=False,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["service", "service_batch", "repository"])
async def test_note_stays_searchable_while_another_worker_refreshes_it(
    engine_factory,
    test_project,
    app_config,
    monkeypatch,
    entry_point: SyncEntryPoint,
):
    engine, session_maker = engine_factory
    if engine.dialect.name == "sqlite":
        pytest.importorskip("sqlite_vec")
    writer, reader = await start_workers(
        session_maker,
        project_id=test_project.id,
        project_path=Path(test_project.path),
        config=semantic_config(app_config),
        count=2,
    )
    note = await seed_embedded_note(session_maker, writer, project_id=test_project.id)
    before = await observe_projection(reader, note.id)
    assert before.found_by_service_fts and before.found_by_vector_search
    assert before.manifest and before.stored_embedding_keys

    window = await observe_refresh_window(
        monkeypatch,
        writer=writer,
        reader=reader,
        note=note,
        content=REFRESHED_BODY,
        replacement_write="bulk_index_items",
        entry_point=entry_point,
    )

    # The note and its observation: two committed rows the sync must have read.
    assert window == unchanged_note_window(before, source_rows=2), (
        "a refresh in progress must read as the last committed projection, not a deletion"
    )

    # --- Both writers finished ---
    after = await observe_projection(reader, note.id)
    assert after == before, "the concurrent sync must not have removed the note's vectors"
    refreshed_hits = await reader.service.search(SearchQuery(text=REFRESHED_TERM))
    assert note.id in hit_entity_ids(refreshed_hits)

    # The derived vectors converge on the next sync, which embeds the refreshed text.
    await reader.service.sync_entity_vectors(note.id)
    converged = await observe_projection(reader, note.id)
    assert converged.found_by_vector_search
    assert converged.manifest != before.manifest
    assert {status for _key, _hash, status in converged.manifest} == {"ready"}


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["service", "service_batch", "repository"])
async def test_file_note_stays_searchable_while_another_worker_refreshes_it(
    engine_factory,
    test_project,
    app_config,
    monkeypatch,
    entry_point: SyncEntryPoint,
):
    engine, session_maker = engine_factory
    if engine.dialect.name == "sqlite":
        pytest.importorskip("sqlite_vec")
    writer, reader = await start_workers(
        session_maker,
        project_id=test_project.id,
        project_path=Path(test_project.path),
        config=semantic_config(app_config),
        count=2,
    )
    scan = await seed_embedded_file(session_maker, writer, project_id=test_project.id)
    before = await observe_projection(reader, scan.id)
    assert before.found_by_service_fts and before.found_by_vector_search
    assert before.manifest and before.stored_embedding_keys

    window = await observe_refresh_window(
        monkeypatch,
        writer=writer,
        reader=reader,
        note=scan,
        content=None,
        replacement_write="index_item",
        entry_point=entry_point,
    )

    # A file's projection is its one entity row.
    assert window == unchanged_note_window(before, source_rows=1), (
        "a file refresh in progress must read as the last committed projection"
    )
    assert await observe_projection(reader, scan.id) == before


# --- Two processes ---


class PausedReplacementRepository(SQLiteSearchRepository):
    """Stop a refresh between its delete and its replacement until told to resume."""

    @override
    async def bulk_index_items(
        self, search_index_rows: list[SearchIndexRow], session: AsyncSession | None = None
    ) -> None:
        assert session is not None
        rows_left = await entity_row_count(
            session,
            project_id=self.project_id,
            entity_id=search_index_rows[0].id,
        )
        print(f"PAUSED {rows_left}", flush=True)
        command = await asyncio.to_thread(sys.stdin.readline)
        if command.strip() != "RESUME":
            raise RuntimeError(f"Unexpected worker command: {command!r}")
        await super().bulk_index_items(search_index_rows, session)


async def run_paused_refresh_worker(spec: dict[str, str | int]) -> None:
    """Child-process entry point: refresh the note, pausing inside the transaction."""
    # stdout carries the PAUSED/RESUME/DONE protocol; keep log lines off it.
    logger.remove()
    project_path = Path(str(spec["project_path"]))
    project_id = int(spec["project_id"])
    config = BasicMemoryConfig(
        env="test",
        projects={"test-project": ProjectEntry(path=str(project_path))},
        default_project="test-project",
        database_backend=DatabaseBackend.SQLITE,
        semantic_search_enabled=True,
        semantic_min_similarity=0.0,
    )
    async with db.engine_session_factory(
        Path(str(spec["db_path"])), db.DatabaseType.FILESYSTEM, config
    ) as (_engine, session_maker):
        repository = PausedReplacementRepository(
            session_maker,
            project_id,
            app_config=config,
            embedding_provider=ConstantEmbeddingProvider(),
        )
        file_service = FileService(project_path, MarkdownProcessor(EntityParser(project_path)))
        service = SearchService(
            repository,
            EntityRepository(project_id=project_id),
            file_service,
            session_maker,
        )
        await service.init_search_index()
        entity = await load_entity(
            session_maker, project_id=project_id, entity_id=int(spec["entity_id"])
        )
        await service.index_entity_data(entity, content=REFRESHED_BODY)
    print("DONE", flush=True)


async def read_worker_line(worker: subprocess.Popen[str], log_path: Path) -> str:
    assert worker.stdout is not None
    line = await asyncio.wait_for(
        asyncio.to_thread(worker.stdout.readline), timeout=STEP_TIMEOUT_SECONDS * 2
    )
    assert line, f"refresh worker exited early:\n{log_path.read_text(encoding='utf-8')}"
    return line.strip()


@pytest.mark.asyncio
async def test_note_stays_searchable_while_another_process_refreshes_it(
    engine_factory,
    test_project,
    app_config,
    monkeypatch,
    tmp_path,
):
    engine, session_maker = engine_factory
    if engine.dialect.name != "sqlite":
        pytest.skip("two OS processes sharing one database file is the SQLite deployment shape")
    pytest.importorskip("sqlite_vec")
    db_path = engine.url.database
    assert db_path is not None

    project_path = Path(test_project.path)
    [reader] = await start_workers(
        session_maker,
        project_id=test_project.id,
        project_path=project_path,
        config=semantic_config(app_config),
        count=1,
    )
    note = await seed_embedded_note(session_maker, reader, project_id=test_project.id)
    before = await observe_projection(reader, note.id)
    assert before.found_by_service_fts and before.found_by_vector_search
    assert before.manifest and before.stored_embedding_keys
    probe = probe_vector_sync(monkeypatch, reader)

    spec = {
        "db_path": db_path,
        "project_id": test_project.id,
        "project_path": str(project_path),
        "entity_id": note.id,
    }
    log_path = tmp_path / "refresh-worker.log"
    with log_path.open("w", encoding="utf-8") as worker_log:
        # The child runs this file as a script, so it imports nothing from pytest's
        # module namespace and works under every --import-mode.
        worker = subprocess.Popen(
            [sys.executable, __file__, json.dumps(spec)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=worker_log,
            text=True,
        )
    assert worker.stdin is not None
    sync: asyncio.Task[None] | None = None
    try:
        paused, rows_visible = (await read_worker_line(worker, log_path)).split()
        assert paused == "PAUSED"

        sync = asyncio.create_task(run_vector_sync(reader, note.id, "service"))
        await asyncio.wait_for(probe.window_prepared.wait(), timeout=STEP_TIMEOUT_SECONDS)
        during = await observe_projection(reader, note.id)
        plan = probe.plan_for(note.id)

        worker.stdin.write("RESUME\n")
        worker.stdin.flush()
        probe.release.set()
        assert await read_worker_line(worker, log_path) == "DONE"
        await asyncio.wait_for(sync, timeout=STEP_TIMEOUT_SECONDS)
        exit_code = await asyncio.to_thread(worker.wait, STEP_TIMEOUT_SECONDS)
    finally:
        probe.release.set()
        if worker.poll() is None:
            worker.kill()
            worker.wait()
        if sync is not None and not sync.done():
            sync.cancel()

    assert exit_code == 0, log_path.read_text(encoding="utf-8")
    window = RefreshWindow(rows_visible_to_refresh=int(rows_visible), during=during, plan=plan)
    assert window == unchanged_note_window(before, source_rows=2), (
        "a refresh in another process must read as the last committed projection"
    )
    assert await observe_projection(reader, note.id) == before
    refreshed_hits = await reader.service.search(SearchQuery(text=REFRESHED_TERM))
    assert note.id in hit_entity_ids(refreshed_hits)


if __name__ == "__main__":
    asyncio.run(run_paused_refresh_worker(json.loads(sys.argv[1])))
