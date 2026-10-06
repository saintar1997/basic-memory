"""A note stays searchable, with one search projection, while other writers refresh it.

Regression tests for behavior main has provided since #1623, which replaces an entity's
search rows in one transaction. A deployment on 0.23.2, with several MCP clients writing
one SQLite project, hit both failures that shape prevents:

- the old rows were deleted in a transaction of their own, so a vector sync that ran
  before the replacement committed read the note as deleted and removed its vectors;
- concurrent refreshes of one note each appended a copy of its rows.

Each test holds one refresh inside its transaction, after its delete and before its
replacement rows, and checks what the other writers see and leave behind, on file-backed
SQLite and on PostgreSQL.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from basic_memory import db
from basic_memory.config import BasicMemoryConfig, DatabaseBackend
from basic_memory.models import Entity, Observation
from basic_memory.repository import EntityRepository
from basic_memory.repository.postgres_search_repository import PostgresSearchRepository
from basic_memory.repository.search_repository_base import SearchRepositoryBase
from basic_memory.repository.sqlite_search_repository import SQLiteSearchRepository
from basic_memory.schemas.search import SearchQuery, SearchRetrievalMode
from basic_memory.services.file_service import FileService
from basic_memory.services.search_service import SearchService

type NoteKind = Literal["markdown", "file"]
type RowKey = tuple[str, int, str | None]

STEP_TIMEOUT_SECONDS = 30
ORIGINAL = "Quartz repair notes stay searchable while another worker refreshes them."
REFRESHED = "Quartz repair notes, refreshed by a lapidary."


class StubEmbeddingProvider:
    """One unit vector for every text, so any vector query matches every chunk."""

    model_name = "refresh-concurrency-stub"
    dimensions = 4

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def runtime_log_attrs(self) -> dict[str, object]:
        return {}


@dataclass(frozen=True)
class Worker:
    """One writer's own repository and service, as each request or MCP process builds."""

    service: SearchService
    repository: SearchRepositoryBase


def build_worker(
    session_maker: async_sessionmaker[AsyncSession],
    project_id: int,
    project_path: Path,
    config: BasicMemoryConfig,
) -> Worker:
    repository_type = (
        PostgresSearchRepository
        if config.database_backend == DatabaseBackend.POSTGRES
        else SQLiteSearchRepository
    )
    repository = repository_type(
        session_maker,
        project_id,
        app_config=config,
        embedding_provider=StubEmbeddingProvider(),
    )
    service = SearchService(
        repository,
        EntityRepository(project_id=project_id),
        FileService(project_path),
        session_maker,
    )
    return Worker(service=service, repository=repository)


async def seed_note(
    session_maker: async_sessionmaker[AsyncSession], project_id: int, kind: NoteKind
) -> Entity:
    """A Markdown note with one observation, or a non-Markdown file without a permalink."""
    now = datetime.now(timezone.utc)
    async with db.scoped_session(session_maker) as session:
        if kind == "markdown":
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
        else:
            note = Entity(
                project_id=project_id,
                title="Quartz repair scan",
                note_type="file",
                permalink=None,
                file_path="files/quartz-repair-scan.pdf",
                content_type="application/pdf",
                created_at=now,
                updated_at=now,
            )
        session.add(note)
        await session.flush()
        if kind == "markdown":
            session.add(
                Observation(
                    project_id=project_id,
                    entity_id=note.id,
                    category="fact",
                    content="Quartz repairs need a steady hand",
                )
            )
        note_id = note.id

    async with db.scoped_session(session_maker) as session:
        entity = await EntityRepository(project_id=project_id).find_by_id(session, note_id)
    assert entity is not None
    return entity


def content_for(kind: NoteKind, body: str) -> str | None:
    """Markdown refreshes carry their text; a file is indexed from its metadata alone."""
    return body if kind == "markdown" else None


def hold_replacement(
    monkeypatch: pytest.MonkeyPatch, worker: Worker, kind: NoteKind
) -> tuple[asyncio.Event, asyncio.Event]:
    """Pause the worker's refresh after its delete, inside its still-open transaction.

    Markdown notes write their replacement through ``bulk_index_items`` and files
    through ``index_item``; both run in ``index_entity_data``'s transaction.
    """
    paused, resume = asyncio.Event(), asyncio.Event()
    method_name = "bulk_index_items" if kind == "markdown" else "index_item"
    write_replacement = getattr(worker.repository, method_name)

    async def write_after_resume(replacement, session: AsyncSession | None = None) -> None:
        paused.set()
        await asyncio.wait_for(resume.wait(), timeout=STEP_TIMEOUT_SECONDS)
        await write_replacement(replacement, session)

    monkeypatch.setattr(worker.repository, method_name, write_after_resume)
    return paused, resume


def watch_prepare_window(monkeypatch: pytest.MonkeyPatch, worker: Worker) -> asyncio.Event:
    """Signal once the worker's vector sync has read and planned the note."""
    planned = asyncio.Event()
    prepare_window = worker.repository._prepare_entity_vector_jobs_window

    async def signalling_window(entity_ids: list[int]):
        prepared = await prepare_window(entity_ids)
        planned.set()
        return prepared

    monkeypatch.setattr(worker.repository, "_prepare_entity_vector_jobs_window", signalling_window)
    return planned


@contextmanager
def count_projection_deletes(engine: AsyncEngine, expected: int) -> Iterator[asyncio.Event]:
    """Signal once ``expected`` refreshes have sent the delete that opens their transaction."""
    all_sent = asyncio.Event()
    sent = 0

    def on_execute(conn, cursor, statement: str, parameters, context, executemany) -> None:
        nonlocal sent
        if statement.startswith("DELETE FROM search_index WHERE entity_id"):
            sent += 1
            if sent == expected:
                all_sent.set()

    event.listen(engine.sync_engine, "before_cursor_execute", on_execute)
    try:
        yield all_sent
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", on_execute)


async def entity_ids_found(worker: Worker, mode: SearchRetrievalMode) -> set[int | None]:
    rows = await worker.repository.search(
        search_text="quartz", retrieval_mode=mode, min_similarity=0.0
    )
    return {row.entity_id for row in rows}


async def projection(
    session_maker: async_sessionmaker[AsyncSession], project_id: int, entity_id: int
) -> list[RowKey]:
    """Every stored search row the entity owns, duplicates included."""
    async with db.scoped_session(session_maker) as session:
        result = await session.execute(
            text(
                "SELECT type, id, permalink FROM search_index "
                "WHERE project_id = :project_id AND entity_id = :entity_id "
                "ORDER BY type, id"
            ),
            {"project_id": project_id, "entity_id": entity_id},
        )
        return [(str(kind), int(row_id), permalink) for kind, row_id, permalink in result.all()]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["markdown", "file"])
async def test_vector_sync_during_a_refresh_keeps_the_note_and_its_vectors(
    engine_factory, test_project, app_config, monkeypatch, kind: NoteKind
):
    _engine, session_maker = engine_factory
    config = app_config.model_copy(update={"semantic_search_enabled": True})
    writer, reader = (
        build_worker(session_maker, test_project.id, Path(test_project.path), config)
        for _ in range(2)
    )
    # Database initialization creates vector storage, as at startup.
    await writer.service.init_search_index()
    note = await seed_note(session_maker, test_project.id, kind)
    await writer.service.index_entity_data(note, content=content_for(kind, ORIGINAL))
    await writer.service.sync_entity_vectors(note.id)
    manifest = await reader.repository.get_entity_chunk_manifest(note.id)
    assert manifest

    paused, resume = hold_replacement(monkeypatch, writer, kind)
    planned = watch_prepare_window(monkeypatch, reader)
    refresh = asyncio.create_task(
        writer.service.index_entity_data(note, content=content_for(kind, REFRESHED))
    )
    await asyncio.wait_for(paused.wait(), timeout=STEP_TIMEOUT_SECONDS)

    # The refresh has deleted the old rows in its open transaction. Every other reader
    # still sees the last committed projection.
    assert note.id in await entity_ids_found(reader, SearchRetrievalMode.FTS)
    found = await reader.service.search(SearchQuery(text="quartz"))
    assert note.id in {row.entity_id for row in found}
    sync = asyncio.create_task(reader.service.sync_entity_vectors(note.id))
    # Trigger: the sync has read and planned while the refresh is still open.
    # Why: on SQLite its remaining writes queue behind the refresh's write lock.
    # Outcome: let the refresh commit, then let the sync finish.
    await asyncio.wait_for(planned.wait(), timeout=STEP_TIMEOUT_SECONDS)
    resume.set()
    await asyncio.wait_for(refresh, timeout=STEP_TIMEOUT_SECONDS)
    await asyncio.wait_for(sync, timeout=STEP_TIMEOUT_SECONDS)

    # The sync planned from an unchanged note, so the vectors it found are still there.
    assert await reader.repository.get_entity_chunk_manifest(note.id) == manifest
    assert note.id in await entity_ids_found(reader, SearchRetrievalMode.VECTOR)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["markdown", "file"])
async def test_refreshes_queued_behind_an_open_refresh_leave_one_projection(
    engine_factory, test_project, app_config, monkeypatch, kind: NoteKind
):
    engine, session_maker = engine_factory
    if kind == "file" and engine.dialect.name == "postgresql":
        # Constraint: at READ COMMITTED a queued delete cannot see the row the first
        #   writer inserted, and a row without a permalink has no upsert target, so the
        #   queued insert fails on search_index_pkey instead of duplicating the row. The
        #   projection stays the first writer's until the next refresh.
        pytest.skip("PostgreSQL rejects a queued file refresh rather than duplicating it")
    first, *queued = (
        build_worker(session_maker, test_project.id, Path(test_project.path), app_config)
        for _ in range(4)
    )
    note = await seed_note(session_maker, test_project.id, kind)
    await first.service.index_entity_data(note, content=content_for(kind, ORIGINAL))
    expected = await projection(session_maker, test_project.id, note.id)

    paused, resume = hold_replacement(monkeypatch, first, kind)
    with count_projection_deletes(engine, expected=4) as all_deletes_sent:
        refreshes = [
            asyncio.create_task(
                first.service.index_entity_data(note, content=content_for(kind, ORIGINAL))
            )
        ]
        await asyncio.wait_for(paused.wait(), timeout=STEP_TIMEOUT_SECONDS)
        # The other writers start while the first one holds its replacement open; each
        # sends its delete and waits behind the first one's lock.
        refreshes += [
            asyncio.create_task(
                worker.service.index_entity_data(note, content=content_for(kind, ORIGINAL))
            )
            for worker in queued
        ]
        await asyncio.wait_for(all_deletes_sent.wait(), timeout=STEP_TIMEOUT_SECONDS)
        resume.set()
        outcomes = await asyncio.wait_for(
            asyncio.gather(*refreshes, return_exceptions=True), timeout=STEP_TIMEOUT_SECONDS
        )

    assert [outcome for outcome in outcomes if outcome is not None] == []
    assert await projection(session_maker, test_project.id, note.id) == expected
