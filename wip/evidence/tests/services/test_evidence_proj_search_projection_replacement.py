"""Refreshing a note replaces its whole search projection, and only its own.

#1623 moved the delete of an entity's old search rows into the transaction that writes
the replacement rows (``SearchService.index_entity_data``). Every production write of
``search_index`` reaches the repository through that method, so the one transaction is
responsible for all of the following, on SQLite and on PostgreSQL:

- rows the note no longer owns disappear: a dropped observation, a dropped relation,
  every row under a renamed permalink, and -- when a Markdown note becomes a plain
  file -- all of its observation and relation rows;
- on PostgreSQL the bounded full-text chunks follow their parent rows (ON DELETE
  CASCADE) and are rebuilt for the replacement rows;
- other notes and other projects keep their rows, and a refresh that fails rolls back
  to the previous projection exactly;
- a failed content read, which happens before the transaction opens, changes nothing:
  the previous rows and their vectors stay searchable;
- an embedding opt-out removes the note's vectors on its next sync but keeps its
  full-text rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory import db
from basic_memory.config import BasicMemoryConfig
from basic_memory.models import Entity
from basic_memory.repository import EntityRepository
from basic_memory.repository.postgres_search_repository import PostgresSearchRepository
from basic_memory.repository.project_repository import ProjectRepository
from basic_memory.repository.search_index_row import SearchIndexRow
from basic_memory.repository.search_repository_base import SearchRepositoryBase
from basic_memory.repository.sqlite_search_repository import SQLiteSearchRepository
from basic_memory.schemas.search import SearchItemType, SearchQuery, SearchRetrievalMode
from basic_memory.services.file_service import FileService
from basic_memory.services.search_service import SearchService

type Backend = Literal["sqlite", "postgres"]
type RowSnapshot = tuple[int, str, int, int | None, str | None, str | None]
type ChunkSnapshot = tuple[int, int, str, int, str]


class ConstantEmbeddingProvider:
    """Embed every text to one unit vector, so any vector query matches every chunk."""

    model_name = "evidence-projection-constant"
    dimensions = 4

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def runtime_log_attrs(self) -> dict[str, object]:
        return {}


class ReplacementFailed(RuntimeError):
    """Stands in for the asyncpg statement timeout seen while writing replacement rows."""


@dataclass(frozen=True, slots=True)
class SearchState:
    """Every search_index row and full-text chunk in the database, across projects."""

    rows: tuple[RowSnapshot, ...]
    chunks: tuple[ChunkSnapshot, ...]


@dataclass(frozen=True, slots=True)
class VectorState:
    """One note's vector manifest and the embeddings actually stored for it."""

    manifest: tuple[tuple[str, str, str], ...]
    stored_embedding_keys: frozenset[str]


# --- Fixtures and helpers ---


def search_repository_for(
    db_backend: Backend,
    session_maker: async_sessionmaker[AsyncSession],
    project_id: int,
    config: BasicMemoryConfig,
    *,
    semantic: bool,
) -> SearchRepositoryBase:
    """Build the backend's repository; semantic repositories embed with the constant stub."""
    provider = ConstantEmbeddingProvider() if semantic else None
    if db_backend == "postgres":
        return PostgresSearchRepository(
            session_maker, project_id, app_config=config, embedding_provider=provider
        )
    return SQLiteSearchRepository(
        session_maker, project_id, app_config=config, embedding_provider=provider
    )


@pytest_asyncio.fixture
async def semantic_search_service(
    db_backend: Backend,
    app_config: BasicMemoryConfig,
    session_maker: async_sessionmaker[AsyncSession],
    test_project,
    entity_repository: EntityRepository,
    file_service: FileService,
) -> SearchService:
    """A search service whose repository embeds notes, initialized like a worker start."""
    if db_backend == "sqlite":
        pytest.importorskip("sqlite_vec")
    config = app_config.model_copy(
        update={"semantic_search_enabled": True, "semantic_min_similarity": 0.0}
    )
    repository = search_repository_for(
        db_backend, session_maker, test_project.id, config, semantic=True
    )
    service = SearchService(repository, entity_repository, file_service, session_maker)
    await service.init_search_index()
    return service


async def reload(
    session_maker: async_sessionmaker[AsyncSession],
    entity_repository: EntityRepository,
    entity_id: int,
) -> Entity:
    """Load a note with the observations and relations its projection is built from."""
    async with db.scoped_session(session_maker) as session:
        entity = await entity_repository.find_by_id(session, entity_id)
    assert entity is not None
    return entity


async def projection(service: SearchService, entity_id: int) -> list[tuple[str, int, str | None]]:
    rows = await service.repository.get_entity_search_rows(entity_id)
    return sorted((row.type, row.id, row.permalink) for row in rows)


async def search_state(
    session_maker: async_sessionmaker[AsyncSession], db_backend: Backend
) -> SearchState:
    """Snapshot the shared search tables, so untouched rows can be compared exactly."""
    async with db.scoped_session(session_maker) as session:
        rows = await session.execute(
            text(
                "SELECT project_id, type, id, entity_id, permalink, content_snippet "
                "FROM search_index ORDER BY project_id, type, id"
            )
        )
        chunks = (
            await session.execute(
                text(
                    "SELECT project_id, search_index_id, search_index_type, chunk_index, "
                    "chunk_text FROM search_index_fts_chunks "
                    "ORDER BY project_id, search_index_type, search_index_id, chunk_index"
                )
            )
            if db_backend == "postgres"
            else None
        )
        return SearchState(
            rows=tuple(
                (
                    int(row.project_id),
                    str(row.type),
                    int(row.id),
                    row.entity_id,
                    row.permalink,
                    row.content_snippet,
                )
                for row in rows
            ),
            chunks=tuple(
                (
                    int(chunk.project_id),
                    int(chunk.search_index_id),
                    str(chunk.search_index_type),
                    int(chunk.chunk_index),
                    str(chunk.chunk_text),
                )
                for chunk in (chunks or ())
            ),
        )


async def vector_state(service: SearchService, entity_id: int) -> VectorState:
    repository = service.repository
    assert isinstance(repository, SearchRepositoryBase)
    manifest = await repository.get_entity_chunk_manifest(entity_id)
    stored = await repository.get_entity_physical_chunk_keys(entity_id)
    return VectorState(
        manifest=tuple((row.chunk_key, row.source_hash, row.embedding_status) for row in manifest),
        stored_embedding_keys=frozenset(stored or ()),
    )


async def found_by(service: SearchService, query: SearchQuery, entity_id: int) -> bool:
    return entity_id in {hit.entity_id for hit in await service.search(query)}


def fail_refresh_after_replacement_rows(
    monkeypatch: pytest.MonkeyPatch, service: SearchService, db_backend: Backend
) -> None:
    """Make the next refresh fail after its delete and replacement rows ran, before commit."""
    repository = service.repository
    if db_backend == "postgres":
        # The #1621 failure: full-text chunk writes timing out under load.
        async def chunks_time_out(session: AsyncSession, rows: list[SearchIndexRow]) -> None:
            raise ReplacementFailed("canceling statement due to statement timeout")

        monkeypatch.setattr(repository, "_replace_fts_chunks", chunks_time_out)
        return

    write_rows = repository.bulk_index_items

    async def write_then_fail(
        search_index_rows: list[SearchIndexRow], session: AsyncSession | None = None
    ) -> None:
        await write_rows(search_index_rows, session)
        raise ReplacementFailed("replacement write timed out")

    monkeypatch.setattr(repository, "bulk_index_items", write_then_fail)


# --- Exact replacement (B7) ---


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["committed", "rolled_back"])
async def test_refresh_replaces_only_the_refreshed_notes_projection(
    monkeypatch,
    outcome: Literal["committed", "rolled_back"],
    db_backend: Backend,
    app_config: BasicMemoryConfig,
    search_service: SearchService,
    full_entity: Entity,
    sample_entity: Entity,
    entity_repository: EntityRepository,
    observation_repository,
    relation_repository,
    session_maker: async_sessionmaker[AsyncSession],
    tmp_path: Path,
):
    # --- Two indexed notes in this project, one row in a sibling project ---
    note = await reload(session_maker, entity_repository, full_entity.id)
    original_permalink = note.permalink
    assert original_permalink is not None
    neighbour = await reload(session_maker, entity_repository, sample_entity.id)
    await search_service.index_entity_data(note, content="original searchable body")
    await search_service.index_entity_data(neighbour, content="neighbour body")
    assert [kind for kind, _id, _permalink in await projection(search_service, note.id)] == [
        "entity",
        "observation",
        "observation",
        "relation",
        "relation",
    ]

    # Entity ids are unique across projects, so a sibling project can only share this
    # note's entity_id through a hand-written row. It makes the delete's project scope
    # observable: an unscoped delete by entity_id would take this row with it.
    async with db.scoped_session(session_maker) as session:
        sibling_project = await ProjectRepository().create(
            session,
            {
                "name": "sibling-project",
                "description": "Shares the search tables with test-project",
                "path": str(tmp_path / "sibling-project"),
                "is_active": True,
                "is_default": False,
            },
        )
    sibling_repository = search_repository_for(
        db_backend, session_maker, sibling_project.id, app_config, semantic=False
    )
    now = datetime.now(timezone.utc)
    await sibling_repository.bulk_index_items(
        [
            SearchIndexRow(
                project_id=sibling_project.id,
                id=note.id,
                type=SearchItemType.ENTITY.value,
                entity_id=note.id,
                title="Sibling project note",
                permalink=original_permalink,
                file_path=note.file_path,
                content_stems="sibling project body",
                content_snippet="sibling project body",
                created_at=now,
                updated_at=now,
            )
        ]
    )
    before = await search_state(session_maker, db_backend)
    if db_backend == "postgres":
        sibling_chunk = (sibling_project.id, note.id, "entity", 0, "sibling project body")
        assert sibling_chunk in before.chunks
        assert {chunk[2] for chunk in before.chunks if chunk[0] != sibling_project.id} == {
            "entity",
            "observation",
        }

    # --- The note drops an observation and a relation, and its permalink moves ---
    kept_observation, dropped_observation = sorted(
        note.observations, key=lambda observation: observation.category != "tech"
    )
    kept_relation, dropped_relation = sorted(
        note.outgoing_relations, key=lambda relation: relation.relation_type != "out1"
    )
    async with db.scoped_session(session_maker) as session:
        await observation_repository.delete(session, dropped_observation.id)
        await relation_repository.delete(session, dropped_relation.id)
        await entity_repository.update(
            session, note.id, {"permalink": "test/search-entity-renamed"}
        )
    note = await reload(session_maker, entity_repository, note.id)

    if outcome == "rolled_back":
        fail_refresh_after_replacement_rows(monkeypatch, search_service, db_backend)
        with pytest.raises(ReplacementFailed):
            await search_service.index_entity_data(note, content="replacement body")
        assert await search_state(session_maker, db_backend) == before
        return

    await search_service.index_entity_data(note, content="replacement body")

    # --- Exactly the current projection, under the new permalink ---
    [kept_observation] = [obs for obs in note.observations if obs.id == kept_observation.id]
    [kept_relation] = [rel for rel in note.outgoing_relations if rel.id == kept_relation.id]
    assert await projection(search_service, note.id) == sorted(
        [
            ("entity", note.id, "test/search-entity-renamed"),
            ("observation", kept_observation.id, kept_observation.permalink),
            ("relation", kept_relation.id, kept_relation.permalink),
        ]
    )
    assert not await found_by(search_service, SearchQuery(permalink=original_permalink), note.id)
    assert await found_by(
        search_service, SearchQuery(permalink="test/search-entity-renamed"), note.id
    )

    # --- Everything the note does not own is untouched ---
    after = await search_state(session_maker, db_backend)
    test_project_id = search_service.repository.project_id

    def owned_by_note(row: RowSnapshot) -> bool:
        project_id, _kind, _row_id, entity_id, _permalink, _snippet = row
        return project_id == test_project_id and entity_id == note.id

    note_keys = {
        (row[1], row[2]) for state in (before, after) for row in state.rows if owned_by_note(row)
    }

    def chunk_owned_by_note(chunk: ChunkSnapshot) -> bool:
        project_id, row_id, kind, _index, _text = chunk
        return project_id == test_project_id and (kind, row_id) in note_keys

    assert [row for row in after.rows if not owned_by_note(row)] == [
        row for row in before.rows if not owned_by_note(row)
    ]
    assert [chunk for chunk in after.chunks if not chunk_owned_by_note(chunk)] == [
        chunk for chunk in before.chunks if not chunk_owned_by_note(chunk)
    ]
    sibling_row = (
        sibling_project.id,
        "entity",
        note.id,
        note.id,
        original_permalink,
        "sibling project body",
    )
    assert sibling_row in after.rows

    # PostgreSQL rebuilds the note's bounded full-text chunks for its current rows only:
    # the dropped observation's chunk cascaded away with its parent row.
    if db_backend == "postgres":
        assert sorted(chunk for chunk in after.chunks if chunk_owned_by_note(chunk)) == sorted(
            [
                (test_project_id, note.id, "entity", 0, "replacement body"),
                (test_project_id, kept_observation.id, "observation", 0, kept_observation.content),
            ]
        )


# --- Markdown note becomes a file (B1b) ---


@pytest.mark.asyncio
async def test_markdown_note_refreshed_as_a_file_keeps_only_its_entity_row(
    db_backend: Backend,
    search_service: SearchService,
    full_entity: Entity,
    entity_repository: EntityRepository,
    session_maker: async_sessionmaker[AsyncSession],
):
    note = await reload(session_maker, entity_repository, full_entity.id)
    await search_service.index_entity_data(note, content="original body")
    assert await found_by(search_service, SearchQuery(text="Tech note"), note.id)
    markdown_state = await search_state(session_maker, db_backend)
    note_keys = {
        (kind, row_id)
        for project_id, kind, row_id, entity_id, _permalink, _snippet in markdown_state.rows
        if project_id == search_service.repository.project_id and entity_id == note.id
    }
    if db_backend == "postgres":
        assert {(kind, row_id) for _p, row_id, kind, _i, _t in markdown_state.chunks} == {
            key for key in note_keys if key[0] != "relation"
        }

    async with db.scoped_session(session_maker) as session:
        await entity_repository.update(
            session,
            note.id,
            {"content_type": "application/pdf", "file_path": "test/Search_Entity.pdf"},
        )
    file_entity = await reload(session_maker, entity_repository, note.id)
    assert not file_entity.is_markdown
    # The graph rows still exist; a file's projection simply does not include them.
    assert file_entity.observations and file_entity.outgoing_relations

    await search_service.index_entity_data(file_entity)

    assert await projection(search_service, note.id) == [("entity", note.id, file_entity.permalink)]
    assert not await found_by(search_service, SearchQuery(text="Tech note"), note.id)
    # A file row carries no body text, so on PostgreSQL every chunk the Markdown
    # projection owned is gone with its parent row and none replaces it.
    assert (await search_state(session_maker, db_backend)).chunks == ()


# --- Failed content read (B1c) ---


@pytest.mark.asyncio
async def test_failed_content_read_keeps_previous_projection_and_vectors(
    semantic_search_service: SearchService,
    sample_entity: Entity,
    entity_repository: EntityRepository,
    session_maker: async_sessionmaker[AsyncSession],
    file_service: FileService,
):
    service = semantic_search_service
    note = await reload(session_maker, entity_repository, sample_entity.id)
    await service.index_entity_data(note, content="Quartz repair notes stay searchable.")
    await service.sync_entity_vectors(note.id)
    rows_before = await service.repository.get_entity_search_rows(note.id)
    vectors_before = await vector_state(service, note.id)
    assert rows_before and vectors_before.manifest and vectors_before.stored_embedding_keys

    # sample_entity has a database row but no file, like a note whose storage read fails.
    assert not file_service.get_entity_path(note).exists()
    with pytest.raises(FileNotFoundError):
        await service.index_entity_data(note)

    assert await service.repository.get_entity_search_rows(note.id) == rows_before
    assert await vector_state(service, note.id) == vectors_before
    assert await found_by(service, SearchQuery(text="quartz"), note.id)
    vector_query = SearchQuery(text="quartz", retrieval_mode=SearchRetrievalMode.VECTOR)
    assert await found_by(service, vector_query, note.id)

    # The next vector sync sees the unchanged rows and keeps the vectors.
    await service.sync_entity_vectors(note.id)
    assert await vector_state(service, note.id) == vectors_before


# --- Embedding opt-out (B1e) ---


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["single", "batch"])
async def test_embedding_opt_out_clears_vectors_but_keeps_full_text(
    entry_point: Literal["single", "batch"],
    semantic_search_service: SearchService,
    sample_entity: Entity,
    entity_repository: EntityRepository,
    session_maker: async_sessionmaker[AsyncSession],
):
    service = semantic_search_service
    note = await reload(session_maker, entity_repository, sample_entity.id)
    await service.index_entity_data(note, content="Quartz repair notes stay searchable.")
    await service.sync_entity_vectors(note.id)
    rows_before = await service.repository.get_entity_search_rows(note.id)
    vector_query = SearchQuery(text="quartz", retrieval_mode=SearchRetrievalMode.VECTOR)
    assert (await vector_state(service, note.id)).manifest
    assert await found_by(service, vector_query, note.id)

    async with db.scoped_session(session_maker) as session:
        await entity_repository.update(session, note.id, {"entity_metadata": {"embed": False}})

    if entry_point == "single":
        await service.sync_entity_vectors(note.id)
    else:
        result = await service.sync_entity_vectors_batch([note.id])
        assert (result.entities_failed, result.entities_skipped) == (0, 1)

    assert await vector_state(service, note.id) == VectorState(
        manifest=(), stored_embedding_keys=frozenset()
    )
    assert not await found_by(service, vector_query, note.id)
    assert await service.repository.get_entity_search_rows(note.id) == rows_before
    assert await found_by(service, SearchQuery(text="quartz"), note.id)
