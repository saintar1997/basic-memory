"""PostgreSQL: which locks the built-in pgvector prepare write transaction holds.

Evidence for the lock-order analysis of vector preparation. One sync is paused
inside its prepare write transaction, after its chunk upsert and before commit,
and a second connection reads ``pg_locks`` for that backend and probes row locks
with ``FOR UPDATE NOWAIT``. Built-in pgvector preparation must stay lock-light:
table locks no stronger than ROW EXCLUSIVE, row locks only on the entity's own
manifest rows, and nothing on the project, entity, or search_index rows that
other writers serialize on.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import override

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory import db
from basic_memory.config import BasicMemoryConfig, DatabaseBackend
from basic_memory.models import Entity
from basic_memory.repository import EntityRepository
from basic_memory.repository.postgres_search_repository import PostgresSearchRepository
from basic_memory.repository.semantic_chunking import VectorChunkRecord
from basic_memory.repository.semantic_vector_sync import PendingEmbeddingJob, VectorChunkState
from basic_memory.services.file_service import FileService
from basic_memory.services.search_service import SearchService

pytestmark = pytest.mark.postgres

PAUSE_TIMEOUT_SECONDS = 60


class FixedEmbeddingProvider:
    model_name = "prepare-lock-evidence"
    dimensions = 4

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def runtime_log_attrs(self) -> dict[str, object]:
        return {}


class PausedInsidePrepareWrite(PostgresSearchRepository):
    """Hold the prepare write transaction open right after the chunk upsert."""

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        project_id: int,
        *,
        app_config: BasicMemoryConfig,
    ) -> None:
        super().__init__(
            session_maker,
            project_id,
            app_config=app_config,
            embedding_provider=FixedEmbeddingProvider(),
        )
        self.inside_write = asyncio.Event()
        self.release = asyncio.Event()
        self.backend_pid: int | None = None

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
        jobs = await super()._upsert_scheduled_chunk_records(
            session,
            entity_id=entity_id,
            scheduled_records=scheduled_records,
            existing_by_key=existing_by_key,
            entity_fingerprint=entity_fingerprint,
            embedding_model=embedding_model,
        )
        self.backend_pid = int(
            (await session.execute(text("SELECT pg_backend_pid()"))).scalar_one()
        )
        self.inside_write.set()
        await asyncio.wait_for(self.release.wait(), PAUSE_TIMEOUT_SECONDS)
        return jobs


@pytest.fixture(autouse=True)
def _require_postgres_backend(db_backend: str) -> None:
    if db_backend != "postgres":
        pytest.skip("Prepare lock evidence requires BASIC_MEMORY_TEST_POSTGRES=1")


async def _relation_locks(
    session_maker: async_sessionmaker[AsyncSession], backend_pid: int
) -> set[tuple[str, str]]:
    async with db.scoped_session(session_maker) as session:
        result = await session.execute(
            text(
                "SELECT c.relname, l.mode FROM pg_locks l "
                "JOIN pg_class c ON c.oid = l.relation "
                "WHERE l.pid = :pid AND l.locktype = 'relation' AND l.granted "
                "AND c.relnamespace = current_schema()::regnamespace"
            ),
            {"pid": backend_pid},
        )
        return {(str(row[0]), str(row[1])) for row in result.fetchall()}


async def _row_lock_is_free(
    session_maker: async_sessionmaker[AsyncSession],
    statement: str,
    params: dict[str, object],
) -> bool:
    """Try to lock the rows without waiting; report whether nobody holds them."""
    async with session_maker() as session:
        try:
            await session.execute(text(statement + " FOR UPDATE NOWAIT"), params)
        except DBAPIError as exc:
            assert "could not obtain lock" in str(exc)
            return False
        finally:
            await session.rollback()
    return True


def _tables(locks: Sequence[tuple[str, str]] | set[tuple[str, str]], mode: str) -> set[str]:
    return {relname for relname, lock_mode in locks if lock_mode == mode}


@pytest.mark.asyncio
async def test_builtin_prepare_write_takes_no_lock_beyond_its_manifest_rows(
    session_maker: async_sessionmaker[AsyncSession],
    test_project,
    entity_repository: EntityRepository,
    file_service: FileService,
    sample_entity: Entity,
) -> None:
    config = BasicMemoryConfig(
        env="test",
        projects={"test-project": "/tmp/basic-memory-test"},
        default_project="test-project",
        database_backend=DatabaseBackend.POSTGRES,
        semantic_search_enabled=True,
    )
    repository = PausedInsidePrepareWrite(session_maker, test_project.id, app_config=config)
    await repository.init_search_index()
    service = SearchService(repository, entity_repository, file_service, session_maker)
    async with db.scoped_session(session_maker) as session:
        entity = await entity_repository.find_by_id(session, sample_entity.id)
    assert entity is not None

    # Embed once, so the paused sync updates an existing (visible) manifest row
    # rather than inserting one no other transaction can see yet.
    await service.index_entity(entity, content="Quartz lock evidence note.")
    repository.release.set()
    await repository.sync_entity_vectors(entity.id)
    repository.inside_write.clear()
    repository.release.clear()
    await service.index_entity(entity, content="Quartz lock evidence note, edited.")

    sync = asyncio.create_task(repository.sync_entity_vectors(entity.id))
    try:
        await asyncio.wait_for(repository.inside_write.wait(), PAUSE_TIMEOUT_SECONDS)
        assert repository.backend_pid is not None
        locks = await _relation_locks(session_maker, repository.backend_pid)
        entity_row_free = await _row_lock_is_free(
            session_maker, "SELECT id FROM entity WHERE id = :id", {"id": entity.id}
        )
        search_rows_free = await _row_lock_is_free(
            session_maker,
            "SELECT id FROM search_index WHERE entity_id = :id",
            {"id": entity.id},
        )
        project_row_free = await _row_lock_is_free(
            session_maker, "SELECT id FROM project WHERE id = :id", {"id": test_project.id}
        )
        manifest_rows_free = await _row_lock_is_free(
            session_maker,
            "SELECT id FROM search_vector_chunks WHERE entity_id = :id",
            {"id": entity.id},
        )
        repository.release.set()
        await asyncio.wait_for(sync, PAUSE_TIMEOUT_SECONDS)
    finally:
        repository.release.set()
        if not sync.done():
            sync.cancel()
            await asyncio.gather(sync, return_exceptions=True)

    print(f"prepare write relation locks: {sorted(locks)}")
    # Table locks: only ACCESS SHARE (reads) and ROW EXCLUSIVE (row writes), which
    # conflict with each other never and with DDL only.
    assert {mode for _relname, mode in locks} <= {"AccessShareLock", "RowExclusiveLock"}
    assert _tables(locks, "RowExclusiveLock") <= {
        "search_vector_chunks",
        "search_vector_chunks_id_seq",
        "search_vector_chunks_pkey",
        "idx_search_vector_chunks_project_entity",
        "search_vector_chunks_project_id_entity_id_chunk_key_key",
    }
    assert "project" not in {relname for relname, _mode in locks}
    # Row locks: the manifest rows it upserted, and nothing another writer orders on.
    assert entity_row_free
    assert search_rows_free
    assert project_row_free
    assert not manifest_rows_free
