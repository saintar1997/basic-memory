"""PostgreSQL characterization: the window a lock-free re-read cannot close.

Re-reading the source and manifest inside the prepare write transaction moves the
staleness check next to the write, but at READ COMMITTED nothing stops a newer
generation from committing between that re-read and the chunk upsert. This test
pauses a sync exactly there -- inside its write transaction, after the re-read,
before ``_upsert_scheduled_chunk_records`` -- and lets the note move on. It
documents the limit of a fix that adds no lock; it is expected to fail on main
and on the lock-free candidate alike.
"""

from __future__ import annotations

import asyncio
from typing import Literal, override

import pytest
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory import db
from basic_memory.config import BasicMemoryConfig, DatabaseBackend
from basic_memory.models import Entity
from basic_memory.repository import EntityRepository
from basic_memory.repository.postgres_search_repository import PostgresSearchRepository
from basic_memory.repository.semantic_chunking import VectorChunkRecord
from basic_memory.repository.semantic_vector_sync import PendingEmbeddingJob, VectorChunkState
from basic_memory.schemas.search import SearchRetrievalMode
from basic_memory.services.file_service import FileService
from basic_memory.services.search_service import SearchService

pytestmark = pytest.mark.postgres

PAUSE_TIMEOUT_SECONDS = 60


class FixedEmbeddingProvider:
    model_name = "residual-window-evidence"
    dimensions = 4

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def runtime_log_attrs(self) -> dict[str, object]:
        return {}


class PausedBeforeChunkUpsert(PostgresSearchRepository):
    """Pause inside the prepare write transaction, after any re-read, before the upsert."""

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
        self.before_upsert = asyncio.Event()
        self.resume_upsert = asyncio.Event()

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
        self.before_upsert.set()
        await asyncio.wait_for(self.resume_upsert.wait(), PAUSE_TIMEOUT_SECONDS)
        return await super()._upsert_scheduled_chunk_records(
            session,
            entity_id=entity_id,
            scheduled_records=scheduled_records,
            existing_by_key=existing_by_key,
            entity_fingerprint=entity_fingerprint,
            embedding_model=embedding_model,
        )


@pytest.fixture(autouse=True)
def _require_postgres_backend(db_backend: str) -> None:
    if db_backend != "postgres":
        pytest.skip("Residual window evidence requires BASIC_MEMORY_TEST_POSTGRES=1")


async def _manifest_texts(
    session_maker: async_sessionmaker[AsyncSession], project_id: int, entity_id: int
) -> list[tuple[str, str]]:
    async with db.scoped_session(session_maker) as session:
        result = await session.execute(
            text(
                "SELECT chunk_text, embedding_status FROM search_vector_chunks "
                "WHERE project_id = :project_id AND entity_id = :entity_id ORDER BY chunk_key"
            ),
            {"project_id": project_id, "entity_id": entity_id},
        )
        return [(str(row[0]), str(row[1])) for row in result.fetchall()]


@pytest.mark.asyncio
@pytest.mark.parametrize("latest_change", ["update", "opt_out"])
async def test_change_committed_between_reread_and_upsert(
    session_maker: async_sessionmaker[AsyncSession],
    test_project,
    entity_repository: EntityRepository,
    file_service: FileService,
    sample_entity: Entity,
    latest_change: Literal["update", "opt_out"],
) -> None:
    config = BasicMemoryConfig(
        env="test",
        projects={"test-project": "/tmp/basic-memory-test"},
        default_project="test-project",
        database_backend=DatabaseBackend.POSTGRES,
        semantic_search_enabled=True,
    )
    current_repository = PostgresSearchRepository(
        session_maker,
        test_project.id,
        app_config=config,
        embedding_provider=FixedEmbeddingProvider(),
    )
    paused_repository = PausedBeforeChunkUpsert(session_maker, test_project.id, app_config=config)
    await current_repository.init_search_index()
    current = SearchService(current_repository, entity_repository, file_service, session_maker)
    async with db.scoped_session(session_maker) as session:
        entity = await entity_repository.find_by_id(session, sample_entity.id)
    assert entity is not None

    await current.index_entity(entity, content="Quartz original wording.")
    await current.sync_entity_vectors(entity.id)
    await current.index_entity(entity, content="Quartz older update awaiting embedding.")

    pending = asyncio.create_task(paused_repository.sync_entity_vectors(entity.id))
    try:
        await asyncio.wait_for(paused_repository.before_upsert.wait(), PAUSE_TIMEOUT_SECONDS)
        match latest_change:
            case "update":
                await current.index_entity(entity, content="Quartz newest update.")
                await current.sync_entity_vectors(entity.id)
            case "opt_out":
                async with db.scoped_session(session_maker) as session:
                    await session.execute(
                        update(Entity)
                        .where(Entity.id == entity.id)
                        .values(entity_metadata={"embed": False})
                    )
                await current.sync_entity_vectors(entity.id)
        latest = await _manifest_texts(session_maker, test_project.id, entity.id)
        paused_repository.resume_upsert.set()
        await asyncio.wait_for(pending, PAUSE_TIMEOUT_SECONDS)
    finally:
        paused_repository.resume_upsert.set()
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    assert await _manifest_texts(session_maker, test_project.id, entity.id) == latest
    hits = await current_repository.search(
        search_text="quartz", retrieval_mode=SearchRetrievalMode.VECTOR, min_similarity=0.0
    )
    expected = [entity.id] if latest_change == "update" else []
    assert sorted({hit.entity_id for hit in hits if hit.entity_id is not None}) == expected
