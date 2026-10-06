import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from basic_memory.config import BasicMemoryConfig
from basic_memory.markdown.entity_parser import EntityParser
from basic_memory.markdown.markdown_processor import MarkdownProcessor
from basic_memory.models import Entity, Observation, Project
from basic_memory.models.base import Base
from basic_memory.repository import EntityRepository
from basic_memory.repository.sqlite_search_repository import SQLiteSearchRepository
from basic_memory.schemas.search import SearchRetrievalMode
from basic_memory.services.file_service import FileService
from basic_memory.services.search_service import SearchService


class SyntheticEmbeddings:
    model_name = "synthetic-index-presence"
    dimensions = 4

    async def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def runtime_log_attrs(self):
        return {}


class PausedRefresh(SearchService):
    """Pause row construction to allow another real indexer to interleave."""

    async def index_entity_markdown(self, entity, content=None):
        self.replacement_started.set()
        await asyncio.wait_for(self.resume_replacement.wait(), timeout=10)
        await super().index_entity_markdown(entity, content)


@pytest_asyncio.fixture
async def indexed_note(tmp_path):
    logger.disable("basic_memory")
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'memory.db'}",
        connect_args={"timeout": 10},
    )
    async with engine.begin() as connection:
        await connection.execute(text("PRAGMA journal_mode=WAL"))
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(timezone.utc)
    entity = Entity(
        id=1, project_id=1, title="Quartz repair note", note_type="note",
        content_type="text/markdown", permalink="quartz-repair", file_path="quartz.md",
        observations=[], outgoing_relations=[], incoming_relations=[],
        created_at=now, updated_at=now,
    )
    async with sessions() as session:
        session.add(Project(id=1, name="synthetic", permalink="synthetic", path=str(tmp_path)))
        session.add(entity)
        await session.commit()
    config = BasicMemoryConfig(
        env="test", semantic_search_enabled=True, reranker_enabled=False,
        semantic_vector_k=20,
    )
    files = FileService(
        tmp_path, MarkdownProcessor(EntityParser(tmp_path), app_config=config), app_config=config,
    )
    repository = SQLiteSearchRepository(
        sessions, 1, app_config=config, embedding_provider=SyntheticEmbeddings(),
    )
    reader = SQLiteSearchRepository(
        sessions, 1, app_config=config, embedding_provider=SyntheticEmbeddings(),
    )
    service = SearchService(repository, EntityRepository(project_id=1), files, sessions)
    await service.init_search_index()
    await reader.init_search_index()
    await service.index_entity(entity, content="Quartz repair notes stay searchable during refresh.")
    await service.sync_entity_vectors(entity.id)
    try:
        yield entity, service, reader
    finally:
        await engine.dispose()


async def vector_ids(repository):
    hits = await repository.search(
        search_text="quartz", retrieval_mode=SearchRetrievalMode.VECTOR, min_similarity=0,
    )
    return [hit.entity_id for hit in hits]


@pytest.mark.asyncio
async def test_note_remains_searchable_when_another_indexer_syncs_during_refresh(indexed_note):
    entity, service, reader = indexed_note
    assert await vector_ids(reader) == [entity.id]
    writer = PausedRefresh(
        service.repository, service.entity_repository, service.file_service, service.session_maker,
    )
    writer.replacement_started = asyncio.Event()
    writer.resume_replacement = asyncio.Event()
    refresh = asyncio.create_task(writer.index_entity(entity, content="Quartz repair notes updated."))
    try:
        await asyncio.wait_for(writer.replacement_started.wait(), timeout=10)
        await reader.sync_entity_vectors(entity.id)
        during_refresh = await vector_ids(reader)
    finally:
        writer.resume_replacement.set()
        await refresh
    assert during_refresh == [entity.id], "Refreshing an existing note must not erase semantic search"
    assert await vector_ids(reader) == [entity.id]


@pytest.mark.asyncio
async def test_file_refresh_removes_previous_markdown_observations(indexed_note):
    entity, service, reader = indexed_note
    entity.observations.append(Observation(
        id=2, project_id=1, category="fact", content="outdatedmarker", context=None,
    ))
    await service.index_entity(entity, content="Quartz note with an observation.")
    assert await reader.search(search_text="outdatedmarker")
    entity.content_type = "application/pdf"
    entity.file_path = "quartz.pdf"
    await service.index_entity(entity)
    assert await reader.search(search_text="outdatedmarker") == []


@pytest.mark.asyncio
async def test_failed_file_read_preserves_searchable_note(indexed_note):
    entity, service, reader = indexed_note
    with pytest.raises(FileNotFoundError):
        await service.index_entity(entity)
    assert await vector_ids(reader) == [entity.id]
    assert await reader.search(search_text="quartz")


@pytest.mark.asyncio
async def test_explicit_deletion_clears_text_and_vectors(indexed_note):
    entity, service, reader = indexed_note
    await service.handle_delete(entity)
    assert await vector_ids(reader) == []
    assert await reader.search(search_text="quartz") == []


@pytest.mark.asyncio
async def test_embedding_opt_out_clears_vectors_but_preserves_text(indexed_note):
    entity, service, reader = indexed_note
    async with service.session_maker() as session:
        saved_entity = await session.get(Entity, entity.id)
        saved_entity.entity_metadata = {"embed": False}
        await session.commit()
    await service.sync_entity_vectors(entity.id)
    assert await vector_ids(reader) == []
    assert await reader.search(search_text="quartz")


class PausedVectorPrepare(SQLiteSearchRepository):
    @asynccontextmanager
    async def _prepare_entity_write_scope(self):
        self.plan_ready.set()
        await asyncio.wait_for(self.resume_prepare.wait(), timeout=10)
        async with super()._prepare_entity_write_scope():
            yield


@pytest.mark.asyncio
@pytest.mark.parametrize("latest_change", ["update", "delete", "opt_out"])
async def test_delayed_vector_plan_preserves_latest_state(indexed_note, latest_change):
    entity, service, reader = indexed_note
    delayed = PausedVectorPrepare(
        service.session_maker, 1,
        app_config=BasicMemoryConfig(env="test", semantic_search_enabled=True),
        embedding_provider=SyntheticEmbeddings(),
    )
    await delayed.init_search_index()
    delayed.plan_ready = asyncio.Event()
    delayed.resume_prepare = asyncio.Event()
    await service.index_entity(entity, content="Quartz older update awaiting embedding.")
    pending = asyncio.create_task(delayed.sync_entity_vectors(entity.id))
    try:
        await asyncio.wait_for(delayed.plan_ready.wait(), timeout=10)
        if latest_change == "delete":
            await service.handle_delete(entity)
        elif latest_change == "opt_out":
            async with service.session_maker() as session:
                saved_entity = await session.get(Entity, entity.id)
                saved_entity.entity_metadata = {"embed": False}
                await session.commit()
            await service.sync_entity_vectors(entity.id)
        else:
            await service.index_entity(entity, content="Quartz newest update must remain indexed.")
            await reader.sync_entity_vectors(entity.id)
        current_manifest = await reader.get_entity_chunk_manifest(entity.id)
    finally:
        delayed.resume_prepare.set()
        await pending
    assert await reader.get_entity_chunk_manifest(entity.id) == current_manifest
    assert await vector_ids(reader) == ([entity.id] if latest_change == "update" else [])
    if latest_change == "opt_out":
        assert await reader.search(search_text="quartz")
