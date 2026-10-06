"""PostgreSQL: a delayed vector plan must not overwrite the entity's latest vector state.

Vector sync reads an entity's search rows and chunk manifest, plans the manifest
mutations from that read, and only then opens its write transaction. Generation
fences (``source_hash`` predicates) already cover a sync that pauses *after* that
write -- ``test_vector_manifest_generation_ownership.py`` pauses one inside
``embed_documents``. These tests pause one sync *between* planning and its write
transaction (``_prepare_entity_write_scope`` is the seam) while the entity moves
on: a newer edit is synced, the note opts out of embeddings, or the note is
deleted. When the delayed sync resumes, the entity's latest vector state must
survive.

Two repository instances share the test database, so the two syncs run on
separate PostgreSQL connections with real READ COMMITTED transactions. Built-in
pgvector and an external adapter are both covered: main re-reads under its
project lock only for external indexes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, override

import pytest
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory import db
from basic_memory.config import BasicMemoryConfig, DatabaseBackend
from basic_memory.models import Entity
from basic_memory.repository import EntityRepository
from basic_memory.repository.postgres_search_repository import PostgresSearchRepository
from basic_memory.repository.search_scope import ProjectScope
from basic_memory.repository.semantic_vector_index import (
    VectorDeletion,
    VectorIndexScope,
    VectorKey,
    VectorMatch,
    VectorRecord,
)
from basic_memory.schemas.search import SearchRetrievalMode
from basic_memory.services.file_service import FileService
from basic_memory.services.search_service import SearchService

pytestmark = pytest.mark.postgres

# Generous: the pause points are synchronized with events, never with sleeps, but the
# machine may be loaded by other suites.
PAUSE_TIMEOUT_SECONDS = 60

ORIGINAL_CONTENT = "Quartz original wording."
DELAYED_CONTENT = "Quartz older update awaiting embedding."
NEWEST_CONTENT = "Quartz newest update must remain indexed."

type LatestChange = Literal["update", "opt_out", "delete"]
type SyncApi = Literal["repository", "service"]
type VectorBackend = Literal["pgvector", "external"]
type ManifestState = list[tuple[str, str, str, str]]
type EmbeddingState = list[tuple[str, str]]


class SearchableExternalVectorIndex:
    """In-memory external adapter behind Core's manifest contract; every record matches."""

    def __init__(self) -> None:
        self._scope = VectorIndexScope(
            namespace="delayed-plan-evidence",
            embedding_identity="test:4",
            dimensions=4,
        )
        self.records: dict[VectorKey, VectorRecord] = {}

    @property
    def scope(self) -> VectorIndexScope:
        return self._scope

    async def initialize(self) -> None:
        return None

    async def upsert(self, project_id: int, records: Sequence[VectorRecord]) -> None:
        for record in records:
            self.records[record.key] = record

    async def delete(self, project_id: int, records: Sequence[VectorDeletion]) -> None:
        for deletion in records:
            current = self.records.get(deletion.key)
            if current is not None and current.source_hash == deletion.source_hash:
                self.records.pop(deletion.key)

    async def delete_entity(self, project_id: int, entity_id: int) -> None:
        self.records = {
            key: record for key, record in self.records.items() if key.entity_id != entity_id
        }

    async def search(
        self,
        query: Sequence[float],
        *,
        limit: int,
        projects: ProjectScope,
    ) -> list[VectorMatch]:
        return [VectorMatch(key=key, similarity=1.0) for key in list(self.records)[:limit]]


class UniformEmbeddingProvider:
    """Embed every text to one vector, so any ready chunk is a vector-search hit.

    With ``started``/``resume`` it pauses inside ``embed_documents``, i.e. after the
    prepare transaction committed its pending manifest rows.
    """

    model_name = "delayed-plan-evidence"
    dimensions = 4

    def __init__(
        self,
        *,
        started: asyncio.Event | None = None,
        resume: asyncio.Event | None = None,
    ) -> None:
        self.started = started
        self.resume = resume

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.started is not None and self.resume is not None:
            self.started.set()
            await asyncio.wait_for(self.resume.wait(), PAUSE_TIMEOUT_SECONDS)
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def runtime_log_attrs(self) -> dict[str, object]:
        return {}


class DelayedPlanRepository(PostgresSearchRepository):
    """Pause after planning from the read snapshot, before the prepare write transaction."""

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        project_id: int,
        *,
        app_config: BasicMemoryConfig,
        external_index: SearchableExternalVectorIndex | None = None,
    ) -> None:
        super().__init__(
            session_maker,
            project_id,
            app_config=app_config,
            embedding_provider=UniformEmbeddingProvider(),
            vector_index_name="test-external" if external_index is not None else None,
            vector_index=external_index,
        )
        self.plan_ready = asyncio.Event()
        self.resume_write = asyncio.Event()

    @asynccontextmanager
    @override
    async def _prepare_entity_write_scope(self) -> AsyncIterator[None]:
        # The read session has closed and the plans exist; no transaction is open.
        self.plan_ready.set()
        await asyncio.wait_for(self.resume_write.wait(), PAUSE_TIMEOUT_SECONDS)
        async with super()._prepare_entity_write_scope():
            yield


@dataclass(frozen=True, slots=True)
class VectorWriters:
    """Two vector writers over one database, plus the note they both sync."""

    session_maker: async_sessionmaker[AsyncSession]
    entity_repository: EntityRepository
    project_id: int
    entity: Entity
    current_repository: PostgresSearchRepository
    current: SearchService
    delayed_repository: DelayedPlanRepository
    delayed: SearchService
    external_index: SearchableExternalVectorIndex | None = None

    async def manifest(self) -> ManifestState:
        async with db.scoped_session(self.session_maker) as session:
            result = await session.execute(
                text(
                    "SELECT chunk_key, source_hash, embedding_status, chunk_text "
                    "FROM search_vector_chunks "
                    "WHERE project_id = :project_id AND entity_id = :entity_id "
                    "ORDER BY chunk_key"
                ),
                {"project_id": self.project_id, "entity_id": self.entity.id},
            )
            return [
                (str(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in result.fetchall()
            ]

    async def embeddings(self) -> EmbeddingState:
        """Stored vectors for the note: adapter records, or pgvector rows by manifest row."""
        if self.external_index is not None:
            return sorted(
                (key.chunk_key, record.source_hash)
                for key, record in self.external_index.records.items()
                if key.entity_id == self.entity.id
            )
        async with db.scoped_session(self.session_maker) as session:
            result = await session.execute(
                text(
                    "SELECT c.chunk_key, e.source_hash "
                    "FROM search_vector_embeddings e "
                    "JOIN search_vector_chunks c ON c.id = e.chunk_id "
                    "WHERE c.project_id = :project_id AND c.entity_id = :entity_id "
                    "ORDER BY c.chunk_key"
                ),
                {"project_id": self.project_id, "entity_id": self.entity.id},
            )
            return [(str(row[0]), str(row[1])) for row in result.fetchall()]

    async def vector_hit_ids(self) -> list[int]:
        hits = await self.current_repository.search(
            search_text="quartz",
            retrieval_mode=SearchRetrievalMode.VECTOR,
            min_similarity=0.0,
        )
        return sorted({hit.entity_id for hit in hits if hit.entity_id is not None})

    async def opt_out_of_embeddings(self) -> None:
        async with db.scoped_session(self.session_maker) as session:
            await session.execute(
                update(Entity)
                .where(Entity.id == self.entity.id)
                .values(entity_metadata={"embed": False})
            )

    async def delete_note(self) -> None:
        """Delete the way EntityService.delete_entity does: search state first, then the row."""
        await self.current.handle_delete(self.entity)
        async with db.scoped_session(self.session_maker) as session:
            assert await self.entity_repository.delete(session, self.entity.id)

    async def apply(self, latest_change: LatestChange) -> None:
        match latest_change:
            case "update":
                await self.current.index_entity(self.entity, content=NEWEST_CONTENT)
                await self.current.sync_entity_vectors(self.entity.id)
            case "opt_out":
                await self.opt_out_of_embeddings()
                await self.current.sync_entity_vectors(self.entity.id)
            case "delete":
                await self.delete_note()

    def delayed_sync(self, api: SyncApi) -> Coroutine[Any, Any, None]:
        if api == "service":
            return self.delayed.sync_entity_vectors(self.entity.id)
        return self.delayed_repository.sync_entity_vectors(self.entity.id)


@pytest.fixture(autouse=True)
def _require_postgres_backend(db_backend: str) -> None:
    """These interleavings need real PostgreSQL connections and transactions."""
    if db_backend != "postgres":
        pytest.skip("Delayed vector plan PostgreSQL tests require BASIC_MEMORY_TEST_POSTGRES=1")


async def _skip_if_pgvector_unavailable(session_maker: async_sessionmaker[AsyncSession]) -> None:
    async with db.scoped_session(session_maker) as session:
        try:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await session.commit()
        except Exception:
            pytest.skip("pgvector extension is unavailable in this PostgreSQL test environment")


def _semantic_config() -> BasicMemoryConfig:
    return BasicMemoryConfig(
        env="test",
        projects={"test-project": "/tmp/basic-memory-test"},
        default_project="test-project",
        database_backend=DatabaseBackend.POSTGRES,
        semantic_search_enabled=True,
    )


async def _vector_writers(
    session_maker: async_sessionmaker[AsyncSession],
    project_id: int,
    entity_repository: EntityRepository,
    file_service: FileService,
    entity_id: int,
    vector_backend: VectorBackend = "pgvector",
) -> VectorWriters:
    """Build both writers and embed the note's original wording."""
    external_index: SearchableExternalVectorIndex | None = None
    if vector_backend == "pgvector":
        await _skip_if_pgvector_unavailable(session_maker)
    else:
        external_index = SearchableExternalVectorIndex()
    config = _semantic_config()
    current_repository = PostgresSearchRepository(
        session_maker,
        project_id,
        app_config=config,
        embedding_provider=UniformEmbeddingProvider(),
        vector_index_name="test-external" if external_index is not None else None,
        vector_index=external_index,
    )
    delayed_repository = DelayedPlanRepository(
        session_maker, project_id, app_config=config, external_index=external_index
    )
    # The database is initialized once, as at startup; both writers then share it.
    await current_repository.init_search_index()

    # Load with the relationships indexing and deletion walk.
    async with db.scoped_session(session_maker) as session:
        entity = await entity_repository.find_by_id(session, entity_id)
    assert entity is not None

    writers = VectorWriters(
        session_maker=session_maker,
        entity_repository=entity_repository,
        project_id=project_id,
        entity=entity,
        current_repository=current_repository,
        current=SearchService(current_repository, entity_repository, file_service, session_maker),
        delayed_repository=delayed_repository,
        delayed=SearchService(delayed_repository, entity_repository, file_service, session_maker),
        external_index=external_index,
    )
    await writers.current.index_entity(entity, content=ORIGINAL_CONTENT)
    await writers.current.sync_entity_vectors(entity.id)
    assert await writers.vector_hit_ids() == [entity.id]
    return writers


async def _delay_plan_across(
    writers: VectorWriters,
    latest_change: LatestChange,
    delayed_api: SyncApi,
) -> tuple[ManifestState, EmbeddingState]:
    """Plan an edit's vector sync, apply ``latest_change``, then let the plan write.

    Returns the vector state the latest change left before the delayed plan resumed.
    """
    await writers.current.index_entity(writers.entity, content=DELAYED_CONTENT)
    pending = asyncio.create_task(writers.delayed_sync(delayed_api))
    try:
        await asyncio.wait_for(writers.delayed_repository.plan_ready.wait(), PAUSE_TIMEOUT_SECONDS)
        await writers.apply(latest_change)
        latest = (await writers.manifest(), await writers.embeddings())

        writers.delayed_repository.resume_write.set()
        await asyncio.wait_for(pending, PAUSE_TIMEOUT_SECONDS)
    finally:
        writers.delayed_repository.resume_write.set()
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    return latest


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_backend", ["pgvector", "external"])
@pytest.mark.parametrize("delayed_api", ["repository", "service"])
@pytest.mark.parametrize("latest_change", ["update", "opt_out", "delete"])
async def test_plan_delayed_before_its_write_preserves_latest_vector_state(
    session_maker: async_sessionmaker[AsyncSession],
    test_project,
    entity_repository: EntityRepository,
    file_service: FileService,
    sample_entity: Entity,
    latest_change: LatestChange,
    delayed_api: SyncApi,
    vector_backend: VectorBackend,
) -> None:
    """A plan read before the latest change must not be applied after it."""
    writers = await _vector_writers(
        session_maker,
        test_project.id,
        entity_repository,
        file_service,
        sample_entity.id,
        vector_backend,
    )

    latest_manifest, latest_embeddings = await _delay_plan_across(
        writers, latest_change, delayed_api
    )

    # --- The latest change owns the final vector state ---
    # What a reader sees first: an opted-out or deleted note answers no vector query.
    expected_hits = [writers.entity.id] if latest_change == "update" else []
    assert await writers.vector_hit_ids() == expected_hits

    final_manifest = await writers.manifest()
    assert final_manifest == latest_manifest, (
        "the delayed plan rewrote the manifest the latest change produced"
    )
    assert await writers.embeddings() == latest_embeddings
    if latest_change == "update":
        assert latest_manifest
        assert all(NEWEST_CONTENT in row[3] for row in final_manifest)
        assert {row[2] for row in final_manifest} == {"ready"}
    else:
        assert latest_manifest == []
    if latest_change == "opt_out":
        # The opt-out removes vectors only; the note stays in full-text search.
        fts_hits = await writers.current_repository.search(search_text="quartz")
        assert [hit.entity_id for hit in fts_hits] == [writers.entity.id]


@pytest.mark.asyncio
@pytest.mark.parametrize("latest_change", ["update", "opt_out", "delete"])
async def test_reindex_converges_whatever_a_delayed_plan_left(
    session_maker: async_sessionmaker[AsyncSession],
    test_project,
    entity_repository: EntityRepository,
    file_service: FileService,
    sample_entity: Entity,
    latest_change: LatestChange,
) -> None:
    """``reindex_vectors`` (bm reindex --embeddings) repairs every state the race leaves."""
    writers = await _vector_writers(
        session_maker, test_project.id, entity_repository, file_service, sample_entity.id
    )
    await _delay_plan_across(writers, latest_change, "service")

    await writers.current.reindex_vectors()

    final_manifest = await writers.manifest()
    if latest_change == "update":
        assert final_manifest
        assert all(NEWEST_CONTENT in row[3] for row in final_manifest)
        assert {row[2] for row in final_manifest} == {"ready"}
        assert await writers.vector_hit_ids() == [writers.entity.id]
    else:
        assert final_manifest == []
        assert await writers.embeddings() == []
        assert await writers.vector_hit_ids() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("latest_change", ["update", "opt_out"])
async def test_next_edit_converges_whatever_a_delayed_plan_left(
    session_maker: async_sessionmaker[AsyncSession],
    test_project,
    entity_repository: EntityRepository,
    file_service: FileService,
    sample_entity: Entity,
    latest_change: Literal["update", "opt_out"],
) -> None:
    """The note's next edit re-syncs its vectors from the current source and policy."""
    writers = await _vector_writers(
        session_maker, test_project.id, entity_repository, file_service, sample_entity.id
    )
    await _delay_plan_across(writers, latest_change, "service")

    next_content = "Quartz next edit after the race."
    await writers.current.index_entity(writers.entity, content=next_content)
    await writers.current.sync_entity_vectors(writers.entity.id)

    final_manifest = await writers.manifest()
    if latest_change == "update":
        assert final_manifest
        assert all(next_content in row[3] for row in final_manifest)
        assert {row[2] for row in final_manifest} == {"ready"}
    else:
        assert final_manifest == []
        assert await writers.vector_hit_ids() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("delayed_api", ["repository", "service"])
@pytest.mark.parametrize("latest_change", ["opt_out", "delete"])
async def test_sync_paused_in_embedding_tolerates_opt_out_and_delete(
    session_maker: async_sessionmaker[AsyncSession],
    test_project,
    entity_repository: EntityRepository,
    file_service: FileService,
    sample_entity: Entity,
    latest_change: Literal["opt_out", "delete"],
    delayed_api: SyncApi,
) -> None:
    """The prepare write already committed; the vectors are removed before they persist."""
    await _skip_if_pgvector_unavailable(session_maker)
    config = _semantic_config()
    embed_started = asyncio.Event()
    resume_embedding = asyncio.Event()

    current_repository = PostgresSearchRepository(
        session_maker,
        test_project.id,
        app_config=config,
        embedding_provider=UniformEmbeddingProvider(),
    )
    paused_repository = PostgresSearchRepository(
        session_maker,
        test_project.id,
        app_config=config,
        embedding_provider=UniformEmbeddingProvider(
            started=embed_started,
            resume=resume_embedding,
        ),
    )
    await current_repository.init_search_index()
    async with db.scoped_session(session_maker) as session:
        entity = await entity_repository.find_by_id(session, sample_entity.id)
    assert entity is not None
    writers = VectorWriters(
        session_maker=session_maker,
        entity_repository=entity_repository,
        project_id=test_project.id,
        entity=entity,
        current_repository=current_repository,
        current=SearchService(current_repository, entity_repository, file_service, session_maker),
        delayed_repository=DelayedPlanRepository(session_maker, test_project.id, app_config=config),
        delayed=SearchService(paused_repository, entity_repository, file_service, session_maker),
    )

    await writers.current.index_entity(entity, content="Quartz note awaiting its first embedding.")
    paused_sync = (
        writers.delayed.sync_entity_vectors(entity.id)
        if delayed_api == "service"
        else paused_repository.sync_entity_vectors(entity.id)
    )
    pending = asyncio.create_task(paused_sync)
    try:
        await asyncio.wait_for(embed_started.wait(), PAUSE_TIMEOUT_SECONDS)
        prepared_manifest = await writers.manifest()
        assert prepared_manifest
        assert {row[2] for row in prepared_manifest} == {"pending"}

        await writers.apply(latest_change)

        resume_embedding.set()
        await asyncio.wait_for(pending, PAUSE_TIMEOUT_SECONDS)
    finally:
        resume_embedding.set()
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    assert await writers.manifest() == []
    assert await writers.embeddings() == []
    assert await writers.vector_hit_ids() == []
