"""Postgres projection lifecycle SQL on SQLite; no PG FTS/concurrency claims."""

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from basic_memory.config import BasicMemoryConfig
from basic_memory.repository.postgres_search_repository import PostgresSearchRepository
from basic_memory.repository.search_index_row import SearchIndexRow


class PortableChunkRepository(PostgresSearchRepository):
    """Adapt only PG array/JSON chunk SQL; execute production bulk SQL unchanged."""

    fail_chunk_write = False

    async def _replace_fts_chunks(self, session, rows):
        for row in rows:
            params = {"project_id": self.project_id, "id": row.id, "type": row.type}
            await session.execute(text("""
                DELETE FROM search_index_fts_chunks
                WHERE project_id = :project_id
                  AND search_index_id = :id AND search_index_type = :type
            """), params)
            if row.content_snippet:
                await session.execute(text("""
                    INSERT INTO search_index_fts_chunks
                        (project_id, search_index_id, search_index_type, chunk_index, chunk_text)
                    VALUES (:project_id, :id, :type, 0, :content)
                """), {**params, "content": row.content_snippet.replace("\x00", "")})
        if self.fail_chunk_write:
            raise RuntimeError("Injected chunk write failure")


def row(row_id, kind="entity", *, entity_id=1, permalink=None, content="Original content"):
    now = datetime.now(timezone.utc)
    return SearchIndexRow(
        project_id=1, id=row_id, type=kind, entity_id=entity_id,
        file_path="quartz.md", permalink=permalink, title="Quartz note",
        content_stems=content, content_snippet=content, created_at=now, updated_at=now,
    )


@pytest_asyncio.fixture
async def postgres_projection(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'projection.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.execute(text("""
            CREATE TABLE search_index (
                id INTEGER NOT NULL, project_id INTEGER NOT NULL,
                title TEXT, content_stems TEXT, content_snippet TEXT,
                permalink TEXT, file_path TEXT, type TEXT,
                from_id INTEGER, to_id INTEGER, relation_type TEXT,
                entity_id INTEGER, category TEXT, metadata TEXT,
                created_at TEXT, updated_at TEXT,
                PRIMARY KEY (id, type, project_id)
            )
        """))
        await connection.execute(text("""
            CREATE UNIQUE INDEX uix_search_index_permalink_project
            ON search_index(permalink, project_id) WHERE permalink IS NOT NULL
        """))
        await connection.execute(text("""
            CREATE TABLE search_index_fts_chunks (
                project_id INTEGER NOT NULL, search_index_id INTEGER NOT NULL,
                search_index_type TEXT NOT NULL, chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                PRIMARY KEY (project_id, search_index_id, search_index_type, chunk_index),
                FOREIGN KEY (search_index_id, search_index_type, project_id)
                    REFERENCES search_index(id, type, project_id)
                    ON UPDATE CASCADE ON DELETE CASCADE
            )
        """))
    config = BasicMemoryConfig(env="test", semantic_search_enabled=False, reranker_enabled=False)
    repository = PortableChunkRepository(sessions, 1, app_config=config)
    other_project = PortableChunkRepository(sessions, 2, app_config=config)
    await repository.bulk_index_items([
        row(1, permalink="quartz"),
        row(101, "observation"),
        row(201, "relation"),
        row(2, entity_id=2, permalink="other-note"),
    ])
    await other_project.bulk_index_items([row(1, permalink="quartz")])
    try:
        yield repository, other_project, sessions
    finally:
        await engine.dispose()


async def stored_chunks(sessions):
    async with sessions() as session:
        result = await session.execute(text("""
            SELECT project_id, search_index_id, search_index_type, chunk_text
            FROM search_index_fts_chunks ORDER BY project_id, search_index_id
        """))
        return [tuple(item) for item in result.all()]


@pytest.mark.asyncio
@pytest.mark.parametrize("new_permalink", ["quartz", "quartz-renamed"])
async def test_bulk_refresh_replaces_owned_rows_and_cascades_removed_chunks(
    postgres_projection, new_permalink,
):
    repository, other_project, sessions = postgres_projection
    await repository.bulk_index_items([
        row(1, permalink=new_permalink, content="Updated\x00 content"),
        row(102, "observation", content="New observation"),
    ])

    actual = await repository.get_entity_search_rows(1)
    assert [(item.id, item.type, item.permalink) for item in actual] == [
        (1, "entity", new_permalink), (102, "observation", None),
    ]
    assert actual[0].content_snippet == "Updated content"
    assert (await repository.get_entity_search_rows(2))[0].permalink == "other-note"
    assert (await other_project.get_entity_search_rows(1))[0].permalink == "quartz"
    assert await stored_chunks(sessions) == [
        (1, 1, "entity", "Updated content"),
        (1, 2, "entity", "Original content"),
        (1, 102, "observation", "New observation"),
        (2, 1, "entity", "Original content"),
    ]


@pytest.mark.asyncio
async def test_chunk_failure_rolls_back_full_projection_replacement(postgres_projection):
    repository, _, sessions = postgres_projection
    before_rows = await repository.get_entity_search_rows(1)
    before_chunks = await stored_chunks(sessions)
    repository.fail_chunk_write = True
    with pytest.raises(RuntimeError, match="Injected chunk write failure"):
        await repository.bulk_index_items([row(1, permalink="quartz", content="New content")])
    assert await repository.get_entity_search_rows(1) == before_rows
    assert await stored_chunks(sessions) == before_chunks
