"""Exercise independent SQLite writers using Windows-compatible spawn workers."""

import asyncio
import multiprocessing
import queue
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pytest
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from basic_memory.config import BasicMemoryConfig
from basic_memory.markdown.entity_parser import EntityParser
from basic_memory.markdown.markdown_processor import MarkdownProcessor
from basic_memory.models import Entity, Project
from basic_memory.models.base import Base
from basic_memory.repository import EntityRepository
from basic_memory.repository.sqlite_search_repository import SQLiteSearchRepository
from basic_memory.services.file_service import FileService
from basic_memory.services.search_service import SearchService
from test_index_refresh import SyntheticEmbeddings, vector_ids


def synthetic_entity():
    now = datetime.now(timezone.utc)
    return Entity(
        id=1, project_id=1, title="Quartz repair note", note_type="note",
        content_type="text/markdown", permalink="quartz-repair", file_path="quartz.md",
        observations=[], outgoing_relations=[], incoming_relations=[],
        created_at=now, updated_at=now,
    )


def index_components(directory, service_type=SearchService):
    logger.disable("basic_memory")
    directory = Path(directory)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{directory / 'memory.db'}",
        connect_args={"timeout": 10},
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    config = BasicMemoryConfig(
        env="test", semantic_search_enabled=True, reranker_enabled=False,
        semantic_vector_k=20,
    )
    files = FileService(
        directory,
        MarkdownProcessor(EntityParser(directory), app_config=config),
        app_config=config,
    )
    repository = SQLiteSearchRepository(
        sessions, 1, app_config=config, embedding_provider=SyntheticEmbeddings(),
    )
    service = service_type(repository, EntityRepository(project_id=1), files, sessions)
    return engine, sessions, service


@pytest.fixture
def multiprocess_note(tmp_path):
    async def seed():
        engine, sessions, service = index_components(tmp_path)
        try:
            async with engine.begin() as connection:
                await connection.execute(text("PRAGMA journal_mode=WAL"))
                await connection.run_sync(Base.metadata.create_all)
            entity = synthetic_entity()
            async with sessions() as session:
                session.add(Project(
                    id=1, name="synthetic", permalink="synthetic", path=str(tmp_path),
                ))
                session.add(entity)
                await session.commit()
            await service.init_search_index()
            await service.index_entity(entity, content="Quartz repair notes before refresh.")
            await service.repository.sync_entity_vectors(entity.id)
            assert await vector_ids(service.repository) == [entity.id]
        finally:
            await engine.dispose()

    asyncio.run(seed())
    return tmp_path


class PausedProcessRefresh(SearchService):
    async def index_entity_markdown(self, entity, content=None):
        self.replacement_started.set()
        resumed = await asyncio.to_thread(self.resume_replacement.wait, 20)
        assert resumed, "Other process did not complete its concurrent vector sync"
        await super().index_entity_markdown(entity, content)


async def run_process_operation(directory, operation, started, resume, barrier):
    service_type = PausedProcessRefresh if operation == "paused_refresh" else SearchService
    engine, _, service = index_components(directory, service_type)
    try:
        await service.init_search_index()
        entity = synthetic_entity()
        if operation == "paused_refresh":
            service.replacement_started = started
            service.resume_replacement = resume
            await service.index_entity(entity, content="Quartz repair notes after refresh.")
            return {"ids": await vector_ids(service.repository)}
        if operation == "sync_during_refresh":
            assert await asyncio.to_thread(started.wait, 20), "Writer did not reach refresh seam"
            try:
                result = await service.repository.sync_entity_vectors_batch([entity.id])
                return {
                    "ids": await vector_ids(service.repository),
                    "failed": result.entities_failed,
                    "errors": result.sample_errors,
                }
            finally:
                resume.set()

        observations = []
        for iteration in range(4):
            await asyncio.to_thread(barrier.wait, 20)
            await service.index_entity(entity, content=f"Quartz repair refresh number {iteration}.")
            await asyncio.to_thread(barrier.wait, 20)
            result = await service.repository.sync_entity_vectors_batch([entity.id])
            observations.append({
                "ids": await vector_ids(service.repository),
                "failed": result.entities_failed,
                "errors": result.sample_errors,
            })
        return observations
    finally:
        await engine.dispose()


def index_worker(directory, operation, started, resume, barrier, results):
    try:
        result = asyncio.run(run_process_operation(directory, operation, started, resume, barrier))
        results.put((operation, result, None))
    except BaseException:
        results.put((operation, None, traceback.format_exc()))
        raise


def run_workers(directory, operations):
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    started, resume = context.Event(), context.Event()
    barrier = context.Barrier(len(operations))
    workers = [
        context.Process(
            target=index_worker,
            args=(str(directory), operation, started, resume, barrier, results),
        )
        for operation in operations
    ]
    try:
        for worker in workers:
            worker.start()
        received = []
        for _ in workers:
            try:
                received.append(results.get(timeout=35))
            except queue.Empty:
                pytest.fail("Spawned index worker timed out")
        for worker in workers:
            worker.join(timeout=10)
        errors = [error for _, _, error in received if error]
        assert not errors, "\n".join(errors)
        assert all(worker.exitcode == 0 for worker in workers)
        return {operation: result for operation, result, _ in received}
    finally:
        resume.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
            worker.close()
        results.close()
        results.join_thread()


def test_vector_search_survives_another_process_syncing_during_refresh(multiprocess_note):
    results = run_workers(multiprocess_note, ["paused_refresh", "sync_during_refresh"])
    reader = results["sync_during_refresh"]
    assert reader["failed"] == 0, reader["errors"]
    assert reader["ids"] == [1], "Concurrent refresh must not erase the existing note's vectors"
    assert results["paused_refresh"]["ids"] == [1]


def test_repeated_concurrent_refreshes_and_vector_syncs_remain_searchable(multiprocess_note):
    results = run_workers(multiprocess_note, ["overlapping_writer_1", "overlapping_writer_2"])
    for observations in results.values():
        for observation in observations:
            assert observation["failed"] == 0, observation["errors"]
            assert observation["ids"] == [1]
