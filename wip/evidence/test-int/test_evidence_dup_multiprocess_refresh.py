"""Several processes refresh one note's search rows in one SQLite file (B3 evidence).

The downstream incident ran several MCP clients -- separate processes -- against one
SQLite project. Here four OS processes refresh the same note through the production
``SearchService.index_entity_data``. The first is held inside its refresh transaction
after its DELETE; the other three start their own refresh and the parent releases the
first only after each of their DELETE statements has started executing in SQLite (it
cannot finish while the first process holds the database write lock).

The worker side is this same file run as a script; parent and workers talk over
stdin/stdout lines, so nothing waits on a timer.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import override

PROTOCOL_PREFIX = "@@ "
QUEUED_PROCESSES = 3
WAIT_SECONDS = 60
NOTE_CONTENT = "Multiprocess refresh body"


def _sqlite_config(home: Path):
    from basic_memory.config import BasicMemoryConfig, DatabaseBackend

    return BasicMemoryConfig(
        env="test",
        projects={"shared": str(home)},
        default_project="shared",
        database_backend=DatabaseBackend.SQLITE,
        semantic_search_enabled=False,
    )


# --- Worker process ---


def _say(message: str) -> None:
    print(f"{PROTOCOL_PREFIX}{message}", flush=True)


async def _next_command() -> str:
    return (await asyncio.to_thread(sys.stdin.readline)).strip()


async def _run_worker(db_path: Path, role: str, project_id: int, entity_id: int) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession

    from basic_memory import db
    from basic_memory.models import Entity
    from basic_memory.repository.entity_repository import EntityRepository
    from basic_memory.repository.search_repository import create_search_repository
    from basic_memory.services.file_service import FileService
    from basic_memory.services.search_service import SearchService

    class PausedAfterDelete(SearchService):
        @override
        async def index_entity_markdown(
            self,
            entity: Entity,
            content: str | None = None,
            session: AsyncSession | None = None,
        ) -> None:
            _say("deleted")
            assert await _next_command() == "resume"
            await super().index_entity_markdown(entity, content, session=session)

    config = _sqlite_config(db_path.parent)
    async with db.engine_session_factory(db_path, db.DatabaseType.FILESYSTEM, config) as (
        _,
        session_maker,
    ):
        entity_repository = EntityRepository(project_id=project_id)
        async with db.scoped_session(session_maker) as session:
            entity = await entity_repository.find_by_id(session, entity_id)
        assert entity is not None
        service_type = PausedAfterDelete if role == "paused" else SearchService
        service = service_type(
            create_search_repository(session_maker, project_id=project_id, app_config=config),
            entity_repository,
            FileService(db_path.parent, app_config=config),
            session_maker=session_maker,
        )

        if role == "queued":
            original = service.repository.delete_by_entity_id

            async def delete_and_report(entity_id: int, session: AsyncSession | None = None):
                async with db.scoped_session(session_maker, session) as active:
                    connection = await active.connection()
                    driver = (await connection.get_raw_connection()).driver_connection
                    assert driver is not None

                    def on_statement(sql: str) -> None:
                        if sql.startswith("DELETE FROM search_index"):
                            _say("delete-started")

                    await driver.set_trace_callback(on_statement)
                    try:
                        await original(entity_id=entity_id, session=active)
                    finally:
                        await driver.set_trace_callback(None)

            setattr(service.repository, "delete_by_entity_id", delete_and_report)

        _say("ready")
        assert await _next_command() == "go"
        await service.index_entity_data(entity, content=NOTE_CONTENT)
        _say("done")


if __name__ == "__main__":
    worker_db_path, worker_role, worker_project_id, worker_entity_id = sys.argv[1:5]
    asyncio.run(
        _run_worker(
            Path(worker_db_path), worker_role, int(worker_project_id), int(worker_entity_id)
        )
    )
    sys.exit(0)


# --- Test (parent process) ---

import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402


class Worker:
    """One worker process; its pipes are read in threads so any event loop can drive it.

    Basic Memory installs the selector event loop on Windows, which cannot run asyncio
    subprocesses, and the downstream report came from Windows.
    """

    def __init__(self, role: str, process: subprocess.Popen[str], stderr_path: Path) -> None:
        self.role = role
        self.process = process
        self.stderr_path = stderr_path
        self.lines: list[str] = []

    async def expect(self, message: str) -> None:
        stdout = self.process.stdout
        assert stdout is not None
        while True:
            line = await asyncio.wait_for(asyncio.to_thread(stdout.readline), timeout=WAIT_SECONDS)
            if not line:
                raise AssertionError(
                    f"{self.role} worker exited before '{message}': {self.lines}\n"
                    f"{await self._stderr_tail()}"
                )
            self.lines.append(line.rstrip("\n"))
            if line.rstrip("\n") == f"{PROTOCOL_PREFIX}{message}":
                return

    async def send(self, command: str) -> None:
        stdin = self.process.stdin
        assert stdin is not None

        def write() -> None:
            stdin.write(f"{command}\n")
            stdin.flush()

        await asyncio.to_thread(write)

    async def wait(self) -> int:
        return await asyncio.to_thread(self.process.wait, WAIT_SECONDS)

    async def _stderr_tail(self) -> str:
        await self.wait()
        return self.stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]


def _spawn(db_path: Path, role: str, project_id: int, entity_id: int) -> Worker:
    import basic_memory

    source_root = Path(basic_memory.__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(source_root), os.environ.get("PYTHONPATH", "")]),
        "HOME": str(db_path.parent),
        "USERPROFILE": str(db_path.parent),
        "BASIC_MEMORY_HOME": str(db_path.parent / "basic-memory"),
        "BASIC_MEMORY_ENV": "test",
    }
    # Worker logs go to a file: an unread stderr pipe could fill and stall the worker.
    stderr_path = db_path.parent / f"{role}-{len(list(db_path.parent.glob('*.stderr')))}.stderr"
    with stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(db_path),
                role,
                str(project_id),
                str(entity_id),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
            text=True,
        )
    return Worker(role, process, stderr_path)


async def _seed_shared_database(db_path: Path) -> tuple[int, int, Counter[tuple[str, int]]]:
    from basic_memory import db
    from basic_memory.models import Entity, Observation, Project, Relation
    from basic_memory.models.base import Base
    from basic_memory.models.search import CREATE_SEARCH_INDEX
    from basic_memory.repository.entity_repository import EntityRepository
    from basic_memory.repository.search_repository import create_search_repository
    from basic_memory.services.file_service import FileService
    from basic_memory.services.search_service import SearchService

    config = _sqlite_config(db_path.parent)
    now = datetime.now(timezone.utc)
    async with db.engine_session_factory(db_path, db.DatabaseType.FILESYSTEM, config) as (
        engine,
        session_maker,
    ):
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.execute(CREATE_SEARCH_INDEX)
        async with db.scoped_session(session_maker) as session:
            project = Project(name="shared", path=str(db_path.parent), is_active=True)
            session.add(project)
            await session.flush()
            note = Entity(
                project_id=project.id,
                title="Shared note",
                note_type="note",
                permalink="shared/note",
                file_path="shared/note.md",
                content_type="text/markdown",
                created_at=now,
                updated_at=now,
            )
            session.add(note)
            await session.flush()
            session.add_all(
                [
                    Observation(
                        project_id=project.id, entity_id=note.id, category="fact", content=fact
                    )
                    for fact in ("first fact", "second fact")
                ]
            )
            session.add(
                Relation(
                    project_id=project.id,
                    from_id=note.id,
                    to_id=None,
                    to_name="Elsewhere",
                    relation_type="links_to",
                )
            )
            project_id, entity_id = project.id, note.id

        entity_repository = EntityRepository(project_id=project_id)
        async with db.scoped_session(session_maker) as session:
            entity = await entity_repository.find_by_id(session, entity_id)
        assert entity is not None
        service = SearchService(
            create_search_repository(session_maker, project_id=project_id, app_config=config),
            entity_repository,
            FileService(db_path.parent, app_config=config),
            session_maker=session_maker,
        )
        await service.index_entity_data(entity, content=NOTE_CONTENT)
        expected = Counter(
            [("entity", entity.id)]
            + [("observation", observation.id) for observation in entity.observations]
            + [("relation", relation.id) for relation in entity.outgoing_relations]
        )
    return project_id, entity_id, expected


async def _stored_rows(db_path: Path, project_id: int, entity_id: int) -> Counter[tuple[str, int]]:
    from basic_memory import db

    config = _sqlite_config(db_path.parent)
    async with db.engine_session_factory(db_path, db.DatabaseType.FILESYSTEM, config) as (
        _,
        session_maker,
    ):
        async with db.scoped_session(session_maker) as session:
            result = await session.execute(
                text(
                    "SELECT type, id FROM search_index "
                    "WHERE project_id = :project_id AND entity_id = :entity_id"
                ),
                {"project_id": project_id, "entity_id": entity_id},
            )
            return Counter((str(kind), int(row_id)) for kind, row_id in result.all())


@pytest.mark.asyncio
async def test_refreshes_from_four_processes_leave_one_projection(tmp_path, db_backend):
    if db_backend != "sqlite":
        pytest.skip("cross-process writer exclusion is SQLite's database lock")

    db_path = tmp_path / "shared.db"
    project_id, entity_id, expected = await _seed_shared_database(db_path)

    paused = _spawn(db_path, "paused", project_id, entity_id)
    queued = [_spawn(db_path, "queued", project_id, entity_id) for _ in range(QUEUED_PROCESSES)]
    workers = [paused, *queued]
    try:
        for worker in workers:
            await worker.expect("ready")

        await paused.send("go")
        await paused.expect("deleted")
        for worker in queued:
            await worker.send("go")
        for worker in queued:
            await worker.expect("delete-started")
        await paused.send("resume")

        for worker in workers:
            await worker.expect("done")
            assert await worker.wait() == 0
    finally:
        for worker in workers:
            if worker.process.poll() is None:
                worker.process.kill()
            await worker.wait()
            for pipe in (worker.process.stdin, worker.process.stdout):
                if pipe is not None:
                    pipe.close()

    assert await _stored_rows(db_path, project_id, entity_id) == expected
