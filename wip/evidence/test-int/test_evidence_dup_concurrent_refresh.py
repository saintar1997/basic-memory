"""Concurrent refreshes of one entity's search projection (B3 / B2 evidence).

The downstream report: on 0.23.2, four concurrent batch writes of one note left four
search_index rows, because the refresh deleted the old rows in one committed transaction
and inserted the replacement in another. On main (#1623) ``SearchService.index_entity_data``
deletes by entity_id and writes the replacement inside one transaction.

These tests drive the real production refresh from several writers, each with its own
repository and sessions, and force the interleaving that used to duplicate rows: writer A
is paused after its DELETE but before its INSERT while writers B-D start their own refresh
transactions. Pause points are asyncio Events; "B is waiting on A" is observed, never
slept on:

- SQLite: a statement trace on B's connection reports that B's DELETE has started
  executing; it cannot finish while A holds the database write lock.
- PostgreSQL: ``pg_blocking_pids`` reports that B's backend is waiting on a lock.

Each test asserts the exact multiset of (type, id, permalink) rows -- the projection of
whichever writer committed last -- and that no writer raised. The number of rows each
queued DELETE removed is recorded so a failure shows the mechanism, not just the result.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import override

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from basic_memory import db
from basic_memory.models import Entity, Observation, Relation
from basic_memory.repository.entity_repository import EntityRepository
from basic_memory.repository.search_repository import create_search_repository
from basic_memory.services.search_service import SearchService

type RowKey = tuple[str, int, str | None]

QUEUED_WRITERS = 3
WAIT_SECONDS = 30


# --- Fixture data ---


async def _seed_note(session_maker, project_id: int, observations: Sequence[str]) -> int:
    now = datetime.now(timezone.utc)
    async with db.scoped_session(session_maker) as session:
        target = Entity(
            project_id=project_id,
            title="Refresh target",
            note_type="note",
            permalink="refresh-race/target",
            file_path="refresh-race/target.md",
            content_type="text/markdown",
            created_at=now,
            updated_at=now,
        )
        note = Entity(
            project_id=project_id,
            title="Refresh race",
            note_type="note",
            permalink="refresh-race/note",
            file_path="refresh-race/note.md",
            content_type="text/markdown",
            created_at=now,
            updated_at=now,
        )
        session.add_all([target, note])
        await session.flush()
        session.add_all(
            [
                Observation(
                    project_id=project_id, entity_id=note.id, category="fact", content=content
                )
                for content in observations
            ]
        )
        session.add_all(
            [
                Relation(
                    project_id=project_id,
                    from_id=note.id,
                    to_id=target.id,
                    to_name=target.title,
                    relation_type="links_to",
                ),
                Relation(
                    project_id=project_id,
                    from_id=note.id,
                    to_id=None,
                    to_name="Not Written Yet",
                    relation_type="mentions",
                ),
            ]
        )
        return note.id


async def _seed_resource(session_maker, project_id: int) -> int:
    """A non-Markdown resource: the batch indexer stores these with permalink=None."""
    now = datetime.now(timezone.utc)
    async with db.scoped_session(session_maker) as session:
        resource = Entity(
            project_id=project_id,
            title="scan.pdf",
            note_type="file",
            permalink=None,
            file_path="refresh-race/scan.pdf",
            content_type="application/pdf",
            created_at=now,
            updated_at=now,
        )
        session.add(resource)
        await session.flush()
        return resource.id


async def _accept_new_observations(
    session_maker, project_id: int, entity_id: int, observations: Sequence[str]
) -> None:
    """Re-create the note's observations the way a newer accepted generation does: new ids.

    The replacements are inserted before the old rows are deleted so SQLite, which reuses
    the highest freed rowid, hands out ids the older version never had.
    """
    async with db.scoped_session(session_maker) as session:
        previous_ids = list(
            (
                await session.execute(
                    text("SELECT id FROM observation WHERE entity_id = :entity_id"),
                    {"entity_id": entity_id},
                )
            ).scalars()
        )
        session.add_all(
            [
                Observation(
                    project_id=project_id, entity_id=entity_id, category="fact", content=content
                )
                for content in observations
            ]
        )
        await session.flush()
        for observation_id in previous_ids:
            await session.execute(
                text("DELETE FROM observation WHERE id = :id"), {"id": observation_id}
            )


async def _move_note(session_maker, entity_id: int, permalink: str) -> None:
    """A move rewrites the entity's permalink and path; observation ids are kept."""
    async with db.scoped_session(session_maker) as session:
        await session.execute(
            text("UPDATE entity SET permalink = :permalink, file_path = :file_path WHERE id = :id"),
            {"permalink": permalink, "file_path": f"{permalink}.md", "id": entity_id},
        )


async def _snapshot(session_maker, project_id: int, entity_id: int) -> Entity:
    async with db.scoped_session(session_maker) as session:
        entity = await EntityRepository(project_id=project_id).find_by_id(session, entity_id)
    assert entity is not None
    return entity


def _projection_of(entity: Entity) -> Counter[RowKey]:
    """The rows one refresh of this snapshot writes: one per (type, id)."""
    rows: Counter[RowKey] = Counter({("entity", entity.id, entity.permalink): 1})
    if entity.is_markdown:
        rows.update(
            ("observation", observation.id, observation.permalink)
            for observation in entity.observations
        )
        rows.update(
            ("relation", relation.id, relation.permalink) for relation in entity.outgoing_relations
        )
    return rows


async def _stored_rows(session_maker, project_id: int, entity_id: int) -> Counter[RowKey]:
    """Every physical row the entity owns -- duplicates are counted, not collapsed."""
    async with db.scoped_session(session_maker) as session:
        result = await session.execute(
            text(
                "SELECT type, id, permalink FROM search_index "
                "WHERE project_id = :project_id AND entity_id = :entity_id"
            ),
            {"project_id": project_id, "entity_id": entity_id},
        )
        return Counter(
            (str(kind), int(row_id), permalink) for kind, row_id, permalink in result.all()
        )


# --- Writers ---


def _writer(search_service: SearchService, app_config, project_id: int) -> SearchService:
    """An independent refresher: its own repository and its own sessions."""
    session_maker = search_service.session_maker
    return SearchService(
        create_search_repository(session_maker, project_id=project_id, app_config=app_config),
        search_service.entity_repository,
        search_service.file_service,
        session_maker=session_maker,
    )


class ReplacementPausedAfterDelete(SearchService):
    """Hold the refresh transaction open after its DELETE and before its INSERT."""

    def __init__(self, writer: SearchService) -> None:
        super().__init__(
            writer.repository,
            writer.entity_repository,
            writer.file_service,
            session_maker=writer.session_maker,
        )
        self.deleted = asyncio.Event()
        self.resume = asyncio.Event()

    async def _pause(self) -> None:
        self.deleted.set()
        await asyncio.wait_for(self.resume.wait(), timeout=WAIT_SECONDS)

    @override
    async def index_entity_markdown(
        self,
        entity: Entity,
        content: str | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        await self._pause()
        await super().index_entity_markdown(entity, content, session=session)

    @override
    async def index_entity_file(self, entity: Entity, session: AsyncSession | None = None) -> None:
        await self._pause()
        await super().index_entity_file(entity, session=session)


@dataclass
class QueuedDeletes:
    """Instrument queued writers so the test knows their DELETE waits on writer A."""

    dialect: str
    started: list[asyncio.Event] = field(default_factory=list)
    backend_pids: list[int] = field(default_factory=list)
    deleted_rows: list[int] = field(default_factory=list)

    def instrument(self, writer: SearchService) -> None:
        started = asyncio.Event()
        self.started.append(started)
        original: Callable[..., Awaitable[None]] = writer.repository.delete_by_entity_id
        session_maker = writer.repository.session_maker

        async def sqlite_delete(entity_id: int, session: AsyncSession | None = None) -> None:
            # Trigger: SQLite runs this DELETE as the first write of B's transaction.
            # Why: the statement trace fires when SQLite starts executing it; while A holds
            #   the database write lock it cannot finish, so B is now queued behind A.
            # Outcome: the test releases A only after every queued DELETE has started.
            async with db.scoped_session(session_maker, session) as active:
                connection = await active.connection()
                driver = (await connection.get_raw_connection()).driver_connection
                assert driver is not None
                loop = asyncio.get_running_loop()

                def on_statement(sql: str) -> None:
                    if sql.startswith("DELETE FROM search_index"):
                        loop.call_soon_threadsafe(started.set)

                await driver.set_trace_callback(on_statement)
                try:
                    await original(entity_id=entity_id, session=active)
                finally:
                    await driver.set_trace_callback(None)
                changes = await active.execute(text("SELECT changes()"))
                self.deleted_rows.append(int(changes.scalar_one()))

        async def postgres_delete(entity_id: int, session: AsyncSession | None = None) -> None:
            async with db.scoped_session(session_maker, session) as active:
                pid = (await active.execute(text("SELECT pg_backend_pid()"))).scalar_one()
                self.backend_pids.append(int(pid))
                started.set()
                await original(entity_id=entity_id, session=active)
                # Rows this transaction has deleted so far: READ COMMITTED decides here
                # whether the queued DELETE saw the rows writer A committed meanwhile.
                deleted = await active.execute(
                    text(
                        "SELECT coalesce(sum(n_tup_del), 0) FROM pg_stat_xact_user_tables "
                        "WHERE relname = 'search_index'"
                    )
                )
                self.deleted_rows.append(int(deleted.scalar_one()))

        delete = sqlite_delete if self.dialect == "sqlite" else postgres_delete
        setattr(writer.repository, "delete_by_entity_id", delete)

    async def wait_until_blocked(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in self.started)), timeout=WAIT_SECONDS
        )
        if self.dialect == "sqlite":
            return
        await asyncio.wait_for(self._postgres_lock_waits(session_maker), timeout=WAIT_SECONDS)

    async def _postgres_lock_waits(self, session_maker) -> None:
        async with session_maker() as observer:
            while True:
                waiting = (
                    await observer.execute(
                        text(
                            "SELECT count(*) FROM unnest(CAST(:pids AS integer[])) AS w(pid) "
                            "WHERE cardinality(pg_blocking_pids(w.pid)) > 0"
                        ),
                        {"pids": self.backend_pids},
                    )
                ).scalar_one()
                await observer.commit()
                if waiting == len(self.backend_pids):
                    return


@dataclass(frozen=True)
class RaceOutcome:
    rows: Counter[RowKey]
    errors: list[str]
    queued_deleted_rows: list[int]


async def _race_behind_open_replacement(
    search_service: SearchService,
    app_config,
    project_id: int,
    *,
    dialect: str,
    paused_snapshot: Entity,
    queued_snapshot: Entity,
    content_for: Callable[[Entity], str | None],
) -> RaceOutcome:
    """Writer A pauses after its DELETE; writers B-D start and queue; then A resumes."""
    session_maker = search_service.session_maker
    paused = ReplacementPausedAfterDelete(_writer(search_service, app_config, project_id))
    queued = QueuedDeletes(dialect=dialect)
    queued_writers = [
        _writer(search_service, app_config, project_id) for _ in range(QUEUED_WRITERS)
    ]
    for writer in queued_writers:
        queued.instrument(writer)

    paused_task = asyncio.create_task(
        paused.index_entity_data(paused_snapshot, content=content_for(paused_snapshot))
    )
    await asyncio.wait_for(paused.deleted.wait(), timeout=WAIT_SECONDS)
    queued_tasks = [
        asyncio.create_task(
            writer.index_entity_data(queued_snapshot, content=content_for(queued_snapshot))
        )
        for writer in queued_writers
    ]
    try:
        await queued.wait_until_blocked(session_maker)
    finally:
        paused.resume.set()
    results = await asyncio.gather(paused_task, *queued_tasks, return_exceptions=True)
    errors = [repr(result) for result in results if isinstance(result, BaseException)]
    return RaceOutcome(
        rows=await _stored_rows(session_maker, project_id, paused_snapshot.id),
        errors=errors,
        queued_deleted_rows=sorted(queued.deleted_rows),
    )


def _note_content(entity: Entity) -> str:
    return f"Refresh race body for {len(entity.observations)} observations"


def _no_content(entity: Entity) -> None:
    return None


# --- Same snapshot, four writers ---


@pytest.mark.asyncio
async def test_four_concurrent_refreshes_of_one_note_leave_one_projection(
    search_service, app_config, test_project
):
    session_maker = search_service.session_maker
    entity_id = await _seed_note(session_maker, test_project.id, ["alpha fact", "beta fact"])
    snapshot = await _snapshot(session_maker, test_project.id, entity_id)
    writers = [_writer(search_service, app_config, test_project.id) for _ in range(4)]

    for round_number in range(5):
        start = asyncio.Event()

        async def refresh(writer: SearchService) -> None:
            await start.wait()
            await writer.index_entity_data(snapshot, content=_note_content(snapshot))

        tasks = [asyncio.create_task(refresh(writer)) for writer in writers]
        start.set()
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=WAIT_SECONDS
        )

        assert [result for result in results if isinstance(result, BaseException)] == []
        assert await _stored_rows(session_maker, test_project.id, entity_id) == (
            _projection_of(snapshot)
        ), f"round {round_number}"


@pytest.mark.asyncio
async def test_refreshes_queued_behind_an_open_replacement_leave_one_projection(
    search_service, app_config, test_project, engine_factory
):
    session_maker = search_service.session_maker
    entity_id = await _seed_note(session_maker, test_project.id, ["alpha fact", "beta fact"])
    snapshot = await _snapshot(session_maker, test_project.id, entity_id)
    await search_service.index_entity_data(snapshot, content=_note_content(snapshot))

    outcome = await _race_behind_open_replacement(
        search_service,
        app_config,
        test_project.id,
        dialect=engine_factory[0].dialect.name,
        paused_snapshot=snapshot,
        queued_snapshot=snapshot,
        content_for=_note_content,
    )

    assert outcome.errors == [], outcome
    assert outcome.rows == _projection_of(snapshot), outcome


# --- Successive versions: observation rows re-created with new ids ---


async def _two_versions(search_service, project_id: int) -> tuple[Entity, Entity]:
    session_maker = search_service.session_maker
    entity_id = await _seed_note(session_maker, project_id, ["version one fact", "shared fact"])
    older = await _snapshot(session_maker, project_id, entity_id)
    await search_service.index_entity_data(older, content=_note_content(older))
    await _accept_new_observations(
        session_maker, project_id, entity_id, ["version two fact", "shared fact"]
    )
    newer = await _snapshot(session_maker, project_id, entity_id)
    assert not {o.id for o in older.observations} & {o.id for o in newer.observations}
    return older, newer


@pytest.mark.asyncio
@pytest.mark.parametrize("paused_version", ["older", "newer"])
async def test_refreshes_of_successive_versions_leave_only_the_last_writers_rows(
    search_service, app_config, test_project, paused_version, engine_factory
):
    """Whichever version commits last owns the projection; nothing of the other survives."""
    older, newer = await _two_versions(search_service, test_project.id)
    paused_snapshot, queued_snapshot = (
        (older, newer) if paused_version == "older" else (newer, older)
    )

    outcome = await _race_behind_open_replacement(
        search_service,
        app_config,
        test_project.id,
        dialect=engine_factory[0].dialect.name,
        paused_snapshot=paused_snapshot,
        queued_snapshot=queued_snapshot,
        content_for=_note_content,
    )

    assert outcome.errors == [], outcome
    # The queued writers commit after the paused one, so their version is the last write.
    assert outcome.rows == _projection_of(queued_snapshot), outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("paused_version", ["older", "newer"])
async def test_successive_version_race_converges_on_the_next_refresh(
    search_service, app_config, test_project, paused_version, engine_factory
):
    """Whatever the race leaves behind, one more refresh of the current note repairs it."""
    older, newer = await _two_versions(search_service, test_project.id)
    paused_snapshot, queued_snapshot = (
        (older, newer) if paused_version == "older" else (newer, older)
    )
    await _race_behind_open_replacement(
        search_service,
        app_config,
        test_project.id,
        dialect=engine_factory[0].dialect.name,
        paused_snapshot=paused_snapshot,
        queued_snapshot=queued_snapshot,
        content_for=_note_content,
    )

    await search_service.index_entity_data(newer, content=_note_content(newer))

    assert await _stored_rows(search_service.session_maker, test_project.id, newer.id) == (
        _projection_of(newer)
    )


@pytest.mark.asyncio
async def test_stale_observation_rows_from_a_race_are_removed_by_the_reindex_sweep(
    search_service, app_config, test_project, engine_factory
):
    """The reindex sweep drops observation rows whose observation no longer exists."""
    older, newer = await _two_versions(search_service, test_project.id)
    await _race_behind_open_replacement(
        search_service,
        app_config,
        test_project.id,
        dialect=engine_factory[0].dialect.name,
        paused_snapshot=older,
        queued_snapshot=newer,
        content_for=_note_content,
    )

    await search_service.repository.purge_stale_search_rows()

    assert await _stored_rows(search_service.session_maker, test_project.id, newer.id) == (
        _projection_of(newer)
    )


# --- A move: same row ids, new permalinks ---


@pytest.mark.asyncio
@pytest.mark.parametrize("paused_version", ["before_move", "after_move"])
async def test_refreshes_racing_a_move_leave_only_the_last_writers_rows(
    search_service, app_config, test_project, paused_version, engine_factory
):
    session_maker = search_service.session_maker
    entity_id = await _seed_note(session_maker, test_project.id, ["moving fact"])
    before_move = await _snapshot(session_maker, test_project.id, entity_id)
    await search_service.index_entity_data(before_move, content=_note_content(before_move))
    await _move_note(session_maker, entity_id, "refresh-race/moved-note")
    after_move = await _snapshot(session_maker, test_project.id, entity_id)
    paused_snapshot, queued_snapshot = (
        (before_move, after_move) if paused_version == "before_move" else (after_move, before_move)
    )

    outcome = await _race_behind_open_replacement(
        search_service,
        app_config,
        test_project.id,
        dialect=engine_factory[0].dialect.name,
        paused_snapshot=paused_snapshot,
        queued_snapshot=queued_snapshot,
        content_for=_note_content,
    )

    assert outcome.errors == [], outcome
    assert outcome.rows == _projection_of(queued_snapshot), outcome


# --- Rows without a permalink: non-Markdown resources ---


@pytest.mark.asyncio
async def test_resource_refreshes_queued_behind_an_open_replacement_keep_one_row(
    search_service, app_config, test_project, engine_factory
):
    session_maker = search_service.session_maker
    entity_id = await _seed_resource(session_maker, test_project.id)
    snapshot = await _snapshot(session_maker, test_project.id, entity_id)
    assert snapshot.permalink is None
    await search_service.index_entity_data(snapshot)

    outcome = await _race_behind_open_replacement(
        search_service,
        app_config,
        test_project.id,
        dialect=engine_factory[0].dialect.name,
        paused_snapshot=snapshot,
        queued_snapshot=snapshot,
        content_for=_no_content,
    )

    assert outcome.rows == Counter({("entity", entity_id, None): 1}), outcome
    assert outcome.errors == [], outcome
