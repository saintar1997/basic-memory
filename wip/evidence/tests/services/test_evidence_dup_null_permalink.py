"""Search rows without a permalink are replaced, never appended (B2 evidence).

Only one kind of search row is written without a permalink on main: the entity row of a
non-Markdown resource (the batch indexer stores those entities with ``permalink=None``) or
of a legacy Markdown row that predates mandatory permalinks. Observation and relation rows
always carry a derived permalink, even when the entity's own permalink is NULL or the
relation target is unresolved.

The row key used by ``index_item``'s replacement DELETE is
``(permalink, type, project_id)``; ``permalink = NULL`` never matches, so the repository
API alone cannot replace such a row. These tests separate the two contracts:

- the production path, ``SearchService.index_entity_data`` (and the accepted-note refresh),
  which deletes the entity's whole projection by ``entity_id`` in the same transaction
  before writing the replacement;
- the bare repository API, ``index_item`` called twice for the same NULL-permalink row.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from basic_memory import db
from basic_memory.indexing.accepted_note_search import build_accepted_note_search_row
from basic_memory.models import Entity, Observation, Relation
from basic_memory.repository.accepted_note_search_repository import (
    AcceptedNoteSearchRepository,
)
from basic_memory.repository.search_index_row import SearchIndexRow
from basic_memory.schemas.search import SearchItemType


type RowKey = tuple[str, int, str | None]


async def _create_entity(
    session_maker,
    entity_repository,
    *,
    content_type: str,
    file_path: str,
    permalink: str | None,
) -> Entity:
    now = datetime.now(timezone.utc)
    async with db.scoped_session(session_maker) as session:
        created = await entity_repository.create(
            session,
            {
                "project_id": entity_repository.project_id,
                "title": file_path.rsplit("/", 1)[-1],
                "note_type": "file" if content_type != "text/markdown" else "note",
                "permalink": permalink,
                "file_path": file_path,
                "content_type": content_type,
                "created_at": now,
                "updated_at": now,
            },
        )
    return created


async def _reload(session_maker, entity_repository, entity_id: int) -> Entity:
    async with db.scoped_session(session_maker) as session:
        entity = await entity_repository.find_by_id(session, entity_id)
    assert entity is not None
    return entity


async def _stored_rows(session_maker, project_id: int, entity_id: int) -> list[RowKey]:
    """Every physical search_index row owned by one entity, duplicates included."""
    async with db.scoped_session(session_maker) as session:
        result = await session.execute(
            text(
                "SELECT type, id, permalink FROM search_index "
                "WHERE project_id = :project_id AND entity_id = :entity_id "
                "ORDER BY type, id"
            ),
            {"project_id": project_id, "entity_id": entity_id},
        )
        return [(str(row[0]), int(row[1]), row[2]) for row in result.all()]


# --- Production path: SearchService ---


@pytest.mark.asyncio
async def test_search_service_refreshes_a_resource_row_without_permalink_in_place(
    search_service, entity_repository, session_maker, test_project
):
    """A re-indexed non-Markdown resource keeps exactly one NULL-permalink row."""
    resource = await _create_entity(
        session_maker,
        entity_repository,
        content_type="application/pdf",
        file_path="assets/report.pdf",
        permalink=None,
    )
    assert resource.permalink is None
    assert not resource.is_markdown

    await search_service.index_entity_data(resource)
    await search_service.index_entity_data(resource)
    await search_service.index_entity_data(resource)

    assert await _stored_rows(session_maker, test_project.id, resource.id) == [
        (SearchItemType.ENTITY.value, resource.id, None)
    ]


@pytest.mark.asyncio
async def test_null_entity_permalink_still_gives_observations_and_relations_an_address(
    search_service, entity_repository, session_maker, test_project
):
    """Only the entity row can lack a permalink; its observation and relation rows cannot.

    A legacy Markdown row without a permalink, an observation, and a relation whose target
    does not exist yet: re-indexing it repeatedly leaves one row per (type, id).
    """
    note = await _create_entity(
        session_maker,
        entity_repository,
        content_type="text/markdown",
        file_path="legacy/no-permalink.md",
        permalink=None,
    )
    async with db.scoped_session(session_maker) as session:
        session.add(
            Observation(
                project_id=test_project.id,
                entity_id=note.id,
                category="fact",
                content="legacy rows keep observations addressable",
            )
        )
        session.add(
            Relation(
                project_id=test_project.id,
                from_id=note.id,
                to_id=None,
                to_name="Not Written Yet",
                relation_type="links_to",
            )
        )
    note = await _reload(session_maker, entity_repository, note.id)
    assert note.permalink is None
    assert note.outgoing_relations[0].to_entity is None

    for _ in range(3):
        await search_service.index_entity_data(note, content="legacy body")

    rows = await _stored_rows(session_maker, test_project.id, note.id)
    assert [(kind, row_id) for kind, row_id, _ in rows] == [
        (SearchItemType.ENTITY.value, note.id),
        (SearchItemType.OBSERVATION.value, note.observations[0].id),
        (SearchItemType.RELATION.value, note.outgoing_relations[0].id),
    ]
    permalinks = {kind: permalink for kind, _, permalink in rows}
    assert permalinks[SearchItemType.ENTITY.value] is None
    assert permalinks[SearchItemType.OBSERVATION.value] is not None
    assert permalinks[SearchItemType.RELATION.value] is not None


@pytest.mark.asyncio
async def test_accepted_note_refresh_replaces_a_row_without_permalink(
    entity_repository, session_maker, test_project
):
    """The accepted-note hot path deletes by entity_id first, inside the caller's session."""
    note = await _create_entity(
        session_maker,
        entity_repository,
        content_type="text/markdown",
        file_path="accepted/no-permalink.md",
        permalink=None,
    )
    repository = AcceptedNoteSearchRepository(project_id=test_project.id)
    for body in ("first accepted body", "second accepted body"):
        row = build_accepted_note_search_row(
            entity_id=note.id,
            title=note.title,
            note_type=note.note_type,
            entity_metadata=None,
            permalink=None,
            file_path=note.file_path,
            search_content=body,
            created_at=note.created_at,
            updated_at=note.updated_at,
            project_id=test_project.id,
        )
        async with db.scoped_session(session_maker) as session:
            await repository.refresh_entity(session, row)

    assert await _stored_rows(session_maker, test_project.id, note.id) == [
        (SearchItemType.ENTITY.value, note.id, None)
    ]


# --- Repository API: index_item on its own ---


@pytest.mark.asyncio
async def test_index_item_twice_without_permalink_keeps_one_row(
    search_repository, entity_repository, session_maker, test_project
):
    """The bare repository contract: a second index_item replaces the first row.

    No production caller relies on this -- every caller deletes by entity_id first in the
    same transaction -- but the method is documented as "Index or update a single item".
    """
    resource = await _create_entity(
        session_maker,
        entity_repository,
        content_type="image/png",
        file_path="assets/diagram.png",
        permalink=None,
    )
    row = SearchIndexRow(
        id=resource.id,
        type=SearchItemType.ENTITY.value,
        title=resource.title,
        permalink=None,
        file_path=resource.file_path,
        entity_id=resource.id,
        metadata={"note_type": "file"},
        created_at=resource.created_at,
        updated_at=resource.updated_at,
        project_id=test_project.id,
    )

    await search_repository.index_item(row)
    await search_repository.index_item(row)

    assert await _stored_rows(session_maker, test_project.id, resource.id) == [
        (SearchItemType.ENTITY.value, resource.id, None)
    ]
