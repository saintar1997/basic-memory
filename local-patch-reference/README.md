# Local patch reference (not for merging)

This folder carries a downstream patch so a cloud session can port what upstream still lacks. It lives
only on the `local-patch-reference` branch of this fork and must never appear in a pull request.

## Where it comes from

One user runs Basic Memory 0.23.2 as a uv tool on Windows, with several MCP clients (Claude Code, Codex,
Copilot) writing to the same SQLite project. Two local repairs were applied to the installed package:

1. **September 2026 — duplicate FTS rows.** Two writes without a permalink produced two `search_index`
   rows, and four concurrent batch writes produced four. The repair restored identity-based single-row
   replacement and entity-scoped batch replacement inside the insert transaction
   (`search_repository_base.py`).
2. **October 2026 — notes disappearing from search during refresh.** Overlapping MCP writers left an
   existing entity with zero vector chunks, and another worker reported vanished manifest rows. Logs
   suggested that an empty FTS interval during refresh was read as an entity deletion. The repair:
   - replaces each entity's search projection in one transaction, file notes included;
   - in SQLite vector preparation, takes the database write lock with `BEGIN IMMEDIATE` and re-reads the
     source and manifest state before mutating, so a delayed, older vector plan cannot overwrite a newer
     manifest;
   - checks current entity eligibility with the existing shared embed policy, so delayed work cannot
     recreate vectors after an embedding opt-out or a deletion;
   - in PostgreSQL's shared bulk path, removes the entity's previous projection inside the insertion
     transaction.

Behavior the repair keeps: a saved note stays retrievable through `SearchService` and the search
repository while its index refreshes; concurrent vector publication completes or safely retries; real
deletions and embedding opt-outs still remove derived vectors.

## Files

- `local-delta-vs-v0.23.2.diff` — the complete difference between the installed package and the
  `v0.23.2` tag, as a patch against `src/basic_memory/` (6 files; it applies cleanly to `v0.23.2`).
- `tests/` — the 13 focused tests used locally, written against the installed 0.23.2 layout:
  - `test_index_refresh.py`: empty-projection race, delayed update, deletion and opt-out, failed file
    reads, Markdown-to-file replacement;
  - `test_multiprocess_refresh.py`: two spawned writers on one SQLite database;
  - `test_postgres_refresh.py`: PostgreSQL-compatible replacement and rollback SQL, run on SQLite with
    matching constraints and an adapted FTS seam.

## What was and was not verified locally

- Before the repair, the refresh tests (one process and two processes), the delayed-generation test and
  the concurrent opt-out test failed. After it, all 13 passed against both the source snapshot and the
  installed package, as did the earlier null-permalink and four-writer duplicate checks.
- Not verified: an actual PostgreSQL runtime or PostgreSQL concurrency, the upstream full test suite, and
  every possible interleaving.

## Upstream since 0.23.2

`main` has refactored search heavily (`FtsBackend`, `SearchReader`, `ProjectScope`, vector adapters) and
includes `fix(core): replace an entity's search rows in one transaction (#1623)` and
`fix(core): create vector storage at initialization, never at runtime (#1656)`. Part of this patch may
already be covered there.
