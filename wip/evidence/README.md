# Investigation evidence (work in progress, remove before review)

Scratch tests written while checking which behaviors of the downstream concurrency repair
upstream main already provides. They are kept here only so the investigation survives the
session; they are not part of the change and live outside `tests/` and `test-int/`, so the
suites and CI do not collect them. Several fail on purpose on main, and some document
PostgreSQL limits that this branch does not change.

To run one, copy it back to the same relative path under `tests/` or `test-int/`:

```bash
cp wip/evidence/test-int/test_evidence_sqlvec_delayed_vector_plan.py test-int/
BASIC_MEMORY_ENV=test uv run pytest -p pytest_mock -q --no-cov --import-mode=importlib \
  test-int/test_evidence_sqlvec_delayed_vector_plan.py
```

For PostgreSQL add `BASIC_MEMORY_TEST_POSTGRES=1` and
`BASIC_MEMORY_TEST_POSTGRES_URL=postgresql://USER:PASSWORD@HOST:PORT/DB` (pgvector 0.8+).

| File | What it probes |
| --- | --- |
| `test-int/test_evidence_sqlvec_delayed_vector_plan.py` | SQLite delayed vector plans (update, delete, opt-out), write lock, two processes |
| `tests/repository/test_evidence_pgvec_delayed_vector_plan.py` | the same on PostgreSQL, pgvector and an external index |
| `tests/repository/test_evidence_pgvec_prepare_write_locks.py` | which locks a PostgreSQL prepare write takes |
| `tests/repository/test_evidence_pgvec_residual_window.py` | the window a lock-free re-read cannot close on PostgreSQL |
| `test-int/test_evidence_proj_refresh_concurrency.py` | a note stays searchable while another worker refreshes it |
| `tests/services/test_evidence_proj_search_projection_replacement.py` | projection replacement for Markdown and file notes |
| `test-int/test_evidence_dup_concurrent_refresh.py` | concurrent refreshes of one note, SQLite and PostgreSQL |
| `test-int/test_evidence_dup_multiprocess_refresh.py` | refreshes from four processes |
| `tests/services/test_evidence_dup_null_permalink.py` | rows without a permalink |
