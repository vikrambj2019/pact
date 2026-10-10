# mvp1_b2e — Pact Enterprise MVP (Phase 1)

Converts meeting transcripts and email chains into an interactive decision brief.
One decision workspace, one confirmed question, source-grounded brief, human records the decision.

See [`docs/Pact_MVP_Technical_Spec.md`](../docs/Pact_MVP_Technical_Spec.md) for the full spec.

## Structure

```
mvp1_b2e/
  api/        FastAPI backend — commands, validation, state transitions
  worker/     Python worker — parsing, extraction, evidence review, exports
  ui/         React/TypeScript — sources, brief, chat, diffs, recording, search
  pact/       Pact core — observe, check, interpret (shared with mvp1_c2c)
```

## Stack

- **UI**: React + TypeScript
- **API**: Python / FastAPI
- **Worker**: Python (document parsing, LLM extraction)
- **DB**: PostgreSQL
- **Storage**: local object storage (Phase 1)
- **LLM**: Anthropic (claude-sonnet-4-6 for extraction, claude-haiku-4-5 for fast ops)

## Start (Phase 1)

```bash
pip install -r mvp1_b2e/requirements.txt
cp .env.example .env
# Create the PostgreSQL database and edit DATABASE_URL in .env first.
(cd mvp1_b2e && python -m alembic upgrade head)
python -m uvicorn mvp1_b2e.api.main:app --port 8506
```

In a second terminal, from the repository root:

```bash
python -m mvp1_b2e.jobs.worker
```

Current status: Step 1 foundation only. The parser intentionally fails with
`NotImplementedError`; ingestion, extraction, deliberation, decision recording,
and React UI are subsequent implementation steps. No LLM calls are implemented.
PDF extracted-word limits must be enforced by the future parser before persisting
units; text uploads currently enforce a conservative workspace word budget.

For regression tests:

```bash
pip install -r mvp1_b2e/requirements-dev.txt
python -m pytest -q mvp1_b2e/tests
```

Tests use an isolated SQLite database by default. For PostgreSQL integration,
set `PACT_TEST_DATABASE_URL` to a **dedicated disposable database**; the tests
create and drop Pact tables there. Never point it at your working database.

Upload retries accept `Idempotency-Key` (the legacy form field also works).
Same key and input return the original source/job; changed input returns 409.
Question confirmation locks the workspace and creates a revision; clients must
send the current revision number. The worker renews leases, recovers expired
attempts, and discards results from superseded attempts. Handler writes must use
the provided session and must not commit independently.
