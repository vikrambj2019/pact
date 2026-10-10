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
pip install -r api/requirements.txt
python -m uvicorn mvp1_b2e.api.main:app --port 8506
```
