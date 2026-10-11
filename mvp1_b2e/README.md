# mvp1_b2e — Pact Enterprise MVP (Phase 1)

Converts meeting transcripts and email chains into an interactive decision brief.
One decision workspace, one confirmed question, source-grounded brief, human records the decision.

See [`docs/Pact_MVP_Technical_Spec.md`](../docs/Pact_MVP_Technical_Spec.md) for the full spec.

## Structure

```
mvp1_b2e/
  api/            FastAPI backend — commands, validation, state transitions
  worker/         Python worker — parsing, extraction, evidence review, exports
  ui/             React/TypeScript — sources, brief, chat, diffs, recording, search
  goldendataset/  Evaluation harness — run pipeline against gold-labeled transcripts
  archive/        Historical snapshots (Pact_Pipeline_Fix, pact_update_v5)
```

## Stack

- **UI**: React + TypeScript
- **API**: Python / FastAPI
- **Worker**: Python (document parsing, LLM extraction)
- **DB**: PostgreSQL
- **Storage**: local object storage (Phase 1)
- **LLM**: Anthropic (claude-sonnet-4-6 for extraction, claude-haiku-4-5 for fast ops)

## Worker pipeline

The `worker/` package runs the full extraction pipeline:

| File | Role |
|---|---|
| `parser.py` | Parse email chains / transcripts / plain text into SourceUnits |
| `dedup.py` | Mark duplicate units (exact-match: same author + timestamp) |
| `chunker.py` | Split units into LLM-sized chunks |
| `extractor.py` | Per-chunk LLM extraction (two-pass: main + coverage) |
| `merger.py` | Merge multi-chunk extractions, reconcile stance supersessions |
| `synthesizer.py` | Build final brief, assign server UUIDs, validate source refs |
| `extraction_reviewer.py` | LLM audit of extraction quality against source units |
| `reviewer.py` | Claim evidence scanner (used in claim_review jobs) |
| `llm.py` | Async Anthropic SDK client |

## Goldendataset evaluation

Run the pipeline against labelled transcripts and score against a gold answer key:

```bash
cd mvp1_b2e
python goldendataset/run_gold.py \
  --credentials /path/to/.env \
  --model claude-sonnet-4-6 \
  --skip-reviewer

python goldendataset/compare_gold.py \
  --credentials /path/to/.env \
  --dataset ent-01-easy-cutover
```

Gold files live in `goldendataset/output/`. Run outputs are written to `goldendataset/runs/`.

## Start (Phase 1)

```bash
pip install -r api/requirements.txt
python -m uvicorn mvp1_b2e.api.main:app --port 8506
```
