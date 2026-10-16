# pact

Decision intelligence pipeline. Two parallel tracks: **enterprise** (email chains, meeting transcripts) and **social** (chat transcripts).

## Structure

```
enterprise/
  run_fast.py          # main entry point — full pipeline, parallel reconcile
  run.py               # sequential reconcile (slower, reference)
  ingest.py            # parse email chain / meeting transcript → source + spans
  summarize.py         # 2-pass Sonnet summary (conversation → structured brief)
  llm.py               # AnthropicLLM wrapper
  eval.py              # batch eval runner
  evals/
    sources/           # ent-08 … ent-10, real-01 … real-03, long-01
    outputs/           # *.fast_result.json (one per source)

social/
  normalize.py         # segment messages → atomic spans
  classify.py          # multi-label span classification
  reconcile.py         # sequential reconcile (Registry + typed ops)
  reconcile_fast.py    # 4-step parallel reconcile (main impl)
  run.py               # social pipeline entry point
  llm.py
  eval_spans.py        # stage-1 eval (normalize + classify)
  eval_reconcile.py    # stage-2 eval (reconcile vs gold)
  score_spans.py
  score_reconcile.py
  tests/
```

---

## Quick start

```bash
# Enterprise — run full pipeline on a source file
cd /path/to/pact
python enterprise/run_fast.py enterprise/evals/sources/real-02-email-benefit-expense-chain.txt

# Output written to enterprise/evals/outputs/<name>.fast_result.json
```

---

## Enterprise pipeline (`run_fast.py`)

```
ingest → normalize (Haiku) → classify (Haiku) → reconcile_fast (parallel Haiku) → summarize (Sonnet)
```

1. **Ingest** — parse email chain or meeting transcript; extract speakers, subject, message metadata
2. **Normalize** — segment each message into atomic spans (Haiku)
3. **Classify** — apply multi-label classification to each span (Haiku)
4. **Reconcile fast** — 4-step parallel pass (see below)
5. **Summarize** — 2-pass Sonnet: conversation → main point → structured brief

Output is a JSON document with `elements`, `events`, `warnings`, `summary`, and `_enterprise` metadata.

---

## 4-step parallel reconcile (`reconcile_fast.py`)

Replaces the sequential O(batches × latency) loop with parallel passes:

| Step | What | Model | Parallelism |
|------|------|-------|-------------|
| 1. Extract | Each batch sees 8 prior spans as context; emits candidate questions/options/claims | Haiku | ThreadPoolExecutor |
| 2. Consolidate | One call merges paraphrase questions, links options to questions, drops non-alternatives | Haiku | Single call |
| 3. Build | Deterministic code pass — creates Registry in transcript order; writes create/attach/alias events | — | Code only |
| 4. Stances | Frozen registry; emits stance/status/decide/supersede/attach/skip per span | Haiku | ThreadPoolExecutor |

**Speedup**: 635-span source: ~29s vs ~20 min sequential.

---

## Eval sources

| id | type | topic |
|----|------|-------|
| ent-08 | email | vendor selection with private note |
| ent-09 | email | offsite reversal / forward |
| ent-10 | meeting | go-live decision |
| real-01 | email | Alpha Evolve |
| real-02 | email | benefit expense chain |
| real-03 | meeting | T-gov status |
| long-01 | meeting | leadership clinical intelligence (635 spans) |

All 7 sources have been run through `run_fast.py`; outputs are in `enterprise/evals/outputs/`.

---

## Social pipeline

```
normalize + classify (stage 1) → reconcile (stage 2)
```

Scored against hand-written gold files in `social/evals/gold/`.

### Stage 1 results (normalize + classify)

| transcript | seg R | P | R | F1 | exact |
|---|---|---|---|---|---|
| ent-01-easy-cutover | 100% | 0.96 | 0.98 | 0.97 | 94% |
| ent-02-medium-support-ai | 100% | 0.88 | 1.00 | 0.94 | 91% |
| ent-03-hard-data-platform | 100% | 0.90 | 0.95 | 0.92 | 90% |
| ent-04-hiking-trip | 100% | 0.85 | 0.94 | 0.89 | 81% |

### Stage 2 results (reconcile)

| transcript | options | opt_status | cells | questions | elements | histories | flags |
|---|---|---|---|---|---|---|---|
| ent-01 | 3/3 | 3/3 | 18/18 | 1/1 | 1/1 | 1/1 | — |
| ent-02 | 3/3 | 3/3 | 18/18 | 1/1 | 5/5 | 2/2 | — |
| ent-03 | 5/5 | 5/5 | 35/35 | 2/2 | 8/8 | 4/4 | 2/2 |
| ent-04 | 7/7 | 7/7 | 39/42 | 3/3 | 11/12 | 5/6 | 1/1 |
| ent-05 | 5/5 | 5/5 | 21/25 | 2/2 | 4/5 | 2/2 | 1/1 |

---

## Hard rules (reconcile)

- **Author-only stances**: participant is always the span's author; model cannot override
- **Inferred-stance check**: stance op rejected unless span is stance-labelled AND names the option; short replies (≤4 words) are exempt
- **Variant dedup**: new option whose name tokens are contained in an existing option becomes an alias
- **Consensus-conflict flag**: claim asserting others' stances is checked against ledger; mismatch raises a flag
- **Duplicate option redirect**: create on existing option (by title/alias) becomes attach
