# pact eval harness — v2

Two-stage pipeline: **normalize + classify** (stage1) → **reconcile** (stage2).

## Structure

```
newtest/
  stage1/   normalize.py  classify.py  eval_all.py  run.py  score.py
            transcripts/  gold/  outputs/
  stage2/   reconcile.py  score_reconcile.py  eval_reconcile.py
            test_reconcile.py  llm.py  gold/
```

Run order:
```
cd stage1 && python eval_all.py          # normalize + classify all transcripts → outputs/
cd stage2 && python eval_reconcile.py --diff   # reconcile all + score vs gold
```

---

## Stage 1 — Normalize + Classify

Segments messages into atomic spans (normalize) then applies multi-label classification
(classify). Scored against hand-written gold span files.

| transcript | seg R | over | under | spur | P | R | F1 | exact |
|---|---|---|---|---|---|---|---|---|
| ent-01-easy-cutover | 100% | 6 | 1 | 0 | 0.96 | 0.98 | 0.97 | 94% |
| ent-02-medium-support-ai | 100% | 21 | 1 | 1 | 0.88 | 1.00 | 0.94 | 91% |
| ent-03-hard-data-platform | 100% | 33 | 2 | 4 | 0.90 | 0.95 | 0.92 | 90% |
| ent-04-hiking-trip | 100% | 15 | 1 | 0 | 0.85 | 0.94 | 0.89 | 81% |

**ent-05 stage1 gold not yet written** — pipeline ran (100 spans from 50 messages) but no label score available.

Columns: **seg R** = segment recall, **over/under** = over/under-splits (diagnostic),
**spur** = spurious spans, **P/R/F1** = label micro-scores, **exact** = all labels match.

---

## Stage 2 — Reconcile

Walks classified spans in batches; LLM emits typed ops (create, stance, status, supersede,
decide, alias, merge, attach, skip). Code validates every op. Hard rules enforced in code:

- **Author-only stances**: participant is always the span's author; model cannot set it
- **Inferred-stance check**: stance op rejected (flagged, not counted) unless span is
  stance-labelled AND names the option by title, alias, poll letter, or context note.
  Short replies (≤4 words: "+1", "A", "None") are exempt — they answer polls/objection checks.
- **Variant dedup**: new option whose distinctive name tokens are contained in an existing
  option becomes an alias instead of a second row
- **Consensus-conflict flag**: claim asserting others' stances is checked against the ledger;
  mismatch raises a flag and marks the claim conflicting
- **Duplicate option redirect**: create on an existing option (by title/alias) becomes attach

| transcript | options | opt_status | cells | nonblank | questions | elements | histories | flags | dups |
|---|---|---|---|---|---|---|---|---|---|
| ent-01-easy-cutover | 3/3 | 3/3 | 18/18 | 10/10 | 1/1 | 1/1 | 1/1 | — | 0 |
| ent-02-medium-support-ai | 3/3 | 3/3 | 18/18 | 11/11 | 1/1 | 5/5 | 2/2 | — | 0 |
| ent-03-hard-data-platform | 5/5 | 5/5 | 35/35 | 22/22 | 2/2 | 8/8 | 4/4 | 2/2 | 0 |
| ent-04-hiking-trip | 7/7 | 7/7 | 39/42 | 21/23 | 3/3 | 11/12 | 5/6 | 1/1 | 0 |
| **ent-05-medium-feature-priority** | **5/5** | **5/5** | **21/25** | **10/10** | **2/2** | **4/5** | **2/2** | **1/1** | **0** |

ent-05 is the generalization transcript (not seen during prompt development).

**ent-05 inferred stances excluded (9)**: shows the check working on a new transcript.
Options, option status, non-blank cells, questions, histories, and the consensus-conflict
flag all pass on first run.

### Known gaps (ent-04 baseline, not fixed)

| check | failure |
|---|---|
| cells | saturday/Leo: blank expected, got support |
| cells | carpool/Maya: support expected, got conditional |
| cells | carpool/Sam: conditional expected, got blank |
| elements | driver blocker (m56): open expected, got resolved |
| histories | sunol/Leo: conditional→support expected, got support only |

These were intentionally left to establish an honest generalization baseline before
changing any prompts or rules.

---

## Transcripts

| id | topic | msgs | difficulty |
|---|---|---|---|
| ent-01 | software cutover date | 39 | easy |
| ent-02 | support AI vendor selection | 53 | medium |
| ent-03 | data platform end-state | 101 | hard |
| ent-04 | hiking trip planning | 75 | medium (multi-question, casual) |
| ent-05 | Q1 feature prioritization | 50 | medium (new domain) |
