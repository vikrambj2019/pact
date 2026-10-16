"""Stage 3 (fast) — 3-pass parallel reconcile.

    python reconcile_fast.py evals/outputs/long-01.output.json

Design
------
Sequential reconcile is a bottleneck on long files: each batch needs the full
registry from prior batches, so nothing can run in parallel.

This module breaks that dependency with three passes:

  Pass 1 — Extract (parallel, Haiku)
    Every batch of spans is processed independently, with no registry context.
    Each batch produces raw element candidates: {kind, title, author, span_id, temp_id}.
    Since there's no shared state, all batches fire at once.

  Pass 2 — Dedup (code, no LLM, ~instant)
    Collect every candidate. Apply the same _find_option_by_name / _norm / _core
    logic from reconcile.py to merge duplicates and assign final IDs.
    Result: a frozen Registry with all elements but no stances.

  Pass 3 — Stances (parallel, Haiku)
    Registry is now stable. Every batch fires at once, only emitting:
    stance, status, decide, supersede, attach, skip.
    (No creates — elements already exist.)

Wall time drops from O(batches × latency) to O(latency) × 3 passes.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm import HAIKU, AnthropicLLM, LLM
from reconcile import (
    BATCH,
    CONTEXT_SPANS,
    DEFAULT_STATUS,
    PREFIX,
    SKIP_ONLY,
    STANCES,
    Registry,
    _core,
    _norm,
    changes,
    dump,
    load_classified,
    matrix,
    render_span,
)

# ── Pass 1: Extract prompt ──────────────────────────────────────────────────

EXTRACT_SYSTEM = """You extract decision elements from a group chat. You receive a set of spans
(already labelled by a classifier). Extract every distinct decision element you can see.

Element kinds
- question   a decision being made ("which platform", "which vendor", "go or no-go")
- option     an alternative for a question ("Databricks", "Nov 14-15", "Q2")
- criterion  a factor for evaluating options
- constraint a hard or soft limit that filters options
- claim      an assertion (fact, estimate, objection, prediction). Set claim_kind:
             factual|predictive|causal|priority|objection|consensus|other
             For objections, list the option titles they target in about_options[].
- blocker    an open question / missing info / dependency blocking a decision.
             Set blocker_kind: open_question|missing_info|unresolved_dependency.

Output one entry per distinct element. Use "temp_id" as a short label (e.g. "q1","opt2")
so you can reference it within this batch. If multiple spans describe the same element,
output it once with the first span_id that introduces it.

Do NOT output stances, status changes, or decisions — only element creation.
If a span contains no new element (filler, greeting, stance-only), omit it."""

EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "temp_id": {"type": "string"},
                    "kind": {"type": "string"},
                    "title": {"type": "string"},
                    "span_id": {"type": "string"},
                    "claim_kind": {"type": "string"},
                    "constraint_kind": {"type": "string"},
                    "blocker_kind": {"type": "string"},
                    "about_options": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["temp_id", "kind", "title", "span_id"],
            },
        }
    },
    "required": ["elements"],
}


def _extract_batch(batch_rows: list[dict], llm: LLM) -> list[dict]:
    user = "SPANS:\n" + "\n".join(render_span(r) for r in batch_rows)
    out = llm.structured(
        system=EXTRACT_SYSTEM, user=user, tool_name="extract_elements",
        schema=EXTRACT_SCHEMA, max_tokens=2048,
    )
    return out.get("elements", [])


# ── Pass 2: Dedup (code only) ───────────────────────────────────────────────

def _build_registry_from_candidates(
    all_candidates: list[dict],
    rows_by_span: dict[str, dict],
) -> Registry:
    """Deterministically assign canonical IDs and merge duplicates.

    Returns a Registry whose .el is populated but has no stance/status events.
    """
    from collections import Counter
    from reconcile import Element

    reg = Registry(rows_by_span)
    n: Counter = Counter()

    for cand in all_candidates:
        kind = cand.get("kind")
        if kind not in PREFIX:
            continue
        title = (cand.get("title") or "").strip()
        if not title:
            continue
        span_id = cand.get("span_id", "")
        author = rows_by_span.get(span_id, {}).get("author", "")

        # For options: deduplicate against existing registry using Registry's method
        if kind == "option":
            dup, _how = reg._find_option_by_name(title, [])
            if dup:
                if title not in dup.aliases and title != dup.title:
                    dup.aliases.append(title)
                continue

        # For non-options: simple exact-title dedup
        elif kind != "option":
            existing = next(
                (e for e in reg.live(kind) if _norm(e.title) == _norm(title)), None
            )
            if existing:
                continue

        n[kind] += 1
        eid = f"{PREFIX[kind]}{n[kind]}"
        attrs: dict = {}
        if cand.get("claim_kind"):
            attrs["claim_kind"] = cand["claim_kind"]
        if cand.get("constraint_kind"):
            attrs["constraint_kind"] = cand["constraint_kind"]
        if cand.get("blocker_kind"):
            attrs["blocker_kind"] = cand["blocker_kind"]

        e = Element(
            id=eid, kind=kind, title=title, author=author,
            created_span=span_id, status=DEFAULT_STATUS[kind],
            attrs=attrs,
        )
        reg.el[eid] = e

    reg._n = n
    return reg


# ── Pass 3: Stances prompt ──────────────────────────────────────────────────

STANCE_SYSTEM = """You assign stances and status changes to a frozen decision registry.
The registry already has all elements with final IDs. Do NOT create new elements.

You receive the current REGISTRY and a batch of NEW SPANS. For each span emit only:
- {"op":"stance","option":"opt2","stance":"proposed|support|conditional|oppose|withdrawn|
   undecided|acknowledged","conditions":"...","span":...,"because":[...]}
  Participant is ALWAYS the span's author — you cannot set it.
- {"op":"status","element":"opt3","status":...,"span":...,"because":[...]}
  option: active|dropped|fallback|reopened   blocker: open|resolved
  claim: unverified|conflicting|confirmed    constraint: active|superseded|satisfied
  question: open|reopened
- {"op":"supersede","old":"clm2","new":"clm7","span":...}
- {"op":"decide","question":"q1","chosen":["opt1"],"ruled_out":["opt3"],"conditional":false,
   "conditions":"...","span":...}
- {"op":"attach","element":"con2","span":...}
- {"op":"skip","span":...,"reason":"..."}

Rules
1. Every span gets at least one op (skip is fine for greetings/filler).
2. A stance is only for the span AUTHOR'S OWN stance on an option.
3. "+1", "same", "None" (to "anyone against X?"), poll letters are stances.
4. Statements about what OTHERS think are NOT stances — skip them here.
5. Decide only when someone explicitly closes the question for the group."""

STANCE_OPS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ops": {"type": "array", "items": {"type": "object"}},
        "notes": {"type": "string"},
    },
    "required": ["ops"],
}


def _render_registry_compact(reg: Registry) -> str:
    from reconcile import render_registry
    return render_registry(reg)


def _stance_batch(
    batch_rows: list[dict],
    reg_snapshot: str,
    participants: list[str],
    llm: LLM,
) -> list[dict]:
    user = (
        f"PARTICIPANTS: {', '.join(participants)}\n\n"
        f"REGISTRY:\n{reg_snapshot}\n\n"
        "NEW SPANS:\n" + "\n".join(render_span(r) for r in batch_rows)
    )
    out = llm.structured(
        system=STANCE_SYSTEM, user=user, tool_name="stance_ops",
        schema=STANCE_OPS_SCHEMA, max_tokens=2048,
    )
    return out.get("ops", [])


# ── Driver ──────────────────────────────────────────────────────────────────

def reconcile_fast(
    rows: list[dict],
    participants: list[str],
    llm: LLM,
    *,
    batch: int = BATCH,
    workers: int = 12,
    log=print,
) -> Registry:
    work = [
        i for i, r in enumerate(rows)
        if r.get("status") != "self_corrected"
        and not {l["type"] for l in r["labels"]} <= SKIP_ONLY
    ]
    batches = [work[k:k + batch] for k in range(0, len(work), batch)]
    n_batches = len(batches)
    rows_by_span = {r["span_id"]: r for r in rows}
    log(f"[pass1] {len(work)} work spans → {n_batches} batches, firing in parallel (Haiku)")

    # ── Pass 1: parallel extraction ─────────────────────────────────────────
    all_candidates: list[dict] = []
    errors_p1 = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_extract_batch, [rows[i] for i in idx], llm): b_idx
            for b_idx, idx in enumerate(batches)
        }
        completed = 0
        for fut in as_completed(futs):
            b_idx = futs[fut]
            completed += 1
            try:
                cands = fut.result()
                all_candidates.extend(cands)
                log(f"  pass1 batch {b_idx+1}/{n_batches} done ({len(cands)} candidates) [{completed}/{n_batches}]")
            except Exception as e:
                errors_p1 += 1
                log(f"  pass1 batch {b_idx+1} ERROR: {e}")

    log(f"[pass2] dedup {len(all_candidates)} raw candidates → stable registry")

    # ── Pass 2: deterministic dedup ─────────────────────────────────────────
    reg = _build_registry_from_candidates(all_candidates, rows_by_span)
    log(f"  {len(reg.live())} elements after dedup  ({len(reg.live('question'))} Q, "
        f"{len(reg.live('option'))} opt, {len(reg.live('claim'))} clm, "
        f"{len(reg.live('blocker'))} blk, {len(reg.live('constraint'))} con, "
        f"{len(reg.live('criterion'))} crit)")

    # ── Pass 3: parallel stance assignment ──────────────────────────────────
    log(f"[pass3] assign stances, {n_batches} batches in parallel (Haiku)")
    reg_snapshot = _render_registry_compact(reg)  # frozen — same for all batches

    # collect ops per batch index so we can apply them in order (temporal ordering)
    ops_by_batch: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _stance_batch,
                [rows[i] for i in idx],
                reg_snapshot,
                participants,
                llm,
            ): b_idx
            for b_idx, idx in enumerate(batches)
        }
        for fut in as_completed(futs):
            b_idx = futs[fut]
            try:
                ops = fut.result()
                ops_by_batch[b_idx] = ops
                log(f"  pass3 batch {b_idx+1}/{n_batches} done ({len(ops)} ops)")
            except Exception as e:
                ops_by_batch[b_idx] = []
                log(f"  pass3 batch {b_idx+1} ERROR: {e}")

    # Apply ops in original batch order to respect temporal ordering
    for b_idx in range(n_batches):
        ops = ops_by_batch.get(b_idx, [])
        allowed = {rows[i]["span_id"] for i in batches[b_idx]}
        reg.apply(ops, allowed, participants)

    log(f"[done] {len(reg.live())} elements, {len(reg.events)} events, "
        f"{len(reg.warnings)} warnings, p1_errors={errors_p1}")
    return reg


def main():
    import sys
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("classified", help="*.output.json from run.py")
    ap.add_argument("--model", default=HAIKU, help="LLM model for extract/stance passes")
    ap.add_argument("--sonnet-model", default=None, help="Sonnet model for summarize (default: claude-sonnet-4-6)")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out")
    ap.add_argument("--no-summary", action="store_true", help="Skip the summarize step")
    a = ap.parse_args()

    from llm import AnthropicLLM, SONNET
    rows, participants = load_classified(a.classified)
    print(f"Loaded {len(rows)} classified spans, {len(participants)} participants")

    t0 = time.time()
    reg = reconcile_fast(
        rows, participants, AnthropicLLM(model=a.model),
        batch=a.batch, workers=a.workers,
    )
    elapsed = time.time() - t0

    result = dump(reg, participants, a.classified, a.model)

    # Attach _enterprise block so make_summary can access spans + classified
    raw = json.load(open(a.classified))
    result["_enterprise"] = {
        "source_type": "transcript",
        "subject": None,
        "speakers": participants,
        "observers": [],
        "msg_meta": [],
        "spans": raw.get("spans", []),
        "classified": raw.get("classified", []),
    }

    # Summarize (two Sonnet calls)
    if not a.no_summary:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "enterprise"))
        from summarize import make_summary
        sonnet_model = a.sonnet_model or SONNET
        sonnet = AnthropicLLM(model=sonnet_model)
        print(f"\n[summarize] two Sonnet passes ({sonnet_model})...")

        def _call_llm(system, user):
            resp = sonnet.client.messages.create(
                model=sonnet.model, max_tokens=2048,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        result["summary"] = make_summary(result, _call_llm)

    out = a.out or re.sub(r"\.output\.json$", "", a.classified) + ".fast_reconcile.json"
    Path(out).write_text(json.dumps(result, indent=1))

    print(matrix(reg, participants))
    print("\nWHAT SHIFTED")
    print("\n".join(changes(reg)))
    if result.get("summary"):
        print("\n── Summary ──────────────────────────────────────────────────────")
        print(result["summary"])
        print("─────────────────────────────────────────────────────────────────")
    print(f"\nwrote {out}")
    print(f"time: {elapsed:.1f}s  ({len(reg.live())} elements, {len(reg.events)} events, "
          f"{len(reg.warnings)} warnings)")


if __name__ == "__main__":
    main()
