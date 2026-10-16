"""Stage 3 (fast) -- parallel reconcile, 4 steps.

    python reconcile_fast.py evals/outputs/long-01.output.json

Why this exists: sequential reconcile needs the registry from the previous batch, so nothing runs in parallel.

  Step 1  Extract      (parallel, LLM)   every batch on its own, with the 8 spans before it as read-only context.
  Step 2  Consolidate  (ONE small call)  the candidate questions and options only (titles, no claims): merge paraphrases,
                                         link each option to its question, drop "options" that are just an argument or
                                         the other side's position. Falls back to code-only matching if the call fails.
  Step 3  Build        (code)            deterministic: candidates are taken in transcript order, never in the order the
                                         threads happened to finish. Writes create/attach/alias events, links options to
                                         questions, resolves "about" on objections, rejects span ids that are not in the batch.
  Step 4  Stances      (parallel, LLM)   registry frozen; every batch emits stance/status/decide/supersede/attach/skip.
                                         Ops are applied in transcript order, so `prev` and history stay correct.

Changes from the 3-pass version (found on real-02, Haiku): 17 questions and 20 options where sequential gave 5 and 5;
0 of 20 options linked to a question (so matrix() printed nothing); 0 of 39 claims linked to the option they object to;
no create events and 50 of 83 elements with no spans; candidate order and therefore IDs, authors and dedup winners
depended on which thread finished first.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm import HAIKU, LLM
from reconcile import (
    BATCH,
    CONTEXT_SPANS,
    DEFAULT_STATUS,
    PREFIX,
    SKIP_ONLY,
    Element,
    Registry,
    _core,
    _norm,
    changes,
    dump,
    load_classified,
    matrix,
    render_registry,
    render_span,
)

CONTEXT = CONTEXT_SPANS   # spans of read-only context shown before each batch


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b and (a | b) else 0.0


# ── Step 1: extract ─────────────────────────────────────────────────────────

EXTRACT_SYSTEM = """You extract decision elements from part of a group discussion (a chat, an email chain or a meeting).
You get CONTEXT (spans just before this part, for understanding only; never extract from them) and SPANS (extract from these).

Element kinds
- question   a decision the group is trying to make ("which platform", "go-live date"). Only when a span frames a decision to be
             made. A question put to one person for information is a blocker, not a question. One entry per DECISION: do not
             make a new question for each way it is phrased.
- option     an alternative someone PUTS FORWARD, recommends or asks the group to choose ("Databricks", "Nov 14-15"). Set
             question_title to the decision it answers. Do NOT make an option out of: a position someone describes only to
             argue against it, a restatement of another person's proposal, the status quo, text quoted from a document, or
             the opposite of an option ("do not X").
- criterion  a factor for judging options
- constraint a hard or soft limit that filters options. constraint_kind: mandatory|negotiable|unknown
- claim      an assertion (fact, estimate, objection, prediction). claim_kind: factual|predictive|causal|priority|objection|
             consensus|other. For an objection, list the option titles it targets in about_options[].
- blocker    an open question, missing information or dependency stopping progress. blocker_kind: open_question|missing_info|
             unresolved_dependency

One entry per distinct element, citing the span_id that introduces it. temp_id is a short label ("q1", "o2").
Do NOT output stances, status changes or decisions. Skip filler, greetings, and spans that only react to something."""

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
                    "question_title": {"type": "string"},
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


def _context_block(rows: list[dict], idx: list[int]) -> str:
    start = max(0, idx[0] - CONTEXT)
    ctx = rows[start:idx[0]]
    return "\n".join(render_span(r) for r in ctx) if ctx else "(none)"


def _extract_batch(rows: list[dict], idx: list[int], llm: LLM) -> list[dict]:
    user = ("CONTEXT (do not extract from these):\n" + _context_block(rows, idx)
            + "\n\nSPANS:\n" + "\n".join(render_span(rows[i]) for i in idx))
    out = llm.structured(system=EXTRACT_SYSTEM, user=user, tool_name="extract_elements",
                         schema=EXTRACT_SCHEMA, max_tokens=2048)
    return out.get("elements", [])


# ── Step 2: consolidate questions and options (one small call) ──────────────

CONSOLIDATE_SYSTEM = """Candidate decision questions and options were extracted separately from different parts of one long
discussion, so the same thing appears several times under different wording, and options are not yet linked to questions.
Clean up the list. Use only the ids given.

1. QUESTIONS: group candidates that are the same decision (paraphrases, sub-questions of one decision). For each group give one
   canonical id (the clearest wording) and the ids merged into it. A genuinely separate decision stays its own group.
2. OPTIONS: for every option give
   - question: the canonical question id it answers (or null if none fits);
   - merge_into: the id of another option that is the same alternative, else null;
   - drop: true if it is NOT a real alternative someone put forward (an argument, the other side described only to oppose it,
     a restatement of the memo's own text, the status quo, the negation of another option).
Be conservative about merging two options that differ in substance (different dates, vendors, owners). Prefer fewer questions."""

CONSOLIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "merge": {"type": "array", "items": {"type": "string"}}},
            "required": ["id"]}},
        "options": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "question": {"type": ["string", "null"]},
            "merge_into": {"type": ["string", "null"]}, "drop": {"type": "boolean"}},
            "required": ["id"]}},
    },
    "required": ["questions", "options"],
}


def _consolidate(cands: list[dict], rows_by_span: dict[str, dict], llm: LLM) -> dict | None:
    qs = [c for c in cands if c["kind"] == "question"]
    os_ = [c for c in cands if c["kind"] == "option"]
    if not qs and not os_:
        return None
    def line(c):
        t = (rows_by_span.get(c["span_id"], {}).get("text") or "").replace("\n", " ")[:140]
        extra = f' | answers: {c["question_title"]}' if c.get("question_title") else ""
        return f'  {c["cid"]} "{c["title"]}" by {c["author"]}{extra} | said: "{t}"'
    user = "QUESTIONS\n" + "\n".join(line(c) for c in qs) + "\n\nOPTIONS\n" + "\n".join(line(c) for c in os_)
    try:
        out = llm.structured(system=CONSOLIDATE_SYSTEM, user=user, tool_name="consolidate",
                             schema=CONSOLIDATE_SCHEMA, max_tokens=4096)
    except Exception:
        return None
    ids = {c["cid"] for c in cands}
    if not isinstance(out, dict) or not all(q.get("id") in ids for q in out.get("questions", [])):
        return None
    return out


# ── Step 3: build the registry (code only, deterministic) ───────────────────

def _fallback_plan(cands: list[dict]) -> dict:
    """Code-only consolidation: paraphrased questions by token overlap, options linked by overlap with a question."""
    canon: dict[str, str] = {}
    kept: list[dict] = []
    for c in (c for c in cands if c["kind"] == "question"):
        hit = next((k for k in kept if _jaccard(_core(k["title"]), _core(c["title"])) >= 0.6), None)
        if hit:
            canon[c["cid"]] = hit["cid"]
        else:
            kept.append(c)
            canon[c["cid"]] = c["cid"]
    plan_q: dict[str, list[str]] = {}
    for c in canon:
        plan_q.setdefault(canon[c], [])
        if canon[c] != c:
            plan_q[canon[c]].append(c)
    opts = []
    for c in (c for c in cands if c["kind"] == "option"):
        qt = _core(c.get("question_title") or "")
        best = max(kept, key=lambda k: _jaccard(_core(k["title"]), qt), default=None) if qt else None
        opts.append({"id": c["cid"], "question": best["cid"] if best and _jaccard(_core(best["title"]), qt) > 0 else None,
                     "merge_into": None, "drop": False})
    return {"questions": [{"id": k, "merge": v} for k, v in plan_q.items()], "options": opts}


def _build(cands: list[dict], plan: dict, rows_by_span: dict[str, dict]) -> Registry:
    reg = Registry(rows_by_span)
    n: Counter = Counter()
    final: dict[str, str] = {}            # candidate id -> element id

    def make(c: dict, qid: str | None = None) -> Element:
        kind = c["kind"]
        n[kind] += 1
        eid = f"{PREFIX[kind]}{n[kind]}"
        attrs = {k: c[k] for k in ("claim_kind", "constraint_kind", "blocker_kind") if c.get(k)}
        e = Element(id=eid, kind=kind, title=c["title"], author=rows_by_span.get(c["span_id"], {}).get("author", ""),
                    created_span=c["span_id"], status=DEFAULT_STATUS[kind], question_id=qid, attrs=attrs)
        reg.el[eid] = e
        reg._event("create", eid, c["span_id"], {"title": c["title"], "kind": kind}, e.author)
        final[c["cid"]] = eid
        return e

    def restate(c: dict, target: Element, alias: bool = True) -> None:
        final[c["cid"]] = target.id
        if alias and c["title"] != target.title and c["title"] not in target.aliases and target.kind in ("question", "option"):
            target.aliases.append(c["title"])
            reg._event("alias", target.id, c["span_id"], {"alias": c["title"]}, rows_by_span.get(c["span_id"], {}).get("author"))
        else:
            reg._event("attach", target.id, c["span_id"])

    by_cid = {c["cid"]: c for c in cands}

    # questions: canonical ones first, in transcript order; merged ones become aliases
    q_canon: dict[str, str] = {}
    for g in plan.get("questions", []):
        for m in g.get("merge") or []:
            if m in by_cid and m != g["id"]:
                q_canon[m] = g["id"]
    for c in (c for c in cands if c["kind"] == "question"):
        root = q_canon.get(c["cid"], c["cid"])
        if root == c["cid"]:
            make(c)
    for c in (c for c in cands if c["kind"] == "question"):
        root = q_canon.get(c["cid"], c["cid"])
        if root != c["cid"] and root in final:
            restate(c, reg.el[final[root]])
        elif c["cid"] not in final:
            make(c)

    # options
    oplan = {o["id"]: o for o in plan.get("options", []) if o.get("id") in by_cid}
    dropped = 0
    for c in (c for c in cands if c["kind"] == "option"):
        o = oplan.get(c["cid"], {})
        if o.get("drop"):
            dropped += 1
            reg.warnings.append(f"consolidate dropped option (not a real alternative): {c['title']}")
            continue
        q = o.get("question")
        qid = final.get(q_canon.get(q, q)) if q else None
        dup, _ = reg._find_option_by_name(c["title"], [])
        mi = o.get("merge_into")
        if not dup and mi and mi in final and reg.el.get(final[mi], None) and reg.el[final[mi]].kind == "option":
            dup = reg.el[final[mi]]
        if dup:
            restate(c, dup)
        else:
            make(c, qid)
    qs = reg.live("question")
    if len(qs) == 1:                                    # one decision in play: every unlinked option belongs to it
        for e in reg.live("option"):
            e.question_id = e.question_id or qs[0].id

    # constraints, criteria, blockers, claims: exact or near-exact restatement is an attach, not a second element
    for c in (c for c in cands if c["kind"] in ("constraint", "criterion", "blocker", "claim")):
        pool = reg.live(c["kind"])
        dup = next((e for e in pool if _norm(e.title) == _norm(c["title"])
                    or _jaccard(_core(e.title), _core(c["title"])) >= 0.75), None)
        if dup:
            restate(c, dup, alias=False)
        else:
            make(c)

    # objections point at the options they target
    for c in (c for c in cands if c["kind"] == "claim" and c.get("about_options")):
        e = reg.el.get(final.get(c["cid"], ""))
        if not e or e.kind != "claim":
            continue
        ids = []
        for t in c["about_options"]:
            o, _ = reg._find_option_by_name(t, [])
            if o and o.id not in ids:
                ids.append(o.id)
        if ids:
            e.attrs["about"] = ids

    reg._n = n
    if dropped:
        reg.warnings.append(f"consolidate dropped {dropped} option candidates")
    return reg


# ── Step 4: stances ─────────────────────────────────────────────────────────

STANCE_SYSTEM = """You assign stances and status changes to a frozen decision registry.
The registry already has every element with its final ID. Do NOT create elements.

You get CONTEXT (earlier spans, for understanding only: emit no ops for them), the REGISTRY, and NEW SPANS. For each NEW SPAN emit:
- {"op":"stance","option":"opt2","stance":"proposed|support|conditional|oppose|withdrawn|undecided|acknowledged",
   "conditions":"...","span":...,"because":[...]}   The participant is ALWAYS the span's author.
- {"op":"status","element":"opt3","status":...,"span":...,"because":[...]}
  option: active|dropped|fallback|reopened   blocker: open|resolved
  claim: unverified|conflicting|confirmed    constraint: active|superseded|satisfied   question: open|reopened
- {"op":"supersede","old":"clm2","new":"clm7","span":...}   old and new must be different elements
- {"op":"decide","question":"q1","chosen":["opt1"],"ruled_out":["opt3"],"conditional":false,"conditions":"...","span":...}
- {"op":"attach","element":"con2","span":...}
- {"op":"skip","span":...,"reason":"..."}

Rules
1. Every NEW SPAN gets at least one op (skip is fine for greetings and filler).
2. A stance is only for the span AUTHOR'S OWN view of an option, and only on the option the span talks about. Agreeing with a
   point ("I agree with your point about X") is not a stance on an option unless the option is the point.
3. "+1", "same", "None" (to "anyone against X?") and poll letters are stances.
4. What OTHERS think is never a stance: skip it.
5. The author of an option does not need a "proposed" stance on it; use support only if they say so again later.
6. decide only when someone explicitly closes the question for the group. Do not decide because a thread ended.
7. If no option fits what the span says, skip. Never stretch a span onto the nearest option."""

STANCE_OPS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ops": {"type": "array", "items": {"type": "object"}}, "notes": {"type": "string"}},
    "required": ["ops"],
}


def _stance_batch(rows: list[dict], idx: list[int], reg_snapshot: str, participants: list[str], llm: LLM) -> list[dict]:
    user = (f"PARTICIPANTS: {', '.join(participants)}\n\nREGISTRY:\n{reg_snapshot}\n\n"
            "CONTEXT (no ops for these):\n" + _context_block(rows, idx)
            + "\n\nNEW SPANS:\n" + "\n".join(render_span(rows[i]) for i in idx))
    out = llm.structured(system=STANCE_SYSTEM, user=user, tool_name="stance_ops",
                         schema=STANCE_OPS_SCHEMA, max_tokens=2048)
    return out.get("ops", [])


# ── Driver ──────────────────────────────────────────────────────────────────

def reconcile_fast(
    rows: list[dict],
    participants: list[str],
    llm: LLM,
    *,
    batch: int = BATCH,
    workers: int = 12,
    consolidate: bool = True,
    log=print,
) -> Registry:
    work = [i for i, r in enumerate(rows)
            if r.get("status") != "self_corrected" and not {l["type"] for l in r["labels"]} <= SKIP_ONLY]
    batches = [work[k:k + batch] for k in range(0, len(work), batch)]
    nb = len(batches)
    rows_by_span = {r["span_id"]: r for r in rows}
    order = {r["span_id"]: i for i, r in enumerate(rows)}
    log(f"[step1] {len(work)} work spans -> {nb} batches, extracting in parallel")

    # ── step 1 ──
    got: dict[int, list[dict]] = {}
    err1 = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_extract_batch, rows, idx, llm): b for b, idx in enumerate(batches)}
        for k, fut in enumerate(as_completed(futs), 1):
            b = futs[fut]
            try:
                got[b] = fut.result()
                log(f"  step1 batch {b + 1}/{nb} ({len(got[b])} candidates) [{k}/{nb}]")
            except Exception as e:
                err1 += 1
                got[b] = []
                log(f"  step1 batch {b + 1} ERROR: {e}")

    # flatten in TRANSCRIPT order, validate the span id belongs to the batch, give every candidate a stable id
    cands: list[dict] = []
    bad_spans = 0
    for b in range(nb):
        allowed = {rows[i]["span_id"] for i in batches[b]}
        for c in got.get(b, []):
            if c.get("kind") not in PREFIX or not (c.get("title") or "").strip():
                continue
            if c.get("span_id") not in allowed:
                bad_spans += 1
                continue
            c["title"] = c["title"].strip()
            c["author"] = rows_by_span[c["span_id"]]["author"]
            cands.append(c)
    cands.sort(key=lambda c: order[c["span_id"]])
    for k, c in enumerate(cands, 1):
        c["cid"] = f"c{k}"
    log(f"[step2] {len(cands)} candidates ({sum(c['kind'] == 'question' for c in cands)} questions, "
        f"{sum(c['kind'] == 'option' for c in cands)} options); span ids rejected: {bad_spans}")

    # ── step 2 ──
    plan = _consolidate(cands, rows_by_span, llm) if consolidate else None
    if plan is None:
        log("  consolidate: code-only fallback" if consolidate else "  consolidate: off, code-only")
        plan = _fallback_plan(cands)

    # ── step 3 ──
    reg = _build(cands, plan, rows_by_span)
    if bad_spans:
        reg.warnings.append(f"extract: {bad_spans} candidates cited a span outside their batch and were ignored")
    log(f"[step3] {len(reg.live())} elements ({len(reg.live('question'))} Q, {len(reg.live('option'))} opt, "
        f"{len(reg.live('claim'))} clm, {len(reg.live('blocker'))} blk, {len(reg.live('constraint'))} con, "
        f"{len(reg.live('criterion'))} crit); options linked: "
        f"{sum(1 for o in reg.live('option') if o.question_id)}/{len(reg.live('option'))}")

    # ── step 4 ──
    snapshot = render_registry(reg)
    ops_by: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_stance_batch, rows, idx, snapshot, participants, llm): b for b, idx in enumerate(batches)}
        for fut in as_completed(futs):
            b = futs[fut]
            try:
                ops_by[b] = fut.result()
                log(f"  step4 batch {b + 1}/{nb} ({len(ops_by[b])} ops)")
            except Exception as e:
                ops_by[b] = []
                log(f"  step4 batch {b + 1} ERROR: {e}")
    for b in range(nb):
        reg.apply(ops_by.get(b, []), {rows[i]["span_id"] for i in batches[b]}, participants)

    log(f"[done] {len(reg.live())} elements, {len(reg.events)} events, {len(reg.warnings)} warnings, step1_errors={err1}")
    return reg


def main():
    import sys
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("classified", help="*.output.json from run.py")
    ap.add_argument("--model", default=HAIKU)
    ap.add_argument("--sonnet-model", default=None)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-consolidate", action="store_true", help="skip the consolidation call (code-only matching)")
    ap.add_argument("--out")
    ap.add_argument("--no-summary", action="store_true")
    a = ap.parse_args()

    from llm import SONNET, AnthropicLLM
    rows, participants = load_classified(a.classified)
    print(f"Loaded {len(rows)} classified spans, {len(participants)} participants")

    t0 = time.time()
    reg = reconcile_fast(rows, participants, AnthropicLLM(model=a.model), batch=a.batch, workers=a.workers,
                         consolidate=not a.no_consolidate)
    elapsed = time.time() - t0
    result = dump(reg, participants, a.classified, a.model)

    raw = json.load(open(a.classified))
    result["_enterprise"] = {"source_type": "transcript", "subject": None, "speakers": participants, "observers": [],
                             "msg_meta": [], "spans": raw.get("spans", []), "classified": raw.get("classified", [])}

    if not a.no_summary:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "enterprise"))
        from summarize import make_summary
        sonnet = AnthropicLLM(model=a.sonnet_model or SONNET)
        print(f"\n[summarize] ({sonnet.model})...")

        def _call_llm(system, user):
            resp = sonnet.client.messages.create(
                model=sonnet.model, max_tokens=2048,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}])
            return resp.content[0].text

        result["summary"] = make_summary(result, _call_llm)

    out = a.out or re.sub(r"\.output\.json$", "", a.classified) + ".fast_reconcile.json"
    Path(out).write_text(json.dumps(result, indent=1))
    print(matrix(reg, participants))
    print("\nWHAT SHIFTED")
    print("\n".join(changes(reg)))
    if result.get("summary"):
        print("\n-- Summary " + "-" * 50)
        print(result["summary"])
    print(f"\nwrote {out}")
    print(f"time: {elapsed:.1f}s  ({len(reg.live())} elements, {len(reg.events)} events, {len(reg.warnings)} warnings)")


if __name__ == "__main__":
    main()
