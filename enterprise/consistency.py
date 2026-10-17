"""Logic check: find statements that cannot both be true.

    from consistency import find_conflicts
    d["conflicts"] = find_conflicts(d, llm)       # llm: anything with .structured(...), e.g. AnthropicLLM(SONNET)

Version 1 looks for two things only, because they are the cheapest to get right:
  decision_vs_constraint   a decision (or timeline) that runs into something a person said earlier or later
  figures                  two different numbers for what looks like the same thing, with no acknowledged correction
(Same-person reversals are version 2: telling a changed mind from a contradiction is the hard part.)

Precision over recall. Code builds the candidate pairs and attaches the quotes, so the model can never invent one.
The model only judges "can both be true?". A pair the model marks as acknowledged or resolved later is dropped.
Pact never says who is right; it shows both statements and the question the group would have to answer.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import Ctx, _live, disp  # noqa: E402

MAX_DECISION_PAIRS = 12
MAX_FIGURE_PAIRS = 10

_STOP = set("the a an and or of to in on for with is are was were be been it this that we our you your they their i my at by as from "
            "if then so but not no can could will would should may might have has had do does did there here what which who when".split())
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s?(?:[kKmMbB]\b)?|\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|k\b|K\b)")
_NUMTOK = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?[kKmMbB]?")
_CORRECTION = re.compile(r"\b(correction|i meant|to correct|actually|sorry|i was wrong|not \$?\d|rather than|instead of the|old quote|updated)\b", re.I)


def _toks(t: str) -> set[str]:
    t = _NUMTOK.sub(" ", t.lower())
    return {w for w in re.findall(r"[a-z][a-z'-]{2,}", t) if w not in _STOP}


def _jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _clock(ts: str) -> str:
    """'2000-01-01T00:02:05' -> '2:05' for transcripts; the date for anything else."""
    m = re.match(r"^2000-01-01T(\d\d):(\d\d):(\d\d)", ts or "")
    if m:
        h, mi, s = map(int, m.groups())
        return f"{h * 60 + mi}:{s:02d}"
    return (ts or "")[:16].replace("T", " ")


def _ref(cx: Ctx, sid: str) -> dict:
    s = cx.spans[cx.order[sid]]
    return dict(span=sid, author=s["author"], name=disp(s["author"]), when=_clock(s.get("ts", "")),
                text=re.sub(r"\s+", " ", s["text"]).strip()[:260])


def _after(cx: Ctx, sid: str, n: int = 3) -> list[str]:
    i = cx.order[sid]
    out = []
    for s in cx.spans[i + 1:i + 1 + n]:
        if (cx.span_scope(s["id"]) or (None,))[0] == "private":
            continue
        out.append(f"{disp(s['author'])}: {re.sub(chr(10), ' ', s['text'])[:140]}")
    return out


def _public(cx: Ctx, sid: str) -> bool:
    return sid in cx.order and (cx.span_scope(sid) or (None,))[0] != "private"


def candidates(d: dict) -> list[dict]:
    cx = Ctx(d)
    els = _live(d)
    out: list[dict] = []

    # 1. the decision against each stated constraint / dependency
    dec_ev = [ev for ev in d.get("events", []) if ev["kind"] == "decide"]
    if dec_ev and _public(cx, dec_ev[-1]["span_id"]):
        ev = dec_ev[-1]
        dsid = ev["span_id"]
        cond = (ev.get("payload") or {}).get("conditions") or ""
        seen = set()
        for e in els:
            if e["kind"] not in ("constraint", "blocker"):
                continue
            sid = e.get("created_span")
            if not sid or sid == dsid or sid in seen or not _public(cx, sid):
                continue
            t = cx.text.get(sid, "").strip()
            if len(t.split()) < 6 or t.endswith("?"):
                continue                       # "Anyone?" and "I'll be out" say too little to judge on their own
            seen.add(sid)
            out.append(dict(kind="decision_vs_constraint", a=sid, b=dsid, note=("Conditions the decision recorded: " + cond) if cond else ""))
            if len([o for o in out if o["kind"] == "decision_vs_constraint"]) >= MAX_DECISION_PAIRS:
                break

    # 2. two different figures for what looks like the same thing
    spans = []
    for e in els:
        if e["kind"] in ("claim", "constraint", "criterion", "blocker"):
            for sid in ([e.get("created_span")] + list(e.get("spans") or [])):
                if sid and sid in cx.order and _public(cx, sid) and sid not in [x[0] for x in spans]:
                    t = cx.text.get(sid, "")
                    nums = {re.sub(r"[\s,]", "", m.group(0).lower()) for m in _MONEY.finditer(t)}
                    if nums:
                        spans.append((sid, nums, _toks(t)))
    pairs = []
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            (a, na, ta), (b, nb, tb) = sorted((spans[i], spans[j]), key=lambda x: cx.order[x[0]])
            if na == nb or na & nb:
                continue
            sim = _jac(ta, tb)
            if sim < 0.3:
                continue
            same = cx.author.get(a) == cx.author.get(b)
            if same and _CORRECTION.search(cx.text.get(b, "")):
                continue                       # an acknowledged correction is not a contradiction
            pairs.append((sim, a, b))
    for sim, a, b in sorted(pairs, reverse=True)[:MAX_FIGURE_PAIRS]:
        out.append(dict(kind="figures", a=a, b=b, note=""))
    return out


JUDGE_SYSTEM = """You check pairs of statements from one discussion and say whether both can be true at the same time.
For each pair you get statement A and statement B (speaker and time shown), what was said right after each, and sometimes a note.
Verdicts:
- conflict: they cannot both hold as stated. Example: someone says their team cannot be away on a date, and a decision puts the
  event on that date; or two different prices for the same item with no sign that one corrects the other.
- tension: they strain against each other or could be reconciled with a change, but are not strictly incompatible.
- compatible: they can both be true. Use this for: a person correcting their own number; a hypothetical or an "if"; different
  scopes or time periods; a dependency the decision already lists as a condition; anything you cannot establish from the words.
Rules:
- Judge only from the words given. Do not assume a year, a weekday or a calendar date that is not stated. If working it out would need
  one, answer compatible.
- Check arithmetic and calendar logic only when every number needed is in the words.
- resolved_later = true if the lines after either statement show someone responded to this exact problem and settled it.
- Never say who is right. why: one plain sentence naming both speakers and what does not fit, 30 words at most.
- resolve: the one question the group would have to answer to clear it, written as a question.
When in doubt, answer compatible. A false alarm costs more than a miss."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {
        "pair": {"type": "integer"},
        "verdict": {"type": "string", "enum": ["conflict", "tension", "compatible"]},
        "resolved_later": {"type": "boolean"},
        "why": {"type": "string"},
        "resolve": {"type": "string"},
    }, "required": ["pair", "verdict"]}}},
    "required": ["results"],
}


def _render(cx: Ctx, i: int, c: dict) -> str:
    a, b = _ref(cx, c["a"]), _ref(cx, c["b"])
    lines = [f"PAIR {i} ({c['kind']})",
             f"A [{a['name']}, {a['when']}]: {a['text']}",
             "  right after A: " + (" | ".join(_after(cx, c['a'], 2)) or "(nothing)"),
             f"B [{b['name']}, {b['when']}]: {b['text']}",
             "  right after B: " + (" | ".join(_after(cx, c['b'], 3)) or "(nothing)")]
    if c.get("note"):
        lines.append("  note: " + c["note"])
    return "\n".join(lines)


_RANK = {"conflict": 0, "tension": 1}


def find_conflicts(d: dict, llm) -> list[dict]:
    cx = Ctx(d)
    cands = candidates(d)
    if not cands:
        return []
    user = "\n\n".join(_render(cx, i, c) for i, c in enumerate(cands))
    out = llm.structured(system=JUDGE_SYSTEM, user=user, tool_name="judge_pairs", schema=JUDGE_SCHEMA, max_tokens=2048)
    found = []
    results = out.get("results") if isinstance(out, dict) else None
    for r in results if isinstance(results, list) else []:
        if not isinstance(r, dict):
            continue
        i = r.get("pair")
        if not isinstance(i, int) or not (0 <= i < len(cands)):
            continue
        v = r.get("verdict")
        if v not in _RANK or r.get("resolved_later"):
            continue
        why, res = (r.get("why") or "").strip(), (r.get("resolve") or "").strip()
        if not why:
            continue
        if v == "conflict" and not res:
            v = "tension"
        c = cands[i]
        found.append(dict(kind=c["kind"], verdict=v, a=_ref(cx, c["a"]), b=_ref(cx, c["b"]), why=why, resolve=res))
    seen, uniq = set(), []
    for f in sorted(found, key=lambda f: (_RANK[f["verdict"]], f["kind"] != "decision_vs_constraint")):
        key = (f["a"]["span"], f["b"]["span"])
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def count_conflicts(d: dict) -> int:
    return sum(1 for c in d.get("conflicts") or [] if c.get("verdict") == "conflict")


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    for i, c in enumerate(candidates(data)):
        print(_render(Ctx(data), i, c), "\n")
