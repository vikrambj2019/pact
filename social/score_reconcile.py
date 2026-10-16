"""Score a reconcile output against an end-state gold file.

    python score_reconcile.py outputs/ent-03.reconcile.json gold/ent-03-hard-data-platform.state.json [--diff]

Checks
- options      found? (regex on title/aliases, minus exclude) and final status
- cells        final stance per participant per option, by class
               (proposed=support; acknowledged=undecided; "blank" = never stated)
- questions    final status
- elements     claims / blockers / constraints located by first message (+ optional regex)
- histories    required stance sequence appears in order for that person/option
- flags        consensus_conflict raised on the right message
- duplicates   gold options matched by >1 live element (a failed merge)
"""
from __future__ import annotations

import argparse
import json
import re

CLASS = {"proposed": "support", "support": "support", "conditional": "conditional",
         "oppose": "oppose", "withdrawn": "withdrawn", "undecided": "undecided",
         "acknowledged": "undecided", None: "blank", "blank": "blank"}


def as_list(x):
    return x if isinstance(x, list) else [x]


def load(out_path):
    d = json.load(open(out_path))
    els = {e["id"]: e for e in d["elements"]}

    def canon(eid):
        seen = set()
        while eid in els and els[eid].get("merged_into") and eid not in seen:
            seen.add(eid)
            eid = els[eid]["merged_into"]
        return eid

    live = [e for e in els.values() if not e.get("merged_into")]
    stances = {}                                    # (opt, participant) -> [stance,...]
    for ev in d["events"]:
        if ev["kind"] == "stance" and not ev["payload"].get("inferred"):
            stances.setdefault((canon(ev["element_id"]), ev["participant"]), []).append(ev["payload"]["stance"])
    return d, els, live, canon, stances


def matches(e, spec):
    names = [e["title"], *e.get("aliases", [])]
    hit = any(re.search(rx, n, re.I) for rx in spec.get("match", []) for n in names)
    bad = any(re.search(rx, n, re.I) for rx in spec.get("exclude", []) for n in names)
    return hit and not bad


def score(out_path, gold_path):
    d, els, live, canon, stances = load(out_path)
    g = json.load(open(gold_path))
    span_msg = d["span_msgs"]
    R = {k: [0, 0] for k in ("options", "option_status", "cells", "cells_nonblank",
                              "questions", "elements", "histories", "flags")}
    errs, dup = [], []

    def tally(k, ok, msg):
        R[k][1] += 1
        R[k][0] += ok
        if not ok:
            errs.append(f"{k:14} {msg}")

    # questions
    for q in g.get("questions", []):
        cands = [e for e in live if e["kind"] == "question" and matches(e, q)]
        ok = any(e["status"] in q["status"] for e in cands)
        got = [e["status"] for e in cands] or "missing"
        tally("questions", ok, f"{q['key']}: want {q['status']} got {got}")

    # options + cells
    opt_map = {}
    for o in g.get("options", []):
        cands = [e for e in live if e["kind"] == "option" and matches(e, o)]
        if not cands:
            if not o.get("optional"):
                tally("options", False, f"{o['key']}: not found")
                for p in g["participants"]:
                    want = [CLASS[x] for x in as_list(o.get("stances", {}).get(p, "blank"))]
                    if want != ["blank"]:
                        tally("cells_nonblank", False, f"{o['key']}/{p}: option missing")
            continue
        if len(cands) > 1:
            dup.append((o["key"], [c["id"] + ":" + c["title"] for c in cands]))
        best = max(cands, key=lambda e: sum(len(v) for (oid, _), v in stances.items() if oid == e["id"]))
        opt_map[o["key"]] = best["id"]
        tally("options", True, "")
        tally("option_status", best["status"] in as_list(o["status"]),
              f"{o['key']}: want {o['status']} got {best['status']}")
        for p in g["participants"]:
            want = {CLASS[x] for x in as_list(o.get("stances", {}).get(p, "blank"))}
            hist = stances.get((best["id"], p), [])
            got = CLASS[hist[-1] if hist else None]
            ok = got in want
            tally("cells", ok, f"{o['key']}/{p}: want {sorted(want)} got {got}")
            if want != {"blank"}:
                tally("cells_nonblank", ok, f"{o['key']}/{p}: want {sorted(want)} got {got}")

    # elements located by message
    for spec in g.get("elements", []):
        kinds = spec.get("kinds", ["claim", "blocker", "constraint"])
        cands = [e for e in live if e["kind"] in kinds
                 and any(span_msg.get(s) == spec["msg"] for s in [e["created_span"], *e.get("spans", [])])
                 and (not spec.get("match") or matches(e, {"match": as_list(spec["match"])}))]
        ok = any(e["status"] in as_list(spec["status"]) for e in cands)
        got = [f"{e['id']}:{e['status']}" for e in cands] or "missing"
        tally("elements", ok, f"{spec['key']} ({spec['msg']}): want {spec['status']} got {got}")

    # histories
    for h in g.get("histories", []):
        oid = opt_map.get(h["option"])
        seq = [CLASS[s] for s in stances.get((oid, h["participant"]), [])] if oid else []
        collapsed = [s for i, s in enumerate(seq) if i == 0 or s != seq[i - 1]]
        i = 0
        for s in collapsed:
            if i < len(h["sequence"]) and s in {CLASS[x] for x in as_list(h["sequence"][i])}:
                i += 1
        tally("histories", i == len(h["sequence"]),
              f"{h['option']}/{h['participant']}: want {h['sequence']} got {collapsed or 'none'}")

    # flags
    flagged = [ev for ev in d["events"] if ev["kind"] == "flag"]
    for f in g.get("flags", []):
        ok = any(ev["payload"].get("type") == f["type"] and span_msg.get(ev["span_id"]) == f["msg"]
                 for ev in flagged)
        tally("flags", ok, f"{f['type']} at {f['msg']}: "
              + ("ok" if ok else f"raised at {[span_msg.get(e['span_id']) for e in flagged] or 'none'}"))

    return R, errs, dup, d


def report(R, errs, dup, d, show):
    print("RECONCILE")
    for k, (ok, n) in R.items():
        if n:
            print(f"  {k:15} {ok:3}/{n:<3} {ok / n:5.0%}")
    inferred = [e for e in d["events"] if e["kind"] == "stance" and e["payload"].get("inferred")]
    print(f"  duplicates      {len(dup)}   warnings {len(d.get('warnings', []))}   "
          f"inferred stances {len(inferred)} (excluded)")
    if show:
        for e in inferred:
            print(f"  ? inferred {e['participant']} {e['payload']['stance']} on {e['element_id']} "
                  f"@ {d['span_msgs'].get(e['span_id'])}: {e['payload'].get('inferred_reason')}")
    if show:
        for e in errs:
            print("  ✗", e)
        for k, c in dup:
            print("  ⧉ duplicate", k, c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("gold")
    ap.add_argument("--diff", action="store_true")
    a = ap.parse_args()
    report(*score(a.output, a.gold), a.diff)
