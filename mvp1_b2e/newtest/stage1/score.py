"""Score a pipeline output against a gold file.

    python score.py output.json gold.json [--diff]

Matching: each predicted span is assigned to the gold span in the same message with the
highest token overlap. Several predictions on one gold span = over-split; a prediction
matching nothing = spurious.

Segmentation
  recall      required gold spans that got >=1 prediction
  over-split  gold spans that got >1 prediction
  spurious    content-labelled predictions matching no gold span
Labels (per gold span: union of predicted content labels vs gold)
  TP = predicted & required, FN = required - predicted,
  FP = predicted - required - allowed   (+ every label on a spurious span)
  exact = spans with FN == FP == 0
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

NON_CONTENT = {"process", "irrelevant"}


def toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def overlap(a: str, b: str) -> float:
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def load_pred(path: str) -> list[dict]:
    d = json.load(open(path))
    spans = {s["id"]: s for s in d["spans"]}
    out = []
    for c in d["classified"]:
        s = spans[c["span_id"]]
        out.append({"id": c["span_id"], "msg": s["source_message_ids"][0], "text": c["text"],
                    "labels": [l["type"] for l in c["labels"]]})
    return out


def score(pred: list[dict], gold: list[dict]):
    by_msg = defaultdict(list)
    for gi, g in enumerate(gold):
        by_msg[g["msg"]].append(gi)

    assigned = defaultdict(list)          # gold idx -> pred list
    group_of = {}                         # gold idx -> set of gold idxs one pred covered
    spurious, undersplit = [], []
    for p in pred:
        pt = toks(p["text"])
        golds = by_msg.get(p["msg"], [])
        # gold spans this prediction swallows almost entirely
        covered = [gi for gi in golds
                   if toks(gold[gi]["text"]) and
                   len(toks(gold[gi]["text"]) & pt) / len(toks(gold[gi]["text"])) >= 0.6]
        if len(covered) >= 2:
            undersplit.append((p, [gold[gi] for gi in covered]))
            for gi in covered:
                assigned[gi].append(p)
                group_of[gi] = set(covered)
            continue
        best = max(((overlap(p["text"], gold[gi]["text"]), gi) for gi in golds), default=(0, None))
        if best[0] >= 0.5:
            assigned[best[1]].append(p)
        else:
            spurious.append(p)

    def accepts(g):                       # every label this gold span tolerates
        return set(g["labels"]) | set(g.get("allowed", [])) | set(g.get("any_of", []))

    tp = fn = fp = 0
    per_type = defaultdict(Counter)
    exact = scored = 0
    missed, oversplit, rows = [], [], []
    for gi, g in enumerate(gold):
        preds = assigned.get(gi, [])
        req = set(g["labels"]) - NON_CONTENT
        any_of = set(g.get("any_of", [])) - NON_CONTENT
        allowed = accepts(g)
        for gj in group_of.get(gi, ()):   # under-split: siblings' labels aren't FPs here
            allowed |= accepts(gold[gj])
        optional = g.get("optional", False)
        if not preds:
            if not optional:
                missed.append(g)
                need = req | ({sorted(any_of)[0]} if any_of else set())
                fn += len(need)
                for t in need:
                    per_type[t]["fn"] += 1
            continue
        if len(preds) > 1:
            oversplit.append((g, preds))
        got = set().union(*(set(p["labels"]) for p in preds)) - NON_CONTENT
        s_tp, s_fn, s_fp = got & req, req - got, got - allowed
        if any_of:                        # at least one alternative must be present
            hit = got & any_of
            if hit:
                s_tp = s_tp | {sorted(hit)[0]}
            else:
                s_fn = s_fn | {sorted(any_of)[0]}
        tp, fn, fp = tp + len(s_tp), fn + len(s_fn), fp + len(s_fp)
        for t in s_tp: per_type[t]["tp"] += 1
        for t in s_fn: per_type[t]["fn"] += 1
        for t in s_fp: per_type[t]["fp"] += 1
        scored += 1
        ok = not s_fn and not s_fp
        exact += ok
        rows.append((g, preds, s_fn, s_fp, ok))

    real_spurious = [p for p in spurious if set(p["labels"]) - NON_CONTENT]
    for p in real_spurious:
        for t in set(p["labels"]) - NON_CONTENT:
            fp += 1
            per_type[t]["fp"] += 1

    required = [g for g in gold if not g.get("optional")]
    return dict(tp=tp, fn=fn, fp=fp, exact=exact, scored=scored, per_type=per_type,
                missed=missed, oversplit=oversplit, undersplit=undersplit,
                spurious=real_spurious, rows=rows,
                seg_recall=(len(required) - len(missed)) / max(1, len(required)),
                n_required=len(required))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def report(res, show_diff: bool):
    p, r, f = prf(res["tp"], res["fp"], res["fn"])
    print("SEGMENTATION")
    print(f"  recall      {res['seg_recall']:.0%}  ({res['n_required'] - len(res['missed'])}/{res['n_required']} required gold spans found)")
    print(f"  over-split  {len(res['oversplit'])}   (diagnostic: labels are scored on the union)")
    print(f"  under-split {len(res['undersplit'])}   (one span swallowed several gold spans)")
    print(f"  spurious    {len(res['spurious'])}")
    print("LABELS")
    print(f"  micro P/R/F1  {p:.2f} / {r:.2f} / {f:.2f}   (tp {res['tp']}, fp {res['fp']}, fn {res['fn']})")
    print(f"  exact spans   {res['exact']}/{res['scored']} = {res['exact'] / max(1, res['scored']):.0%}")
    print("  per type          P     R     F1   tp fp fn")
    for t in sorted(res["per_type"]):
        c = res["per_type"][t]
        tp_, fp_, fn_ = c["tp"], c["fp"], c["fn"]
        pp, rr, ff = prf(tp_, fp_, fn_)
        print(f"    {t:18}{pp:5.2f} {rr:5.2f} {ff:5.2f}  {tp_:3}{fp_:3}{fn_:3}")
    if not show_diff:
        return
    print("\nERRORS")
    for g in res["missed"]:
        print(f"  MISSED   {g['msg']:4} {g['text'][:60]!r}  gold={g['labels']}")
    for g, preds in res["oversplit"]:
        print(f"  SPLIT    {g['msg']:4} {g['text'][:60]!r} -> {len(preds)} spans")
    for p, gs in res["undersplit"]:
        print(f"  MERGED   {p['msg']:4} {p['text'][:60]!r} covers {len(gs)} gold spans")
    for p in res["spurious"]:
        print(f"  SPURIOUS {p['msg']:4} {p['text'][:60]!r}  pred={p['labels']}")
    for g, preds, s_fn, s_fp, ok in res["rows"]:
        if ok:
            continue
        got = sorted(set().union(*(set(p["labels"]) for p in preds)))
        print(f"  LABEL    {g['msg']:4} {g['text'][:50]!r:54} gold={g['labels']} pred={got}"
              + (f"  missing={sorted(s_fn)}" if s_fn else "") + (f"  extra={sorted(s_fp)}" if s_fp else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("gold")
    ap.add_argument("--diff", action="store_true")
    a = ap.parse_args()
    report(score(load_pred(a.output), json.load(open(a.gold))["spans"]), a.diff)
