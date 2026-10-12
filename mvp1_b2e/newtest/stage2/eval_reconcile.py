"""Run + score the reconciler on every transcript that has a *.state.json gold file.

Reuses the extraction outputs already in outputs/ (run eval_all.py first), so you can
iterate on the reconciler without re-running normalize/classify.

    python eval_reconcile.py                 # reconcile + score
    python eval_reconcile.py --score-only    # rescore outputs/*.reconcile.json
    python eval_reconcile.py --diff          # list every failed check
    python eval_reconcile.py --only ent-03   # one transcript
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from score_reconcile import report, score

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--only")
    a = ap.parse_args()

    rows = []
    for gold in sorted((ROOT / "gold").glob("*.state.json")):
        name = gold.name.removesuffix(".state.json")
        if a.only and a.only not in name:
            continue
        cls = ROOT.parent / "stage1" / "outputs" / f"{name}.output.json"
        rec = ROOT.parent / "stage1" / "outputs" / f"{name}.reconcile.json"
        if not a.score_only:
            if not cls.exists():
                print(f"skip {name}: run eval_all.py first (no {cls.name})")
                continue
            cmd = [sys.executable, str(ROOT / "reconcile.py"), str(cls), "--out", str(rec)]
            if a.model:
                cmd += ["--model", a.model]
            print(f"\n===== {name} =====")
            subprocess.run(cmd, check=True)
        if not rec.exists():
            print(f"skip {name}: no {rec.name}")
            continue
        R, errs, dup, d = score(str(rec), str(gold))
        if a.diff:
            print(f"\n----- {name} checks -----")
            report(R, errs, dup, d, True)
        pct = lambda k: (f"{R[k][0]}/{R[k][1]}" if R[k][1] else "-")
        rows.append((name, pct("options"), pct("option_status"), pct("cells"), pct("cells_nonblank"),
                     pct("questions"), pct("elements"), pct("histories"), pct("flags"), len(dup)))

    cols = ["options", "opt_stat", "cells", "nonblank", "questions", "elements", "histories", "flags", "dups"]
    print(f"\n{'transcript':28} " + " ".join(f"{c:>9}" for c in cols))
    for r in rows:
        print(f"{r[0]:28} " + " ".join(f"{str(x):>9}" for x in r[1:]))


if __name__ == "__main__":
    main()
