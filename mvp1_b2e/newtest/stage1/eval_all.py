"""Run + score every transcript that has a gold file, print one comparison table.

    python eval_all.py                 # run the pipeline, then score
    python eval_all.py --score-only    # rescore existing outputs/*.output.json
    python eval_all.py --diff          # also print per-transcript error lists
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from score import load_pred, prf, report, score

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--model")
    a = ap.parse_args()

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for gold in sorted((ROOT / "gold").glob("*.gold.json")):
        name = gold.name.removesuffix(".gold.json")
        txt, out = ROOT / "transcripts" / f"{name}.txt", out_dir / f"{name}.output.json"
        if not a.score_only:
            cmd = [sys.executable, str(ROOT / "run.py"), str(txt), "--out", str(out)]
            if a.model:
                cmd += ["--model", a.model]
            subprocess.run(cmd, check=True)
        if not out.exists():
            print(f"skip {name}: no output")
            continue
        res = score(load_pred(str(out)), json.load(open(gold))["spans"])
        if a.diff:
            print(f"\n===== {name} =====")
            report(res, True)
        p, r, f = prf(res["tp"], res["fp"], res["fn"])
        rows.append((name, res["seg_recall"], len(res["oversplit"]), len(res["undersplit"]),
                     len(res["spurious"]), p, r, f, res["exact"] / max(1, res["scored"])))

    print(f"\n{'transcript':28} {'segR':>5} {'over':>5} {'under':>5} {'spur':>5}  {'P':>5} {'R':>5} {'F1':>5} {'exact':>6}")
    for n, sr, ov, un, sp, p, r, f, ex in rows:
        print(f"{n:28} {sr:5.0%} {ov:5} {un:5} {sp:5}  {p:5.2f} {r:5.2f} {f:5.2f} {ex:6.0%}")


if __name__ == "__main__":
    main()
