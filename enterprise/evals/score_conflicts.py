"""Score result["conflicts"] in saved outputs against gold/conflicts.gold.json.

    python3 evals/score_conflicts.py                 # every source that has gold and a *.fast_result.json
    python3 evals/score_conflicts.py ent-12          # one
Outputs must come from run_fast.py (which runs the logic check). An output without a "conflicts" key is reported as 'not run'.
"""
import json, sys
from pathlib import Path

H = Path(__file__).resolve().parent
gold = json.load(open(H / "gold" / "conflicts.gold.json"))
only = sys.argv[1] if len(sys.argv) > 1 else ""
bad = 0
for sid, g in gold.items():
    if sid == "note" or only not in sid:
        continue
    f = H / "outputs" / f"{sid}.fast_result.json"
    if not f.exists():
        print(f"SKIP {sid}: no output"); continue
    d = json.load(open(f))
    if "conflicts" not in d:
        print(f"NOT RUN {sid}: output predates the logic check"); continue
    hard = [c for c in d["conflicts"] if c["verdict"] == "conflict"]
    if g.get("forbid"):
        ok = not hard
        print(("PASS" if ok else "FAIL"), sid, "- no conflict expected;", len(hard), "returned")
        for c in hard: print("   ", c["a"]["text"][:70], "<>", c["b"]["text"][:70])
        bad += not ok
    for e in g.get("expect", []):
        hit = [c for c in hard if c["kind"] == e["kind"] and e["a_contains"].lower() in c["a"]["text"].lower()
               and e["b_contains"].lower() in c["b"]["text"].lower() and (not e.get("a_author") or c["a"]["author"] == e["a_author"])]
        print(("PASS" if hit else "FAIL"), sid, "- expected conflict:", e["a_contains"], "vs", e["b_contains"])
        bad += not hit
    extra = [c for c in hard if g.get("expect") and not any(e["a_contains"].lower() in c["a"]["text"].lower() for e in g["expect"])]
    if extra:
        print("   extra conflicts (check for false alarms):", len(extra))
sys.exit(1 if bad else 0)
