"""Eval all enterprise sources against gold and print a summary table.

    python3 eval.py                        # run all, score all
    python3 eval.py --only ent-08          # run one
    python3 eval.py --score-only           # skip pipeline, score cached outputs/
    python3 eval.py --diff                 # show per-check detail for failures

Gold files live in evals/gold/, raw sources in evals/sources/.
Results are cached in outputs/ and reused unless --rerun is passed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import types

# ---------------------------------------------------------------------------
# Bootstrap (same as run.py)
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent
_SERVICES_ROOT = _HERE / "pipeline"
sys.path.insert(0, str(_SERVICES_ROOT))

_api = types.ModuleType("pact.api")
_models = types.ModuleType("pact.api.models")
for _n in ("BriefRevision", "Event", "Job", "SourceUnit", "Workspace"):
    setattr(_models, _n, type(_n, (), {}))
_models.JobStatus = types.SimpleNamespace(done="done", failed="failed", pending="pending")
_models.JobType = types.SimpleNamespace()
sys.modules.setdefault("pact.api", _api)
sys.modules.setdefault("pact.api.models", _models)

sys.path.insert(0, str(_HERE / "evals"))
from score import score as score_result  # noqa: E402

from run import run  # noqa: E402 (picks up the pipeline)

# ---------------------------------------------------------------------------
# Manifest: raw source, gold file, optional question override
# ---------------------------------------------------------------------------
_RAW = _HERE / "evals" / "sources"
_GOLD = _HERE / "evals" / "gold"

SOURCES = [
    {
        "id": "ent-08-email-vendor-private-note",
        "raw": _RAW / "ent-08-email-vendor-private-note.txt",
        "gold": _GOLD / "ent-08-email-vendor-private-note.gold.json",
        "question": "Should the Q4 contract review pilot buy Clauseline (A) or be built in-house (B)?",
    },
    {
        "id": "ent-09-email-offsite-reversal-forward",
        "raw": _RAW / "ent-09-email-offsite-reversal-forward.txt",
        "gold": _GOLD / "ent-09-email-offsite-reversal-forward.gold.json",
        "question": "Should the Q1 offsite be in Austin (Feb 9-11) or Denver (Feb 23-25)?",
    },
    {
        "id": "ent-10-meeting-golive-decision",
        "raw": _RAW / "ent-10-meeting-golive-decision.txt",
        "gold": _GOLD / "ent-10-meeting-golive-decision.gold.json",
        "question": "When should the claims dashboard go live: November 2 or November 16?",
    },
    {
        "id": "real-01-email-alpha-evolve",
        "raw": _RAW / "real-01-email-alpha-evolve.txt",
        "gold": _GOLD / "real-01-email-alpha-evolve.gold.json",
        "question": "What should Wednesday's (Oct 14) session with RL cover: only the Alpha Evolve progress update, or also the broader always-on learning assessment?",
    },
    {
        "id": "real-02-email-benefit-expense-chain",
        "raw": _RAW / "real-02-email-benefit-expense-chain.txt",
        "gold": _GOLD / "real-02-email-benefit-expense-chain.gold.json",
        "question": "How should benefit-expense projection and management (BEPMM) be funded and organised: one coordinated project funded by reallocation, or integrated into existing work such as Clinical Intelligence?",
    },
    {
        "id": "real-03-meeting-tgov-status",
        "raw": _RAW / "real-03-meeting-tgov-status.txt",
        "gold": _GOLD / "real-03-meeting-tgov-status.gold.json",
        "question": "Where should the chart tool be deployed for the pilot: inside our own environment or the vendor's hosted environment?",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _out_path(sid: str) -> pathlib.Path:
    return _HERE / "evals" / "outputs" / f"{sid}.result.json"


def _summary_path(sid: str) -> pathlib.Path:
    return _HERE / "evals" / "outputs" / f"{sid}.summary.txt"


def _extract_summary(brief_body: dict) -> str:
    es = brief_body.get("executive_summary") or ""
    if isinstance(es, str):
        return es
    return es.get("text") or es.get("body") or ""


async def _run_one(s: dict) -> dict:
    result = await run(s["raw"], question=s.get("question"))
    out = _out_path(s["id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = _extract_summary(result)
    _summary_path(s["id"]).write_text(summary, encoding="utf-8")
    return result


def _score_one(s: dict, diff: bool) -> tuple[int, int]:
    out = _out_path(s["id"])
    if not out.exists():
        print(f"  {s['id']}: no output (run without --score-only first)")
        return 0, 1

    result = json.load(out.open())
    gold = json.load(s["gold"].open())
    summary_path = _summary_path(s["id"])
    summary = summary_path.read_text() if summary_path.exists() else ""

    rep = score_result(gold, result, summary)
    passed = sum(1 for row in rep.rows if row[0] in ("PASS", "ok  "))
    total = len(rep.rows)

    if diff:
        print(f"\n  {s['id']}:")
        for row in rep.rows:
            if row[0] in ("FAIL", "warn") or diff:
                print(f"    {row[0]}  {row[1]}" + (f"  -- {row[2]}" if row[2] else ""))
    return passed, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run/score only this transcript id prefix")
    ap.add_argument("--score-only", action="store_true", help="skip pipeline, score cached results")
    ap.add_argument("--rerun", action="store_true", help="rerun even if output already exists")
    ap.add_argument("--diff", action="store_true", help="print per-check detail")
    a = ap.parse_args()

    sources = [s for s in SOURCES if not a.only or s["id"].startswith(a.only)]
    if not sources:
        sys.exit(f"no sources match '{a.only}'")

    # Run pipeline
    if not a.score_only:
        for s in sources:
            out = _out_path(s["id"])
            if out.exists() and not a.rerun:
                print(f"  {s['id']}: cached, skipping (--rerun to force)")
                continue
            print(f"\n[{s['id']}]")
            asyncio.run(_run_one(s))

    # Score
    print("\n" + "=" * 72)
    print(f"{'transcript':<45} {'pass/total':>12}  {'hard_fail':>10}")
    print("-" * 72)
    all_pass = all_total = all_hard = 0
    for s in sources:
        passed, total = _score_one(s, diff=False)
        out = _out_path(s["id"])
        if not out.exists():
            continue
        result = json.load(out.open())
        gold = json.load(s["gold"].open())
        summary_path = _summary_path(s["id"])
        summary = summary_path.read_text() if summary_path.exists() else ""
        rep = score_result(gold, result, summary)
        hard_fail = rep.hard_fail
        all_pass += passed
        all_total += total
        all_hard += hard_fail
        print(f"  {s['id']:<43} {passed:>5}/{total:<5}   {hard_fail:>6}")
        if a.diff:
            for row in rep.rows:
                indent = "    "
                print(f"{indent}{row[0]}  {row[1]}" + (f"  -- {row[2]}" if row[2] else ""))
    print("-" * 72)
    print(f"  {'total':<43} {all_pass:>5}/{all_total:<5}   {all_hard:>6}")
    print()
    if all_hard == 0:
        print("all hard checks pass")
    else:
        print(f"{all_hard} hard check(s) failed")
    sys.exit(0 if all_hard == 0 else 1)


if __name__ == "__main__":
    main()
