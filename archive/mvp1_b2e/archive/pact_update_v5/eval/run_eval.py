"""Consistency eval: run a chat through Pact and through a single-pass baseline N times each,
score every run against the answer key, and report how often each check passes.

    python -m eval.run_eval --runs 5                      # both modes, the anniversary chat
    python -m eval.run_eval --runs 10 --mode pact --out eval/runs
    python -m eval.run_eval --runs 5 --key eval/answer_key_portal.json   # the portal-launch chat

Uses the same LLMClient and model settings as the worker (LLM_API_KEY / llm_model from
pact.api.config). No database is needed: the parser, extract_and_merge and build_brief
run directly. Each run's brief and score are written to --out for inspection.

What to look at: a check that passes in some runs and fails in others is the
inconsistency the answer key warns about. "stable" checks pass in every run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path

from pact.worker.llm import LLMClient
from pact.worker.parser import _detect_format, _parse_email_chain, _parse_plain_text, _parse_transcript, _parse_whatsapp
from pact.worker.synthesizer import _SYNTH_TOOL, _unique, build_brief, extract_and_merge

from .score_brief import score_brief

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Baseline: one prompt, same output schema, no rules or checks
# ---------------------------------------------------------------------------

_BASELINE_SYSTEM = """\
You summarize a group chat into a decision card for the group. Read the whole chat and fill in
the structured brief: the options considered, who supports what, constraints, claims, objections,
blockers, actions with owners, and unrelated side topics. Then write `card`: a short plain-text
card with the sections NEEDS ATTENTION, PROPOSED, COMMITTED and UNVERIFIED, one line per item
starting with "· ", each line ending with the source in brackets, e.g. [Leila, 10/5/26 9:48 AM].
"""

_BASELINE_TOOL = {
    "name": "produce_card",
    "description": "Produce the brief and the card.",
    "input_schema": {
        "type": "object",
        "required": _SYNTH_TOOL["input_schema"]["required"] + ["card"],
        "properties": {**_SYNTH_TOOL["input_schema"]["properties"], "card": {"type": "string"}},
    },
}


def _format_chat(units: list[dict]) -> str:
    return "\n".join(f"[unit:{u['id']}] [{u['locator'].get('timestamp_str', '')}] {u['author_label']}: "
                     f"{u['normalized_text']}" for u in units)


async def run_baseline(units: list[dict], llm) -> dict:
    return await llm.extract(_BASELINE_SYSTEM, _format_chat(units), _BASELINE_TOOL, timeout=240, max_tokens=16000)


async def run_pact(units: list[dict], llm, question: str | None) -> dict:
    merged, _ = await extract_and_merge(units, llm)
    unit_text = {u["id"]: u["normalized_text"] for u in units}
    unit_meta = {u["id"]: {"author": u["author_label"] or "", "ts": u["locator"].get("timestamp_str", "")} for u in units}
    participants = _unique(u["author_label"] for u in units if u.get("author_label"))
    q = question or (merged.get("candidate_questions") or [{}])[0].get("text")
    brief, notes, warnings, report = await build_brief(merged, q, None, unit_text, unit_meta, participants, llm)
    brief["_eval"] = {"restore_notes": notes, "ref_warnings": warnings, "dropped_at_extraction": merged.get("dropped_items", [])}
    return brief


# ---------------------------------------------------------------------------

def load_units(path: Path) -> list[dict]:
    text = path.read_text()
    fmt = _detect_format(text, "")
    parse = {"whatsapp": _parse_whatsapp, "transcript": _parse_transcript,
             "email_chain": _parse_email_chain}.get(fmt, _parse_plain_text)
    units = parse(text)
    for u in units:
        u["id"] = str(uuid.uuid4())
    return units


def summarize(results: list[dict]) -> dict:
    scores = [r["score"] for r in results]
    names = [c["name"] for c in results[0]["checks"]] if results else []
    per_check = []
    for i, name in enumerate(names):
        passes = [r["checks"][i]["passed"] for r in results if i < len(r["checks"]) and r["checks"][i]["name"] == name]
        per_check.append({"name": name, "category": results[0]["checks"][i]["category"],
                          "pass_rate": sum(passes) / len(passes) if passes else 0.0})
    return {
        "runs": len(results),
        "mean_score": statistics.mean(scores) if scores else 0.0,
        "stdev_score": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "min_score": min(scores, default=0.0),
        "mean_false_agreements": statistics.mean(r["false_agreements"] for r in results) if results else 0.0,
        "always_pass": sum(1 for c in per_check if c["pass_rate"] == 1.0),
        "flaky": [c for c in per_check if 0.0 < c["pass_rate"] < 1.0],
        "always_fail": [c for c in per_check if c["pass_rate"] == 0.0],
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=str(ROOT / "eval" / "answer_key.json"))
    ap.add_argument("--chat", default=None, help="defaults to the key's source")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--mode", choices=("pact", "baseline", "both"), default="both")
    ap.add_argument("--out", default=str(ROOT / "eval" / "runs"))
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text())
    units = load_units(Path(args.chat or ROOT / key["source"]))
    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    llm = LLMClient()

    modes = ("pact", "baseline") if args.mode == "both" else (args.mode,)
    report = {}
    for mode in modes:
        results = []
        for i in range(args.runs):
            try:
                brief = await (run_pact(units, llm, key.get("question")) if mode == "pact" else run_baseline(units, llm))
            except Exception as exc:  # noqa: BLE001 - a failed run is a result too
                print(f"{mode} run {i + 1}: FAILED {exc}")
                (out / f"{mode}-{i + 1}.error.txt").write_text(repr(exc))
                continue
            scored = score_brief(brief, key)
            results.append(scored)
            (out / f"{mode}-{i + 1}.brief.json").write_text(json.dumps(brief, indent=2, default=str))
            (out / f"{mode}-{i + 1}.score.json").write_text(json.dumps(scored, indent=2))
            print(f"{mode} run {i + 1}: {scored['passed']}/{scored['total']}  "
                  f"false agreements {scored['false_agreements']}  {scored['by_category']}")
        report[mode] = summarize(results)

    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print()
    for mode, s in report.items():
        print(f"== {mode}: {s['runs']} runs, score {s['mean_score']:.2f} ± {s['stdev_score']:.2f} "
              f"(min {s['min_score']:.2f}), false agreements/run {s['mean_false_agreements']:.1f}, "
              f"{s['always_pass']} checks always pass, {len(s['flaky'])} flaky, {len(s['always_fail'])} always fail")
        for c in s["flaky"]:
            print(f"   flaky {c['pass_rate']:.0%}  [{c['category']}] {c['name']}")
        for c in s["always_fail"]:
            print(f"   FAIL        [{c['category']}] {c['name']}")
    print(f"\nRuns written to {out}")


if __name__ == "__main__":
    asyncio.run(main())
