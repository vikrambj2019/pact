"""Reapply final reconciliation/rendering to independently obtained live results.

This is not a fresh model call. It isolates deterministic changes while preserving
the exact model-generated state and source evidence for reproducible evaluation.
"""
import argparse
import json
from pathlib import Path
from load_pipeline import parser, merger, synthesizer
from run_live import score

def replay(paths, output, fixture):
    units = parser._parse_whatsapp(Path(fixture).read_text())
    for index, unit in enumerate(units):
        unit["id"] = f"fixture-{index:03d}"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scores = []
    for index, path in enumerate(paths, 1):
        original = json.loads(Path(path).read_text())
        body = original["brief"]
        options = {o["id"]: o for o in body["options"]}
        for position in body["positions"]:
            position["option_title"] = options[position["option_id"]]["title"]
        merged = merger.reconcile_explicit_replies(body, units)
        body = synthesizer._assign_server_ids(synthesizer._build_brief(merged))
        body["question"] = {"text": "Plan the parents' 40th anniversary dinner"}
        report = {}
        body["executive_summary"] = synthesizer._render_decision_card(body,
            {u["id"]: u["normalized_text"] for u in units},
            {u["id"]: {"author": u["author_label"], "ts": u["locator"]["timestamp_str"]} for u in units}, report)
        result = score(body, units)
        scores.append({"input": str(path), "original_score": original["score"]["passed"], **result})
        (output / f"replayed-{index:02d}.json").write_text(json.dumps({
            "mode": "deterministic replay of a saved live extraction result", "original_result": str(path),
            "model": original.get("model"), "score": result, "brief": body, "report": report}, indent=2))
        (output / f"replayed-{index:02d}-card.txt").write_text(body["executive_summary"])
        print(json.dumps({"replay": index, "passed": result["passed"], "total": result["total"], "failed": result["failed"]}))
    (output / "scores.json").write_text(json.dumps(scores, indent=2))
    return all(s["passed"] == s["total"] for s in scores)

if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("results", nargs="+")
    cli.add_argument("--output", default="validation/replays")
    cli.add_argument("--fixture", default=str(Path(__file__).with_name("anniversary.txt")))
    args = cli.parse_args()
    raise SystemExit(0 if replay(args.results, args.output, args.fixture) else 1)
