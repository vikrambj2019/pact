"""Replay saved real portal extractions through current deterministic functions.

This is explicitly a replay, not a new model run.
"""
import argparse
import json
from pathlib import Path
from load_pipeline import parser, merger, synthesizer
from score_portal import score_portal

def replay(path, output):
    original = json.loads(Path(path).read_text())
    units = parser._parse_teams(Path(__file__).with_name("portal.txt").read_text())
    for i, u in enumerate(units):
        u["id"] = f"fixture-{i:03d}"
    body = original["brief"]
    options = {o["id"]: o for o in body["options"]}
    for p in body["positions"]:
        p["option_title"] = options[p["option_id"]]["title"]
    for obj in body["objections"]:
        obj["option_titles"] = [options[i]["title"] for i in obj.get("option_ids", []) if i in options]
    merged = merger.reconcile_explicit_replies(body, units)
    body = synthesizer._assign_server_ids(synthesizer._build_brief(merged))
    body["question"] = {"text": "Member portal launch: Nov 16 with vendor, security and legal dependencies"}
    report = {}
    body["executive_summary"] = synthesizer._render_decision_card(body,
        {u["id"]: u["normalized_text"] for u in units},
        {u["id"]: {"author": u["author_label"], "ts": u["locator"]["timestamp_str"]} for u in units}, report)
    score = score_portal(body, units)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"result.json").write_text(json.dumps({"mode": "deterministic replay of live extraction", "input": str(path),
        "model": original["model"], "score": score, "brief": body, "report": report}, indent=2))
    (output/"card.txt").write_text(body["executive_summary"])
    print(json.dumps(score))
    return score

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    replay(args.input, args.output)
