"""Native Teams portal fixture evaluation. Credentials are never saved.

python tests/run_portal.py --credentials /path/to/env --model MODEL --runs 1
Requires only Python's standard library. The app uses its normal Anthropic SDK.
"""
import argparse
import asyncio
import json
import re
import urllib.request
from pathlib import Path
from load_pipeline import chunker, extractor, merger, parser, synthesizer, reviewer

class LiveLLM:
    def __init__(self, key, model):
        self.key, self.model = key, model

    async def extract(self, system, user, tool_schema, timeout=None):
        def call():
            payload = {"model": self.model, "max_tokens": 8192, "system": system,
                       "messages": [{"role": "user", "content": user}], "tools": [tool_schema],
                       "tool_choice": {"type": "tool", "name": tool_schema["name"]}}
            request = urllib.request.Request("https://api.anthropic.com/v1/messages",
                data=json.dumps(payload).encode(), headers={"x-api-key": self.key,
                "anthropic-version": "2023-06-01", "content-type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=timeout or 120) as response:
                    result = json.load(response)
            except Exception as exc:
                # Never propagate request headers, keys or a credential file.
                raise RuntimeError(f"Anthropic call failed ({type(exc).__name__}, HTTP {getattr(exc, 'code', 'n/a')})") from None
            if result.get("stop_reason") == "max_tokens":
                raise RuntimeError("Structured response truncated")
            for block in result.get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == tool_schema["name"]:
                    return block["input"]
            raise RuntimeError("No structured response")
        return await asyncio.to_thread(call)

async def run(args):
    values = {}
    for line in Path(args.credentials).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip("\"'")
    key = values.get("ANTHROPIC_API_KEY") or values.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("Credential file has no Anthropic key")
    llm = LiveLLM(key, args.model)
    text = Path(args.fixture).read_text()
    units = parser._parse_teams(text)
    for i, u in enumerate(units):
        u["id"] = f"fixture-{i:03d}"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for run_index in range(args.runs):
        results, known_options, known_decisions, open_items, known_actions, known_agreements, known_state = [], [], [], {}, [], [], {}
        for index, chunk in enumerate(chunker.chunk_units(units)):
            extracted = await extractor.extract_chunk(chunk, llm, index, known_options, known_decisions,
                list(open_items.values()), known_actions, known_agreements, known_state)
            results.append(extracted)
            interim = merger.merge_extractions(results, {u["id"]: u["sequence"] for u in units})
            known_options = [{"title": o["title"], "decision": o.get("decision", "")} for o in interim["options"]]
            known_decisions = list(dict.fromkeys(o["decision"] for o in known_options if o["decision"]))
            known_actions = [{k: a[k] for k in ("text", "owner_label", "owner_kind", "status", "due", "external_due") if k in a} for a in interim["actions"]]
            known_agreements = list(dict.fromkeys(a["subject"] for a in interim["agreements"]))
            known_state = {s: [{k: v for k, v in i.items() if k not in ("quote", "source_unit_ids", "id")}
                              for i in interim[s]] for s in ("options", "positions", "claims", "actions", "agreements")}
            open_items = {merger._norm_text(i["text"]): {"kind": kind, "text": i["text"]}
                          for section, kind in (("blockers", "blocker"), ("objections", "objection"))
                          for i in interim[section] if i.get("resolution_status", "open") == "open"}
            print(json.dumps({"run": run_index + 1, "chunk": index + 1, "dropped": len(extracted.get("_dropped", []))}), flush=True)
        interim = merger.reconcile_explicit_replies(interim, units)
        merged_review = await reviewer.review_extraction(interim, units, llm, stage="merged_extraction")
        body = synthesizer._assign_server_ids(synthesizer._build_brief(interim))
        body["question"] = {"text": "Plan the member portal launch"}
        report = {}
        body["executive_summary"] = synthesizer._render_decision_card(body, {u["id"]: u["normalized_text"] for u in units},
            {u["id"]: {"author": u["author_label"], "ts": u["locator"].get("timestamp_str", "")} for u in units}, report)
        from score_portal import score_portal
        body["extraction_review_report"] = await reviewer.review_extraction(body, units, llm, stage="final_structured_output")
        report["semantic_review"] = await reviewer.review_summary(body["executive_summary"], body, units, llm)
        body["executive_summary_report"] = report
        result = score_portal(body, units)
        (output / f"run-{run_index+1}.json").write_text(json.dumps({"model": args.model, "score": result, "brief": body, "report": report}, indent=2))
        (output / f"run-{run_index+1}-card.txt").write_text(body["executive_summary"])
        print(json.dumps({"run": run_index + 1, **result}), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--credentials", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", default=str(Path(__file__).with_name("portal.txt")))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--output", default="live-results")
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        print(json.dumps({"error": str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__}))
        raise SystemExit(1)
