"""Live fixture evaluation. Credentials are read locally and never saved.

python tests/run_live.py --credentials /path/to/env --model MODEL --runs 3
Requires only Python's standard library. The app uses its normal Anthropic SDK.
"""
import argparse
import asyncio
import json
import re
import urllib.request
from pathlib import Path
from load_pipeline import chunker, extractor, merger, parser, synthesizer

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

def score(body, units):
    """Semantic sets/status checks for the supplied answer key, not exact prose."""
    checks = {}
    options = body["options"]
    def matches(title, needle):
        return merger._same_option(title, needle)
    def positions(needle):
        ids = {o["id"] for o in options if matches(o["title"], needle)}
        return [p for p in body["positions"] if p.get("option_id") in ids]
    def support(needle):
        return {p["participant_label"] for p in positions(needle) if p["stance"] == "support"}
    checks["date_affirmers"] = support("Dec 12") == {"Nadia", "Leila", "Omar", "Tariq", "Yusuf", "Hana"}
    checks["venue_affirmers"] = support("Casa Verde") == {"Nadia", "Leila", "Omar", "Tariq", "Hana"}
    checks["yusuf_not_venue_support"] = "Yusuf" not in support("Casa Verde")
    checks["initial_date_vote_superseded"] = "Yusuf" not in support("Dec 5")
    def affirmers(number):
        return {a["participant_label"] for a in body["agreements"]
                if number in merger._anchors(a["subject"]) and a.get("stance") == "affirm" and not a.get("private")}
    checks["budget_affirmers"] = affirmers("400") == {"Nadia", "Omar", "Leila", "Yusuf", "Hana"}
    checks["guest_cap_affirmers"] = affirmers("20") == {"Nadia", "Leila"}
    for title in ("Harbor Hall", "Alma", "Seoul House"):
        checks["dropped_" + title] = any(matches(o["title"], title) and o.get("status") == "dropped" for o in options)
    checks["dropped_home_catering"] = any(o.get("status") == "dropped" and
        re.search(r"cater|nadia.*(?:home|house)|(?:home|house).*nadia", o["title"], re.I) for o in options)
    checks["step_free_constraint"] = any(re.search(r"step.free|no stairs|stair.free|accessible", c["text"], re.I) for c in body["constraints"])
    open_text = " ".join(i["text"] for section in ("blockers", "objections") for i in body[section]
                         if i.get("resolution_status", "open") == "open" and not i.get("moot"))
    checks["drinks_open"] = "drink" in open_text.lower()
    checks["overage_owner_open"] = bool(re.search(r"(?:who|owner|cover|pay|fund).*(?:drink|over|extra)|(?:drink|overage).*(?:who|owner|cover|pay|fund)", open_text, re.I))
    actions = [a for a in body["actions"] if not a.get("private")]
    def action(owner, pattern, status):
        return [a for a in actions if a.get("owner_label") == owner and a.get("status") == status and re.search(pattern, a["text"], re.I)]
    deposit = action("Nadia", "deposit", "committed")
    checks["deposit_committed_monday"] = any("monday" in a.get("due", "").lower() for a in deposit)
    checks["deposit_external_deadline"] = any("16" in a.get("external_due", "") for a in deposit)
    checks["trim_committed_sunday"] = any("sunday" in a.get("due", "").lower() for a in action("Leila", "guest|list", "committed"))
    checks["cake_committed"] = bool(action("Omar", "cake", "committed"))
    checks["booking_requested_not_accepted"] = bool(action("Leila", "book|reserv", "requested")) and not action("Leila", "book|reserv", "committed")
    checks["omar_not_drinks_owner"] = not action("Omar", "drink", "committed")
    separate = " ".join(i["title"] for i in body["separate_issues"]).lower()
    checks["photographer_separate"] = "photo" in separate
    checks["anniversary_dispute_separate"] = bool(re.search(r"anniversary|8th|9th", separate))
    checks["secrecy_separate"] = bool(re.search(r"secret|surprise|parent.*chat|confidential", separate))
    card = body["executive_summary"]
    checks["private_details_absent"] = not re.search(r"knee|surgery|between jobs|cover.{0,25}(?:100|hana)|pay.{0,10}back", card, re.I)
    facts = [c for c in body["claims"] if re.search(r"parking|ground.floor|20.*head|drinks.*20", c["text"], re.I)
             and c.get("evidence_status") != "superseded"]
    checks["venue_reports_not_verified_facts"] = bool(facts) and all(c.get("evidence_status") in ("unverified", "conflicting") for c in facts)
    by_id = {u["id"]: u for u in units}
    checks["all_items_have_valid_sources"] = all(i.get("source_unit_ids") and all(uid in by_id for uid in i["source_unit_ids"])
        for section in synthesizer._BRIEF_SECTIONS for i in body[section])
    checks["all_quotes_exist"] = all(i.get("quote") and any(extractor.quote_in_text(i["quote"], by_id[uid]["normalized_text"])
        for uid in i.get("source_unit_ids", []) if uid in by_id)
        for section in extractor._QUOTED_SECTIONS if section != "candidate_questions" for i in body.get(section, []))
    return {"passed": sum(checks.values()), "total": len(checks), "checks": checks,
            "failed": [name for name, ok in checks.items() if not ok]}

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
    units = parser._parse_whatsapp(Path(args.fixture).read_text())
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
        body = synthesizer._assign_server_ids(synthesizer._build_brief(interim))
        body["question"] = {"text": "Plan the parents' 40th anniversary dinner"}
        report = {}
        body["executive_summary"] = synthesizer._render_decision_card(body, {u["id"]: u["normalized_text"] for u in units},
            {u["id"]: {"author": u["author_label"], "ts": u["locator"].get("timestamp_str", "")} for u in units}, report)
        result = score(body, units)
        (output / f"run-{run_index+1}.json").write_text(json.dumps({"model": args.model, "score": result, "brief": body, "report": report}, indent=2))
        (output / f"run-{run_index+1}-card.txt").write_text(body["executive_summary"])
        print(json.dumps({"run": run_index + 1, **result}), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--credentials", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", default=str(Path(__file__).with_name("anniversary.txt")))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--output", default="live-results")
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        print(json.dumps({"error": str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__}))
        raise SystemExit(1)
