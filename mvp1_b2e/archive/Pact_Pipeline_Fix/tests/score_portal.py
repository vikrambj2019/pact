"""Semantic expectations for the user-supplied portal transcript/card."""
import re

def score_portal(body, units):
    checks = {}
    options = body["options"]
    a = next((o for o in options if "Acme" in o["title"] and "OTP" not in o["title"]), {})
    b = next((o for o in options if "OTP" in o["title"]), {})
    c = next((o for o in options if "January" in o["title"]), {})
    ps = [p for p in body["positions"] if p.get("option_id") == a.get("id")]
    checks["teams_messages_and_authors"] = len(units) == 68 and len({u["author_label"] for u in units}) == 7
    checks["three_options"] = len(options) == 3
    checks["january_dropped"] = c.get("status") == "dropped"
    checks["otp_fallback"] = b.get("status") == "fallback"
    checks["explicit_A_supporters"] = {p["participant_label"] for p in ps if p["stance"] in ("support", "conditional")} == {"Marcus Webb", "Rosa Alvarez", "Elena Ruiz"}
    checks["elena_conditional"] = any(p["participant_label"] == "Elena Ruiz" and p["stance"] == "conditional" and p.get("conditions") for p in ps)
    checks["priya_proposer"] = any(p["participant_label"] == "Priya Nair" and (p.get("proposer") or p["stance"] == "proposed") for p in ps)
    checks["hal_not_option_vote"] = not any(p["participant_label"] == "Hal Okonkwo" and p["stance"] in ("support", "conditional") for p in ps)
    blockers = [x for x in body["blockers"] if x.get("resolution_status", "open") == "open"]
    checks["dpa_open"] = any("DPA" in x["text"] for x in blockers)
    checks["review_open"] = any(re.search(r"review|sign.off", x["text"], re.I) and "SOC" in x["text"] for x in blockers)
    checks["finding_open"] = any(re.search(r"finding.*3", x["text"], re.I) for x in blockers)
    objs = [x for x in body["objections"] if x.get("resolution_status", "open") == "open" and not x.get("moot")]
    checks["exact_open_gate_counts"] = len(blockers) == 3 and len(objs) == 2
    checks["funding_objection_open"] = any(re.search(r"budget|cost|approve|fund|215", x["text"], re.I) for x in objs)
    checks["recovery_objection_open"] = any("recovery" in x["text"].lower() for x in objs)
    acts = body["actions"]
    checks["exact_unowned_action_count"] = len([x for x in acts if x.get("owner_kind") != "named" and x.get("status") in ("requested", "committed")]) == 2
    checks["checkpoint_unowned"] = any("checkpoint" in x["text"].lower() and x.get("owner_kind") != "named" and "30" in x.get("due", "") for x in acts)
    checks["funding_ask_unowned"] = any(re.search(r"ask|request", x["text"], re.I) and "Hal" in x["text"] and x.get("owner_kind") != "named" and "30" in x.get("due", "") for x in acts)
    for owner, pattern, deadline in [("Marcus Webb", "code complete", "6"), ("Rosa Alvarez", "DPA", "20"), ("Elena Ruiz", "review", "14"), ("Dev Patel", "fix", "")]:
        matches = [x for x in acts if x.get("owner_label") == owner and x.get("status") == "committed" and re.search(pattern, x["text"], re.I)]
        checks["commitment_" + owner.split()[0]] = any(deadline in x.get("due", "") for x in matches) if deadline else bool(matches) and all(not x.get("due") for x in matches)
    card = body["executive_summary"]
    checks["revised_B_cost_visible"] = bool(re.search(r"\$?65(?:,000|k)", card, re.I)) and bool(re.search(r"\$?215(?:,000|k)", card, re.I))
    checks["disputed_effort_visible"] = bool(re.search(r"3 weeks", card, re.I)) and bool(re.search(r"5(?: weeks| once)", card, re.I))
    checks["stale_arrival_hidden"] = "Oct 23" not in card
    by_id = {u["id"]: u for u in units}
    from load_pipeline import extractor
    checks["citation_integrity"] = all(x.get("source_unit_ids") and all(s in by_id for s in x["source_unit_ids"]) and any(extractor.quote_in_text(x.get("quote", ""), by_id[s]["normalized_text"]) for s in x["source_unit_ids"])
        for section in extractor._QUOTED_SECTIONS if section != "candidate_questions" for x in body.get(section, []))
    return {"passed": sum(checks.values()), "total": len(checks), "checks": checks, "failed": [k for k,v in checks.items() if not v]}
