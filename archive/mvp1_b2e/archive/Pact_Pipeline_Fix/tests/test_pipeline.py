import asyncio
import copy
import unittest
from datetime import datetime, timezone
from load_pipeline import chunker, dedup, extractor, merger, parser, synthesizer

def unit(uid, author, text, seq=0):
    return {"id": uid, "author_label": author, "normalized_text": text,
            "original_text": text, "sequence": seq, "locator": {}}

def item(text, uid, quote=None, **kw):
    return {"text": text, "quote": quote or text, "source_unit_ids": [uid], **kw}

class PipelineTests(unittest.TestCase):
    def test_public_budget_affirmation_is_separate_from_private_arrangement(self):
        units = [unit("h", "Hana", "thank you. 400 is the number then", 0)]
        merged = merger.merge_extractions([{"agreements": [{"subject": "$400 per sibling cap", "participant_label": "Hana",
            "stance": "affirm", "private": True, "source_unit_ids": ["private"], "quote": "between jobs"}]}])
        result = merger.reconcile_explicit_replies(merged, units)
        self.assertFalse(result["agreements"][0]["private"])
        self.assertEqual(result["agreements"][0]["source_unit_ids"], ["h"])

    def test_source_reconciliation_preserves_explicit_reopening(self):
        units = [unit("1", "A", "Harbor Hall is out", 0), unit("2", "B", "Reopen Harbor Hall", 1)]
        merged = merger.merge_extractions([{"options": [{"title": "Harbor Hall", "decision": "venue", "source_unit_ids": ["1"]}]}])
        self.assertEqual(merger.reconcile_explicit_replies(merged, units)["options"][0]["status"], "active")

    def test_budget_as_option_does_not_erase_requirement_affirmers(self):
        units = [unit("n", "Nadia", "400 each is the cap", 0), unit("y", "Yusuf", "ok", 1),
                 unit("h", "Hana", "thank you. 400 is the number then", 2)]
        merged = merger.merge_extractions([{"options": [{"title": "$400 per sibling cap", "decision": "budget", "source_unit_ids": ["n"]}],
            "positions": [{"participant_label": "Yusuf", "option_title": "$400 per sibling cap", "decision": "budget",
                           "stance": "support", "source_unit_ids": ["y"], "quote": "ok"}]}])
        result = merger.reconcile_explicit_replies(merged, units)
        self.assertEqual(result["options"], [])
        self.assertEqual({a["participant_label"] for a in result["agreements"]}, {"Nadia", "Yusuf", "Hana"})

    def test_step_free_requirement_and_explicit_withdrawals_are_recovered(self):
        units = [unit("y", "Yusuf", "whatever place we pick has to be step-free", 0),
                 unit("n", "Nadia", "my house has 5 steps so catering at mine is out", 1),
                 unit("l", "Leila", "Harbor Hall is out then. Casa Verde is ground floor", 2)]
        merged = merger.merge_extractions([{"options": [{"title": t, "decision": "venue", "status": "active", "source_unit_ids": ["n"]}
            for t in ("Cater at Nadia's", "Harbor Hall", "Casa Verde")]}])
        result = merger.reconcile_explicit_replies(merged, units)
        self.assertEqual({o["title"] for o in result["options"] if o["status"] == "dropped"}, {"Cater at Nadia's", "Harbor Hall"})
        self.assertEqual(result["constraints"][0]["text"], "Step-free access required")

    def test_budget_ceiling_and_missing_booking_request_are_recovered(self):
        units = [unit("o", "Omar", "I'm not paying more than $400 each", 0),
                 unit("n", "Nadia", "Casa Verde, Dec 12. Leila book it", 1),
                 unit("l", "Leila", "I'll trim the guest list", 2)]
        merged = merger.merge_extractions([{"options": [{"title": "Casa Verde", "decision": "venue", "source_unit_ids": ["n"]}]}])
        result = merger.reconcile_explicit_replies(merged, units)
        self.assertTrue(any(a["participant_label"] == "Omar" and "400" in a["subject"] for a in result["agreements"]))
        self.assertTrue(any(a["owner_label"] == "Leila" and a["status"] == "requested" for a in result["actions"]))

    def test_yes_to_guest_task_does_not_accept_booking(self):
        units = [unit("n", "Nadia", "Leila trims the list to 20 by Sunday", 0), unit("l", "Leila", "yes", 1)]
        result = extractor.validate_extraction({"actions": [item("Book Casa Verde", "l", "yes",
            owner_label="Leila", owner_kind="named", status="committed")]}, units)
        self.assertEqual(result["actions"][0]["status"], "requested")

    def test_unique_quote_repairs_mistyped_citation_only(self):
        units = [unit("real", "A", "The private room only holds 14")]
        result = extractor.validate_extraction({"claims": [item("Capacity 14", "typo", "only holds 14")]}, units)
        self.assertEqual(result["claims"][0]["source_unit_ids"], ["real"])
        self.assertTrue(result["claims"][0]["citation_repaired"])
        ambiguous = [unit("one", "A", "yes"), unit("two", "B", "yes", 1)]
        result = extractor.validate_extraction({"claims": [item("A claim", "typo", "yes")]}, ambiguous)
        self.assertEqual(result["claims"], [])

    def test_health_reason_is_hidden_even_if_flag_is_missing(self):
        source = {"h": "Dad's knee surgery went fine", "p": "parking needed"}
        body = synthesizer._build_brief({"claims": [item("Dad's knee surgery went fine", "h")],
            "constraints": [item("Dad won't walk far, so parking is needed", "p", "parking needed", sensitive=True)]})
        card = synthesizer._render_decision_card(body, source)
        self.assertNotIn("knee", card)
        self.assertNotIn("Dad", card)
        self.assertIn("parking is needed", card)

    def test_capacity_comparison_requires_intended_guest_target(self):
        units = [unit("a", "Tariq", "called Alma. the private room only holds 14")]
        base = {"options": [{"title": "Alma", "decision": "venue", "status": "active", "source_unit_ids": ["a"]}]}
        merged = merger.merge_extractions([base])
        self.assertEqual(merger.reconcile_explicit_replies(merged, units)["options"][0]["status"], "active")
        merged["actions"] = [item("Trim the guest list to 20", "l", owner_label="Leila", status="committed")]
        self.assertEqual(merger.reconcile_explicit_replies(merged, units)["options"][0]["status"], "dropped")

    def test_adjacent_bundle_yes_and_deadline_are_resolved(self):
        units = [unit("l", "Leila", "final from me: Dec 12, Casa Verde, 20 max, $400 each", 0),
                 unit("n", "Nadia", "yes. I'll send the $500 deposit Monday", 1),
                 unit("c", "Leila", "$500 deposit needed by Oct 16", 2)]
        merged = merger.merge_extractions([{"options": [
            {"title": "Dec 12", "decision": "date", "source_unit_ids": ["l"], "quote": "Dec 12"},
            {"title": "Casa Verde", "decision": "venue", "source_unit_ids": ["l"], "quote": "Casa Verde"}],
            "actions": [item("Send $500 deposit", "n", "I'll send the $500 deposit Monday", owner_label="Nadia", owner_kind="named", status="committed", due="Monday")],
            "claims": [item("$500 deposit needed by Oct 16", "c")]}])
        reconciled = merger.reconcile_explicit_replies(merged, units)
        self.assertEqual({p["option_title"] for p in reconciled["positions"] if p["participant_label"] == "Nadia"}, {"Dec 12", "Casa Verde"})
        self.assertEqual({a["participant_label"] for a in reconciled["agreements"] if "20" in a["subject"]}, {"Leila", "Nadia"})
        self.assertEqual(reconciled["actions"][0]["external_due"], "Oct 16")
        self.assertEqual(reconciled["actions"][0]["due"], "Monday")
        self.assertIn("c", reconciled["actions"][0]["source_unit_ids"])

    def test_qualified_yes_and_other_document_do_not_create_votes(self):
        opts = [{"title": "Option A", "decision": "venue", "source_unit_ids": ["1"]}]
        merged = merger.merge_extractions([{"options": opts}])
        for reply in ("sure whatever", "ok", "yes if parking is available", "yes but not A"):
            units = [unit("1", "X", "Option A", 0), unit("2", "Y", reply, 1)]
            self.assertEqual(merger.reconcile_explicit_replies(merged, units)["positions"], [])
        units = [{**unit("1", "X", "Option A", 0), "document_id": "one"},
                 {**unit("2", "Y", "yes", 1), "document_id": "two"}]
        self.assertEqual(merger.reconcile_explicit_replies(merged, units)["positions"], [])

    def test_short_quote_requires_word_boundaries(self):
        self.assertFalse(extractor.quote_in_text("ok", "booked my flight"))
        self.assertFalse(extractor.quote_in_text("yes", "yesterday"))
        self.assertTrue(extractor.quote_in_text("ok yes", "OK yes. Casa Verde"))

    def test_focused_pass_recovers_omitted_commitment(self):
        class LLM:
            async def extract(self, system, user, tool, **kw):
                if tool["name"] == "review_elements":
                    import json
                    keys = [e["key"] for e in json.loads(user)["elements"]]
                    return {"verdicts": [{"key": k, "verdict": "pass", "reason": "Owner explicitly accepts cake task"} for k in keys], "consistency": "pass", "issues": []}
                if tool["name"] == "extract_candidates":
                    return {}
                return {"actions": [item("Bring the cake", "o", "cake is mine", owner_label="Omar",
                                        owner_kind="named", status="committed")]}
        result = asyncio.run(extractor.extract_chunk([unit("o", "Omar", "cake is mine")], LLM()))
        self.assertEqual(result["actions"][0]["status"], "committed")

    def test_fixture_preserves_all_valid_turns(self):
        from pathlib import Path
        units = parser._parse_whatsapp(Path(__file__).with_name("anniversary.txt").read_text())
        self.assertEqual(len(units), 87)
        self.assertEqual(len(chunker.chunk_units(units)), 3)
        self.assertTrue(any("example.com" in u["normalized_text"] for u in units))

    def test_distinct_chat_turns_are_not_duplicates(self):
        units = [unit("1", "Nadia", "Casa Verde Dec 12 $400 each ground floor private room"),
                 unit("2", "Leila", "Casa Verde Dec 12 $400 each ground floor private room", 1),
                 unit("3", "Nadia", "Casa Verde Dec 12 NOT $400 each ground floor private room", 2)]
        dedup.mark_duplicates(units)
        self.assertTrue(all(u["_dup_sequence"] is None for u in units))

    def test_only_exact_dated_copies_are_duplicates(self):
        now = datetime.now(timezone.utc)
        u = unit("1", "Nadia", "An identical email message")
        u["timestamp"] = now
        units = [u, {**u, "sequence": 1}, {**u, "sequence": 2, "author_label": "Leila"}]
        dedup.mark_duplicates(units)
        self.assertEqual([u["_dup_sequence"] for u in units], [None, 0, None])

    def test_whatsapp_continuations_links_and_skips(self):
        text = "10/6/26, 12:05 PM - Leila: confirmed parking https://example.com\nabout 12 spots\n10/6/26, 12:06 PM - Tariq: <Media omitted>\n10/6/26, 12:07 PM - Tariq: ok yes. Casa Verde"
        units = parser._parse_whatsapp(text)
        self.assertEqual(len(units), 2)
        self.assertIn("about 12 spots", units[0]["normalized_text"])
        self.assertIn("https://example.com", units[0]["original_text"])

    def test_overlap_is_context_only(self):
        units = [unit(str(i), "A", "hello", i) for i in range(4)]
        chunks = chunker.chunk_units(units, max_units=2, overlap_units=1)
        self.assertTrue(chunks[1][0]["context_only"])
        out = extractor.validate_extraction({"claims": [item("hello", "1")]}, chunks[1])
        self.assertEqual(out["claims"], [])

    def test_speaker_must_own_quote_not_another_citation(self):
        units = [unit("n", "Nadia", "Yusuf is fine with anything"), unit("y", "Yusuf", "I care")]
        p = {"participant_label": "Yusuf", "option_title": "Casa Verde", "stance": "support",
             "source_unit_ids": ["n", "y"], "quote": "Yusuf is fine with anything"}
        result = extractor.validate_extraction({"positions": [p]}, units)
        self.assertEqual(result["positions"], [])

    def test_assignment_is_not_commitment(self):
        a = item("Book Casa Verde", "n", "Leila book it", owner_label="Leila", owner_kind="named", status="committed")
        result = extractor.validate_extraction({"actions": [a]}, [unit("n", "Nadia", "Leila book it")])
        self.assertEqual(result["actions"][0]["status"], "requested")

    def test_date_switch_does_not_erase_venue_support(self):
        opts = [{"title": t, "decision": d, "source_unit_ids": ["s"], "quote": t}
                for t, d in [("Dec 5", "date"), ("Dec 12", "date"), ("Casa Verde", "venue")]]
        ps = [{"participant_label": "Yusuf", "option_title": t, "decision": d, "stance": stance,
               "source_unit_ids": [str(i)], "_seq": i}
              for i, (t, d, stance) in enumerate([
                  ("Dec 5", "date", "support"), ("Casa Verde", "venue", "support"),
                  ("Dec 12", "date", "support"), ("Casa Verde", "venue", "acknowledged")])]
        merged = merger.merge_extractions([{"options": opts, "positions": ps}])
        self.assertEqual({p["option_title"]: p["stance"] for p in merged["positions"]},
                         {"Casa Verde": "support", "Dec 12": "support"})
        self.assertEqual(len(merged["superseded_positions"]), 1)

    def test_same_title_in_different_decisions(self):
        opts = [{"title": "A", "decision": d, "source_unit_ids": ["s"]} for d in ("date", "venue")]
        ps = [{"participant_label": "X", "option_title": "A", "decision": d,
               "stance": "support", "source_unit_ids": ["s"]} for d in ("date", "venue")]
        merged = merger.merge_extractions([{"options": opts, "positions": ps}])
        body = synthesizer._assign_server_ids(synthesizer._build_brief(merged))
        self.assertEqual(len(body["options"]), 2)
        self.assertEqual(len({p["option_id"] for p in body["positions"]}), 2)

    def test_mislabeled_stance_does_not_invent_option(self):
        merged = merger.merge_extractions([{"options": [{"title": "Casa Verde", "decision": "venue", "source_unit_ids": ["s"]}],
            "positions": [{"participant_label": "Nadia", "option_title": "Casa Verde", "decision": "date",
                           "stance": "support", "source_unit_ids": ["s"]}]}])
        self.assertEqual(len(merged["options"]), 1)
        self.assertEqual(merged["positions"][0]["decision"], "venue")

    def test_month_spelling_does_not_split_same_date(self):
        self.assertTrue(merger._same_option("December 12", "Dec 12"))
        self.assertFalse(merger._same_option("December 12", "Dec 5"))

    def test_dropped_does_not_reopen_on_mention(self):
        chunks = [{"options": [{"title": "Harbor Hall", "decision": "venue", "status": status,
                                "source_unit_ids": [str(i)]}]} for i, status in enumerate(["active", "dropped", "active"])]
        merged = merger.merge_extractions(chunks)
        self.assertEqual(merged["options"][0]["status"], "dropped")
        chunks.append({"options": [{"title": "Harbor Hall", "decision": "venue", "status": "reopened", "source_unit_ids": ["4"]}]})
        self.assertEqual(merger.merge_extractions(chunks)["options"][0]["status"], "active")

    def test_actions_keep_owner_acceptance_and_separate_tasks(self):
        def action(owner, status, seq):
            return item("Book the venue", str(seq), owner_label=owner, owner_kind="named", status=status, _seq=seq)
        merged = merger.merge_extractions([{"actions": [action("Leila", "requested", 0),
                                                       action("Leila", "committed", 1),
                                                       action("Leila", "requested", 2),
                                                       action("Nadia", "committed", 3)]}])
        self.assertEqual(len(merged["actions"]), 2)
        self.assertEqual(merged["actions"][0]["status"], "committed")

    def test_requirements_keep_all_affirmers(self):
        agreements = [{"subject": "$400 per sibling cap", "participant_label": name,
                       "stance": "affirm", "source_unit_ids": [name]} for name in ("Nadia", "Omar", "Leila", "Yusuf", "Hana")]
        merged = merger.merge_extractions([{"agreements": agreements}])
        self.assertEqual({a["participant_label"] for a in merged["agreements"]}, {"Nadia", "Omar", "Leila", "Yusuf", "Hana"})

    def test_legacy_entailment_failure_removes_items(self):
        class LLM:
            async def extract(self, system, user, tool, **kw):
                if tool["name"] == "check_support":
                    raise RuntimeError("offline")
                return {"title": "-", "status": "-", "proposed": [
                    {"text": "Invented support", "verbatim_quote": "ok", "source_unit_id": "s"}]}
        report = {}
        card = asyncio.run(synthesizer._generate_exec_summary({}, LLM(), {"s": "ok"}, report=report))
        self.assertNotIn("Invented support", card)
        self.assertFalse(report["verified"])

    def test_public_card_acceptance_privacy_and_grounding(self):
        source = {"n": "I'll send the $500 deposit Monday", "l": "Leila book it",
                  "h": "between jobs", "o": "I'll cover the 100", "p": "step-free access required",
                  "c": "free lot behind the building", "d": "drinks still open"}
        merged = {"actions": [
            item("Send $500 deposit", "n", source["n"], owner_label="Nadia", owner_kind="named", status="committed", due="Monday", external_due="Oct 16"),
            item("Book venue", "l", source["l"], owner_label="Leila", owner_kind="named", status="requested"),
            item("Cover Hana's share", "o", source["o"], owner_label="Omar", owner_kind="named", status="committed", private=True)],
            "claims": [item("Hana is between jobs", "h", source["h"], private=True),
                       item("Free parking lot", "c", source["c"], author_label="Leila", evidence_status="unverified"),
                       item("Made-up claim", "fake")],
            "constraints": [item("step-free access required", "p", sensitive=True)],
            "blockers": [item("Drinks funding is unresolved", "d", source["d"], resolution_status="open")]}
        body = synthesizer._assign_server_ids(synthesizer._build_brief(merged))
        report = {}
        card = synthesizer._render_decision_card(body, source, report=report)
        self.assertIn("REQUESTED, NOT ACCEPTED", card)
        self.assertIn("external deadline Oct 16", card)
        self.assertIn("promised Monday", card)
        self.assertIn("unverified", card)
        self.assertIn("step-free access required", card)
        for secret in ("between jobs", "Hana's share", "Made-up claim"):
            self.assertNotIn(secret, card)
        self.assertEqual(report["omitted_items"], 1)

if __name__ == "__main__":
    unittest.main()
