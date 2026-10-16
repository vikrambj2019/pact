import unittest
from load_pipeline import parser, merger, extractor, synthesizer
from test_pipeline import unit, item

class PortalRules(unittest.TestCase):
    def test_checkpoint_is_not_a_fourth_launch_option(self):
        units = [unit("1", "Q", "A) Blue launch. B) Red launch", 0), unit("2", "R", "Good with A and the checkpoint.", 1)]
        merged = merger.merge_extractions([{"options": [
            {"title": "Blue launch", "decision": "launch plan", "source_unit_ids": ["1"]},
            {"title": "Red launch", "decision": "launch plan", "source_unit_ids": ["1"]},
            {"title": "Go/no-go checkpoint May 30", "decision": "launch plan", "source_unit_ids": ["1"]}],
            "positions": [{"participant_label": "R", "option_title": "Go/no-go checkpoint May 30", "decision": "launch plan", "stance": "support", "quote": units[1]["normalized_text"], "source_unit_ids": ["2"]}]}])
        result = merger.reconcile_explicit_replies(merged, units)
        self.assertEqual(len(result["options"]), 2)
        self.assertEqual(result["positions"][0]["option_title"], "Blue launch")

    def test_fallback_status_is_recovered_from_explicit_label(self):
        units = [unit("1", "Q", "A) Blue launch. B) Red launch", 0), unit("2", "Q", "noted. so B is a fallback pending approval", 1)]
        merged = merger.merge_extractions([{"options": [{"temp_id": "arbitrary", "title": "Red launch", "decision": "plan", "source_unit_ids": ["1"], "status": "active"}]}])
        result = merger.reconcile_explicit_replies(merged, units)
        self.assertEqual(result["options"][0]["status"], "fallback")

    def test_effort_dispute_is_not_a_mandatory_gate(self):
        units = [unit("1", "Q", "three weeks versus five weeks")]
        merged = merger.merge_extractions([{"blockers": [item("Conflicting effort estimates", "1", "three weeks versus five weeks", resolution_status="open")]}])
        result = merger.reconcile_explicit_replies(merged, units)
        self.assertEqual(result["blockers"], [])
        self.assertEqual(result["claims"][0]["evidence_status"], "conflicting")

    def test_relative_deadline_before_then_uses_unique_date(self):
        units = [unit("1", "R", "I'll chase the contract. I'm out from May 20 so I need it signed before then.")]
        merged = merger.merge_extractions([{"actions": [item("Chase the contract", "1", "I'll chase the contract.", owner_label="R", owner_kind="named", status="committed")]}])
        result = merger.reconcile_explicit_replies(merged, units)
        self.assertEqual(result["actions"][0]["due"], "before May 20")

    def test_same_ticket_gate_is_not_counted_twice(self):
        units = [unit("1", "Q", "Finding #8 must close")]
        result = merger.merge_extractions([{"blockers": [item("Finding #8 must close before launch", "1", resolution_status="open"),
            item("Finding #8 remains open on auth service", "1", resolution_status="open")]}])
        result = merger.reconcile_explicit_replies(result, units)
        self.assertEqual(len(result["blockers"]), 1)

    def test_contract_gates_for_different_vendors_remain_distinct(self):
        units = [unit("1", "Q", "Contracts still open")]
        result = merger.merge_extractions([{"blockers": [item("DPA with Atlas not signed", "1", resolution_status="open"),
            item("DPA with Delta not signed", "1", resolution_status="open")]}])
        result = merger.reconcile_explicit_replies(result, units)
        self.assertEqual(len(result["blockers"]), 2)

    def test_funding_objection_does_not_hide_security_approval_gate(self):
        units = [unit("1", "Q", "Both are pending")]
        result = merger.merge_extractions([{"blockers": [item("Security approval still pending", "1", resolution_status="open")],
            "objections": [item("Budget cannot be approved", "1", resolution_status="open")]}])
        result = merger.reconcile_explicit_replies(result, units)
        self.assertEqual(len(result["blockers"]), 1)

    def test_unaccepted_date_is_removed_from_public_task_title(self):
        units = [unit("1", "D", "ok")]
        merged = merger.merge_extractions([{"actions": [item("Close finding #7 by Jun 6", "1", "ok", owner_label="D", owner_kind="named", status="committed", due="Jun 6")]}])
        result = merger.reconcile_explicit_replies(merged, units)
        self.assertNotIn("Jun 6", result["actions"][0]["text"])
        self.assertNotIn("due", result["actions"][0])

    def test_confirmation_omitted_from_model_citations_is_recovered(self):
        units = [unit("1", "L", "I'll trim the list", 0), unit("2", "P", "L trims the list by Sunday", 1), unit("3", "L", "yes", 2)]
        merged = merger.merge_extractions([{"actions": [item("Trim the list", "1", "I'll trim the list", owner_label="L", owner_kind="named", status="committed", due="Sunday")]}])
        result = merger.reconcile_explicit_replies(merged, units)
        self.assertEqual(result["actions"][0]["due"], "Sunday")
        self.assertIn("3", result["actions"][0]["source_unit_ids"])

    def test_owner_ok_does_not_accept_earlier_completion_date(self):
        units = [unit("1", "M", "Can we close #7 by Jun 6?", 0), unit("2", "D", "I can start tomorrow", 1),
                 unit("3", "P", "D, is that a yes?", 2), unit("4", "D", "ok", 3)]
        result = extractor.validate_extraction({"actions": [item("Close finding #7", "4", "ok", owner_label="D", owner_kind="named", status="committed", due="Jun 6")]}, units)
        self.assertNotIn("due", result["actions"][0])
        self.assertEqual(result["actions"][0]["requested_due"], "Jun 6")

    def test_named_confirmation_accepts_named_milestone_date(self):
        units = [unit("1", "P", "M confirms Jun 6 code complete, D on fix #7", 0), unit("2", "M", "confirmed", 1)]
        merged = merger.merge_extractions([{"actions": [item("Confirm Jun 6 code complete", "2", "confirmed", owner_label="M", owner_kind="named", status="committed")]}])
        result = merger.reconcile_explicit_replies(merged, units)
        self.assertEqual(result["actions"][0]["due"], "Jun 6")

    def test_update_promise_is_not_issue_resolution(self):
        units = [unit("1", "Q", "I'll know more Monday")]
        result = extractor.validate_extraction({"actions": [item("Resolve the retention clause", "1", "I'll know more Monday", owner_label="Q", owner_kind="named", status="committed", due="Monday")]}, units)
        self.assertEqual(result["actions"][0]["text"], "Provide a status update")

    def test_funding_ask_at_checkpoint_does_not_run_checkpoint(self):
        units = [unit("1", "Q", "Go/no-go checkpoint May 30.")]
        merged = merger.merge_extractions([{"actions": [item("Send funding ask by go/no-go checkpoint", "1", owner_kind="unassigned", status="requested", due="May 30")]}])
        result = merger.reconcile_explicit_replies(merged, units)
        self.assertTrue(any(a["text"] == "Run the go/no-go checkpoint" for a in result["actions"]))

    def test_teams_native_header_and_multiline(self):
        text = "Project title\n[1/2/26 9:05 AM] One Person: first\ncontinued\n[1/2/26 9:06 AM] Other Person: yes"
        self.assertEqual(parser._detect_format(text, "plain_text"), "teams")
        units = parser._parse_teams(text)
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["author_label"], "One Person")
        self.assertEqual(units[0]["normalized_text"], "first\ncontinued")
        self.assertTrue(units[0]["locator"]["timezone_unspecified"])

    def test_conditional_not_released_by_bare_confirmation(self):
        options = [{"title": "Red", "decision": "plan", "id": "red"}]
        ps, _ = merger._merge_positions([
            {"participant_label": "Q", "option_title": "Red", "stance": "conditional", "conditions": "review complete", "source_unit_ids": ["1"], "quote": "Red only if review complete"},
            {"participant_label": "Q", "option_title": "Red", "stance": "support", "source_unit_ids": ["2"], "quote": "yes"}], options)
        self.assertEqual(ps[0]["stance"], "conditional")
        self.assertEqual(ps[0]["conditions"], "review complete")

    def test_substantive_reconfirmation_can_update_condition(self):
        options = [{"title": "Red", "decision": "plan", "id": "red"}]
        ps, _ = merger._merge_positions([
            {"participant_label": "Q", "option_title": "Red", "stance": "conditional", "conditions": "flight booked", "source_unit_ids": ["1"], "quote": "Red only if flight booked"},
            {"participant_label": "Q", "option_title": "Red", "stance": "support", "source_unit_ids": ["2"], "quote": "booked my flight, Red works"}], options)
        self.assertEqual(ps[0]["stance"], "support")

    def test_escalation_is_not_approval(self):
        units = [unit("1", "Q", "someone has to take that to leadership")]
        result = extractor.validate_extraction({"objections": [item("Funds not approved", "1", "someone has to take that to leadership", resolution_status="resolved")]}, units)
        self.assertEqual(result["objections"][0]["resolution_status"], "open")

    def test_independent_legal_and_security_dependencies_do_not_merge(self):
        result = merger.merge_extractions([{"blockers": [item("Acme DPA not signed before launch", "1", resolution_status="open"),
            item("Acme SOC 2 report not received before launch", "2", resolution_status="open")]}])
        self.assertEqual(len(result["blockers"]), 2)

    def test_pending_review_does_not_become_received_gate(self):
        units = [unit("1", "Q", "Report received. I'll finish review by May 12")]
        result = merger.merge_extractions([{"actions": [item("Finish review of vendor audit by May 12", "1", owner_label="Q", owner_kind="named", status="committed", due="May 12")]}])
        result = merger.reconcile_explicit_replies(result, units)
        self.assertTrue(any("Review remains pending" in b["text"] and b["resolution_status"] == "open" for b in result["blockers"]))

    def test_checkpoint_proposal_has_no_owner(self):
        units = [unit("1", "Q", "Proposal: go with Z. Go/no-go checkpoint May 30.")]
        result = merger.reconcile_explicit_replies(merger.merge_extractions([{}]), units)
        self.assertEqual(result["actions"][0]["owner_kind"], "unassigned")
        self.assertEqual(result["actions"][0]["due"], "May 30")

    def test_estimate_is_rendered_without_full_option_title(self):
        body = synthesizer._assign_server_ids(synthesizer._build_brief({"options": [{"title": "May launch with internal verification", "decision": "plan", "source_unit_ids": ["1"], "quote": "May launch with internal verification"}],
            "claims": [item("Additional build cost estimated at $12k", "2")]}))
        card = synthesizer._render_decision_card(body, {"1": "May launch with internal verification", "2": "Additional build cost estimated at $12k"})
        self.assertIn("$12k", card)

    def test_generic_sponsorship_is_not_specific_vote(self):
        result = extractor.validate_extraction({"positions": [{"participant_label": "Q", "option_title": "Red", "stance": "conditional", "quote": "I'll back whichever you pick", "source_unit_ids": ["1"]}]},
            [unit("1", "Q", "I'll back whichever you pick")])
        self.assertEqual(result["positions"], [])

if __name__ == "__main__":
    unittest.main()
