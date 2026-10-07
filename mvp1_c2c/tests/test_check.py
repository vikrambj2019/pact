"""Pact function 2 — check: what is checked, the verdict, and that only quoted, cited evidence counts."""
import pytest

from fakes import FakePactLLM, act, claim
from mvp1_c2c.pact import PactSession
from mvp1_c2c.pact import check as check_module
from mvp1_c2c.pact.check import ClaimChecker, cited_sources, validate_report

MEMBERS = [{"id": "alex", "name": "Alex", "mapping_allowed": True, "share_allowed": True},
           {"id": "priya", "name": "Priya", "mapping_allowed": True, "share_allowed": True}]


def session_with_claims(intent=None):
    texts = {"Entry is $35 per car.": [act("add_claim", "Entry is $35 per car", text="Yosemite entry is $35 per car",
                                           kind="fact", checkable=True, check_type="price")],
             "The trail is 12 miles.": [claim("The trail is 12 miles")],
             "We'll all love it.": [claim("We'll all love it", checkable=False, kind="prediction")],
             "FAIL claim here.": [claim("FAIL claim here")]}
    session = PactSession.create("Where to hike?", "Sam", [dict(m) for m in MEMBERS], FakePactLLM(texts, intent))
    for speaker, text in [("alex", "Entry is $35 per car."), ("priya", "The trail is 12 miles."),
                          ("priya", "We'll all love it.")]:
        session.add_message(speaker, text, "human")
    return session


# ── what is checked ──────────────────────────────────────────────────────────

def test_claims_carry_a_check_type():
    session = session_with_claims()
    assert [c["check_type"] for c in session.card["claims"]] == ["price", "other", None]


def test_unknown_check_type_is_rejected():
    session = PactSession.create("Q?", "Sam", [dict(m) for m in MEMBERS], FakePactLLM(
        {"Flights $900": [act("add_claim", "Flights $900", text="Flights $900", kind="fact", checkable=True,
                              check_type="vibes")]}))
    session.add_message("alex", "Flights $900", "human")
    assert session.card["claims"] == []


# ── recording a check ────────────────────────────────────────────────────────

def test_check_records_verdict_on_claim_with_who_asked_and_when():
    session = session_with_claims()
    entry = session.check_claim("claim_1")
    request = session.card["messages"][-1]
    assert request["source"] == "explicit_request" and entry["request_message_id"] == request["id"]
    assert entry["verdict"] == "contradicted" and entry["checked_statement"] == "Yosemite entry is $35 per car"
    verification = session.card["claims"][0]["verification"]
    assert (verification["status"], verification["verdict"], verification["check_ids"]) == \
        ("checked", "contradicted", [entry["id"]])
    assert session.card["claims"][0]["history"][-1]["event"] == "checked"
    assert session.card["claims"][0]["history"][-1]["at"] == request["recorded_at"]
    assert session.card["evidence"][0]["claim_id"] == "claim_1"


def test_check_one_persons_claims_skips_the_uncheckable():
    session = session_with_claims()
    outcome = session.check_claims(made_by="priya")
    assert [c["claim_id"] for c in outcome["checks"]] == ["claim_2"]
    assert outcome["skipped_not_checkable"] == ["claim_3"]
    assert len([m for m in session.card["messages"] if m["source"] == "explicit_request"]) == 1


def test_check_all_runs_every_checkable_claim_and_keeps_card_order():
    session = session_with_claims()
    session.add_message("alex", "FAIL claim here.", "human")
    outcome = session.check_claims()
    assert [c["claim_id"] for c in outcome["checks"]] == ["claim_1", "claim_2", "claim_4"]
    assert "error" in outcome["checks"][2]  # one failure is logged and does not block the others
    assert [c["verification"]["status"] for c in session.card["claims"]] == \
        ["checked", "checked", "not_checked", "not_checked"]
    assert session.card["audit"][-2]["event"] == "claim_check_error"


def test_check_refuses_uncheckable_or_unknown_claims():
    session = session_with_claims()
    with pytest.raises(ValueError, match="not checkable"):
        session.check_claim("claim_3")
    with pytest.raises(ValueError, match="None of those claims can be checked"):
        session.check_claims(claim_ids=["claim_3"])
    with pytest.raises(ValueError, match="Not claims on this card"):
        session.check_claims(claim_ids=["claim_99"])
    assert session.llm.checker.checked == []


def test_corrected_claim_is_checked_as_corrected():
    session = session_with_claims()
    session.card["claims"][1].update(status="corrected", corrected_to="The trail is 14 miles")
    session.check_claim("claim_2")
    assert session.llm.checker.checked == ["The trail is 14 miles"]


# ── 'pact, ...' commands ─────────────────────────────────────────────────────

def test_command_records_admin_words_and_checks_one_persons_claims():
    session = session_with_claims({"action": "check_claims", "claim_ids": [], "made_by": "alex", "check_all": False,
                                   "statement": "", "reasoning": ""})
    assert session.handle_command("Pact, check Alex's claims") is True
    request = [m for m in session.card["messages"] if m["source"] == "pact_command"]
    assert [m["text"] for m in request] == ["Pact, check Alex's claims"]
    assert session.card["claim_checks"][0]["request_message_id"] == request[0]["id"]
    assert session.llm.checker.checked == ["Yosemite entry is $35 per car"]


def test_ad_hoc_check_becomes_an_admin_claim_sourced_to_the_command():
    session = session_with_claims({"action": "check_statement", "claim_ids": [], "made_by": "", "check_all": False,
                                   "statement": "Tioga Pass closes in November", "reasoning": ""})
    session.handle_command("pact, does Tioga Pass close in November?")
    new = session.card["claims"][-1]
    assert (new["statement"], new["made_by"], new["verification"]["verdict"]) == \
        ("Tioga Pass closes in November", "human_admin", "contradicted")
    assert new["source_message_ids"] == [session.card["messages"][-1]["id"]]


def test_unknown_command_is_left_for_normal_handling():
    session = session_with_claims({"action": "unknown", "claim_ids": [], "made_by": "", "check_all": False,
                                   "statement": "", "reasoning": ""})
    before = len(session.card["messages"])
    assert session.handle_command("pact, hello") is False
    assert len(session.card["messages"]) == before


# ── the checker: research → verdict ──────────────────────────────────────────

CITED_TEXT = "The entrance fee for a private vehicle is $35 and is valid for 7 days."


class FakeClient:
    """Stands in for the Anthropic API: a paused search turn, a final turn, then the verdict call."""
    def __init__(self, report):
        self.report, self.bodies = report, []
        self.turns = [
            {"stop_reason": "pause_turn", "content": [
                {"type": "web_search_tool_result", "content": [{"type": "web_search_result", "url": "https://nps.gov/yose"},
                                                               {"type": "web_search_result", "url": "https://blog.example/x"}]},
                {"type": "text", "text": "NPS says $35.", "citations": [
                    {"url": "https://nps.gov/yose", "title": "Fees", "cited_text": CITED_TEXT}]}]},
            {"stop_reason": "end_turn", "content": [{"type": "text", "text": "No newer change found."}]},
        ]

    def call(self, body, model, **kwargs):
        self.bodies.append(body)
        return self.turns.pop(0)

    def structured(self, instructions, payload, name, schema, model, **kwargs):
        self.payload = payload
        return self.report


def report(**overrides):
    base = {"checked_statement": "Yosemite private-vehicle entry costs $35 (2026)", "assumptions": [],
            "verdict": "supported", "summary": "NPS lists $35 per private vehicle, valid 7 days.",
            "as_of": "2026", "evidence": [{"url": "https://nps.gov/yose", "quote": "private vehicle is $35",
                                           "stance": "supports"}]}
    return {**base, **overrides}


def run_checker(rep, check_type="price"):
    client = FakeClient(rep)
    claim_ = {"id": "claim_1", "statement": "Yosemite entry is $35 per car", "made_by": "alex",
              "check_type": check_type, "source_message_ids": [], "challenges": []}
    card = PactSession.create("Where to hike?", "Sam", [dict(m) for m in MEMBERS], FakePactLLM()).card
    return ClaimChecker(client, "claude-test").check(card, claim_, "Where to hike?"), client


def test_checker_keeps_citations_across_paused_turns_and_only_cited_pages():
    result, client = run_checker(report())
    assert len(client.bodies) == 2  # resumed after pause_turn
    assert [s["url"] for s in client.payload["cited_sources"]] == ["https://nps.gov/yose"]  # blog was not cited
    assert result["verdict"] == "supported" and result["evidence"][0]["title"] == "Fees"
    assert result["search_results"] == 2


def test_evidence_must_quote_a_cited_source_or_the_verdict_falls_to_insufficient():
    made_up = report(evidence=[{"url": "https://nps.gov/yose", "quote": "entry is free", "stance": "supports"},
                               {"url": "https://other.example/", "quote": "$35", "stance": "supports"}])
    result, _ = run_checker(made_up)
    assert result["evidence"] == [] and result["evidence_dropped"] == 2
    assert result["verdict"] == "insufficient"


def test_no_citations_means_insufficient_without_a_verdict_call():
    client = FakeClient(report())
    client.turns = [{"stop_reason": "end_turn", "content": [{"type": "text", "text": "Couldn't find anything."}]}]
    claim_ = {"id": "c", "statement": "x", "made_by": "alex", "check_type": "other"}
    card = PactSession.create("Q?", "Sam", [dict(m) for m in MEMBERS], FakePactLLM()).card
    result = ClaimChecker(client, "m").check(card, claim_, "Q?")
    assert result["verdict"] == "insufficient" and not hasattr(client, "payload")


def test_search_is_limited_to_the_sites_for_that_claim_type(monkeypatch):
    monkeypatch.setitem(check_module.DOMAINS_BY_TYPE, "price", ["nps.gov"])
    _, client = run_checker(report())
    assert client.bodies[0]["tools"][0]["allowed_domains"] == ["nps.gov"]
    _, client = run_checker(report(), check_type="weather")
    assert "allowed_domains" not in client.bodies[0]["tools"][0]


def test_quote_matching_ignores_case_and_spacing():
    cited, _ = cited_sources([{"type": "text", "citations": [{"url": "https://a.gov", "cited_text": "Fee:  $35\\nper car"}]}])
    kept = validate_report(report(evidence=[{"url": "https://a.gov", "quote": "fee: $35 per car", "stance": "supports"}]),
                           {"https://a.gov": {**cited["https://a.gov"], "cited_text": ["Fee:  $35\nper car"]}})
    assert len(kept["evidence"]) == 1
