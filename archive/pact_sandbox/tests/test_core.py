import json
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from core import LiveBackend, MockBackend, Patch, Proposal, Simulation, Turn, evidence_from_response, load_scenario


def turn(**updates):
    return Turn(**(dict(message="A message", action="say", request="", confirm_ids=[], withdraw_ids=[], consent="unchanged") | updates))


def sim():
    return Simulation(load_scenario(), MockBackend())


def agreement(s):
    msg = s.append("Sarah", "I propose November 14–21.")
    p = Proposal(kind="agreement", text="November 14–21", person="", quote=msg["text"], related_ids=[], relation="none")
    s.apply(Patch(proposals=[p]), msg)
    return s.items[-1]


def test_all_six_confirmations_required_and_withdrawal_reopens():
    s = sim()
    item = agreement(s)
    for actor in s.names[:-1]:
        s.decisions(actor, turn(confirm_ids=[item["id"]]))
    assert item["status"] == "Proposed"
    s.decisions(s.names[-1], turn(confirm_ids=[item["id"]]))
    assert item["status"] == "Agreed"
    s.decisions("Maya", turn(withdraw_ids=[item["id"]]))
    assert item["status"] == "Proposed" and "Maya" not in item["confirmations"]


def test_conditional_support_and_understanding_do_not_confirm():
    s = sim()
    item = agreement(s)
    s.accept("Leo", turn(message="I support it only if Maya attends."))
    s.accept("Maya", turn(message="I understand Leo's concern."))
    assert item["confirmations"] == [] and item["status"] == "Proposed"


@pytest.mark.parametrize("updates", [{"quote": "Invented quote"}, {"kind": "preference", "person": "Alex"},
    {"related_ids": ["I999"], "relation": "supports"}, {"related_ids": [], "relation": "blocks"}])
def test_invalid_patches_rejected(updates):
    s = sim()
    msg = s.append("Sarah", "My budget is $2,000.")
    fields = dict(kind="constraint", text="Budget $2,000", person="Sarah", quote=msg["text"], related_ids=[], relation="none")
    s.apply(Patch(proposals=[Proposal(**(fields | updates))]), msg)
    assert not s.items and s.audit[-1]["status"] == "rejected"


def test_withdrawal_and_unknown_ids_reject():
    s = sim()
    item = agreement(s)
    s.decisions("Alex", turn(withdraw_ids=[item["id"]], confirm_ids=["I999"]))
    assert item["active"] and not item["confirmations"]
    assert all(x["status"] == "rejected" for x in s.audit[-2:])
    s.decisions("Sarah", turn(confirm_ids=[item["id"]], withdraw_ids=[item["id"]]))
    assert item["active"] and not item["confirmations"]


def test_consent_gating_and_revocation():
    s = sim()
    s.accept("Sarah", turn(action="resolve", request="Help us resolve the trip."))
    assert s.audit[-1]["status"] == "blocked"
    s.accept("Ben", turn(action="check", request="Are huts open?"))
    assert len(s.evidence) == 1 and s.evidence[0]["status"] == "Unverified"
    for name in s.names:
        s.accept(name, turn(consent="grant"))
    s.accept("Priya", turn(action="resolve", request="What blocks the trip?"))
    assert s.interpretation
    s.accept("Maya", turn(consent="deny"))
    assert s.interpretation is None
    s.accept("Priya", turn(action="resolve", request="What blocks the trip?"))
    assert s.audit[-1]["status"] == "blocked"


def test_private_profiles_not_in_public_or_exports():
    s = sim()
    text = json.dumps(s.public()) + json.dumps(s.export())
    assert "private_information" not in text and "negotiating tactic" not in text


def test_research_explicit_action_budget_and_source():
    s = sim()
    s.max_research = 1
    s.accept("Ben", turn(message="I heard huts are closed."))
    assert s.research_count == 0
    s.accept("Ben", turn(action="check", request="Are huts open?", consent="grant"))
    request_id = next(m["id"] for m in s.messages if m["action"] == "check")
    assert s.evidence[0]["request_message_id"] == request_id
    s.accept("Ben", turn(action="check", request="Are other huts open?"))
    assert s.research_count == 1 and len(s.evidence) == 1


@pytest.mark.parametrize("searched", [False, True])
def test_research_provenance(searched):
    class Response:
        output_text = "Sourced finding"
        def model_dump(self):
            return {"output": ([{"type": "web_search_call", "status": "completed"}] if searched else []) +
                [{"type": "message", "content": [{"text": "Sourced finding", "annotations": [
                    {"type": "url_citation", "url": "https://example.org", "title": "Source", "start_index": 0, "end_index": 15}]}]}]}
    ev = evidence_from_response(Response())
    assert ev["status"] == ("Sourced research" if searched else "Unverified")
    assert bool(ev["sources"]) == searched


def test_extraction_failure_retains_state_and_message():
    s = sim()
    item = agreement(s)
    def fail(*args):
        raise RuntimeError("refusal")
    s.backend.extract = fail
    s.accept("Maya", turn(message="I am unsure."))
    assert s.messages[-1]["text"] == "I am unsure." and s.items == [item]
    assert s.audit[-1]["status"] == "error"


def test_invalid_interpreter_sources_reject():
    s = sim()
    s.consent = dict.fromkeys(s.names, True)
    original = s.backend.interpret
    s.backend.interpret = lambda *args: original(*args).model_copy(update={"source_message_ids": ["M999"]})
    s.accept("Priya", turn(action="resolve", request="Help."))
    assert s.interpretation is None and s.audit[-1]["status"] == "error"


def test_mock_end_to_end():
    s = Simulation(load_scenario(), MockBackend(), max_turns=18)
    while s.step():
        pass
    assert s.turns == 18 and not s.step()
    assert s.evidence and all(e["simulated"] and not e["sources"] for e in s.evidence)
    assert any(i["status"] == "Agreed" for i in s.items)
    assert all(len(i["confirmations"]) == 6 for i in s.items if i["status"] == "Agreed")


def test_real_sdk_request_contract_without_network():
    import httpx
    from openai import OpenAI
    seen = []
    def transport(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "resp_test", "object": "response", "created_at": 0,
            "status": "completed", "model": "gpt-4.1-mini", "output": [{"id": "msg_test", "type": "message",
                "role": "assistant", "status": "completed", "content": [{"type": "output_text",
                "text": turn(message="My budget is firm.").model_dump_json(), "annotations": []}]}],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}})
    backend = LiveBackend.__new__(LiveBackend)
    backend.client = OpenAI(api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(transport)))
    backend.model = backend.research_model = "gpt-4.1-mini"
    backend.calls = backend.tokens = 0
    backend.max_calls = 1
    assert isinstance(backend.participant(load_scenario()["participants"][0], sim().public()), Turn)
    assert seen[0]["text"]["format"]["strict"] is True and seen[0]["store"] is False
    assert "tools" not in seen[0] and backend.tokens == 30
    with pytest.raises(RuntimeError, match="budget"):
        backend.participant({}, {})


def test_streamlit_mock_flow():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    next(b for b in app.button if b.label == "Next round (6)").click().run()
    assert not app.exception
    assert app.session_state.sim.turns == 6
    next(b for b in app.button if b.label == "Next round (6)").click().run()
    assert not app.exception
    assert app.session_state.sim.turns == 12 and app.session_state.sim.evidence
