"""Pact on its own: session basics, check, interpret, and the admin's authority. Observe is in test_observe.py."""
import json

import pytest

from fakes import FakePactLLM, act, claim
from mvp1_c2c.pact import AnthropicClient, PactSession

MEMBERS = [{"id": "p_1", "name": "Alex", "mapping_allowed": True, "share_allowed": True},
           {"id": "p_2", "name": "Priya", "mapping_allowed": True, "share_allowed": True}]


def make_session(tmp_path, responses=None, intent=None, members=MEMBERS):
    return PactSession.create("Choose a project", "Admin", [dict(m) for m in members],
                              FakePactLLM(responses, intent), path=tmp_path / "session.json")


def say(session, speaker_id, text):
    return session.add_message(speaker_id, text, "human")


# ── session ──────────────────────────────────────────────────────────────────

def test_create_persist_and_reload(tmp_path):
    session = make_session(tmp_path)
    loaded = PactSession.load(tmp_path / "session.json", FakePactLLM())
    assert loaded.card["topic"]["title"] == "Choose a project"
    assert [p["id"] for p in loaded.card["participants"]] == ["human_admin", "p_1", "p_2"]
    assert loaded.card["decision_process"]["approver_ids"] == ["human_admin"]
    assert loaded.card["run_settings"]["pact_model"] == "fake-pact-model"
    assert session.session_id == loaded.session_id


def test_card_holds_no_simulation_bookkeeping(tmp_path):
    card = make_session(tmp_path).card
    assert not {"bot_turns", "max_bot_turns", "bot_order", "next_bot_index"} & set(card)


def test_json_file_does_not_contain_backend_credentials(tmp_path):
    make_session(tmp_path)
    text = (tmp_path / "session.json").read_text()
    assert "ANTHROPIC_API_KEY" not in text
    assert "api_key" not in text


# ── 2. check ─────────────────────────────────────────────────────────────────

def claims_session(tmp_path, intent=None):
    texts = {"Entry is $35 per car.": [claim("Entry is $35 per car")],
             "The trail is 12 miles.": [claim("The trail is 12 miles")],
             "We'll all love it.": [claim("We'll all love it", checkable=False, kind="prediction")]}
    session = make_session(tmp_path, texts, intent)
    for speaker, text in [("p_1", "Entry is $35 per car."), ("p_2", "The trail is 12 miles."),
                          ("p_2", "We'll all love it.")]:
        say(session, speaker, text)
    return session


def test_check_one_claim_records_request_evidence_and_verification(tmp_path):
    session = claims_session(tmp_path)
    entry = session.check_claim("claim_1")
    assert entry["status"] == "test_fixture"
    assert entry["request_message_id"] == session.card["messages"][-1]["id"]
    assert session.card["messages"][-1]["source"] == "explicit_request"
    assert session.card["claims"][0]["verification"] == {"status": "checked", "check_ids": [entry["id"]]}
    assert session.card["audit"][-1]["event"] == "claim_check"


def test_check_one_persons_claims_skips_the_uncheckable(tmp_path):
    session = claims_session(tmp_path)
    outcome = session.check_claims(made_by="p_2")
    assert [c["claim_id"] for c in outcome["checks"]] == ["claim_2"]
    assert outcome["skipped_not_checkable"] == ["claim_3"]
    assert session.llm.checker.checked == ["The trail is 12 miles"]


def test_check_all_claims(tmp_path):
    session = claims_session(tmp_path)
    outcome = session.check_claims()
    assert [c["claim_id"] for c in outcome["checks"]] == ["claim_1", "claim_2"]
    assert outcome["skipped_not_checkable"] == ["claim_3"]


def test_check_refuses_uncheckable_or_unknown_claims(tmp_path):
    session = claims_session(tmp_path)
    with pytest.raises(ValueError, match="not checkable"):
        session.check_claim("claim_3")
    with pytest.raises(ValueError, match="None of those claims can be checked"):
        session.check_claims(claim_ids=["claim_3"])
    with pytest.raises(ValueError, match="Not claims on this card"):
        session.check_claims(claim_ids=["claim_99"])
    assert session.llm.checker.checked == []


def test_pact_command_routes_person_check(tmp_path):
    intent = {"action": "check_claims", "claim_ids": [], "made_by": "p_1", "check_all": False,
              "statement": "", "reasoning": "Asked for Alex's claims."}
    session = claims_session(tmp_path, intent)
    assert session.handle_command("Pact, check Alex's claims") is True
    assert session.llm.checker.checked == ["Entry is $35 per car"]
    assert session.handle_command("not a command") is False


# ── 3. interpret ─────────────────────────────────────────────────────────────

def test_interpret_is_explicit_and_cites_real_messages(tmp_path):
    session = claims_session(tmp_path)
    item = session.interpret()
    assert item["output"]["summary"] == "Fixture state."
    assert item["request_message_id"] == session.card["messages"][-1]["id"]
    assert session.card["audit"][-1]["event"] == "help"


# ── admin authority ──────────────────────────────────────────────────────────

def test_decision_snapshot_preserves_decision_and_source(tmp_path):
    message = "I propose a small pilot."
    session = make_session(tmp_path, {message: [act("add_option", "a small pilot", text="Small pilot")]})
    session.add_admin_message(message)
    option = session.card["options"][0]
    session.record_decision(option["id"], "Test feasibility before expanding.")
    assert session.card["decision"]["status"] == "recorded"
    assert session.card["snapshots"][0]["decision"]["selected_option_id"] == option["id"]
    assert session.card["snapshots"][0]["options"][0]["text"] == "Small pilot"
    assert session.card["decision"]["recorded_snapshot_id"] == "snapshot_1"
    assert session.card["decision"]["rationale"]["source_message_ids"][-1] == session.card["messages"][-1]["id"]


def test_pending_interpretation_requires_recheck_after_card_changes(tmp_path):
    statement = "I may prefer the pilot."
    pending = act("set_preference", statement, stance="pilot", ambiguous=True)
    session = make_session(tmp_path, {statement: [pending]})
    session.add_admin_message(statement)
    pending = session.card["individual_mapping"]["pending_interpretations"][0]
    say(session, "p_1", "What about the cost?")
    with pytest.raises(ValueError, match="card changed"):
        session.confirm_interpretation(pending["id"], "Confirm this interpretation.")


def test_old_card_format_is_refused(tmp_path):
    make_session(tmp_path)
    path = tmp_path / "session.json"
    data = json.loads(path.read_text())
    data["card"]["schema_version"] = "mvp1_c2c.2"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="older card format"):
        PactSession.load(path, FakePactLLM())


# ── transport ────────────────────────────────────────────────────────────────

def test_live_transport_uses_anthropic_headers_and_keeps_key_out_of_payload(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b'{"type":"message","content":[],"stop_reason":"end_turn"}'

    def fake_urlopen(request, timeout, **kwargs):
        seen["request"] = request
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr("mvp1_c2c.pact.llm_client.urllib.request.urlopen", fake_urlopen)
    client = AnthropicClient(api_key="test-only-secret")
    client.call({"input": "test"}, model="claude-test-model")
    body = json.loads(seen["request"].data)
    assert "test-only-secret" not in seen["request"].data.decode()
    assert seen["request"].full_url == "https://api.anthropic.com/v1/messages"
    assert seen["request"].headers["X-api-key"] == "test-only-secret"
    assert seen["request"].headers["Anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-test-model"
    assert seen["timeout"] == 60
    assert client.calls == 1
