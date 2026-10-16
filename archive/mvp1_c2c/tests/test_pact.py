"""Pact on its own: session basics, the admin's authority, transport. Each function has its own test file."""
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
