import json

import pytest

from mvp1_c2c.core import AnthropicBackend, Simulation


class FakeLLMBackend:
    """Fixed model responses used only to verify core contracts; never used by the app."""
    backend_name = "live"

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = 0

    def bot_message(self, profile, card, history):
        return profile["guidance"]

    def observe(self, card, message):
        return self.responses.get(message["text"], [])

    def check_claim(self, card, claim, question):
        self.calls += 1
        return {"finding": "Fixture response; no external research.", "sources": [], "status": "test_fixture"}

    def help(self, card):
        self.calls += 1
        return {"decision_state": "Fixture state.", "factual_unknowns": [],
                "preference_differences": [], "tradeoffs": [], "next_step": "Fixture next step.",
                "source_message_ids": [m["id"] for m in card["messages"]]}


def change(section, entity_id, quote, value, *, participant=None, needs_confirmation=False):
    return {"section": section, "operation": "add", "entity_id": entity_id,
            "participant_id": participant, "quote": quote,
            "needs_confirmation": needs_confirmation, "reason": "Fixed test fixture.",
            "value_json": json.dumps(value)}


def setup_sim(tmp_path, bots=2, turns=4):
    profiles = [{"name": f"Bot {i}", "role": f"role {i}", "guidance": f"I prefer option {i}.",
                 "mapping_allowed": True, "share_allowed": True} for i in range(1, bots + 1)]
    return Simulation.create("Choose a project", "Admin", profiles, turns, FakeLLMBackend(),
                             path=tmp_path / "session.json", mapping_allowed=True)


def test_create_persist_and_reload(tmp_path):
    sim = setup_sim(tmp_path)
    loaded = Simulation.load(tmp_path / "session.json", FakeLLMBackend())
    assert loaded.card["topic"]["title"] == "Choose a project"
    assert len(loaded.profiles) == 2
    assert loaded.card["decision_process"]["approver_ids"] == ["human_admin"]
    assert loaded.card["run_settings"]["backend"] == "live"


def test_load_rejects_backend_mismatch(tmp_path):
    setup_sim(tmp_path)

    class OtherBackend(FakeLLMBackend):
        backend_name = "other"

    with pytest.raises(ValueError, match="Session backend is live"):
        Simulation.load(tmp_path / "session.json", OtherBackend())


def test_three_pact_actions_and_user_is_admin_participant(tmp_path):
    message = "I propose the small pilot. It could save $2 million annually if teams adopt it."
    responses = {message: [
        change("proposals", "proposal_1", "I propose the small pilot",
               {"title": "Small pilot", "proposed_by": "human_admin", "status": "proposed"}),
        change("claims", "claim_1", "It could save $2 million annually",
               {"statement": "It could save $2 million annually", "made_by": "human_admin",
                "verification": {"status": "not_checked", "check_ids": []}}),
    ]}
    profiles = [{"name": "Bot", "role": "planner", "guidance": "Ask about cost."}]
    sim = Simulation.create("Choose a project", "Admin", profiles, 4, FakeLLMBackend(responses),
                            path=tmp_path / "session.json", mapping_allowed=True)
    sim.add_human_message(message)
    assert sim.card["proposals"] and sim.card["claims"]
    claim_check = sim.check_claim(sim.card["claims"][0]["id"])
    assert claim_check["status"] == "test_fixture"
    help_result = sim.ask_for_help()
    assert "Fixture" in help_result["output"]["decision_state"]
    assert [x["event"] for x in sim.card["audit"] if x["event"] in {"claim_check", "help"}] == ["claim_check", "help"]
    assert [m["source"] for m in sim.card["messages"][-2:]] == ["explicit_request", "explicit_request"]
    assert claim_check["request_message_id"] == sim.card["messages"][-2]["id"]


def test_round_robin_and_turn_limit(tmp_path):
    sim = setup_sim(tmp_path, bots=2, turns=3)
    first_round = sim.next_round()
    assert [m["speaker_id"] for m in first_round] == ["sim_1", "sim_2"]
    assert sim.card["bot_turns"] == 2
    final_turn = sim.bot_turn()
    assert final_turn["speaker_id"] == "sim_1"
    with pytest.raises(ValueError, match="limit"):
        sim.bot_turn()


def test_model_changes_require_exact_quote_and_correct_claim_attribution(tmp_path):
    sim = setup_sim(tmp_path, bots=1)
    msg = sim.add_human_message("I prefer the pilot.")
    before = len(sim.card["change_log"])
    malformed = {"section": "claims", "operation": "add", "entity_id": "claim_bad",
                 "participant_id": None, "quote": "fabricated quote", "needs_confirmation": False,
                 "reason": "bad", "value_json": json.dumps({"statement": "claim", "made_by": "sim_1"})}
    with pytest.raises(ValueError):
        sim.apply_change(malformed, msg)
    assert len(sim.card["change_log"]) == before


def test_personal_mapping_requires_opt_in_and_ambiguous_bot_items_are_skipped(tmp_path):
    sim = setup_sim(tmp_path, bots=1)
    bot = sim.card["participants"][1]
    sim.card["individual_mapping"]["participant_permissions"][1]["mapping"]["status"] = "not_requested"
    msg = {"id": "msg_1", "speaker_id": bot["id"], "text": "I prefer this option."}
    change = {"section": "positions", "operation": "add", "entity_id": "position_1",
              "participant_id": bot["id"], "quote": "I prefer this option", "needs_confirmation": True,
              "reason": "ambiguous", "value_json": json.dumps({"statement": "prefers option"})}
    sim.apply_change(change, msg)
    assert sim.card["individual_mapping"]["pending_interpretations"] == []
    sim.card["individual_mapping"]["participant_permissions"][1]["mapping"]["status"] = "granted"
    sim.apply_change(change, msg)
    assert not sim.card["positions"]
    assert sim.card["individual_mapping"]["pending_interpretations"] == []
    assert sim.card["audit"][-1]["event"] == "interpretation_skipped"


def test_decision_snapshot_preserves_decision_and_source(tmp_path):
    message = "I propose a small pilot."
    profiles = [{"name": "Bot", "role": "planner", "guidance": "Ask about cost."}]
    backend = FakeLLMBackend({message: [change("proposals", "proposal_1", message,
                                                   {"title": "Small pilot", "status": "proposed",
                                                    "proposed_by": "human_admin"})]})
    sim = Simulation.create("Choose a project", "Admin", profiles, 2, backend,
                            path=tmp_path / "session.json", mapping_allowed=True)
    sim.add_human_message(message)
    proposal = sim.card["proposals"][0]
    sim.record_decision(proposal["id"], "Test feasibility before expanding.")
    assert sim.card["decision"]["status"] == "recorded"
    assert sim.card["snapshots"][0]["decision"]["selected_proposal_id"] == proposal["id"]
    assert sim.card["decision"]["recorded_snapshot_id"] == "snapshot_1"
    assert sim.card["decision"]["rationale"]["source_message_ids"][-1] == sim.card["messages"][-1]["id"]


def test_human_can_confirm_own_pending_interpretation(tmp_path):
    statement = "I might prefer the pilot."
    profiles = [{"name": "Bot", "role": "participant", "guidance": "Wait."}]
    pending_change = change("positions", "position_pending", "I might prefer the pilot",
                            {"stance": "prefers", "statement": "might prefer pilot"},
                            participant="human_admin", needs_confirmation=True)
    sim = Simulation.create("Choose a project", "Admin", profiles, 2,
                            FakeLLMBackend({statement: [pending_change]}),
                            path=tmp_path / "session.json", mapping_allowed=True)
    sim.add_human_message("I might prefer the pilot.")
    pending = sim.card["individual_mapping"]["pending_interpretations"][0]
    assert not sim.card["positions"]
    sim.confirm_interpretation(pending["id"], "Confirm this interpretation.")
    assert sim.card["positions"][0]["statement"] == "might prefer pilot"
    stored = sim.card["individual_mapping"]["pending_interpretations"][0]
    assert stored["included_in_confirmed_state"] is True


def test_pending_interpretation_requires_recheck_after_card_changes(tmp_path):
    statement = "I may prefer the pilot."
    profiles = [{"name": "Bot", "role": "participant", "guidance": "Wait."}]
    pending_change = change("positions", "position_pending", statement,
                            {"statement": statement}, participant="human_admin", needs_confirmation=True)
    sim = Simulation.create("Choose a project", "Admin", profiles, 2,
                            FakeLLMBackend({statement: [pending_change]}),
                            path=tmp_path / "session.json", mapping_allowed=True)
    sim.add_human_message("I may prefer the pilot.")
    pending = sim.card["individual_mapping"]["pending_interpretations"][0]
    sim.add_message("sim_1", "What about the cost?", "simulated", observe=False)
    with pytest.raises(ValueError, match="card changed"):
        sim.confirm_interpretation(pending["id"], "Confirm this interpretation.")


def test_json_file_does_not_contain_backend_credentials(tmp_path):
    sim = setup_sim(tmp_path)
    text = (tmp_path / "session.json").read_text()
    assert "ANTHROPIC_API_KEY" not in text
    assert "api_key" not in text


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
        seen["context"] = kwargs.get("context")
        return Response()

    monkeypatch.setattr("mvp1_c2c.core.urllib.request.urlopen", fake_urlopen)
    backend = AnthropicBackend(api_key="test-only-secret")
    backend.call({"input": "test"})
    body = json.loads(seen["request"].data)
    assert "test-only-secret" not in seen["request"].data.decode()
    assert seen["request"].full_url == "https://api.anthropic.com/v1/messages"
    assert seen["request"].headers["X-api-key"] == "test-only-secret"
    assert seen["request"].headers["Anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-haiku-4-5-20251001"
    assert seen["timeout"] == 60
