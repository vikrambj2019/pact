"""The simulation harness: it drives a Pact session but keeps its own state off the card."""
import json

import pytest

from fakes import FakeParticipants, FakePactLLM
from mvp1_c2c.simulation import Simulation, saved_models


def setup_sim(tmp_path, bots=2, turns=4):
    profiles = [{"name": f"Bot {i}", "role": f"role {i}", "guidance": f"I prefer option {i}.",
                 "mapping_allowed": True, "share_allowed": True} for i in range(1, bots + 1)]
    return Simulation.create("Choose a project", "Admin", profiles, turns, FakePactLLM(), FakeParticipants(),
                             path=tmp_path / "session.json")


def test_round_robin_and_turn_limit(tmp_path):
    sim = setup_sim(tmp_path, bots=2, turns=3)
    first_round = sim.next_round()
    assert [m["speaker_id"] for m in first_round] == ["sim_1", "sim_2"]
    assert sim.state["bot_turns"] == 2
    assert sim.bot_turn()["speaker_id"] == "sim_1"
    with pytest.raises(ValueError, match="limit"):
        sim.bot_turn()


def test_simulation_state_is_saved_beside_the_card_not_in_it(tmp_path):
    sim = setup_sim(tmp_path)
    sim.bot_turn()
    data = json.loads((tmp_path / "session.json").read_text())
    assert set(data) == {"card", "simulation"}
    assert data["simulation"]["bot_turns"] == 1
    assert data["simulation"]["profiles"][0]["guidance"] == "I prefer option 1."
    assert "guidance" not in json.dumps(data["card"])
    assert [p["participant_type"] for p in data["card"]["participants"]] == ["human", "simulated", "simulated"]
    loaded = Simulation.load(tmp_path / "session.json", FakePactLLM(), FakeParticipants())
    assert loaded.state["next_bot_index"] == 1
    assert saved_models(tmp_path / "session.json") == (None, None)  # fake model names are not claude-*


def test_loads_session_saved_before_the_split(tmp_path):
    sim = setup_sim(tmp_path)
    card = sim.pact.card
    card.update({"bot_turns": 3, "max_bot_turns": 9, "bot_order": ["sim_1", "sim_2"], "next_bot_index": 3,
                 "run_settings": {"backend": "live", "participant_model": "claude-haiku-x",
                                  "research_model": "claude-sonnet-x"},
                 "claims": [{"id": "claim_1", "statement": "Old claim", "made_by": "sim_1"}]})
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"card": card, "profiles": sim.profiles}))
    assert saved_models(path) == ("claude-sonnet-x", "claude-haiku-x")
    loaded = Simulation.load(path, FakePactLLM(), FakeParticipants())
    assert loaded.state["bot_turns"] == 3 and loaded.state["max_bot_turns"] == 9
    assert "bot_turns" not in loaded.pact.card
    assert loaded.pact.card["claims"][0]["checkable"] is True
    assert loaded.bot_turn()["speaker_id"] == "sim_2"


def test_setup_validation(tmp_path):
    with pytest.raises(ValueError, match="one and five"):
        setup_sim(tmp_path, bots=0)
    with pytest.raises(ValueError, match="limit"):
        setup_sim(tmp_path, turns=0)
