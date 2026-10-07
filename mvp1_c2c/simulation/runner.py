"""Drives a Pact session with simulated participants, to see whether Pact works.

The harness owns who speaks and when (profiles, round-robin order, turn limit). It talks to Pact
only through PactSession, exactly as a real chat integration would. Its state is saved next to the
card under "simulation" and never enters the card itself.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..pact import PactSession


class Simulation:
    def __init__(self, pact: PactSession, participants):
        self.pact, self.participants = pact, participants
        self.state = pact.extras["simulation"]

    @classmethod
    def create(cls, question, human_name, bot_profiles, max_bot_turns, pact_llm, participant_llm,
               path=None, mapping_allowed=True):
        if not 1 <= len(bot_profiles) <= 5:
            raise ValueError("Add between one and five simulated participants.")
        if not 1 <= max_bot_turns <= 60:
            raise ValueError("Bot turn limit must be between 1 and 60.")
        members, profiles = [], []
        for i, profile in enumerate(bot_profiles, 1):
            name = profile["name"].strip()
            guidance = profile["guidance"].strip()
            if not name or not guidance:
                raise ValueError("Each simulated participant needs a name and high-level guidance.")
            person = {"id": f"sim_{i}", "name": name, "role": profile.get("role", "participant"),
                      "mapping_allowed": bool(profile.get("mapping_allowed", False)),
                      "share_allowed": bool(profile.get("share_allowed", False))}
            members.append({**person, "participant_type": "simulated"})
            profiles.append({**person, "guidance": guidance})
        state = {"profiles": profiles, "bot_order": [p["id"] for p in profiles], "next_bot_index": 0,
                 "bot_turns": 0, "max_bot_turns": max_bot_turns,
                 "participant_model": getattr(participant_llm, "model", None)}
        pact = PactSession.create(question, human_name, members, pact_llm, path,
                                  admin_mapping_allowed=mapping_allowed, extras={"simulation": state})
        return cls(pact, participant_llm)

    @classmethod
    def load(cls, path: Path, pact_llm, participant_llm):
        return cls(PactSession.load(path, pact_llm), participant_llm)

    @property
    def profiles(self) -> list[dict]:
        return self.state["profiles"]

    def bot_turn(self, participant_id: str | None = None, observe: bool = True):
        if self.state["bot_turns"] >= self.state["max_bot_turns"]:
            raise ValueError("Bot turn limit reached.")
        bots = self.state["bot_order"]
        if not bots:
            raise ValueError("No simulated participants configured.")
        expected = bots[self.state["next_bot_index"] % len(bots)]
        if participant_id and participant_id != expected:
            raise ValueError("Participants speak in the configured round-robin order.")
        profile = next(p for p in self.profiles if p["id"] == expected)
        text = self.participants.bot_message(profile, self.pact.card, self.pact.card["messages"])
        msg = self.pact.add_message(expected, text, "simulated", observe=observe)
        self.state["bot_turns"] += 1
        self.state["next_bot_index"] += 1
        self.pact.save()
        return msg

    def next_round(self, observe: bool = True):
        remaining = min(len(self.profiles), self.state["max_bot_turns"] - self.state["bot_turns"])
        return [self.bot_turn(observe=observe) for _ in range(remaining)]


def saved_models(path: Path) -> tuple[str | None, str | None]:
    """(pact_model, participant_model) recorded in a saved session."""
    data = json.loads(path.read_text())
    pact_model = data["card"].get("run_settings", {}).get("pact_model")
    participant_model = data.get("simulation", {}).get("participant_model")
    keep = lambda m: m if m and m.startswith("claude-") else None
    return keep(pact_model), keep(participant_model)
