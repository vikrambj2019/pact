"""Fixed LLM-shaped responses used only by the test suite; the app never uses these."""
import json


class FakeObserver:
    def __init__(self, responses=None):
        self.responses = responses or {}

    def observe(self, card, message):
        return self.responses.get(message["text"], [])


class FakeChecker:
    def __init__(self):
        self.checked = []

    def check(self, card, claim, question):
        self.checked.append(claim["statement"])
        return {"finding": "Fixture response; no external research.", "sources": [], "status": "test_fixture"}


class FakeInterpreter:
    def interpret(self, card):
        return {"summary": "Fixture state.", "source_message_ids": [m["id"] for m in card["messages"]]}


class FakeCommands:
    def __init__(self, intent=None):
        self.intent = intent or {"action": "unknown"}

    def parse(self, text, card):
        return self.intent


class FakePactLLM:
    model = "fake-pact-model"

    def __init__(self, responses=None, intent=None):
        self.observer = FakeObserver(responses)
        self.checker = FakeChecker()
        self.interpreter = FakeInterpreter()
        self.commands = FakeCommands(intent)


class FakeParticipants:
    model = "fake-participant-model"

    def bot_message(self, profile, card, history):
        return profile["guidance"]


def change(section, entity_id, quote, value, *, participant=None, needs_confirmation=False):
    return {"section": section, "operation": "add", "entity_id": entity_id,
            "participant_id": participant, "quote": quote,
            "needs_confirmation": needs_confirmation, "reason": "Fixed test fixture.",
            "value_json": json.dumps(value)}


def claim(entity_id, quote, made_by, checkable=True):
    return change("claims", entity_id, quote,
                  {"statement": quote, "made_by": made_by, "checkable": checkable,
                   "verification": {"status": "not_checked", "check_ids": []}})
