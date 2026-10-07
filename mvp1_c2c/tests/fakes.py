"""Fixed LLM-shaped responses used only by the test suite; the app never uses these."""
import json


class FakeObserver:
    """Returns fixed actions keyed by message text, stamping each with that message's id."""
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def propose(self, view, new_messages, context):
        self.calls.append({"view": view, "new": new_messages, "context": context})
        out = []
        for message in new_messages:
            out += [{**a, "message_id": a.get("message_id", message["id"])}
                    for a in self.responses.get(message["text"], [])]
        return out


class FakeChecker:
    def __init__(self):
        self.checked = []

    def check(self, card, claim, question):
        self.checked.append(claim["statement"])
        if claim["statement"].startswith("FAIL"):
            raise RuntimeError("search down")
        return {"checked_statement": claim["statement"], "assumptions": [], "verdict": "contradicted",
                "summary": "Fixture: no external research.", "as_of": "2026",
                "evidence": [{"url": "https://example.gov/a", "title": "A", "quote": "q", "stance": "contradicts"}],
                "evidence_dropped": 0, "search_results": 1, "domains": []}


class FakeInterpreter:
    def __init__(self):
        self.result = {"headline": "Nothing leads yet.", "open_items": [], "next_step": "Alex proposes dates.",
                       "cited_ids": []}

    def interpret(self, card):
        return dict(self.result)


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


def act(action, quote, *, ambiguous=False, **fields):
    """One observer action as the model would return it."""
    return {"action": action, "quote": quote, "ambiguous": ambiguous, "reason": "Fixed test fixture.", **fields}


def claim(quote, checkable=True, kind="fact", ref=None):
    fields = {"text": quote, "kind": kind, "checkable": checkable}
    if ref:
        fields["ref"] = ref
    return act("add_claim", quote, **fields)
