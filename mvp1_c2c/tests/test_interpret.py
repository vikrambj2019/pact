"""Observe → check → interpret: what observe records is what check and interpret need, and interpret stays short."""
import pytest

from fakes import FakePactLLM, act, claim
from mvp1_c2c.pact import PactSession
from mvp1_c2c.pact.check import check_context
from mvp1_c2c.pact.interpret import interpret_view

MEMBERS = [{"id": "alex", "name": "Alex", "mapping_allowed": True, "share_allowed": True},
           {"id": "priya", "name": "Priya", "mapping_allowed": True, "share_allowed": True},
           {"id": "leo", "name": "Leo", "mapping_allowed": True, "share_allowed": False}]

CHAT = [
    ("human_admin", "Dates Nov 14 to 21, budget $2,000 each.",
     [act("add_constraint", "Dates Nov 14 to 21", text="Dates Nov 14-21", kind="dates"),
      act("add_constraint", "budget $2,000 each", text="Budget $2,000 per person", kind="budget")]),
    ("alex", "Patagonia! Flights alone are $1,300.",
     [act("add_option", "Patagonia", text="Patagonia", ref="new_pat"),
      act("set_preference", "Patagonia!", stance="Patagonia", option_id="new_pat", leaning="for"),
      act("add_claim", "Flights alone are $1,300", kind="fact", checkable=True, option_id="new_pat",
          text="Round-trip flights to Patagonia for Nov 14-21 cost about $1,300")]),
    ("priya", "No way, flights are more like $1,800. Dolomites instead.",
     [act("challenge_claim", "flights are more like $1,800", target_id="claim_1", text="Flights cost about $1,800"),
      act("add_option", "Dolomites", text="Dolomites", ref="new_dol"),
      act("set_preference", "Dolomites instead", stance="Dolomites", option_id="new_dol", leaning="for")]),
    ("human_admin", "Patagonia works for me", [act("record_affirmation", "works for me", target_id="opt_1")]),
    ("leo", "Patagonia is fine I guess", [act("set_preference", "Patagonia is fine", stance="Patagonia", option_id="opt_1", leaning="for")]),
    ("priya", "Who checks the hut schedule?", [act("add_issue", "Who checks the hut schedule?",
                                                     text="Check Dolomites hut schedule", participant_ids=["priya"])]),
]


def chat_session(tmp_path):
    session = PactSession.create("Where should we hike in November?", "Sam", [dict(m) for m in MEMBERS],
                                 FakePactLLM({text: actions for _, text, actions in CHAT}),
                                 path=tmp_path / "s.json")
    for speaker, text, _ in CHAT:
        session.add_message(speaker, text, "human")
    return session


def test_claim_carries_what_check_needs(tmp_path):
    session = chat_session(tmp_path)
    claim_ = session.card["claims"][0]
    assert claim_["option_id"] == "opt_1"
    ctx = check_context(session.card, claim_, session.card["topic"]["title"])
    assert ctx["claim"] == "Round-trip flights to Patagonia for Nov 14-21 cost about $1,300"
    assert ctx["as_said"][0]["text"] == "Patagonia! Flights alone are $1,300." and ctx["as_said"][0]["at"]
    assert ctx["about_option"] == "Patagonia"
    assert {c["kind"] for c in ctx["current_constraints"]} == {"dates", "budget"}
    assert ctx["disputed_by"] == [{"speaker": "Priya", "text": "Flights cost about $1,800"}]
    assert "recent_conversation" not in ctx  # no unrelated chatter or Pact's own requests


def test_check_uses_the_corrected_statement(tmp_path):
    session = chat_session(tmp_path)
    session.card["claims"][0].update(status="corrected", corrected_to="Flights cost about $1,500")
    session.check_claim("claim_1")
    assert session.llm.checker.checked == ["Flights cost about $1,500"]


def test_interpret_view_counts_support_from_the_card(tmp_path):
    view = interpret_view(chat_session(tmp_path).card)
    pat, dol = view["options"]
    assert pat["said_yes"] == ["Sam"] and pat["agreement"]["status"] == "partial"
    assert [s["who"] for s in pat["stances"]] == ["Alex"]  # Leo's stance is not shared
    assert [s["who"] for s in dol["stances"]] == ["Priya"]
    assert view["claims"][0]["status"] == "disputed" and view["claims"][0]["disputed_by"] == ["Priya"]
    assert view["open_issues"] == [{"id": "issue_1", "text": "Check Dolomites hut schedule", "involves": ["Priya"]}]
    assert view["no_shared_stance"] == ["Leo"]  # Leo has a stance, but it is private
    assert "messages" not in view and "history" not in str(view)


def test_interpret_view_includes_latest_check_and_drops_retracted(tmp_path):
    session = chat_session(tmp_path)
    session.check_claim("claim_1")
    view = interpret_view(session.card)
    assert view["claims"][0]["check"] == "contradicted"
    assert view["options"][0]["claims"] == [{"id": "claim_1", "check": "contradicted", "status": "disputed"}]
    session.card["claims"][0]["status"] = "retracted"
    assert interpret_view(session.card)["claims"] == []


def test_no_stance_lists_people_who_have_not_weighed_in(tmp_path):
    session = PactSession.create("Q?", "Sam", [dict(m) for m in MEMBERS], FakePactLLM())
    assert interpret_view(session.card)["no_shared_stance"] == ["Sam", "Alex", "Priya", "Leo"]


def test_interpretation_is_short_composed_and_cites_real_items(tmp_path):
    session = chat_session(tmp_path)
    session.llm.interpreter.result = {
        "headline": "Patagonia leads on stated support; nothing is decided.",
        "open_items": ["Flight cost disputed: Alex $1,300 vs Priya $1,800.", "Priya prefers Dolomites.",
                       "Hut schedule unchecked (Priya).", "a fourth item"],
        "next_step": "Check claim_1 before Alex and Priya go further.",
        "cited_ids": ["opt_1", "claim_1", "issue_1"]}
    item = session.interpret()
    assert len(item["output"]["open_items"]) == 3
    assert item["output"]["summary"].splitlines()[0] == "Patagonia leads on stated support; nothing is decided."
    assert item["output"]["summary"].splitlines()[-1].startswith("Next: ")
    assert item["request_message_id"] == session.card["messages"][-1]["id"]


def test_interpretation_citing_unknown_items_is_rejected(tmp_path):
    session = chat_session(tmp_path)
    session.llm.interpreter.result = {"headline": "Banff leads.", "open_items": [], "next_step": "Book Banff.",
                                      "cited_ids": ["opt_9"]}
    with pytest.raises(ValueError, match="not on the card"):
        session.interpret()
    assert session.card["facilitation"] == []
    assert session.card["audit"][-1]["event"] == "help_rejected"


# ── standing is computed, not judged by the model ────────────────────────────

def test_standing_counts_backers_minus_opposers(tmp_path):
    view = interpret_view(chat_session(tmp_path).card)
    pat, dol = view["options"]
    assert pat["backers"] == ["Alex", "Sam"] and dol["backers"] == ["Priya"]  # Leo's private stance not counted
    assert view["standing"] == {"leading": "opt_1", "tied": []}


def stance_session(stances):
    """stances: (speaker, option_id, leaning) on two options, Banff and Dolomites."""
    texts = {"Banff or Dolomites?": [act("add_option", "Banff", text="Banff"),
                                     act("add_option", "Dolomites", text="Dolomites")]}
    session = PactSession.create("Where?", "Sam", [dict(m) for m in MEMBERS], FakePactLLM(texts))
    session.add_message("human_admin", "Banff or Dolomites?", "human")
    for i, (speaker, option_id, leaning) in enumerate(stances):
        text = f"stance {i}"
        session.llm.observer.responses[text] = [act("set_preference", text, stance=text, option_id=option_id,
                                                    leaning=leaning)]
        session.add_message(speaker, text, "human")
    return session


def test_opposition_counts_against_and_ties_are_reported():
    view = interpret_view(stance_session([("alex", "opt_1", "for"), ("priya", "opt_2", "for")]).card)
    assert view["standing"] == {"leading": None, "tied": ["opt_1", "opt_2"]}
    view = interpret_view(stance_session([("alex", "opt_1", "for"), ("priya", "opt_1", "against"),
                                          ("human_admin", "opt_2", "for")]).card)
    assert view["options"][0]["opposers"] == ["Priya"]
    assert view["standing"]["leading"] == "opt_2"


def test_nothing_leads_without_support_and_dropped_options_never_lead():
    assert interpret_view(stance_session([]).card)["standing"] == {"leading": None, "tied": []}
    session = stance_session([("alex", "opt_1", "for")])
    session.card["options"][0]["status"] = "dropped"
    assert interpret_view(session.card)["standing"] == {"leading": None, "tied": []}


def test_stance_on_an_option_must_lean_for_or_against():
    texts = {"Banff?": [act("add_option", "Banff", text="Banff")],
             "Hmm Banff": [act("set_preference", "Hmm Banff", stance="unsure", option_id="opt_1")]}
    session = PactSession.create("Where?", "Sam", [dict(m) for m in MEMBERS], FakePactLLM(texts))
    session.add_message("alex", "Banff?", "human")
    session.add_message("alex", "Hmm Banff", "human")
    assert session.card["preferences"] == []


def test_recorded_decision_is_in_the_view(tmp_path):
    session = chat_session(tmp_path)
    session.record_decision("opt_2", "Cheaper and Priya will check the huts.")
    decision = interpret_view(session.card)["decision"]
    assert decision == {"status": "recorded", "option": "Dolomites",
                        "rationale": "Cheaper and Priya will check the huts."}


# ── length is enforced ───────────────────────────────────────────────────────

class ScriptedClient:
    def __init__(self, *answers):
        self.answers, self.payloads = list(answers), []

    def structured(self, instructions, payload, name, schema, model, **kwargs):
        self.payloads.append(payload)
        return self.answers.pop(0)


SHORT = {"headline": "Patagonia leads; not agreed.", "open_items": ["Priya disputes flight cost."],
         "next_step": "Alex checks flights.", "cited_ids": ["opt_1"]}


def test_long_answer_gets_one_retry_with_what_to_shorten(tmp_path):
    from mvp1_c2c.pact.interpret import Interpreter
    long = {**SHORT, "headline": " ".join(["word"] * 40)}
    client = ScriptedClient(long, SHORT)
    result = Interpreter(client, "m").interpret(chat_session(tmp_path).card)
    assert result["headline"] == SHORT["headline"] and result["over_length"] == []
    assert client.payloads[1]["shorten"] == ["headline over 25 words"]


def test_short_answer_is_not_retried(tmp_path):
    from mvp1_c2c.pact.interpret import Interpreter
    client = ScriptedClient(SHORT)
    assert Interpreter(client, "m").interpret(chat_session(tmp_path).card) == SHORT
    assert len(client.payloads) == 1
