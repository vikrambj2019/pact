"""Pact function 1 — observe. Every rule the card enforces on observer actions, using the hiking traps."""
import pytest

from fakes import FakePactLLM, act, claim
from mvp1_c2c.pact import PactSession

MEMBERS = [{"id": "alex", "name": "Alex", "mapping_allowed": True, "share_allowed": True},
           {"id": "priya", "name": "Priya", "mapping_allowed": True, "share_allowed": True},
           {"id": "leo", "name": "Leo", "mapping_allowed": True, "share_allowed": False}]


def make(responses=None, members=MEMBERS):
    return PactSession.create("Where should we hike in November?", "Sam", [dict(m) for m in members],
                              FakePactLLM(responses))


def say(session, speaker, text, reply_to=None):
    return session.add_message(speaker, text, "human", reply_to=reply_to)


def audit_events(session, event):
    return [a for a in session.card["audit"] if a["event"] == event]


# ── history: who, when, exact words ─────────────────────────────────────────

def test_every_change_records_who_when_and_quote():
    session = make({"I vote Patagonia. Torres del Paine.": [act("add_option", "I vote Patagonia", text="Patagonia")]})
    msg = say(session, "alex", "I vote Patagonia. Torres del Paine.")
    option = session.card["options"][0]
    assert option["id"] == "opt_1" and option["proposed_by"] == "alex" and option["status"] == "considering"
    assert option["history"] == [{"event": "added", "message_id": msg["id"], "speaker_id": "alex",
                                  "at": msg["recorded_at"], "quote": "I vote Patagonia"}]
    log = session.card["change_log"][-1]
    assert (log["speaker_id"], log["at"], log["quote"], log["action"]) == ("alex", msg["recorded_at"],
                                                                          "I vote Patagonia", "add_option")


def test_ids_are_assigned_by_pact_and_duplicates_are_not_added():
    session = make({"Patagonia!": [act("add_option", "Patagonia", text="Patagonia", aliases=["Torres"])],
                    "What about Torres?": [act("add_option", "Torres", text="Torres")]})
    say(session, "alex", "Patagonia!")
    say(session, "priya", "What about Torres?")
    assert [o["id"] for o in session.card["options"]] == ["opt_1"]
    assert audit_events(session, "duplicate")


# ── agreement ────────────────────────────────────────────────────────────────

def option_session(extra=None):
    responses = {"Nov 14 to 21 then?": [act("add_constraint", "Nov 14 to 21", text="Dates Nov 14-21", kind="dates")]}
    responses.update(extra or {})
    session = make(responses)
    say(session, "human_admin", "Nov 14 to 21 then?")
    return session


@pytest.mark.parametrize("reply", ["Agreed", "fine", "ok", "Works for me", "👍"])
def test_short_replies_count_as_explicit_affirmation(reply):
    session = option_session({reply: [act("record_affirmation", reply, target_id="con_1")]})
    say(session, "alex", reply)
    agreement = session.card["agreements"][0]
    assert agreement["subject_id"] == "con_1"
    assert [a["participant_id"] for a in agreement["affirmers"]] == ["alex"]
    assert agreement["affirmers"][0]["quote"] == reply
    assert (agreement["affirmed_count"], agreement["participant_count"], agreement["status"]) == (1, 4, "partial")


def test_agreed_by_all_needs_every_participant_and_silence_never_counts():
    session = option_session({t: [act("record_affirmation", t, target_id="con_1")] for t in ["Yes", "Works"]})
    for speaker, text in [("human_admin", "Yes"), ("alex", "Works"), ("priya", "Yes")]:
        say(session, speaker, text)
    assert session.card["agreements"][0]["status"] == "partial"  # Leo never said yes
    say(session, "leo", "Works")
    assert session.card["agreements"][0]["status"] == "agreed_by_all"


def test_objection_contests_a_constraint_and_replaces_that_persons_yes():
    session = option_session({"Works": [act("record_affirmation", "Works", target_id="con_1")],
                              "Actually I can only do the 14th to 18th":
                                  [act("record_objection", "I can only do the 14th to 18th", target_id="con_1")]})
    say(session, "priya", "Works")
    say(session, "priya", "Actually I can only do the 14th to 18th")
    agreement = session.card["agreements"][0]
    assert agreement["affirmers"] == [] and agreement["objectors"][0]["participant_id"] == "priya"
    assert session.card["constraints"][0]["status"] == "contested"
    assert [h["event"] for h in agreement["history"]] == ["added", "affirmed", "objected"]


def test_ambiguous_reply_from_a_member_is_never_recorded():
    session = option_session({"Sounds good": [act("record_affirmation", "Sounds good", target_id="con_1",
                                                  ambiguous=True)]})
    say(session, "leo", "Sounds good")
    assert session.card["agreements"] == []
    assert audit_events(session, "interpretation_skipped")


def test_admin_can_confirm_their_own_ambiguous_reply():
    session = option_session({"Sounds good": [act("record_affirmation", "Sounds good", target_id="con_1",
                                                  ambiguous=True)]})
    say(session, "human_admin", "Sounds good")
    assert session.card["agreements"] == []
    pending = session.card["individual_mapping"]["pending_interpretations"][0]
    session.confirm_interpretation(pending["id"], "Confirm.")
    assert session.card["agreements"][0]["affirmers"][0]["participant_id"] == "human_admin"


# ── options, constraints, preferences ────────────────────────────────────────

def test_constraint_superseded_keeps_old_and_points_to_current():
    session = make({"Budget $2,500 a head": [act("add_constraint", "Budget $2,500", text="Budget $2,500 per person",
                                                 kind="budget")],
                    "Can we drop it to $2,000?": [act("supersede_constraint", "drop it to $2,000", target_id="con_1",
                                                      text="Budget $2,000 per person")]})
    say(session, "human_admin", "Budget $2,500 a head")
    msg = say(session, "priya", "Can we drop it to $2,000?")
    old, new = session.card["constraints"]
    assert (old["status"], old["superseded_by"]) == ("superseded", "con_2")
    assert (new["status"], new["supersedes"], new["kind"], new["stated_by"]) == ("stated", "con_1", "budget", "priya")
    assert old["history"][-1]["event"] == "superseded" and old["history"][-1]["message_id"] == msg["id"]


def test_option_dropped_then_revived_keeps_its_story():
    session = make({"Banff?": [act("add_option", "Banff", text="Banff")],
                    "Banff is out, too cold": [act("set_option_status", "Banff is out", target_id="opt_1",
                                                   status="dropped"),
                                               act("add_reason", "too cold", target_id="opt_1", stance="against",
                                                   text="Too cold in November")],
                    "Let's put Banff back": [act("set_option_status", "put Banff back", target_id="opt_1",
                                                 status="considering")]})
    for speaker, text in [("alex", "Banff?"), ("priya", "Banff is out, too cold"), ("alex", "Let's put Banff back")]:
        say(session, speaker, text)
    option = session.card["options"][0]
    assert option["status"] == "considering"
    assert [h["event"] for h in option["history"]] == ["added", "dropped", "reason_against", "revived"]
    assert option["reasons"][0]["participant_id"] == "priya"


def test_observer_cannot_choose_an_option():
    session = make({"Banff?": [act("add_option", "Banff", text="Banff")],
                    "Banff it is": [act("set_option_status", "Banff it is", target_id="opt_1", status="chosen")]})
    say(session, "alex", "Banff?")
    say(session, "alex", "Banff it is")
    assert session.card["options"][0]["status"] == "considering"
    assert "only the admin records a decision" in audit_events(session, "observation_rejected")[0]["detail"]


def test_changed_mind_replaces_stance_and_keeps_previous():
    session = make({"I want Banff": [act("set_preference", "I want Banff", stance="Banff")],
                    "Switching to Dolomites, it's cheaper": [act("set_preference", "Switching to Dolomites",
                                                                 stance="Dolomites, because of cost")]})
    say(session, "priya", "I want Banff")
    msg = say(session, "priya", "Switching to Dolomites, it's cheaper")
    [pref] = session.card["preferences"]
    assert pref["participant_id"] == "priya" and pref["stance"] == "Dolomites, because of cost"
    assert pref["history"][-1] == {"event": "changed", "previous_stance": "Banff", "previous_leaning": None,
                                   "message_id": msg["id"],
                                   "speaker_id": "priya", "at": msg["recorded_at"],
                                   "quote": "Switching to Dolomites"}


def test_stances_are_always_the_speakers_own():
    # Alex reporting Priya's view can only ever become Alex's own entry, never Priya's.
    session = make({"Priya wants the Dolomites": [act("set_preference", "Priya wants the Dolomites",
                                                      stance="Dolomites")]})
    say(session, "alex", "Priya wants the Dolomites")
    assert [p["participant_id"] for p in session.card["preferences"]] == ["alex"]


def test_preference_needs_mapping_permission():
    members = [{"id": "alex", "name": "Alex", "mapping_allowed": False}]
    session = make({"I want Banff": [act("set_preference", "I want Banff", stance="Banff")]}, members)
    say(session, "alex", "I want Banff")
    assert session.card["preferences"] == [] and audit_events(session, "mapping_skipped")


def test_public_card_hides_unshared_preferences():
    session = make({"I want Banff": [act("set_preference", "I want Banff", stance="Banff")]})
    say(session, "leo", "I want Banff")
    say(session, "alex", "I want Banff")
    from mvp1_c2c.pact import public_card
    assert [p["participant_id"] for p in public_card(session.card)["preferences"]] == ["alex"]


def test_new_items_can_be_referenced_within_one_observation():
    text = "Dolomites, but only if the huts are open"
    session = make({text: [act("add_option", "Dolomites", text="Dolomites", ref="new_dolo"),
                           act("set_preference", text, stance="Dolomites", option_id="new_dolo", leaning="for",
                               conditional="only if the huts are open")]})
    say(session, "priya", text)
    assert session.card["preferences"][0]["option_id"] == "opt_1"
    assert session.card["preferences"][0]["conditional"] == "only if the huts are open"


def test_open_issue_resolved():
    session = make({"Who books the flights?": [act("add_issue", "Who books the flights?", text="Who books flights",
                                                   participant_ids=["alex", "nobody"])],
                    "I'll book them": [act("resolve_issue", "I'll book them", target_id="issue_1")]})
    say(session, "priya", "Who books the flights?")
    say(session, "alex", "I'll book them")
    issue = session.card["open_issues"][0]
    assert issue["involves"] == ["alex"] and issue["status"] == "resolved"


# ── claims ───────────────────────────────────────────────────────────────────

def test_claim_disputed_by_someone_else_and_corrected_by_its_author():
    session = make({"Huts close Sep 20": [claim("Huts close Sep 20")],
                    "No, some stay open": [act("challenge_claim", "some stay open", target_id="claim_1",
                                               text="Some huts stay open")],
                    "Right, most close Sep 20": [act("correct_claim", "most close Sep 20", target_id="claim_1",
                                                     text="Most huts close Sep 20")]})
    say(session, "alex", "Huts close Sep 20")
    say(session, "priya", "No, some stay open")
    assert session.card["claims"][0]["status"] == "disputed"
    say(session, "alex", "Right, most close Sep 20")
    c = session.card["claims"][0]
    assert (c["status"], c["corrected_to"], c["statement"]) == ("corrected", "Most huts close Sep 20", "Huts close Sep 20")
    assert [h["event"] for h in c["history"]] == ["added", "challenged", "corrected"]


def test_only_the_author_can_retract_and_nobody_challenges_themselves():
    session = make({"Huts close Sep 20": [claim("Huts close Sep 20")],
                    "Forget that": [act("retract_claim", "Forget that", target_id="claim_1")],
                    "Hmm, not sure": [act("challenge_claim", "not sure", target_id="claim_1", text="unsure")]})
    say(session, "alex", "Huts close Sep 20")
    say(session, "priya", "Forget that")
    say(session, "alex", "Hmm, not sure")
    assert session.card["claims"][0]["status"] == "unchallenged"
    assert len(audit_events(session, "observation_rejected")) == 2
    say(session, "alex", "Forget that")
    assert session.card["claims"][0]["status"] == "retracted"


def test_claims_need_kind_and_checkable_and_opinions_are_never_checkable():
    session = make({"It'll be the best trip ever": [claim("best trip ever", checkable=True, kind="opinion")],
                    "Flights are $1,300": [act("add_claim", "Flights are $1,300", text="Flights are $1,300",
                                               kind="fact")]})
    say(session, "alex", "It'll be the best trip ever")
    say(session, "alex", "Flights are $1,300")
    assert [c["checkable"] for c in session.card["claims"]] == [False]
    assert "checkable" in audit_events(session, "observation_rejected")[0]["detail"]


# ── robustness ───────────────────────────────────────────────────────────────

def test_one_bad_action_does_not_block_the_others():
    session = make({"Patagonia, flights $1,300": [act("add_option", "not in the message", text="Patagonia"),
                                                  claim("flights $1,300"),
                                                  act("set_option_status", "Patagonia", target_id="opt_9",
                                                      status="dropped")]})
    say(session, "alex", "Patagonia, flights $1,300")
    assert session.card["options"] == [] and len(session.card["claims"]) == 1
    details = [a["detail"] for a in audit_events(session, "observation_rejected")]
    assert any("exact quote" in d for d in details) and any("'opt_9'" in d for d in details)


def test_observation_raises_version_and_errors_leave_card_unchanged():
    session = make()
    before = session.card["card_version"]
    say(session, "alex", "hello")
    assert session.card["card_version"] == before + 2  # message, then observation

    def boom(*args):
        raise RuntimeError("model down")
    session.llm.observer.propose = boom
    say(session, "alex", "Patagonia")
    assert session.card["options"] == [] and audit_events(session, "observation_error")


def test_observer_gets_recent_context_reply_links_and_a_compact_card():
    session = make()
    for i in range(20):
        say(session, "alex", f"message {i}")
    msg = say(session, "priya", "Works", reply_to="msg_20")
    call = session.llm.observer.calls[-1]
    assert [m["id"] for m in call["context"]] == [f"msg_{i}" for i in range(6, 21)]
    assert call["new"][0]["reply_to"] == "msg_20" and call["new"][0]["id"] == msg["id"]
    assert "messages" not in call["view"] and "history" not in str(call["view"])


def test_batch_observation_rejects_actions_citing_other_messages():
    session = make({"a": [act("add_option", "a", text="A")],
                    "b": [act("add_option", "b", text="B", message_id="msg_99")]})
    a = session.add_message("alex", "a", "human", observe=False)
    b = session.add_message("priya", "b", "human", observe=False)
    result = session.observe_messages([a, b])
    assert result == {"applied": 1, "rejected": 1}
    assert [o["text"] for o in session.card["options"]] == ["A"]
