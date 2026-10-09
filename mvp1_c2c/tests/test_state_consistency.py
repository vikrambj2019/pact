"""Regression cases for evolving claims, queued observation and immutable interpretation."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from fakes import FakePactLLM, act, claim
from mvp1_c2c.pact import PactSession
from mvp1_c2c.pact.interpret import interpret_view

MEMBERS = [{"id": "alex", "name": "Alex", "mapping_allowed": True, "share_allowed": True}]


def make(tmp_path, responses=None):
    return PactSession.create("Trip?", "Sam", MEMBERS, FakePactLLM(responses), path=tmp_path / "session.json")


def fee_session(tmp_path):
    session = make(tmp_path, {"Fee is $20": [claim("Fee is $20")],
                             "Fee is $200": [act("correct_claim", "Fee is $200", target_id="claim_1", text="Fee is $200")],
                             "Retract the fee": [act("retract_claim", "Retract the fee", target_id="claim_1")]})
    session.add_message("alex", "Fee is $20", "human")
    return session


def test_correction_invalidates_verdict_and_preserves_check_history(tmp_path):
    session = fee_session(tmp_path)
    first = session.check_claim("claim_1")
    session.add_message("alex", "Fee is $200", "human")
    revised = session.card["claims"][0]
    assert revised["verification"] == {"status": "not_checked", "check_ids": [first["id"]]}
    assert revised["revision"] == 1
    assert interpret_view(session.card)["claims"][0]["check"] == "not_checked"
    assert session.card["claim_checks"][0]["checked_statement"] == "Fee is $20"
    assert session.card["claim_checks"][0]["stale"] is True
    second = session.check_claim("claim_1")
    assert second["checked_statement"] == "Fee is $200" and not second["stale"]
    assert revised["verification"]["check_ids"] == [first["id"], second["id"]]


def test_retracted_claim_loses_current_verdict_and_is_never_researched(tmp_path):
    session = fee_session(tmp_path)
    session.check_claim("claim_1")
    session.add_message("alex", "Retract the fee", "human")
    assert session.card["claims"][0]["verification"]["status"] == "not_checked"
    with pytest.raises(ValueError, match="None of those claims"):
        session.check_claim("claim_1")
    assert session.llm.checker.checked == ["Fee is $20"]


@pytest.mark.parametrize("change", ["Fee is $200", "Retract the fee"])
def test_check_finishing_after_claim_change_cannot_endorse_current_claim(tmp_path, change):
    session = fee_session(tmp_path)
    started, release = Event(), Event()
    original = session.llm.checker.check

    def delayed(card, claim_, question):
        started.set()
        assert release.wait(3)
        return original(card, claim_, question)

    session.llm.checker.check = delayed
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(session.check_claim, "claim_1")
        try:
            assert started.wait(3)
            session.add_message("alex", change, "human")
        finally:
            release.set()
        result = future.result(timeout=3)
    assert result["stale"] is True and result["claim_revision"] == 0
    assert session.card["claims"][0]["verification"]["status"] == "not_checked"
    assert session.card["claims"][0]["verification"]["check_ids"] == [result["id"]]


def stance_session(tmp_path):
    responses = {"Banff": [act("add_option", "Banff", text="Banff")],
                 "Yes": [act("set_preference", "Yes", stance="yes", option_id="opt_1", leaning="for")],
                 "No": [act("set_preference", "No", stance="no", option_id="opt_1", leaning="against")]}
    session = make(tmp_path, responses)
    session.add_message("human_admin", "Banff", "human")
    return session


def test_later_background_task_drains_earlier_messages_and_replay_is_noop(tmp_path):
    session = stance_session(tmp_path)
    earlier = session.add_message("alex", "Yes", "simulated", observe=False)
    later = session.add_message("alex", "No", "simulated", observe=False)
    result = session.observe(later)
    assert result["applied"] == 2
    assert session.card["preferences"][0]["leaning"] == "against"
    assert session.observe(earlier) == {"applied": 0, "rejected": 0}
    loaded = PactSession.load(session.path, FakePactLLM())
    assert loaded.observe(later) == {"applied": 0, "rejected": 0}
    assert loaded.card["observation"]["pending"] == []


def test_overlapping_observations_are_serial_and_newer_stance_wins(tmp_path):
    session = stance_session(tmp_path)
    started, release = Event(), Event()
    original = session.llm.observer.propose
    active = []

    def delayed(view, messages, context):
        active.append(messages[0]["id"])
        assert len(active) == 1
        try:
            if messages[0]["text"] == "Yes":
                started.set()
                assert release.wait(3)
            return original(view, messages, context)
        finally:
            active.pop()

    session.llm.observer.propose = delayed
    earlier = session.add_message("alex", "Yes", "human", observe=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(session.observe, earlier)
        try:
            assert started.wait(3)
            later = session.add_message("alex", "No", "human", observe=False)
            second = pool.submit(session.observe, later)
        finally:
            release.set()
        assert first.result(timeout=3)["applied"] == 1
        assert second.result(timeout=3)["applied"] == 1
    assert session.card["preferences"][0]["leaning"] == "against"
    assert session.card["observation"]["pending"] == []
    assert PactSession.load(session.path, FakePactLLM()).card == session.snapshot()


def test_reverse_model_action_order_cannot_overwrite_the_newer_stance(tmp_path):
    session = stance_session(tmp_path)
    earlier = session.add_message("alex", "Yes", "human", observe=False)
    later = session.add_message("alex", "No", "human", observe=False)
    original = session.llm.observer.propose
    session.llm.observer.propose = lambda *args: list(reversed(original(*args)))
    session.observe_messages([later, earlier])
    assert session.card["preferences"][0]["leaning"] == "against"


def test_stale_observer_proposals_are_discarded_and_recomputed(tmp_path):
    session = stance_session(tmp_path)
    message = session.add_message("alex", "Yes", "human", observe=False)
    original = session.llm.observer.propose
    calls = []

    def changing(view, messages, context):
        calls.append(view)
        if len(calls) == 1:
            session.add_admin_claim("New fee information")
        return original(view, messages, context)

    session.llm.observer.propose = changing
    assert session.observe(message)["applied"] == 1
    assert len(calls) == 2
    assert calls[0]["claims"] == [] and calls[1]["claims"][0]["statement"] == "New fee information"
    assert len(session.card["preferences"][0]["history"]) == 1


def test_failed_observation_remains_pending_and_retries_without_losing_later_work(tmp_path):
    session = stance_session(tmp_path)
    earlier = session.add_message("alex", "Yes", "human", observe=False)
    original = session.llm.observer.propose

    def fail(*args):
        raise RuntimeError("provider unavailable")

    session.llm.observer.propose = fail
    assert session.observe(earlier)["error"] == "RuntimeError"
    assert earlier["id"] in session.card["observation"]["pending"]
    later = session.add_message("alex", "No", "human", observe=False)
    session.llm.observer.propose = original
    assert session.observe(later)["applied"] == 2
    assert session.card["preferences"][0]["leaning"] == "against"


def test_interpretation_rejects_a_card_changed_during_selection(tmp_path):
    session = stance_session(tmp_path)

    def changing(card):
        session.add_admin_claim("New information")
        return {"cited_ids": []}

    session.llm.interpreter.interpret = changing
    with pytest.raises(ValueError, match="card changed during interpretation"):
        session.interpret()
    assert session.card["facilitation"] == []


def test_real_citation_cannot_authorize_fabricated_consensus(tmp_path):
    session = stance_session(tmp_path)
    session.llm.interpreter.result = {"cited_ids": ["opt_1"],
                                      "headline": "Everyone agreed to Banff.",
                                      "open_items": ["No concerns remain."], "next_step": "Book Banff."}
    result = session.interpret()["output"]
    assert result["headline"] == "Nothing leads on stated support yet."
    assert result["open_items"] == []
    assert "Book Banff" not in result["summary"]
