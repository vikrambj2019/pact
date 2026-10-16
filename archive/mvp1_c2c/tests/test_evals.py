"""Offline coverage of the evaluation runner's API, replay and scoring."""
import importlib.util
import json
from pathlib import Path

import pytest

from fakes import FakePactLLM, act

PATH = Path(__file__).resolve().parents[1] / "pact-evals" / "run_evals.py"
spec = importlib.util.spec_from_file_location("pact_run_evals", PATH)
evals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evals)


@pytest.mark.parametrize("batch_size", [1, 2, 0])
def test_eval_runner_replays_current_api_with_reply_metadata(tmp_path, monkeypatch, batch_size):
    transcript = {"id": "fixture", "title": "Where?", "participants": ["Sam", "Alex", "Leo"],
                  "messages": [{"id": "m001", "sender": "Sam", "text": "Banff?", "ts": "2026-10-08"},
                               {"id": "m002", "sender": "Alex", "text": "Yes", "reply_to": "m001"}]}
    gold = {"options": [{"name": "Banff"}], "agreements": [
        {"text": "Banff", "explicit_affirmers": ["Alex"], "msgs": ["m002"]}]}
    out, expected, results = (tmp_path / name for name in ("out", "gold", "results"))
    out.mkdir(); expected.mkdir()
    (out / "fixture.json").write_text(json.dumps(transcript))
    (expected / "fixture.gold.json").write_text(json.dumps(gold))
    monkeypatch.setattr(evals, "OUT_DIR", out)
    monkeypatch.setattr(evals, "GOLD_DIR", expected)
    monkeypatch.setattr(evals, "RESULTS_DIR", results)
    llm = FakePactLLM({"Banff?": [act("add_option", "Banff", text="Banff", ref="banff")],
                       "Yes": [act("record_affirmation", "Yes", target_id="opt_1" if batch_size == 1 else "banff")]})
    result = evals.run_transcript("fixture", llm, batch_size)
    assert result["errors"] == []
    assert result["card"]["agreements"][0]["status"] == "partial"  # Leo's silence never counted
    assert result["card"]["messages"][1]["reply_to"] == "m001"
    assert result["card"]["observation"]["completed"] == ["m001", "m002"]
    scores = evals.score(result)
    assert scores["options"] == "1/1" and scores["false_affirmations"] == 0
    assert scores["provenance_violations"] == 0
    assert (results / "fixture.json").exists()
    agreement = result["card"]["agreements"][0]
    agreement["affirmers"].append({"participant_id": "p_2", "message_id": "m002"})
    assert evals.score(result)["false_affirmations"] == 1


def test_all_five_committed_transcripts_can_be_replayed_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(evals, "RESULTS_DIR", tmp_path)
    for stem in evals.STEMS:
        result = evals.run_transcript(stem, FakePactLLM())
        assert result["errors"] == []
        assert not result["card"]["observation"]["pending"]
        scores = evals.score(result)
        assert scores["false_affirmations"] == 0
        assert scores["provenance_violations"] == 0
        assert scores["observer_errors"] == 0


def test_failed_observation_is_an_eval_error_not_an_empty_success(tmp_path, monkeypatch):
    monkeypatch.setattr(evals, "RESULTS_DIR", tmp_path)
    llm = FakePactLLM()

    def fail(*args):
        raise RuntimeError("offline failure")

    llm.observer.propose = fail
    result = evals.run_transcript(evals.STEMS[0], llm)
    assert result["errors"][0]["error"] == "RuntimeError"
    assert evals.score(result)["observer_errors"] == 1
    assert result["card"]["observation"]["pending"]


def test_empty_option_text_never_matches_a_gold_option():
    assert not evals.similar("", "Banff")


def test_different_budget_amounts_do_not_match():
    assert not evals.similar("Budget $2,500", "Budget $2,000")
