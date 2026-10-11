"""End-to-end run of the DB-free pipeline (what eval/run_eval.py runs) with a scripted LLM,
scored against eval/answer_key.json. Also checks the scorer catches the answer key's traps.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from eval.score_brief import score_brief

from .scripted_llm import ScriptedLLM, cited as _cited, run_pipeline

from .test_anniversary_chat import L, chunk0, chunk1, chunk2, synthesized_with_traps, units

KEY = json.loads((Path(__file__).resolve().parent.parent / "eval" / "answer_key.json").read_text())


def cited(text, line, quote):
    return _cited(L, text, line, quote)


GOOD_SUMMARY = {
    "title": "-", "status": "-",
    "needs_attention": [
        cited("Drinks push the Casa Verde total to about $2,460, over the $2,400 cap", 44, "2,100 + 360 is 2,460"),
        cited("Who covers the drinks: owner not named", 89, "we'll sort the drinks at the end"),
        cited("Booking Casa Verde: asked of Leila, not accepted", 64, "Leila book it"),
        cited("Omar declined paying for drinks", 65, "I said cake. not drinks"),
    ],
    "proposed": [
        cited("Dec 12: explicit support from Nadia, Yusuf, Omar, Leila, Tariq and Hana", 73, "so the 12th works"),
        cited("Casa Verde: explicit support from Nadia, Leila, Omar, Tariq and Hana; acknowledged by Yusuf", 79, "ok yes. Casa Verde"),
        cited("$400 per person: explicit support from Omar, Nadia, Leila, Yusuf and Hana", 29, "400 each is the cap"),
        cited("Guest cap of 20: explicit support from Leila and Nadia", 50, "20 or nothing"),
        cited("Alma dropped: private room holds only 14", 74, "the private room only holds 14"),
        cited("Harbor Hall dropped: second floor", 56, "Harbor Hall is out then"),
    ],
    "committed": [
        cited("Nadia sends the $500 deposit Monday; deadline Oct 16", 87, "I'll send the $500 deposit Monday"),
        cited("Leila trims the guest list to 20 by Sunday", 50, "I'll trim the list"),
        cited("Omar brings the cake", 85, "cake is mine"),
    ],
    "unverified_claims": [
        cited("Free parking behind Casa Verde (one person's report)", 78, "free lot behind the building, about 12 spots"),
        cited("Private room is on the ground floor (one person's report)", 56, "Casa Verde's private room is ground floor"),
        cited("Drinks about $20 a head (not written down)", 43, "she said about $20 a head"),
        cited("Everyone's on board: Tariq disputed this", 64, "everyone's on board"),
    ],
}


def run(summary: dict) -> tuple[dict, ScriptedLLM]:
    llm = ScriptedLLM([chunk0(), chunk1(), chunk2()], synthesized_with_traps, summary)
    return run_pipeline(units(), llm, KEY["question"]), llm


def failed(result):
    return [c["name"] for c in result["checks"] if not c["passed"]]


def test_pipeline_scores_full_marks_on_answer_key():
    brief, llm = run(GOOD_SUMMARY)
    assert llm.calls[:4] == ["extract_candidates"] * 3 + ["produce_brief"]
    assert brief["executive_summary_report"]["omitted_items"] == 0
    result = score_brief(brief, KEY)
    assert failed(result) == [], failed(result)
    assert result["false_agreements"] == 0
    card = brief["executive_summary"]
    assert "[Leila, 10/6/26 12:05 PM]" in card                   # readable citation
    assert "no stance on budget: Tariq" in card                    # silence is shown, not counted


def test_card_traps_are_caught():
    bad = copy.deepcopy(GOOD_SUMMARY)
    bad["committed"].append(cited("Omar pays for drinks", 64, "Omar's paying for cake and drinks"))
    bad["committed"].append(cited("Leila books Casa Verde", 64, "Leila book it"))
    bad["needs_attention"].append(cited("Anniversary date disputed: 8th vs 9th", 7, "its the 8th"))
    bad["proposed"].append(cited("Alma at $150 a head", 23, "Alma is $150 a head"))
    bad["unverified_claims"] = []
    brief, _ = run(bad)
    # the scripted entailment check passes everything, so these reach the card
    names = failed(score_brief(brief, KEY))
    for expected in ("committed does not match /omar.*drink/", "Leila's booking is shown as asked, not agreed",
                     "needs-attention does not raise /8th/", "card does not mention '$150'",
                     "alma appears only as dropped", "unverified section flags /parking/"):
        assert expected in names, (expected, names)


def test_structural_traps_are_caught_in_a_baseline_style_brief():
    """A single-pass brief that takes 'everyone's on board' at face value."""
    m = {"options": [{"title": t, "status": "active"} for t in ("Dec 12", "Casa Verde", "$400 each", "Alma")],
         "positions": [{"participant_label": n, "option_title": t, "stance": "support"}
                       for n in KEY["participants"] for t in ("Dec 12", "Casa Verde", "$400 each")],
         "actions": [{"text": "Omar pays for cake and drinks", "owner_label": "Omar", "owner_kind": "named"}],
         "blockers": [{"text": "Anniversary date unclear (8th or 9th)", "resolution_status": "open"}],
         "card": "PROPOSED\n· Everyone's on board with Casa Verde"}
    result = score_brief(m, KEY)
    names = failed(result)
    assert result["false_agreements"] == 2                         # Yusuf on Casa Verde, Tariq on $400
    for expected in ("Casa Verde: Yusuf is not counted as agreeing", "$400 per person cap: Tariq is not counted as agreeing",
                     "Alma is dropped, not live", "Omar did not take on drinks", "'8th' is not an open blocker",
                     "/everyone'?s on board/ is not stated as fact"):
        assert expected in names, (expected, names)
