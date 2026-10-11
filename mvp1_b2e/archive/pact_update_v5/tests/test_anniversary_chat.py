"""Regression test: the siblings' 40th-anniversary dinner chat (tests/fixtures/anniversary_chat.txt).

The LLM is replaced by scripted outputs: per-chunk extractions written the way the
extractor prompt asks for them, and a synthesis response that falls into the traps a
plain summarizer falls into. The assertions are the answer key (eval/answer_key.json).
"""
from __future__ import annotations

import copy
from pathlib import Path

from pact.worker.chunker import chunk_units
from pact.worker.extractor import validate_extraction
from pact.worker.merger import merge_extractions
from pact.worker.parser import _detect_format, _parse_whatsapp
from pact.worker.synthesizer import (
    _assign_server_ids, _compute_preference, _compute_status_line, _restore_dropped, _summary_state,
)

CHAT = (Path(__file__).parent / "fixtures" / "anniversary_chat.txt").read_text()
_SKIPPED_LINES = (35, 37)          # "<Media omitted>" and "This message was deleted"
PARTICIPANTS = ["Nadia", "Yusuf", "Omar", "Leila", "Tariq", "Hana"]


def L(line: int) -> str:
    """Unit id for a 1-based line of the chat file."""
    return f"u{line - 1 - sum(1 for s in _SKIPPED_LINES if s < line)}"


def units():
    us = _parse_whatsapp(CHAT)
    for u in us:
        u["id"] = f"u{u['sequence']}"
    return us


def text_of(line: int) -> str:
    return next(u["normalized_text"] for u in units() if u["id"] == L(line))


def opt(title, decision, line, quote, status="active", reason=None):
    o = {"temp_id": title, "title": title, "decision": decision, "status": status,
         "source_unit_ids": [L(line)], "quote": quote}
    if reason:
        o["status_reason"] = reason
    return o


def pos(who, title, stance, line, quote, conditions=None):
    p = {"participant_label": who, "option_title": title, "stance": stance,
         "source_unit_ids": [L(line)], "quote": quote}
    if conditions:
        p["conditions"] = conditions
    return p


def item(text, line, quote, lines=(), **kw):
    return {"text": text, "source_unit_ids": [L(line), *[L(x) for x in lines]], "quote": quote, **kw}


def side(title, rationale, *lines):
    return {"title": title, "rationale": rationale, "source_unit_ids": [L(x) for x in lines]}


def empty():
    return {k: [] for k in ("candidate_questions", "options", "criteria", "constraints", "claims",
                            "positions", "objections", "blockers", "separate_issues", "actions")}


# --------------------------------------------------------------------------- scripted extractions

def chunk0():
    r = empty()
    r["candidate_questions"] = [item("Which date and venue for the parents' 40th anniversary dinner?", 6, "dinner. date + place")]
    r["options"] = [
        opt("Dec 5", "date", 10, "Dec 5 or Dec 12"), opt("Dec 12", "date", 10, "Dec 5 or Dec 12"),
        opt("Seoul House", "venue", 16, "Seoul House"),
        opt("Seoul House", "venue", 17, "lol no", status="dropped", reason="rejected"),
        opt("Casa Verde", "venue", 18, "Casa Verde"),
        opt("Catering at Nadia's house", "venue", 18, "or I cater at mine"),
        opt("Alma", "venue", 22, "Alma downtown"), opt("Harbor Hall", "venue", 25, "Harbor Hall"),
        opt("$400 each", "budget", 28, "I'm not paying more than $400 each"),
    ]
    r["positions"] = [            # "Dec 5 or Dec 12. poll" is a question: no position for Nadia
        pos("Tariq", "Dec 12", "conditional", 9, "the 12th might work if I can get a flight", "if he can get a flight"),
        pos("Hana", "Dec 12", "support", 11, "12th"), pos("Leila", "Dec 12", "support", 12, "12th"),
        pos("Omar", "Dec 12", "support", 13, "12th"), pos("Yusuf", "Dec 5", "support", 14, "5th"),
        pos("Omar", "Seoul House", "proposed", 16, "Seoul House"),
        pos("Nadia", "Seoul House", "oppose", 17, "lol no"),
        pos("Nadia", "Casa Verde", "proposed", 18, "I'd say Casa Verde"),
        pos("Nadia", "Catering at Nadia's house", "proposed", 18, "or I cater at mine"),
        pos("Tariq", "Alma", "proposed", 22, "or Alma downtown"),
        pos("Yusuf", "Harbor Hall", "proposed", 25, "what about Harbor Hall"),
        pos("Omar", "$400 each", "proposed", 28, "I'm not paying more than $400 each"),
        pos("Nadia", "$400 each", "support", 29, "400 each is the cap"),
        pos("Leila", "$400 each", "support", 30, "ok"),     # bare ok right after one proposal
    ]
    r["claims"] = [
        item("Catering for 18 costs around $1,400", 18, "around $1,400 for 18", kind="factual", author_label="Nadia"),
        item("Catering for 18 is closer to $2,200 with rentals", 19, "closer to 2,200 once you add rentals", kind="factual", author_label="Omar"),
        item("Alma is $150 a head", 23, "Alma is $150 a head", kind="factual", author_label="Nadia"),
        item("Alma prix fixe is $95 a head", 24, "prix fixe is $95", kind="factual", author_label="Tariq"),
    ]
    r["constraints"] = [item("Budget cap of $400 per sibling ($2,400 for six)", 28, "thats 2,400 for six of us", kind="mandatory", author_label="Omar")]
    r["objections"] = [
        item("Harbor Hall costs $3,200", 26, "$3,200 though", option_titles=["Harbor Hall"], author_label="Omar", resolution_status="open"),
        item("Harbor Hall is on the second floor", 27, "its on the second floor", option_titles=["Harbor Hall"], author_label="Leila", resolution_status="open"),
    ]
    r["separate_issues"] = [
        side("Keeping the plan secret from the parents", "about the group, not the decision", 2),
        side("Whether the anniversary is the 8th or the 9th", "does not change which date works", 7, 8),
    ]
    return r


def chunk1():
    r = empty()
    r["options"] = [
        opt("Guest cap of 20", "guest count", 50, "20 or nothing"),
        opt("Catering at Nadia's house", "venue", 54, "catering at mine is out", status="dropped", reason="5 steps at the front"),
        opt("Harbor Hall", "venue", 56, "Harbor Hall is out then", status="dropped", reason="second floor"),
    ]
    r["positions"] = [
        pos("Yusuf", "$400 each", "support", 31, "ok"),
        pos("Leila", "Guest cap of 20", "proposed", 50, "I'll trim the list. 20 or nothing"),
        pos("Omar", "Casa Verde", "conditional", 60, "Casa Verde is fine with me as long as I can bring my own cake", "outside cake allowed"),
    ]
    r["claims"] = [
        item("Casa Verde private room: $2,100 for 18 with food, drinks extra", 40, "$2,100 for 18 people with food", kind="factual", author_label="Leila"),
        item("Drinks at Casa Verde are about $20 a head", 43, "about $20 a head", kind="factual", author_label="Leila"),
        item("Casa Verde's private room is on the ground floor", 56, "Casa Verde's private room is ground floor", kind="factual", author_label="Leila"),
        item("Yusuf is fine with anything", 62, "Yusuf is fine with anything", kind="other", author_label="Nadia"),
    ]
    r["constraints"] = [
        item("Casa Verde needs a $500 deposit by Oct 16 to hold the 12th", 40, "they need a $500 deposit by Oct 16", kind="mandatory", author_label="Leila"),
        item("Casa Verde private room fits 20 max", 40, "private room fits 20 max", kind="mandatory", author_label="Leila"),
        item("Venue must be step-free", 53, "has to be step-free", kind="mandatory", author_label="Yusuf", sensitive=True),
    ]
    r["blockers"] = [
        item("With drinks the Casa Verde total is $2,460, over the $2,400 budget", 44, "2,100 + 360 is 2,460", kind="open_question", resolution_status="open"),
        item("Guest list is 22 but the room holds 20", 46, "the list is 22", kind="unresolved_dependency", resolution_status="open"),
        item("Guest list is 22 but the room holds 20", 50, "I'll trim the list. 20 or nothing", kind="unresolved_dependency", resolution_status="resolved"),
        item("Does Alma have an elevator", 57, "Alma has an elevator I'm pretty sure", kind="missing_info", resolution_status="open"),
    ]
    r["actions"] = [
        item("Leila trims the guest list to 20", 50, "I'll trim the list", owner_label="Leila", owner_kind="named", acceptance="committed"),
        item("Tariq calls Alma", 59, "I'll call them", owner_label="Tariq", owner_kind="named", acceptance="committed"),
        item("Omar brings the cake", 60, "I'm doing the cake", owner_label="Omar", owner_kind="named", acceptance="committed"),
        item("Leila asks Casa Verde if outside cake is ok", 61, "I'll ask if outside cake is ok", owner_label="Leila", owner_kind="named", acceptance="committed"),
    ]
    return r


def chunk2():
    r = empty()
    r["options"] = [opt("Alma", "venue", 74, "the private room only holds 14", status="dropped", reason="private room holds only 14")]
    r["positions"] = [
        pos("Yusuf", "Casa Verde", "acknowledged", 66, "sure whatever"),
        pos("Tariq", "Dec 12", "support", 73, "so the 12th works"),
        pos("Tariq", "Casa Verde", "support", 79, "ok yes. Casa Verde"),
        pos("Yusuf", "Dec 12", "support", 80, "12th works"),
        pos("Hana", "Casa Verde", "support", 81, "I'm ok with Casa Verde and the 12th"),
        pos("Hana", "$400 each", "support", 83, "400 is the number then"),
        pos("Omar", "Casa Verde", "support", 85, "then yes, Casa Verde, 12th"),
        pos("Leila", "Casa Verde", "support", 86, "Casa Verde private room"),
        pos("Leila", "Guest cap of 20", "support", 86, "20 max"),
        pos("Nadia", "Dec 12", "support", 87, "yes"),
        pos("Nadia", "Casa Verde", "support", 87, "yes"),
        pos("Nadia", "Guest cap of 20", "support", 87, "Leila trims the list to 20 by Sunday"),
    ]
    r["claims"] = [
        item("Everyone is on board with Casa Verde, Dec 12, $400 each", 64, "everyone's on board", kind="other", author_label="Nadia"),
        item("Casa Verde has a free lot behind the building, about 12 spots", 78, "free lot behind the building, about 12 spots", kind="factual", author_label="Leila"),
        item("Private arrangement between two participants", 81, "I can do 300 right now", lines=(82,), kind="other", author_label="Hana", sensitive=True),
    ]
    r["blockers"] = [
        item("Does Alma have an elevator", 74, "there's an elevator but the private room only holds 14", kind="missing_info", resolution_status="resolved"),
        item("Does Casa Verde have parking close by", 75, "does it have parking though?", kind="missing_info", resolution_status="open"),
        item("Does Casa Verde have parking close by", 78, "free lot behind the building, about 12 spots", kind="missing_info", resolution_status="resolved"),
    ]
    r["actions"] = [
        item("Leila books Casa Verde", 64, "Leila book it", owner_label="Leila", owner_kind="named", acceptance="requested"),
        item("Nadia sends the deposit", 64, "I'll send the deposit Monday", owner_label="Nadia", owner_kind="named", due="Monday", acceptance="committed"),
        item("Omar pays for drinks", 64, "Omar's paying for cake and drinks", owner_label="Omar", owner_kind="named", acceptance="requested"),
        item("Omar pays for drinks", 65, "I said cake. not drinks", owner_label="Omar", owner_kind="named", status="declined"),
        item("Tariq calls Alma", 74, "called Alma", owner_label="Tariq", owner_kind="named", status="done"),
        item("Leila asks Casa Verde if outside cake is ok", 84, "outside cake is fine, I asked", owner_label="Leila", owner_kind="named", status="done"),
        item("Nadia sends the $500 deposit", 87, "I'll send the $500 deposit Monday", owner_label="Nadia", owner_kind="named", due="Monday", acceptance="committed"),
        item("Leila trims the guest list to 20", 87, "Leila trims the list to 20 by Sunday", owner_label="Leila", owner_kind="named", due="Sunday", acceptance="requested"),
        item("Sort out who covers the drinks", 89, "we'll sort the drinks at the end", owner_kind="team"),
    ]
    r["separate_issues"] = [side("Photographer", "deferred: 'later. 1 thing at a time'", 69, 70, 71)]
    return r


def merged_draft():
    us = units()
    chunks = chunk_units(us)
    assert len(chunks) == 3
    results = [validate_extraction(copy.deepcopy(f()), c, i) for i, (f, c) in enumerate(zip((chunk0, chunk1, chunk2), chunks))]
    for r in results:
        assert r["_dropped"] == [], r["_dropped"]      # every scripted quote is real
    return merge_extractions(results, unit_sequence={u["id"]: u["sequence"] for u in us})


def stance_map(positions):
    return {(p["participant_label"], p["option_title"]): p["stance"] for p in positions}


def affirmers(positions, title):
    return {p["participant_label"] for p in positions
            if p["option_title"] == title and p["stance"] in ("support", "proposed")}


# --------------------------------------------------------------------------- the brief's data

def test_parse():
    assert _detect_format(CHAT, "auto") == "whatsapp"
    assert len(units()) == 87
    assert text_of(80) == "fine, I moved my thing. 12th works"
    assert "example.com" not in text_of(78)


def test_agreements_match_answer_key():
    m = merged_draft()
    ps = m["positions"]
    assert affirmers(ps, "Dec 12") == set(PARTICIPANTS)
    assert ("Yusuf", "Dec 5") not in stance_map(ps)
    assert any(p["participant_label"] == "Yusuf" and p["option_title"] == "Dec 5" for p in m["superseded_positions"])
    assert affirmers(ps, "Casa Verde") == {"Nadia", "Leila", "Omar", "Tariq", "Hana"}
    assert stance_map(ps)[("Yusuf", "Casa Verde")] == "acknowledged"          # "sure whatever"
    assert affirmers(ps, "$400 each") == {"Omar", "Nadia", "Leila", "Yusuf", "Hana"}
    assert not any(p["participant_label"] == "Tariq" and p["option_title"] == "$400 each" for p in ps)
    assert affirmers(ps, "Guest cap of 20") == {"Leila", "Nadia"}
    # Proposals of options the proposer later moved away from are gone
    for who, title in (("Tariq", "Alma"), ("Omar", "Seoul House"), ("Nadia", "Catering at Nadia's house")):
        assert (who, title) not in stance_map(ps), (who, title)


def test_dropped_options_and_reasons():
    opts = {o["title"]: o for o in merged_draft()["options"]}
    for t in ("Harbor Hall", "Alma", "Catering at Nadia's house", "Seoul House"):
        assert opts[t]["status"] == "dropped", t
    assert "14" in opts["Alma"]["status_reason"]
    assert opts["Casa Verde"]["status"] == opts["Dec 12"]["status"] == "active"
    objs = merged_draft()["objections"]
    assert len(objs) == 2 and all(o.get("moot") for o in objs)    # Omar's cost and Leila's stairs, both Harbor Hall


def test_constraint_step_free_is_private():
    c = next(c for c in merged_draft()["constraints"] if "step-free" in c["text"])
    assert c["sensitive"] and c["author_label"] == "Yusuf"
    assert "knee" not in c["text"].lower() and "surgery" not in c["text"].lower()


def test_open_items_are_only_drinks():
    m = merged_draft()
    open_blockers = [b["text"] for b in m["blockers"] if b["resolution_status"] == "open"]
    assert open_blockers == ["With drinks the Casa Verde total is $2,460, over the $2,400 budget"]
    drinks = [a for a in m["actions"] if "drinks" in a["text"] and a["status"] == "open"]
    assert len(drinks) == 1 and drinks[0]["owner_kind"] == "team" and "owner_label" not in drinks[0]


def test_actions_match_answer_key():
    acts = {a["text"]: a for a in merged_draft()["actions"]}
    deposit = acts["Nadia sends the $500 deposit"]
    assert (deposit["owner_label"], deposit["due"], deposit["deadline"], deposit["acceptance"]) == \
        ("Nadia", "Monday", "Oct 16", "committed")
    assert deposit["deadline_source_unit_ids"] == [L(40)]                     # per Leila
    trim = acts["Leila trims the guest list to 20"]
    assert (trim["due"], trim["acceptance"]) == ("Sunday", "committed")      # Leila accepted at 10:07
    assert acts["Omar brings the cake"]["acceptance"] == "committed"
    assert acts["Leila books Casa Verde"]["acceptance"] == "requested"        # never accepted
    assert acts["Omar pays for drinks"]["status"] == "declined"               # the misquote
    assert acts["Tariq calls Alma"]["status"] == "done"


def test_separate_issues():
    titles = {i["title"] for i in merged_draft()["separate_issues"]}
    assert titles == {"Photographer", "Whether the anniversary is the 8th or the 9th",
                      "Keeping the plan secret from the parents"}
    m = merged_draft()
    assert not any("8th" in b["text"] or "9th" in b["text"] for b in m["blockers"])


# --------------------------------------------------------------------------- synthesis traps

def synthesized_with_traps(m):
    """A synthesis response that falls into the traps from the answer key."""
    b = copy.deepcopy({k: m[k] for k in ("criteria", "constraints", "options", "claims", "positions",
                                         "objections", "blockers", "separate_issues", "actions")})
    for o in b["options"]:
        if o["title"] == "Dec 12":
            o["title"] = "December 12"
        if o["title"] == "Alma":
            o["status"] = "active"                                          # Alma still live
    b["positions"] = [p for p in b["positions"] if p["participant_label"] != "Hana"]
    b["positions"] += [
        {"participant_label": "Yusuf", "option_title": "Casa Verde", "stance": "support", "source_unit_ids": [L(64)]},
        {"participant_label": "Tariq", "option_title": "$400 each", "stance": "support", "source_unit_ids": [L(64)]},
    ]                                                                       # from "everyone's on board"
    b["blockers"].append({"text": "Anniversary date disputed (8th vs 9th)", "kind": "open_question",
                          "resolution_status": "open", "source_unit_ids": [L(7), L(8)]})
    for a in b["actions"]:
        if a["text"] == "Omar pays for drinks":
            a["status"] = "open"
        if a["text"] == "Leila books Casa Verde":
            a["acceptance"] = "committed"
        if a["text"] == "Sort out who covers the drinks":
            a["owner_kind"], a["owner_label"] = "named", "Omar"
    b["actions"].append({"text": "Omar covers $100 of Hana's share", "owner_kind": "named", "owner_label": "Omar",
                         "source_unit_ids": [L(82)]})
    return b


def built_brief():
    m = merged_draft()
    body = synthesized_with_traps(m)
    notes = _restore_dropped(body, m)
    body = _assign_server_ids(body)
    body["participants"] = PARTICIPANTS
    return m, body, notes


def test_synthesis_traps_are_corrected():
    _, body, notes = built_brief()
    opt_title = {o["id"]: o["title"] for o in body["options"]}
    ps = [{**p, "option_title": opt_title[p["option_id"]]} for p in body["positions"]]
    assert stance_map(ps)[("Yusuf", "Casa Verde")] == "acknowledged"
    assert not any(p["participant_label"] == "Tariq" and p["option_title"] == "$400 each" for p in ps)
    assert next(o for o in body["options"] if o["title"] == "Alma")["status"] == "dropped"
    assert not any("8th" in b["text"] for b in body["blockers"])
    acts = {a["text"]: a for a in body["actions"]}
    assert acts["Omar pays for drinks"]["status"] == "declined"
    assert acts["Leila books Casa Verde"]["acceptance"] == "requested"
    assert "owner_label" not in acts["Sort out who covers the drinks"]
    assert not any("Hana" in t for t in acts)
    assert any("removed unsupported" in n for n in notes)


def test_status_line_reads_like_the_answer_key():
    status = _compute_status_line(built_brief()[1])
    assert status.startswith("9 options proposed (4 dropped: ")
    assert "Alma — private room holds only 14" in status
    assert "date — Dec 12: 6 explicit supporters: " in status
    assert "venue — Casa Verde: 5 explicit supporters: " in status and "acknowledged only: Yusuf" in status
    assert "budget — $400 each: 4 explicit supporters: Nadia, Leila, Yusuf, Hana; proposed by Omar" in status
    assert "no stance on budget: Tariq" in status
    assert "guest count — Guest cap of 20: 2 explicit supporters: Leila, Nadia" in status
    assert "0 open objections" in status and "1 open blocker" in status
    assert "1 open action without a named owner" in status
    assert "1 task asked but not accepted (Leila)" in status
    assert "1 assignment declined" in status


def test_preference_counts_only_explicit_affirmers():
    pref = _compute_preference(built_brief()[1])["summary"]
    assert "Casa Verde has explicit support from" in pref and "acknowledged only by Yusuf" in pref
    assert "$400 each has explicit support from Nadia, Leila, Yusuf and Hana, plus Omar (proposer)" in pref
    assert "Yusuf" not in pref.split("Casa Verde has explicit support from")[1].split(";")[0]


def test_card_inputs_exclude_private_details():
    _, body, _ = built_brief()
    state = _summary_state(body)
    blob = repr(state).lower()
    for leak in ("knee", "surgery", "between jobs", "do 300", "cover the 100", "$100", "private arrangement"):
        assert leak not in blob, leak
    assert any("step-free" in c["text"] for c in state["constraints"])     # the requirement stays
