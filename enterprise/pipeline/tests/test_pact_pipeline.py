"""Regression tests for the extraction / merge / synthesis fixes.

The scenario is the sibling anniversary-dinner chat (date + venue + budget decided in
one thread). The extractor output is scripted by hand the way a well-behaved model would
return it, so the tests exercise everything AFTER the model: validation, merging,
positions, status counts, preference, verification and rendering.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pact.services import chunker
from pact.services import synthesizer as S
from pact.services.extractor import SECTIONS, extract_chunk, normalize_quote_text, validate_extraction
from pact.services.merger import _same_option, _similar, merge_extractions


# ---------------------------------------------------------------------------
# Fixtures: the anniversary chat as source units
# ---------------------------------------------------------------------------

_RAW = [
    ("tariq_cond", "Tariq", "10/4/26, 7:48 PM", "I'm in Denver for work till the 11th. the 5th is out for me. the 12th might work if I can get a flight"),
    ("poll", "Nadia", "10/4/26, 7:49 PM", "ok Dec 5 or Dec 12. poll"),
    ("hana_vote", "Hana", "10/4/26, 7:52 PM", "12th"),
    ("leila_vote", "Leila", "10/4/26, 7:52 PM", "12th"),
    ("omar_vote", "Omar", "10/4/26, 7:53 PM", "12th"),
    ("yusuf_vote", "Yusuf", "10/4/26, 7:55 PM", "5th, I have a thing on the 12th"),
    ("nadia_cv", "Nadia", "10/4/26, 8:01 PM", "I'd say Casa Verde, the italian place they love. they have a private room."),
    ("leila_call", "Leila", "10/5/26, 9:48 AM", "called Casa Verde. private room fits 20 max. $2,100 for 18 people with food. drinks extra. they need a $500 deposit by Oct 16 to hold the 12th"),
    ("nadia_all", "Nadia", "10/5/26, 6:30 PM", "ok so we're all good: Casa Verde, Dec 12, $400 each. everyone's on board. Omar's paying for cake and drinks. Leila book it, I'll send the deposit Monday"),
    ("omar_cake", "Omar", "10/5/26, 6:31 PM", "I said cake. not drinks. LOL no"),
    ("yusuf_whatever", "Yusuf", "10/5/26, 6:33 PM", "sure whatever"),
    ("tariq_booked", "Tariq", "10/6/26, 8:15 AM", "booked my flight, landing 4pm on the 11th. so the 12th works"),
    ("tariq_cv", "Tariq", "10/6/26, 12:07 PM", "ok yes. Casa Verde"),
    ("yusuf_moved", "Yusuf", "10/6/26, 12:40 PM", "fine, I moved my thing. 12th works"),
    ("hana_ok", "Hana", "10/6/26, 1:15 PM", "I'm ok with Casa Verde and the 12th. on the 400, I can do 300 right now, between jobs, don't make it weird"),
    ("omar_yes", "Omar", "10/6/26, 1:26 PM", "then yes, Casa Verde, 12th. cake is mine"),
    ("leila_final", "Leila", "10/6/26, 1:30 PM", "ok. final from me: Dec 12, Casa Verde private room, 20 max, ground floor, $400 each"),
    ("nadia_yes", "Nadia", "10/6/26, 1:31 PM", "yes. I'll send the $500 deposit Monday and we settle up after. Leila trims the list to 20 by Sunday"),
]


def ID(key: str) -> str:
    return f"u-{key}"


UNITS = [
    {"id": ID(k), "sequence": i + 1, "author_label": a, "normalized_text": t, "locator": {"timestamp_str": ts}}
    for i, (k, a, ts, t) in enumerate(_RAW)
]
UNIT_TEXT = {u["id"]: u["normalized_text"] for u in UNITS}
UNIT_META = {u["id"]: {"author": u["author_label"], "ts": u["locator"]["timestamp_str"]} for u in UNITS}
CHUNK0_UNITS = UNITS[:9]
CHUNK1_UNITS = [{**u, "context_only": True} for u in UNITS[6:9]] + UNITS[9:]


def _o(title, decision, key, quote, status="active"):
    return {"temp_id": title, "title": title, "decision": decision, "status": status,
            "source_unit_ids": [ID(key)], "quote": quote}


def _p(who, opt, stance, key, quote, **kw):
    return {"participant_label": who, "option_title": opt, "stance": stance,
            "source_unit_ids": [ID(key)], "quote": quote, **kw}


def _blank():
    return {s: [] for s in SECTIONS}


def raw_chunks():
    """What a well-behaved model returns for the two chunks (plus three deliberately bad items)."""
    c0 = _blank()
    c0["options"] = [
        _o("Dec 5", "date", "poll", "Dec 5 or Dec 12"),
        _o("Dec 12", "date", "poll", "Dec 5 or Dec 12"),
        _o("Casa Verde", "venue", "nadia_cv", "I'd say Casa Verde"),
    ]
    c0["positions"] = [
        _p("Tariq", "Dec 12", "conditional", "tariq_cond", "the 12th might work if I can get a flight",
           conditions="if he can get a flight"),
        _p("Hana", "12th", "support", "hana_vote", "12th"),
        _p("Leila", "12th", "support", "leila_vote", "12th"),
        _p("Omar", "12th", "support", "omar_vote", "12th"),
        _p("Yusuf", "Dec 5", "support", "yusuf_vote", "5th, I have a thing on the 12th"),
        _p("Nadia", "Casa Verde", "proposed", "nadia_cv", "I'd say Casa Verde"),
        # bad: Nadia's claim about everyone is not Omar's position
        _p("Omar", "Casa Verde", "support", "nadia_all", "everyone's on board"),
    ]
    c0["claims"] = [{"text": "Casa Verde private room fits 20 max", "kind": "factual", "author_label": "Leila",
                     "source_unit_ids": [ID("leila_call")], "quote": "private room fits 20 max"}]
    c0["constraints"] = [{"text": "Budget cap $400 each", "kind": "mandatory",
                          "source_unit_ids": [ID("nadia_all")], "quote": "$400 each"}]
    c0["blockers"] = [{"text": "Tariq's flight for Dec 12 is not confirmed", "kind": "unresolved_dependency",
                       "resolution_status": "open", "source_unit_ids": [ID("tariq_cond")],
                       "quote": "if I can get a flight"}]
    c0["actions"] = [
        # the venue's deadline, extracted as an ownerless action
        {"text": "Pay the $500 deposit by Oct 16", "owner_kind": "unassigned",
         "source_unit_ids": [ID("leila_call")], "quote": "they need a $500 deposit by Oct 16"},
        # bad: invented quote
        {"text": "Siblings finalise the date by Oct 16", "owner_kind": "unassigned",
         "source_unit_ids": [ID("leila_call")], "quote": "finalise and confirm the date no later than October 16"},
    ]

    c1 = _blank()
    c1["positions"] = [
        _p("Yusuf", "Casa Verde", "acknowledged", "yusuf_whatever", "sure whatever"),
        _p("Tariq", "Dec 12", "support", "tariq_booked", "so the 12th works"),
        _p("Tariq", "Casa Verde", "support", "tariq_cv", "ok yes. Casa Verde"),
        _p("Yusuf", "Dec 12", "support", "yusuf_moved", "12th works"),
        _p("Hana", "Casa Verde", "support", "hana_ok", "I'm ok with Casa Verde"),
        _p("Hana", "Dec 12", "support", "hana_ok", "and the 12th"),
        _p("Omar", "Casa Verde", "support", "omar_yes", "yes, Casa Verde"),
        _p("Omar", "Dec 12", "support", "omar_yes", "12th"),
        _p("Leila", "Dec 12", "support", "leila_final", "Dec 12, Casa Verde"),
        _p("Leila", "Casa Verde", "support", "leila_final", "Casa Verde private room"),
        _p("Nadia", "Dec 12", "support", "nadia_yes", "yes."),
        _p("Nadia", "Casa Verde", "support", "nadia_yes", "yes."),
        # bad: cites a [context] unit, which is not allowed
        _p("Leila", "Dec 12", "support", "leila_call", "Casa Verde"),
    ]
    c1["actions"] = [
        {"text": "Send the $500 deposit Monday", "owner_label": "Nadia", "owner_kind": "named", "due": "Monday",
         "source_unit_ids": [ID("nadia_yes")], "quote": "I'll send the $500 deposit Monday"},
        {"text": "Trim the guest list to 20", "owner_label": "Leila", "owner_kind": "named", "due": "by Sunday",
         "source_unit_ids": [ID("nadia_yes")], "quote": "Leila trims the list to 20 by Sunday"},
    ]
    c1["blockers"] = [{"text": "Tariq's flight for Dec 12 is not confirmed", "kind": "unresolved_dependency",
                       "resolution_status": "resolved", "source_unit_ids": [ID("tariq_booked")],
                       "quote": "booked my flight"}]
    c1["claims"] = [{"text": "Omar is not paying for drinks", "kind": "factual", "author_label": "Omar",
                     "source_unit_ids": [ID("omar_cake")], "quote": "I said cake. not drinks."}]
    c1["constraints"] = [{"text": "Budget stays at $400 per person", "kind": "mandatory", "sensitive": True,
                          "source_unit_ids": [ID("hana_ok")], "quote": "on the 400"}]
    return c0, c1


def run_pipeline():
    c0, c1 = raw_chunks()
    r0 = validate_extraction(c0, CHUNK0_UNITS, 0)
    r1 = validate_extraction(c1, CHUNK1_UNITS, 1)
    seq = {u["id"]: u["sequence"] for u in UNITS}
    return merge_extractions([r0, r1], unit_sequence=seq)


def brief_from_merged(m: dict) -> dict:
    b = {k: json.loads(json.dumps(m[k])) for k in
         ("options", "criteria", "constraints", "claims", "positions", "objections",
          "blockers", "separate_issues", "actions")}
    return S._assign_server_ids(b)


def by(m, option, stance):
    return sorted(p["participant_label"] for p in m["positions"]
                  if p["option_title"] == option and p["stance"] == stance)


class FakeLLM:
    """Scripted stand-in for LLMClient: one handler per tool name."""

    def __init__(self, handlers):
        self.handlers = handlers
        self.calls: list[tuple[str, str]] = []

    async def extract(self, system, user, tool, timeout=60):
        name = tool["name"]
        self.calls.append((name, user))
        n = sum(1 for c in self.calls if c[0] == name)
        return self.handlers[name](user, n)

    def count(self, name):
        return sum(1 for c in self.calls if c[0] == name)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# chunker
# ---------------------------------------------------------------------------

def test_chunk_overlap_is_flagged_context_and_does_not_mutate_inputs():
    units = [{"id": f"u{i}", "sequence": i, "author_label": "A", "normalized_text": "w " * 50, "locator": {}}
             for i in range(1, 7)]
    chunks = chunker.chunk_units(units, max_words=120, overlap_units=2)
    assert [sum(1 for u in c if not u.get("context_only")) for c in chunks] == [2, 2, 2]
    assert not any(u.get("context_only") for u in chunks[0])
    assert [u["id"] for u in chunks[1][:2]] == ["u1", "u2"] and all(u["context_only"] for u in chunks[1][:2])
    assert not any("context_only" in u for u in units)            # originals untouched
    plain = chunker.chunk_units(units, max_words=120, overlap_units=0)
    assert not any(u.get("context_only") for c in plain for u in c)


# ---------------------------------------------------------------------------
# extractor validation
# ---------------------------------------------------------------------------

def test_validation_drops_unsupported_items_and_tags_the_rest():
    m = run_pipeline()
    reasons = sorted(d["reason"] for d in m["dropped_items"])
    assert reasons == ["no_valid_source", "quote_not_in_source", "speaker_mismatch"]
    # nothing from the dropped items leaks into the draft
    assert not any("finalise" in a["text"].lower() for a in m["actions"])
    assert not any(p["participant_label"] == "Omar" and p["option_title"] == "Casa Verde"
                   and ID("nadia_all") in p["source_unit_ids"] for p in m["positions"])


def test_participant_label_is_canonicalised_to_the_unit_author():
    units = [{"id": "a", "sequence": 1, "author_label": "Yusuf Khan", "normalized_text": "12th works",
              "locator": {}}]
    raw = _blank()
    raw["positions"] = [{"participant_label": "Yusuf", "option_title": "Dec 12", "stance": "support",
                         "source_unit_ids": ["a"], "quote": "12th works"}]
    out = validate_extraction(raw, units, 0)
    assert out["positions"][0]["participant_label"] == "Yusuf Khan"
    assert out["positions"][0]["_seq"] == 1 and out["positions"][0]["_chunk_index"] == 0


def test_quote_normalisation_handles_curly_quotes_and_whitespace():
    assert normalize_quote_text("I’ll  send it") == "i'll send it"


def test_extract_chunk_retries_once_then_raises_with_context():
    good = _blank()
    state = {"n": 0}

    def flaky(user, n):
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError("slow")
        return good

    llm = FakeLLM({"extract_candidates": flaky})
    out = run(extract_chunk(CHUNK0_UNITS, llm, chunk_index=4))
    assert llm.count("extract_candidates") == 2 and out["_chunk_index"] == 4

    def always_bad(user, n):
        raise ValueError("bad schema")

    with pytest.raises(RuntimeError, match="chunk 7"):
        run(extract_chunk(CHUNK0_UNITS, FakeLLM({"extract_candidates": always_bad}), chunk_index=7))


def test_prompt_marks_context_and_carries_known_items():
    llm = FakeLLM({"extract_candidates": lambda u, n: _blank()})
    run(extract_chunk(CHUNK1_UNITS, llm, chunk_index=1,
                      known_options=[{"title": "Dec 12", "decision": "date"}],
                      known_decisions=["date"],
                      known_open_items=[{"kind": "blocker", "text": "Tariq's flight for Dec 12 is not confirmed"}]))
    prompt = llm.calls[0][1]
    assert "[context]" in prompt and "KNOWN OPTIONS" in prompt and "KNOWN DECISIONS" in prompt
    assert "OPEN ITEMS FROM EARLIER SEGMENTS" in prompt and "Tariq's flight" in prompt


# ---------------------------------------------------------------------------
# merger: similarity guards
# ---------------------------------------------------------------------------

def test_numbers_and_months_stop_false_merges():
    assert not _similar("Budget cap $400 each", "Budget $2,400 total")
    assert not _same_option("Dec 5 (Saturday)", "Dec 12 (Saturday)")
    assert not _same_option("Option A", "Option B")
    assert _same_option("12th", "Dec 12 (Saturday)")
    assert _same_option("Casa Verde", "Casa Verde private room")
    assert _similar("Pay the $500 deposit by Oct 16", "Send the $500 deposit Monday")
    assert _similar("Tariq's flight is booked, landing 4pm on the 11th", "Tariq's flight is booked, landing 4pm on the 11th")


# ---------------------------------------------------------------------------
# merger: the anniversary regression
# ---------------------------------------------------------------------------

def test_anniversary_positions_survive_across_decisions():
    m = run_pipeline()
    assert sorted(o["title"] for o in m["options"]) == ["Casa Verde", "Dec 12", "Dec 5"]  # "12th" did not make a 4th
    assert by(m, "Dec 12", "support") == ["Hana", "Leila", "Nadia", "Omar", "Tariq", "Yusuf"]
    assert by(m, "Dec 5", "support") == []                       # Yusuf switched within the same decision
    assert any(s["participant_label"] == "Yusuf" and s["option_title"] == "Dec 5" for s in m["superseded_positions"])
    # venue votes were NOT wiped out by the date votes (old code kept one option per person)
    assert by(m, "Casa Verde", "support") == ["Hana", "Leila", "Nadia", "Omar", "Tariq"]
    assert by(m, "Casa Verde", "acknowledged") == ["Yusuf"]
    nadia = next(p for p in m["positions"] if p["participant_label"] == "Nadia" and p["option_title"] == "Casa Verde")
    assert nadia["stance"] == "support" and nadia["proposer"] is True


def test_deposit_action_keeps_its_owner_and_all_sources():
    m = run_pipeline()
    deposit = [a for a in m["actions"] if "deposit" in a["text"].lower()]
    assert len(deposit) == 1
    assert deposit[0]["owner_kind"] == "named" and deposit[0]["owner_label"] == "Nadia"
    assert {ID("leila_call"), ID("nadia_yes")} <= set(deposit[0]["source_unit_ids"])
    trim = next(a for a in m["actions"] if "guest list" in a["text"].lower())
    assert trim["owner_label"] == "Leila" and trim["due"] == "by Sunday"


def test_owner_survives_even_when_the_ownerless_version_comes_last():
    named = {"text": "Send the $500 deposit Monday", "owner_label": "Nadia", "owner_kind": "named",
             "source_unit_ids": ["a"], "_seq": 1, "_chunk_index": 0}
    ownerless = {"text": "Pay the $500 deposit by Oct 16", "owner_kind": "unassigned",
                 "source_unit_ids": ["b"], "_seq": 2, "_chunk_index": 0}
    m = merge_extractions([{"actions": [named, ownerless]}])
    assert len(m["actions"]) == 1 and m["actions"][0]["owner_label"] == "Nadia"


def test_blocker_resolution_follows_chronology():
    m = run_pipeline()
    assert len(m["blockers"]) == 1 and m["blockers"][0]["resolution_status"] == "resolved"

    def blk(status, seq):
        return {"text": "DPA unsigned", "kind": "unresolved_dependency", "resolution_status": status,
                "source_unit_ids": [f"u{seq}"], "_seq": seq, "_chunk_index": 0}

    reopened = merge_extractions([{"blockers": [blk("open", 1), blk("resolved", 2), blk("open", 3)]}])
    assert reopened["blockers"][0]["resolution_status"] == "open"      # old "resolved always wins" rule is gone


def test_sensitive_flag_and_claim_originator_are_kept():
    m = run_pipeline()
    assert next(c for c in m["constraints"] if "400 per person" in c["text"])["sensitive"] is True
    a = {"text": "Casa Verde private room fits 20 max", "kind": "factual", "author_label": "Leila",
         "source_unit_ids": ["a"], "_seq": 1, "_chunk_index": 0}
    b = {"text": "the room is 20 max", "kind": "factual", "author_label": "Nadia",
         "source_unit_ids": ["b"], "_seq": 2, "_chunk_index": 0}
    merged = merge_extractions([{"claims": [a, b]}])
    assert len(merged["claims"]) == 1 and merged["claims"][0]["author_label"] == "Leila"


def test_positions_are_ordered_by_unit_sequence_not_list_order():
    def pos(opt, seq):
        return {"participant_label": "Yusuf", "option_title": opt, "stance": "support",
                "source_unit_ids": [f"u{seq}"], "quote": "x", "_seq": seq, "_chunk_index": 0}

    chunk = {"options": [
        {"title": "Dec 5", "decision": "date", "status": "active", "source_unit_ids": ["u1"], "_seq": 1},
        {"title": "Dec 12", "decision": "date", "status": "active", "source_unit_ids": ["u1"], "_seq": 1}],
        "positions": [pos("Dec 12", 20), pos("Dec 5", 10)]}       # listed newest-first
    m = merge_extractions([chunk])
    assert [p["option_title"] for p in m["positions"]] == ["Dec 12"]


def test_acknowledged_never_downgrades_and_proposer_is_remembered():
    def pos(who, opt, stance, seq):
        return {"participant_label": who, "option_title": opt, "stance": stance,
                "source_unit_ids": [f"u{seq}"], "quote": "x", "_seq": seq, "_chunk_index": 0}

    chunk = {"options": [{"title": "Plan A", "decision": "plan", "status": "active", "source_unit_ids": ["u1"], "_seq": 1}],
             "positions": [pos("Marcus", "Plan A", "support", 1), pos("Marcus", "Plan A", "acknowledged", 2),
                           pos("Dev", "Plan A", "acknowledged", 3),
                           pos("Priya", "Plan A", "proposed", 4), pos("Priya", "Plan A", "support", 5)]}
    m = merge_extractions([chunk])
    stance = {p["participant_label"]: p for p in m["positions"]}
    assert stance["Marcus"]["stance"] == "support"
    assert stance["Dev"]["stance"] == "acknowledged"
    assert stance["Priya"]["stance"] == "support" and stance["Priya"]["proposer"] is True


def test_dropped_options_are_sticky_and_objections_against_them_are_moot():
    def opt(status, seq):
        return {"title": "Alma", "decision": "venue", "status": status, "source_unit_ids": [f"u{seq}"],
                "_seq": seq, "_chunk_index": 0}

    objection = {"text": "Alma's private room is too small", "option_titles": ["Alma"],
                 "resolution_status": "open", "source_unit_ids": ["u3"], "_seq": 3, "_chunk_index": 0}
    m = merge_extractions([{"options": [opt("active", 1), opt("dropped", 5), opt("active", 8)],
                            "objections": [objection]}])
    assert m["options"][0]["status"] == "dropped"          # merely mentioning it again does not revive it
    assert m["objections"][0]["moot"] is True
    revived = merge_extractions([{"options": [opt("active", 1), opt("dropped", 5), opt("reopened", 9)],
                                  "objections": [objection]}])
    assert revived["options"][0]["status"] == "active" and not revived["objections"][0].get("moot")


# ---------------------------------------------------------------------------
# synthesizer: counts, preference, banned words, restore, ids, quotes
# ---------------------------------------------------------------------------

def test_status_line_counts_come_from_the_state():
    brief = brief_from_merged(run_pipeline())
    line = S._compute_status_line(brief)
    assert line.startswith("3 options;")
    assert "Dec 12: explicit support from" in line and "(6)" in line
    assert "Casa Verde: explicit support from" in line and "(5)" in line
    assert "acknowledged only: Yusuf" in line
    assert "0 open objections" in line and "0 open blockers" in line
    assert "without a named owner" not in line               # Nadia's deposit is owned


def test_status_line_excludes_dropped_options_moot_objections_and_team_owned_actions():
    brief = {
        "options": [{"id": "a", "title": "A", "status": "active"}, {"id": "b", "title": "B", "status": "dropped"}],
        "positions": [{"participant_label": "X", "option_id": "b", "stance": "support"}],
        "objections": [{"resolution_status": "open", "moot": True}, {"resolution_status": "open"}],
        "blockers": [{"resolution_status": "resolved"}, {"resolution_status": "open"}],
        "actions": [{"text": "t", "owner_kind": "team"}, {"text": "u", "owner_kind": "named", "owner_label": "Z"}],
    }
    line = S._compute_status_line(brief)
    assert line == "1 option (1 dropped); 1 open objection; 1 open blocker; 1 action without a named owner."


def test_preference_is_computed_from_positions_with_a_strict_gate():
    brief = brief_from_merged(run_pipeline())
    pref = S._compute_preference(brief)
    assert "Dec 12 has explicit support from" in pref["summary"]
    assert all(n in pref["summary"] for n in ("Hana", "Leila", "Nadia", "Omar", "Tariq", "Yusuf"))
    assert "acknowledged only by Yusuf" in pref["summary"]            # the venue line
    assert not S._check_banned_words(pref["summary"])

    def mk(stances, objection=None, status="active"):
        b = {"options": [{"id": "a", "title": "A", "status": status}],
             "positions": [{"participant_label": n, "option_id": "a", "stance": s, "source_unit_ids": []}
                           for n, s in stances],
             "objections": [objection] if objection else []}
        return S._compute_preference(b)

    assert mk([("P", "support")]) is None                                       # one supporter
    assert mk([("P", "support"), ("Q", "acknowledged")]) is None                # acknowledged is not support
    assert mk([("P", "support"), ("Q", "support")]) is not None
    assert mk([("P", "support"), ("Q", "support")],
              {"option_ids": ["a"], "resolution_status": "open"}) is None       # open objection blocks it
    assert mk([("P", "support"), ("Q", "support")],
              {"option_ids": ["a"], "resolution_status": "open", "moot": True}) is not None
    assert mk([("P", "support"), ("Q", "support")], status="dropped") is None


def test_banned_words_match_whole_words_only():
    assert S._check_banned_words("They disagreed, stayed undecided, and the dog cleans up") == []
    assert S._check_banned_words("The group agreed.") == ["agreed"]
    assert S._check_banned_words("Effectively CONSENSUS (source)") == ["consensus", "effectively", "(source)"]


def test_verify_items_normalises_quotes_and_enforces_the_word_limit():
    ok = {"text": "t", "verbatim_quote": "I’ll send the $500 deposit Monday", "source_unit_id": ID("nadia_yes")}
    too_long = {"text": "t", "verbatim_quote": " ".join(["word"] * 16), "source_unit_id": "x"}
    wrong_unit = {"text": "t", "verbatim_quote": "booked my flight", "source_unit_id": ID("nadia_yes")}
    verified, rejected = S._verify_items([ok, too_long, wrong_unit], {**UNIT_TEXT, "x": " ".join(["word"] * 16)})
    assert verified == [ok] and len(rejected) == 2


def test_restore_dropped_repairs_what_the_synthesis_model_changed():
    merged = run_pipeline()
    llm_brief = {
        "options": [{"title": "Dec 12 (Saturday)", "decision": "date", "status": "active", "source_unit_ids": []},
                    {"title": "Casa Verde", "decision": "venue", "status": "active", "source_unit_ids": []}],
        "positions": [
            {"participant_label": "Yusuf", "option_title": "Casa Verde", "stance": "support", "source_unit_ids": []},
            {"participant_label": "Leila", "option_title": "Dec 12", "stance": "support", "source_unit_ids": []}],
        "blockers": [{"text": "Tariq's flight for Dec 12 is not confirmed", "kind": "unresolved_dependency",
                      "resolution_status": "open", "source_unit_ids": []}],
        "objections": [],
        "actions": [{"text": "Send the $500 deposit Monday", "owner_label": "Leila", "owner_kind": "named",
                     "source_unit_ids": []}],
    }
    notes = S._restore_dropped(llm_brief, merged)
    assert any("restored option 'Dec 5'" in n for n in notes)
    yusuf = next(p for p in llm_brief["positions"]
                 if p["participant_label"] == "Yusuf" and p["option_title"] == "Casa Verde")
    assert yusuf["stance"] == "acknowledged"                                   # "sure whatever" is not support
    assert any(p["participant_label"] == "Hana" and p.get("restored") for p in llm_brief["positions"])
    assert llm_brief["blockers"][0]["resolution_status"] == "resolved"        # chronology wins over the model
    deposit = next(a for a in llm_brief["actions"] if "deposit" in a["text"].lower())
    assert deposit["owner_label"] == "Nadia"                                   # invented owner replaced
    assert next(a for a in llm_brief["actions"] if "guest list" in a["text"].lower()).get("restored")


def test_assign_server_ids_resolves_loosely_worded_option_titles():
    brief = {"options": [{"title": "Dec 12 (Saturday)", "source_unit_ids": []}],
             "positions": [{"participant_label": "Leila", "option_title": "Dec 12", "stance": "support",
                            "source_unit_ids": []}],
             "objections": [{"text": "x", "option_titles": ["12th"], "resolution_status": "open",
                             "source_unit_ids": []}]}
    out = S._assign_server_ids(brief)
    oid = out["options"][0]["id"]
    assert out["positions"][0]["option_id"] == oid and out["objections"][0]["option_ids"] == [oid]


# ---------------------------------------------------------------------------
# executive summary: verification pipeline
# ---------------------------------------------------------------------------

def _anniversary_brief():
    b = brief_from_merged(run_pipeline())
    b["question"] = {"text": "When and where should the anniversary dinner be held?", "scope": None}
    b["emerging_preference"] = S._compute_preference(b)
    return b


def _entail(user, n):
    items = json.loads(user)
    return {"verdicts": [{"id": i["id"], "supported": "confirm his dec 12 conflict" not in i["text"].lower()}
                         for i in items]}


GOOD_ATTENTION = {"text": "Drinks are extra and their cost is unconfirmed (Leila)", "verbatim_quote": "drinks extra",
                  "source_unit_id": ID("leila_call")}
GOOD_COMMIT = {"text": "Nadia sends the $500 deposit Monday", "verbatim_quote": "I'll send the $500 deposit Monday",
               "source_unit_id": ID("nadia_yes")}
INVENTED_COMMIT = {"text": "Yusuf to confirm his Dec 12 conflict is cleared",
                   "verbatim_quote": "I moved my thing. 12th works", "source_unit_id": ID("yusuf_moved")}
FABRICATED = {"text": "Siblings must finalise the date by Oct 16",
              "verbatim_quote": "finalise and confirm the date no later than October 16",
              "source_unit_id": ID("leila_call")}
BANNED = {"text": "The group is converging on Dec 12", "verbatim_quote": "so the 12th works",
          "source_unit_id": ID("tariq_booked")}


def _exec_response(**sections):
    base = {"title": "-", "status": "-", "needs_attention": [], "proposed": [], "committed": [],
            "unverified_claims": [], "preference": "Dec 5 is ruled out"}
    base.update(sections)
    return base


def test_exec_summary_removes_invented_fabricated_and_banned_items_then_retries():
    brief = _anniversary_brief()

    def exec_handler(user, n):
        if n == 1:
            return _exec_response(needs_attention=[GOOD_ATTENTION, FABRICATED], proposed=[BANNED],
                                  committed=[GOOD_COMMIT, INVENTED_COMMIT])
        assert "removed" in user                                          # retry feedback was sent
        return _exec_response(needs_attention=[GOOD_ATTENTION], committed=[GOOD_COMMIT])

    llm = FakeLLM({"produce_exec_summary": exec_handler, "check_support": _entail})
    report: dict = {}
    text = run(S._generate_exec_summary(brief, llm, UNIT_TEXT, unit_meta=UNIT_META, report=report))

    assert llm.count("produce_exec_summary") == 2
    assert "Drinks are extra" in text and "Nadia sends the $500 deposit Monday" in text
    for gone in ("finalise", "converging", "conflict is cleared", "ruled out"):
        assert gone not in text
    assert text.splitlines()[0] == "When and where should the anniversary dinner be held?"   # title is deterministic
    assert S._compute_status_line(brief) in text                                              # status is deterministic
    assert "[Leila, 10/5/26, 9:48 AM]" in text and "[Nadia, 10/6/26, 1:31 PM]" in text         # readable refs
    assert "[m" not in text
    assert "Dec 12 has explicit support from" in text.split("PREFERENCE")[1]                  # preference from code
    assert report["verified"] is True and report["attempts"] == 2 and report["omitted_items"] == 0


def test_exec_summary_fails_closed_when_the_model_keeps_misbehaving():
    brief = _anniversary_brief()
    decided = {"text": "The group decided on Dec 12", "verbatim_quote": "so the 12th works",
               "source_unit_id": ID("tariq_booked")}

    def always_bad(user, n):
        return _exec_response(needs_attention=[GOOD_ATTENTION], committed=[GOOD_COMMIT, INVENTED_COMMIT],
                              proposed=[decided])

    llm = FakeLLM({"produce_exec_summary": always_bad, "check_support": _entail})
    report: dict = {}
    text = run(S._generate_exec_summary(brief, llm, UNIT_TEXT, unit_meta=UNIT_META, report=report))
    assert "decided" not in text and "conflict is cleared" not in text
    assert "Drinks are extra" in text                                  # good content is kept
    assert report["omitted_items"] >= 2
    assert any("banned" in i for i in report["issues"]) and any("omitted" in i for i in report["issues"])


def test_exec_summary_reports_when_the_support_check_is_unavailable():
    brief = _anniversary_brief()

    def boom(user, n):
        raise RuntimeError("llm down")

    llm = FakeLLM({"produce_exec_summary": lambda u, n: _exec_response(needs_attention=[GOOD_ATTENTION]),
                   "check_support": boom})
    report: dict = {}
    run(S._generate_exec_summary(brief, llm, UNIT_TEXT, unit_meta=UNIT_META, report=report))
    assert report["verified"] is False
    assert any("entailment check unavailable" in i for i in report["issues"])


def test_sensitive_source_messages_are_tagged_for_the_summary_model():
    brief = _anniversary_brief()
    llm = FakeLLM({"produce_exec_summary": lambda u, n: _exec_response(), "check_support": _entail})
    run(S._generate_exec_summary(brief, llm, UNIT_TEXT, unit_meta=UNIT_META, report={}))
    prompt = llm.calls[0][1]
    hana_line = next(l for l in prompt.splitlines() if l.startswith(f"{ID('hana_ok')}:"))
    assert "[SENSITIVE" in hana_line
    nadia_line = next(l for l in prompt.splitlines() if l.startswith(f"{ID('nadia_yes')}:"))
    assert "[SENSITIVE" not in nadia_line


def test_render_falls_back_to_a_short_id_without_metadata():
    out = S._render_verified_summary(_exec_response(needs_attention=[GOOD_ATTENTION], preference="p"))
    assert "[mu-leil]" in out


# ---------------------------------------------------------------------------
# extract job: carry-forward across chunks
# ---------------------------------------------------------------------------

def test_run_extract_brief_carries_options_and_open_items_forward(monkeypatch):
    filler = " filler" * 1200                                    # one unit per chunk -> 3 chunks
    rows = [SimpleNamespace(id=uuid.uuid4(), sequence=i + 1, author_label=a, normalized_text=t + filler,
                            locator={"timestamp_str": f"t{i + 1}"})
            for i, (a, t) in enumerate([("Nadia", "Dec 12 or Dec 5 for the dinner"),
                                        ("Tariq", "the 12th might work if I can get a flight"),
                                        ("Tariq", "booked my flight, landing on the 11th")])]
    ids = [str(r.id) for r in rows]
    blocker_text = "Tariq's flight for Dec 12 is not confirmed"

    def handler(user, n):
        out = _blank()
        if n == 1:
            out["options"] = [{"temp_id": "o", "title": "Dec 12", "decision": "date", "status": "active",
                               "source_unit_ids": [ids[0]], "quote": "Dec 12 or Dec 5"}]
        elif n == 2:
            out["blockers"] = [{"text": blocker_text, "kind": "unresolved_dependency",
                                "resolution_status": "open", "source_unit_ids": [ids[1]],
                                "quote": "if I can get a flight"}]
        else:
            out["blockers"] = [{"text": blocker_text, "kind": "unresolved_dependency",
                                "resolution_status": "resolved", "source_unit_ids": [ids[2]],
                                "quote": "booked my flight"}]
        return out

    llm = FakeLLM({"extract_candidates": handler})
    session = MagicMock()
    session.execute = AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows)))
    monkeypatch.setattr(S, "select", lambda *a, **k: MagicMock())
    monkeypatch.setattr(S, "SourceUnit", MagicMock())
    fake_llm_module = types.ModuleType("pact.services.llm")
    fake_llm_module.LLMClient = lambda: llm
    monkeypatch.setitem(sys.modules, "pact.services.llm", fake_llm_module)

    out = run(S.run_extract_brief({"workspace_id": str(uuid.uuid4())}, session))

    assert out["chunk_count"] == 3
    prompts = [c[1] for c in llm.calls]
    assert "KNOWN OPTIONS" in prompts[1] and "Dec 12" in prompts[1]
    assert "OPEN ITEMS FROM EARLIER SEGMENTS" in prompts[2] and blocker_text in prompts[2]
    assert "[context]" in prompts[1] and "[context]" in prompts[2]
    blockers = out["merged_draft"]["blockers"]
    assert len(blockers) == 1 and blockers[0]["resolution_status"] == "resolved"
