"""Regression test: the member-portal launch chat (tests/fixtures/portal_launch_chat.txt, a Teams export).

Scripted per-chunk extractions follow the extractor prompt's rules; the synthesis script
falls into the traps a plain summarizer falls into on this chat. Assertions follow
eval/answer_key_portal.json.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from eval.score_brief import score_brief
from pact.worker.chunker import chunk_units
from pact.worker.extractor import validate_extraction
from pact.worker.merger import merge_extractions
from pact.worker.parser import _detect_format, _parse_whatsapp
from pact.worker.synthesizer import _compute_status_line

from .scripted_llm import ScriptedLLM, cited as _cited, run_pipeline

HERE = Path(__file__).resolve().parent
CHAT = (HERE / "fixtures" / "portal_launch_chat.txt").read_text()
KEY = json.loads((HERE.parent / "eval" / "answer_key_portal.json").read_text())

PRIYA, MARCUS, ELENA, ROSA, JON, DEV, HAL = KEY["participants"]
A, B, C = "Nov 16 launch with Acme", "Nov 16 with own email OTP, switch to Acme later", "Slip to January"


def L(line: int) -> str:
    """Unit id for a 1-based line of the chat file (lines 1-2 are the export header)."""
    return f"u{line - 3}"


def units():
    us = _parse_whatsapp(CHAT)
    for u in us:
        u["id"] = f"u{u['sequence']}"
    return us


def opt(title, line, quote, decision="launch plan", label=None, status="active", reason=None):
    o = {"temp_id": title, "title": title, "decision": decision, "status": status,
         "source_unit_ids": [L(line)], "quote": quote}
    if label:
        o["label"] = label
    if reason:
        o["status_reason"] = reason
    return o


def pos(who, title, stance, line, quote, conditions=None):
    p = {"participant_label": who, "option_title": title, "stance": stance, "source_unit_ids": [L(line)], "quote": quote}
    if conditions:
        p["conditions"] = conditions
    return p


def item(text, line, quote, **kw):
    return {"text": text, "source_unit_ids": [L(line)], "quote": quote, **kw}


def act(text, owner, line, quote, **kw):
    kw.setdefault("owner_kind", "named" if owner else "unassigned")
    return item(text, line, quote, **({"owner_label": owner} if owner else {}), **kw)


def empty():
    return {k: [] for k in ("candidate_questions", "options", "criteria", "constraints", "claims",
                            "positions", "objections", "blockers", "separate_issues", "actions")}


# --------------------------------------------------------------------------- scripted extractions

def chunk0():                                            # lines 3-32
    r = empty()
    r["candidate_questions"] = [item("Which launch plan for the member portal?", 3, "We need to lock the plan for the member portal launch")]
    r["options"] = [
        opt(A, 14, "A) Nov 16 with Acme as planned", label="A"),
        opt(B, 14, "B) Nov 16 with our own email OTP verification, switch to Acme later", label="B"),
        opt(C, 14, "C) slip to January", label="C"),
        opt("C", 23, "so C is out", status="dropped", reason="must be live before the Dec 15 change freeze"),
    ]
    r["positions"] = [                                   # "Options as I see them" lists choices: no stance
        pos(HAL, "C", "oppose", 22, "January is a non-starter"),
        pos(PRIYA, "C", "oppose", 23, "so C is out"),
        pos(MARCUS, "C", "oppose", 24, "+1"),
        pos(ELENA, "C", "oppose", 25, "fine with dropping C"),
        pos(ROSA, "C", "oppose", 26, "me too"),
    ]
    r["claims"] = [
        item("Portal code complete by Nov 6 if nothing slips", 4, "Code complete for the portal by Nov 6", kind="predictive", author_label=MARCUS),
        item("Acme IDV integration is done in staging", 4, "Acme IDV integration is done in staging", kind="factual", author_label=MARCUS),
        item("Acme's rep said the SOC 2 report would come Oct 23", 6, "Their rep told me Oct 23", kind="predictive", author_label=ELENA),
        item("Data pipeline run costs are in the ops budget, not the launch budget", 12, "those are in ops budget", kind="factual", author_label=JON),
        item("B is about 3 weeks of work", 15, "B is about 3 weeks of work", kind="predictive", author_label=MARCUS),
        item("B is more like 5 weeks with account recovery", 16, "more like 5 once you build the account recovery flows", kind="predictive", author_label=DEV),
        item("B costs about $40k extra", 19, "I'd guess $40k for the extra build", kind="factual", author_label=JON),
        item("B costs closer to $65k extra, putting the launch at $215k", 27, "closer to $65k, not $40k", kind="factual", author_label=JON),
    ]
    r["constraints"] = [
        item("No member data to Acme until the DPA is signed", 9, "we can't go live with member data flowing to them without it signed", kind="mandatory", author_label=ROSA),
        item("Launch budget is $180k all in (about $150k committed)", 10, "the launch envelope is $180k all in", kind="mandatory", author_label=JON),
        item("B must include account recovery", 18, "absolutely not skipping recovery", kind="mandatory", author_label=ELENA),
        item("Must be live before the Dec 15 change freeze", 22, "exec team wants this live before the Dec 15 change freeze", kind="mandatory", author_label=HAL),
    ]
    r["objections"] = [item("Skipping account recovery in B invites account takeover", 18, "that's how accounts get taken over",
                            option_titles=[B], author_label=ELENA, resolution_status="open")]
    r["blockers"] = [
        item("Acme SOC 2 Type II report not received", 6, "Acme's SOC 2 Type II report isn't in yet", kind="missing_info", resolution_status="open"),
        item("Acme DPA not signed", 7, "the DPA with Acme is not signed", kind="unresolved_dependency", resolution_status="open"),
    ]
    return r


def chunk1():                                            # lines 33-62, context 30-32
    r = empty()
    r["options"] = [
        opt(A, 51, "go with A. Nov 16 launch with Acme", label="A"),
        opt("B", 56, "so B is a fallback pending Hal's approval", status="fallback", reason="pending Hal's approval"),
    ]
    r["positions"] = [                                   # "compliance first" is a criterion, not an option
        pos(PRIYA, "A", "proposed", 51, "Proposal: go with A"),
        pos(MARCUS, "A", "support", 52, "I'm good with that"),
        pos(ROSA, "A", "support", 53, "Good with A and the checkpoint"),
        pos(ELENA, "A", "conditional", 54, "A is fine for me, as long as I see the SOC 2 report in writing",
            "sees the SOC 2 report in writing first"),
        pos(DEV, "A", "acknowledged", 57, "fine"),
    ]                                                    # 59-61 reply to a bundled summary: tasks, not votes
    r["criteria"] = [item("Compliance first, speed second", 43, "compliance first, speed second")]
    r["claims"] = [
        item("Marcus said the portal is done by Nov 16", 34, "Marcus said we're done by Nov 16", kind="other", author_label=HAL),
        item("Launch also needs security sign-off and the signed DPA", 35, "launch also needs security sign-off and the DPA", kind="factual", author_label=MARCUS),
        item("The team is aligned on A with the Oct 30 checkpoint", 58, "ok so we're aligned", kind="other", author_label=PRIYA),
    ]
    r["constraints"] = [
        item("Pen-test finding #3 must be closed before any launch", 37, "has to be closed before any launch, whichever vendor", kind="mandatory", author_label=ELENA),
        item("Go/no-go checkpoint Oct 30: SOC 2 in hand, DPA signed, finding #3 on track, else B", 51, "Go/no-go checkpoint Oct 30", kind="mandatory", author_label=PRIYA),
        item("Elena will not sign off without the SOC 2 report in writing", 54, "I won't sign off on a rep's word", kind="mandatory", author_label=ELENA),
        item("Jon cannot approve B at $215k; it needs Hal", 55, "I can't approve B at $215k", kind="mandatory", author_label=JON),
        item("Hal needs the funding ask for B by Oct 30", 62, "if B needs money, send me the ask by Oct 30", kind="mandatory", author_label=HAL),
    ]
    r["objections"] = [item("B at $215k is over the cap; Jon can't approve it", 55, "I can't approve B at $215k",
                            option_titles=["B"], author_label=JON, resolution_status="open")]
    r["blockers"] = [item("Pen-test finding #3 open on the auth service", 37, "One open finding on the existing auth service", kind="unresolved_dependency", resolution_status="open")]
    r["actions"] = [
        act("Dev's team fixes the token issue for finding #3", DEV, 40, "I can start tomorrow", acceptance="committed"),
        act("Dev's team fixes the token issue for finding #3", DEV, 42, "ok", acceptance="committed"),
        act("Rosa gets the Acme DPA signed", ROSA, 53, "I'll chase the DPA", due="before Oct 20", acceptance="committed"),
        act("Run the Oct 30 go/no-go checkpoint", None, 51, "Go/no-go checkpoint Oct 30", due="Oct 30"),
        act("Take the B funding ask to Hal", None, 55, "someone has to take that to Hal"),
        act("Marcus delivers code complete", MARCUS, 59, "confirmed", due="Nov 6", acceptance="committed"),
        act("Elena reviews the Acme SOC 2 report", ELENA, 61, "yes", acceptance="committed"),
    ]
    return r


def chunk2():                                            # lines 63-70, context 60-62
    r = empty()
    r["positions"] = [pos(JON, "A", "acknowledged", 64, "I want the $180k cap noted as the budget for A")]
    r["constraints"] = [item("Launch budget for A is the $180k cap", 64, "I want the $180k cap noted as the budget for A", kind="mandatory", author_label=JON)]
    r["claims"] = [
        item("Acme sent the SOC 2 Type II report Oct 9", 66, "Acme sent the SOC 2 Type II report today", kind="factual", author_label=ELENA),
        item("One DPA clause (data retention) is still open", 67, "One clause still open, data retention", kind="factual", author_label=ROSA),
        item("Token fix for #3 is on track", 68, "token fix started Wednesday, on track", kind="factual", author_label=DEV),
    ]
    r["blockers"] = [
        item("Acme SOC 2 Type II report not received", 66, "Acme sent the SOC 2 Type II report today", kind="missing_info", resolution_status="resolved"),
        item("SOC 2 report review not finished", 66, "I'll finish the review by Oct 14", kind="unresolved_dependency", resolution_status="open"),
        item("Acme DPA not signed", 67, "One clause still open, data retention", kind="unresolved_dependency", resolution_status="open"),
    ]
    r["actions"] = [act("Elena reviews the Acme SOC 2 report", ELENA, 66, "I'll finish the review by Oct 14", due="Oct 14", acceptance="committed")]
    return r


CHUNKS = (chunk0, chunk1, chunk2)


def merged_draft():
    us = units()
    chunks = chunk_units(us)
    assert len(chunks) == 3
    results = [validate_extraction(copy.deepcopy(f()), c, i) for i, (f, c) in enumerate(zip(CHUNKS, chunks))]
    for r in results:
        assert r["_dropped"] == [], r["_dropped"]
    return merge_extractions(results, unit_sequence={u["id"]: u["sequence"] for u in us})


def by_option(m, title):
    return {p["participant_label"]: p["stance"] for p in m["positions"] if p["option_title"] == title}


# --------------------------------------------------------------------------- data

def test_parse_teams_export():
    assert _detect_format(CHAT, "") == "whatsapp"        # same line shape: "[date time] Name: text"
    us = units()
    assert len(us) == 68 and us[0]["author_label"] == PRIYA and us[-1]["normalized_text"] == "👍"


def test_labelled_options_merge():
    opts = {o["title"]: o for o in merged_draft()["options"]}
    assert set(opts) == {A, B, C}
    assert opts[A]["label"] == "A" and opts[A]["status"] == "active"
    assert opts[B]["status"] == "fallback" and opts[B]["status_reason"] == "pending Hal's approval"
    assert opts[C]["status"] == "dropped" and "freeze" in opts[C]["status_reason"]


def test_positions():
    m = merged_draft()
    assert by_option(m, A) == {PRIYA: "proposed", MARCUS: "support", ROSA: "support", ELENA: "conditional",
                               DEV: "acknowledged", JON: "acknowledged"}   # Hal: "whichever" is not a vote
    assert by_option(m, B) == {}


def test_objection_stays_on_option_b():
    """B's title shares 'Nov 16' and 'Acme' with A; the objection must not move to A."""
    objs = merged_draft()["objections"]
    assert len(objs) == 2 and all(o["option_titles"] == [B] for o in objs)


def test_actions_and_blockers():
    m = merged_draft()
    acts = {a["text"]: a for a in m["actions"]}
    assert acts["Elena reviews the Acme SOC 2 report"]["due"] == "Oct 14"
    assert acts["Rosa gets the Acme DPA signed"]["due"] == "before Oct 20"
    assert acts["Marcus delivers code complete"]["due"] == "Nov 6"
    assert acts["Run the Oct 30 go/no-go checkpoint"]["owner_kind"] == "unassigned"   # Priya proposed it, nobody runs it
    ask = acts["Take the B funding ask to Hal"]
    assert ask["owner_kind"] == "unassigned" and ask["deadline"] == "Oct 30"
    assert ask["deadline_source_unit_ids"] == [L(62)]                    # per Hal
    open_b = sorted(b["text"] for b in m["blockers"] if b["resolution_status"] == "open")
    assert open_b == ["Acme DPA not signed", "Pen-test finding #3 open on the auth service", "SOC 2 report review not finished"]


# --------------------------------------------------------------------------- full pipeline, scored

def synthesized_with_traps(m):
    b = copy.deepcopy({k: m[k] for k in ("criteria", "constraints", "options", "claims", "positions",
                                         "objections", "blockers", "separate_issues", "actions")})
    for o in b["options"]:
        if o["title"] == C:
            o["status"] = "active"                                         # January still live
        if o["title"] == B:
            o["status"] = "dropped"                                        # fallback read as rejected
    a = next(o for o in b["options"] if o["title"] == A)
    a["title"] = "Option A — Nov 16 launch with Acme"                      # renamed
    for p in b["positions"]:
        if p["participant_label"] == ELENA and p["option_title"] == A:
            p["stance"] = "support"                                        # condition dropped
    for who in (JON, DEV, HAL):                                            # from "we're aligned" / "fine" / "whichever"
        b["positions"].append({"participant_label": who, "option_title": A, "stance": "support", "source_unit_ids": [L(58)]})
    for bl in b["blockers"]:
        if bl["text"].startswith("SOC 2 report review"):
            bl["resolution_status"] = "resolved"                            # "received" read as "cleared"
    for x in b["actions"]:
        if x["text"] == "Take the B funding ask to Hal":
            x["owner_kind"], x["owner_label"] = "named", PRIYA             # invented owner
    b["separate_issues"] += [{"title": "On-call Thursday", "rationale": "", "source_unit_ids": [L(32)]}]
    return b


def cited(text, line, quote):
    return _cited(L, text, line, quote)


GOOD_SUMMARY = {                                        # the user's card for this chat
    "title": "-", "status": "-",
    "needs_attention": [
        cited("Blocker: Acme DPA unsigned, one clause open (data retention); no member data can flow until "
              "it's signed, and Rosa is out from Oct 20", 67, "One clause still open, data retention"),
        cited("Blocker: SOC 2 report arrived Oct 9, but Elena's review isn't finished (due Oct 14); "
              "she won't sign off without it", 66, "I'll finish the review by Oct 14"),
        cited("Blocker: pen test finding #3 must close before any launch; fix started, Dev hasn't agreed to Nov 6",
              37, "has to be closed before any launch, whichever vendor"),
        cited("No owner: funding ask to Hal if B is triggered (~$215k vs. $180k cap), due Oct 30", 55,
              "someone has to take that to Hal"),
        cited("No owner: running the Oct 30 go/no-go checkpoint (Priya proposed it)", 51, "Go/no-go checkpoint Oct 30"),
    ],
    "proposed": [
        cited("A, Nov 16 with Acme (Priya): explicit support from Marcus, Rosa and Elena (conditional on the "
              "written SOC 2 report); Dev and Jon only acknowledged; Hal backs whichever, as long as it's live "
              "before the Dec 15 freeze", 51, "Proposal: go with A"),
        cited("B, in-house OTP, fallback only: Jon objects (over cap); Elena objects to skipping recovery flows",
              56, "so B is a fallback pending Hal's approval"),
        cited("C, January: dropped", 23, "so C is out"),
    ],
    "committed": [
        cited("Marcus: code complete Nov 6", 59, "confirmed"),
        cited("Rosa: DPA signed before Oct 20", 53, "I'll chase the DPA"),
        cited("Elena: SOC 2 review by Oct 14", 66, "I'll finish the review by Oct 14"),
        cited("Dev: fix #3 (started, no due date agreed)", 40, "I can start tomorrow"),
    ],
    "unverified_claims": [
        cited("B effort: 3 weeks (Marcus) vs. 5 (Dev), disputed", 16, "more like 5 once you build the account recovery flows"),
        cited("B cost $65k, revised from $40k (Jon), an estimate", 27, "closer to $65k, not $40k"),
    ],
}


def run(summary):
    llm = ScriptedLLM([f() for f in CHUNKS], synthesized_with_traps, summary)
    return run_pipeline(units(), llm, KEY["question"])


def failed(result):
    return [c["name"] for c in result["checks"] if not c["passed"]]


def test_pipeline_scores_full_marks_on_answer_key():
    brief = run(GOOD_SUMMARY)
    assert brief["executive_summary_report"]["omitted_items"] == 0
    result = score_brief(brief, KEY)
    assert failed(result) == [], failed(result)
    status = _compute_status_line(brief)
    # the user's card: "3 options proposed (1 dropped), 3 explicit supporters of Option A (1 conditional),
    # 2 open objections (both on B), 3 blockers, 2 actions without an owner"
    assert status.startswith("3 options proposed (1 dropped: C) Slip to January")
    assert ("launch plan — A) Nov 16 launch with Acme: 3 explicit supporters (1 conditional): "
            "Marcus Webb, Rosa Alvarez, Elena Ruiz (conditional); proposed by Priya Nair; "
            "acknowledged only: Dev Patel, Jon Becker; no stance on launch plan: Hal Okonkwo") in status
    assert "B) Nov 16 with own email OTP, switch to Acme later: fallback (pending Hal's approval)" in status
    assert "2 open objections (on B)" in status
    assert "3 open blockers" in status and "2 open actions without a named owner" in status
    # "Option A has explicit support from Marcus, Rosa and Elena (conditional), plus Priya" — but two open
    # objections on B do not block A's line, and A has no open objection of its own.
    pref = brief["emerging_preference"]["summary"]
    assert pref == (f"{A} has explicit support from Marcus Webb, Rosa Alvarez and Elena Ruiz (conditional), "
                    "plus Priya Nair (proposer); acknowledged only by Dev Patel and Jon Becker.")


def test_card_traps_are_caught():
    bad = copy.deepcopy(GOOD_SUMMARY)
    bad["proposed"][0] = cited("A) Nov 16 with Acme: the team is aligned", 58, "ok so we're aligned")
    bad["committed"].append(cited("Hal approves funding for B", 62, "I'll back whichever you pick"))
    bad["needs_attention"] = bad["needs_attention"][:3]                    # both no-owner items lost
    bad["unverified_claims"] = [cited("B costs $40k", 19, "I'd guess $40k for the extra build")]
    names = failed(score_brief(run(bad), KEY))
    for expected in ("committed does not match /hal.*(approv|fund)/", "needs-attention raises /no owner|owner not named/",
                     "needs-attention raises /checkpoint/",
                     "/\\baligned\\b/ is not stated as fact", "unverified section flags /\\$65k/"):
        assert expected in names, (expected, names)
