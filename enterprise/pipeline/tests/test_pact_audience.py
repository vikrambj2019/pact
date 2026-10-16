"""Email / transcript ingestion, audience tracking and viewer-specific summaries.

Fixtures are sanitised copies of real internal files (addresses moved to dummycompany.com).
"""
import asyncio
import json
import pathlib

from pact.services import audience as A
from pact.services import extractor as X
from pact.services import ingest as I
from pact.services import synthesizer as S

FIX = pathlib.Path(__file__).parent / "fixtures"


def fx(name):
    return (FIX / name).read_text(encoding="utf-8")


def run(coro):
    return asyncio.run(coro)


class FakeLLM:
    def __init__(self, handlers):
        self.handlers, self.calls = handlers, []

    async def extract(self, system, user, tool, timeout=60):
        self.calls.append((tool["name"], user))
        n = sum(1 for c in self.calls if c[0] == tool["name"])
        return self.handlers[tool["name"]](user, n)


def with_ids(units):
    return [{**u, "id": f"u{u['sequence']}"} for u in units]


# ---------------------------------------------------------------------------
# source detection and email parsing
# ---------------------------------------------------------------------------

def test_source_types_are_detected():
    assert I.detect_source_type(fx("ent-07-email-chain-1.txt")) == "email"
    assert I.detect_source_type(fx("ent-04-email-thread-example1.txt")) == "email"
    assert I.detect_source_type(fx("ent-05-meeting-transcript-1.txt")) == "teams_transcript"
    assert I.detect_source_type("9/14/26, 7:40 PM - Sam: hello") == "chat"


def test_email_chain_becomes_twelve_messages_plus_a_cover_note_oldest_first():
    units = I.email_units(fx("ent-07-email-chain-1.txt"))
    assert len(units) == 13
    sent = [u["locator"]["sent_at"] for u in units[:12]]
    assert sent == sorted(sent)
    kinds = [u["locator"]["msg_kind"] for u in units]
    assert kinds[0] == "new" and kinds[5] == "forward" and kinds[11] == "forward" and kinds[12] == "cover_note"
    assert units[-1]["locator"]["audience_unknown"] and units[-1]["author_label"] == "Unknown"
    assert [u["sequence"] for u in units] == list(range(1, 14))


def test_audience_grows_with_every_message_that_quotes_a_mail():
    units = I.email_units(fx("ent-07-email-chain-1.txt"))
    first, side, last_fwd = units[0]["locator"], units[6]["locator"], units[11]["locator"]
    assert len(first["audience"]) == 9 and "Lavu, Ratnakar" not in first["audience"]      # original To + Cc
    assert "Lavu, Ratnakar" in first["effective_audience"]                                # the forward carried it to him
    assert "Chennuru, Ashok" in first["effective_audience"]                               # ... and on to Ashok
    assert "Weber, Kyle" not in side["effective_audience"]                                # the wide group never saw the side thread
    assert sorted(last_fwd["effective_audience"]) == ["Chennuru, Ashok", "Lavu, Ratnakar", "Smith, Monique"]


def test_a_privacy_request_covers_everything_quoted_beneath_it():
    units = I.email_units(fx("ent-07-email-chain-1.txt"))
    assert all(u["locator"]["restricted"] for u in units)
    assert "keep it to yourself" in units[0]["locator"]["restricted_reason"]
    assert "cover note" in units[-1]["locator"]["restricted_reason"]
    # a thread without any such request is not restricted
    assert not any(u["locator"]["restricted"] for u in I.email_units(fx("ent-04-email-thread-example1.txt"))[:5])


def test_late_recipients_can_read_the_history_quoted_under_their_first_mail():
    units = I.email_units(fx("ent-04-email-thread-example1.txt"))
    assert len(units) == 6 and units[-1]["locator"]["audience_unknown"]
    first = units[0]["locator"]
    assert "Vasantham, Maddy" not in first["audience"] and "Vasantham, Maddy" in first["effective_audience"]
    assert len(units[4]["locator"]["audience"]) == 10                                     # Monisha's mail went to ten people


def test_the_same_message_quoted_twice_is_one_unit():
    mail = ("From: A, Ann <ann@x.test>\nSent: Monday, October 5, 2026 9:00 AM\nTo: B, Bob <bob@x.test>\nSubject: Re: Plan\n"
            "Looks fine to me.\n\n"
            "From: B, Bob <bob@x.test>\nSent: Monday, October 5, 2026 8:00 AM\nTo: A, Ann <ann@x.test>\nSubject: Plan\n"
            "Shall we meet on the 12th?\n\n"
            "From: B, Bob <bob@x.test>\nSent: Monday, October 5, 2026 8:00 AM\nTo: A, Ann <ann@x.test>\nSubject: Plan\n"
            "Shall we meet on the 12th?\n")
    units = I.email_units(mail)
    assert [u["author_label"] for u in units] == ["B, Bob", "A, Ann"]


# ---------------------------------------------------------------------------
# Teams transcripts
# ---------------------------------------------------------------------------

def test_teams_transcript_turns_speakers_and_times():
    units = I.parse_teams_transcript(fx("ent-05-meeting-transcript-1.txt"))
    assert len(units) == 48
    assert {u["author_label"] for u in units} == {
        "Bandugula, Vikram", "Benjamin, Danielle T.", "Camarro, Jennifer", "Garnett, Robert T.",
        "Haskins, James", "O'Hara, Brandon M."}
    assert units[0]["locator"]["timestamp_str"] == "0:04" and units[0]["normalized_text"] == "Yeah."
    assert all(a["author_label"] != b["author_label"] for a, b in zip(units, units[1:]))   # same-speaker blocks merged
    for u in units:                                                                       # no export decoration left in the text
        assert " minutes " not in u["normalized_text"][:40] and "Transcript." not in u["normalized_text"]


def test_a_jump_in_an_excerpted_transcript_is_flagged():
    units = I.parse_teams_transcript(fx("ent-06-meeting-transcript-2.txt"))
    assert len(units) == 29 and units[0]["author_label"] == "Robinson, Edwige"
    gaps = [(u["sequence"], u["locator"]["gap_before_s"]) for u in units if "gap_before_s" in u["locator"]]
    assert gaps == [(10, 3785)] and units[9]["locator"]["timestamp_str"] == "1:03:40"


# ---------------------------------------------------------------------------
# extraction prompt
# ---------------------------------------------------------------------------

def test_prompt_shows_recipients_restrictions_and_gaps():
    units = with_ids(I.email_units(fx("ent-07-email-chain-1.txt")))
    text = X._format_units([units[6]])
    assert "[email reply]" in text and "To: Tanal, Stephen" in text and "Cc: Kaye, Mark" in text and "[RESTRICTED]" in text
    assert "recipients unknown" in X._format_units([units[12]])
    talk = with_ids(I.parse_teams_transcript(fx("ent-06-meeting-transcript-2.txt")))
    assert "[GAP: about 63 minutes" in X._format_units([talk[9]]) and "GAP" not in X._format_units([talk[8]])
    assert "EMAIL units" in X._SYSTEM and "MEETING TRANSCRIPT units" in X._SYSTEM


def test_email_extraction_is_validated_against_the_sender():
    units = with_ids(I.email_units(fx("ent-07-email-chain-1.txt")))
    lavu = units[6]                                        # sequence 7: Lavu's reply
    quote = " ".join(lavu["normalized_text"].split()[:6])
    assert X.quote_in_text(quote, lavu["normalized_text"])
    result = {s: [] for s in X.SECTIONS}
    result["positions"] = [
        {"participant_label": "Ratnakar Lavu", "option_title": "x", "stance": "conditional", "quote": quote,
         "source_unit_ids": [lavu["id"]]},
        {"participant_label": "Stephen Tanal", "option_title": "x", "stance": "support", "quote": quote,
         "source_unit_ids": [lavu["id"]]},                 # Tanal is not the sender
    ]
    out = X.validate_extraction(result, [lavu], chunk_index=0)
    assert [p["participant_label"] for p in out["positions"]] == ["Lavu, Ratnakar"]
    assert out["_dropped"][0]["reason"] == "speaker_mismatch"


# ---------------------------------------------------------------------------
# audience helpers
# ---------------------------------------------------------------------------

def _meta(units):
    return {u["id"]: {"author": u["author_label"], "ts": u["locator"]["timestamp_str"],
                      "effective_audience": u["locator"]["effective_audience"],
                      "audience_unknown": u["locator"]["audience_unknown"],
                      "restricted": u["locator"]["restricted"]} for u in units}


def test_name_matching_ignores_order_punctuation_and_case():
    assert A.same_person("Lavu, Ratnakar", "Ratnakar Lavu") and A.same_person("ratnakar.lavu", "Lavu, Ratnakar")
    assert not A.same_person("Smith, Monique", "Smith, Paul")


def test_readers_are_the_people_who_saw_every_cited_message():
    units = with_ids(I.email_units(fx("ent-07-email-chain-1.txt")))
    meta = _meta(units)
    names, unknown = A.readers(meta, [u["id"] for u in units])
    assert sorted(names) == ["Chennuru, Ashok", "Lavu, Ratnakar", "Smith, Monique"] and unknown == 1
    names_side, _ = A.readers(meta, [units[6]["id"], units[7]["id"]])                 # only the side thread
    assert "Tanal, Stephen" in names_side and "Kaye, Mark" in names_side
    line = A.scope_line(meta, [u["id"] for u in units])
    assert "13 cited messages" in line and "13 cited messages are marked private" in line and "no recipient information" in line


def _chain_brief(units):
    u = {x["sequence"]: x["id"] for x in units}
    return {
        "question": {"text": "Should benefit-expense work be funded as one project?"},
        "options": [
            {"id": "oA", "title": "Fund as a separate project", "decision": "funding", "status": "active",
             "source_unit_ids": [u[5], u[6]]},
            {"id": "oB", "title": "Fold into existing capability work", "decision": "funding", "status": "active",
             "source_unit_ids": [u[7]]},
        ],
        "positions": [
            {"participant_label": "Ali, Kanwal (Walle)", "option_id": "oA", "stance": "proposed", "source_unit_ids": [u[5]]},
            {"participant_label": "Tanal, Stephen", "option_id": "oA", "stance": "support", "source_unit_ids": [u[6]]},
            {"participant_label": "Lavu, Ratnakar", "option_id": "oB", "stance": "support", "source_unit_ids": [u[7]]},
        ],
        "blockers": [{"text": "Alignment of the funding owner is not confirmed", "kind": "open_question",
                      "resolution_status": "open", "source_unit_ids": [u[5], u[7]]}],
        "actions": [], "objections": [], "claims": [], "criteria": [], "constraints": [], "separate_issues": [],
    }


def test_items_are_tagged_with_who_saw_them_and_whether_they_are_private():
    units = with_ids(I.email_units(fx("ent-07-email-chain-1.txt")))
    brief = _chain_brief(units)
    A.attach_audience(brief, _meta(units))
    blocker = brief["blockers"][0]
    assert blocker["restricted"] is True and "Weber, Kyle" not in blocker["seen_by"]       # u7 was never sent to the wide group
    assert "Lavu, Ratnakar" in blocker["seen_by"] and "Tanal, Stephen" in blocker["seen_by"]


def test_a_viewer_only_gets_what_they_could_see():
    units = with_ids(I.email_units(fx("ent-07-email-chain-1.txt")))
    meta = _meta(units)
    brief = _chain_brief(units)

    weber, stats = A.filter_for_viewer(brief, meta, "Kyle Weber")                          # on the wide thread only
    assert [o["id"] for o in weber["options"]] == ["oA"]                                   # option B came only from the side thread
    assert weber["options"][0]["partial"] and len(weber["options"][0]["source_unit_ids"]) == 1
    assert [p["participant_label"] for p in weber["positions"]] == ["Ali, Kanwal (Walle)"]  # side-thread positions removed
    assert weber["blockers"][0]["partial"] and stats["removed"] >= 2

    kaye, _ = A.filter_for_viewer(brief, meta, "Mark Kaye")                                # on the side thread and (via the forward) the history
    assert len(kaye["options"]) == 2 and len(kaye["positions"]) == 3
    assert A.filter_for_viewer(brief, meta, "Somebody Else")[0]["options"] == []           # not on anything

    assert len(brief["options"]) == 2 and "partial" not in brief["options"][0]            # the original is untouched


def test_viewer_summary_never_sends_hidden_text_to_the_model():
    units = with_ids(I.email_units(fx("ent-07-email-chain-1.txt")))
    meta = _meta(units)
    texts = {u["id"]: u["normalized_text"] for u in units}
    brief = _chain_brief(units)
    brief["emerging_preference"] = S._compute_preference(brief)

    quote = "confirm that Ratnakar is aligned with this ask"
    assert quote in texts[units[4]["id"]]

    def produce(user, n):
        return {"title": "-", "status": "-", "needs_attention": [
            {"text": "The funding owner's alignment is unconfirmed", "verbatim_quote": quote, "source_unit_id": units[4]["id"]}],
            "proposed": [], "committed": [], "unverified_claims": [], "preference": "-"}

    def entail(user, n):
        return {"verdicts": [{"id": i["id"], "supported": True} for i in json.loads(user)]}

    llm = FakeLLM({"produce_exec_summary": produce, "check_support": entail})
    out = run(S.exec_summary_for_viewer(brief, "Kyle Weber", llm, texts, meta))
    sent_to_model = " ".join(u for _, u in llm.calls)
    assert "duplication" not in sent_to_model and "Clinical Intelligence" not in sent_to_model      # side-thread wording stayed out
    assert "The funding owner's alignment is unconfirmed" in out["summary"] and out["partial"] >= 1
    assert "Scope: email" in out["summary"]

    nothing = run(S.exec_summary_for_viewer(brief, "Somebody Else", llm, texts, meta))
    assert nothing["summary"] == "" and "nothing" in nothing["note"]


# ---------------------------------------------------------------------------
# synthetic enterprise fixtures with answer keys (pact-evals/gold)
# ---------------------------------------------------------------------------

def test_a_replys_privacy_request_covers_only_that_reply_but_a_forwards_covers_the_pile():
    units = I.email_units(fx("ent-08-email-vendor-private-note.txt"))
    assert [u["locator"]["restricted"] for u in units] == [False, False, False, False, True, False, False, False]
    assert units[4]["author_label"] == "Okafor, Jun" and units[4]["locator"]["audience"] == ["Nair, Priya"]
    # Priya's reply-all quoted Jun's note, so it reached everyone: the restriction is flagged, not forgotten
    assert len(units[4]["locator"]["effective_audience"]) == 6
    fwd = "From: A, Ann <a@x.test>\nSent: Monday, October 5, 2026 9:00 AM\nTo: B, Bob <b@x.test>\nSubject: FW: Plan\nPlease don't share this.\n\n" \
          "From: C, Cy <c@x.test>\nSent: Monday, October 5, 2026 8:00 AM\nTo: A, Ann <a@x.test>\nSubject: Plan\nShall we meet?\n"
    assert all(u["locator"]["restricted"] for u in I.email_units(fwd))


def test_forward_to_a_new_person_widens_the_audience_of_everything_quoted_under_it():
    units = I.email_units(fx("ent-09-email-offsite-reversal-forward.txt"))
    assert len(units) == 9 and [u["locator"]["msg_kind"] for u in units][6:] == ["reply", "forward", "reply"]
    first, fwd, reply = units[0]["locator"], units[7]["locator"], units[8]["locator"]
    assert "Okoye, Dana" in first["effective_audience"] and "Okoye, Dana" not in first["audience"]
    assert "Frey, Nate" in first["audience"] and first["cc"] == ["Frey, Nate"]               # cc counts as audience
    assert sorted(fwd["effective_audience"]) == ["Hale, Rosa", "Okoye, Dana"] and fwd["msg_kind"] == "forward"
    assert sorted(reply["effective_audience"]) == ["Hale, Rosa", "Okoye, Dana"]
    # the wide group never saw Dana's reply, so a group member's view stops before it
    meta = _meta(with_ids(units))
    assert not A.unit_visible_to(meta["u9"], "Andre Pike") and A.unit_visible_to(meta["u1"], "Andre Pike")


def test_meeting_with_a_decision_parses_cleanly():
    units = I.parse_teams_transcript(fx("ent-10-meeting-golive-decision.txt"))
    assert len(units) == 25 and not any("gap_before_s" in u["locator"] for u in units)
    assert {u["author_label"] for u in units} == {"Reyes, Camila", "Lindqvist, Erik", "Abara, Nneka", "Voss, Jonas", "Tran, Mei"}
    assert units[3]["author_label"] == "Voss, Jonas" and units[3]["normalized_text"] == "Yeah."
    assert units[6]["normalized_text"].startswith("Nikka,")                                     # the mis-heard name is kept as said
    assert units[-1]["locator"]["timestamp_str"] == "4:30"
    assert I.detect_source_type(fx("ent-10-meeting-golive-decision.txt")) == "teams_transcript"
