"""Standalone test for parser logic — no DB, no storage, no FastAPI required."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mvp1_b2e.worker.parser import (
    _detect_format,
    _parse_email_chain,
    _parse_transcript,
    _parse_plain_text,
)
from mvp1_b2e.worker.dedup import mark_duplicates

INPUT = ROOT / "mvp1_b2e" / "input"
EMAIL_FILE = INPUT / "emails" / "vendor-selection-01.eml"
TRANSCRIPT_FILE = INPUT / "transcripts" / "product-roadmap-q1-meeting.txt"


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def test_email_chain():
    section("Email chain — vendor-selection-01.eml")
    text = EMAIL_FILE.read_text()
    fmt = _detect_format(text, "email_chain")
    print(f"  detected format : {fmt}")
    units = _parse_email_chain(text)
    units = mark_duplicates(units)
    print(f"  units parsed    : {len(units)}")
    for u in units:
        dup = f"  [dup of seq {u['_dup_sequence']}]" if u.get("_dup_sequence") is not None else ""
        ts = u["timestamp"].isoformat() if u["timestamp"] else "—"
        print(f"  [{u['sequence']}] {u['author_label'] or '(no author)':<20}  {ts}{dup}")
        print(f"       original ({len(u['original_text'])} chars) | normalized ({len(u['normalized_text'])} chars)")
        print(f"       locator: {json.dumps(u['locator'])}")

    # Assertions
    assert fmt == "email_chain", f"Wrong format: {fmt}"
    assert len(units) == 8, f"Expected 8 units, got {len(units)}"
    authors = [u["author_label"] for u in units]
    assert "Sarah Chen" in authors, "Missing Sarah Chen"
    assert "Marcus Webb" in authors, "Missing Marcus Webb"
    assert all(u.get("_dup_sequence") is None for u in units), "No dups expected in email chain"
    print("\n  PASS")


def test_transcript():
    section("Transcript — product-roadmap-q1-meeting.txt")
    text = TRANSCRIPT_FILE.read_text()
    fmt = _detect_format(text, "txt")
    print(f"  detected format : {fmt}")
    units = _parse_transcript(text)
    units = mark_duplicates(units)
    print(f"  units parsed    : {len(units)}")
    for u in units:
        ts = u["timestamp"].strftime("%H:%M:%S") if u["timestamp"] else "—"
        print(f"  [{u['sequence']:02d}] {ts}  {u['author_label']:<18}  {u['normalized_text'][:70]!r}")

    assert fmt == "transcript", f"Wrong format: {fmt}"
    assert len(units) > 0, "No units parsed"
    speakers = {u["author_label"] for u in units}
    assert "Rachel Kim" in speakers, f"Missing Rachel Kim, got {speakers}"
    assert "Dev Patel" in speakers
    assert "Chris Abara" in speakers
    print("\n  PASS")


def test_plain_text():
    section("Plain text — synthetic two-paragraph input")
    text = "The team agreed on a budget of $50,000 for Q1.\n\nImplementation will begin in January and run through March."
    fmt = _detect_format(text, "txt")
    print(f"  detected format : {fmt}")
    units = _parse_plain_text(text)
    units = mark_duplicates(units)
    print(f"  units parsed    : {len(units)}")
    for u in units:
        print(f"  [{u['sequence']}] {u['normalized_text']!r}")

    assert fmt == "plain_text"
    assert len(units) == 2
    assert units[0]["author_label"] is None
    print("\n  PASS")


def test_dedup():
    section("Dedup — identical normalized texts should be flagged")
    shared = "The budget is fifty thousand dollars for the quarter"
    units = [
        {"sequence": 0, "normalized_text": shared, "locator": {}},
        {"sequence": 1, "normalized_text": "Completely different content about something else entirely here now", "locator": {}},
        {"sequence": 2, "normalized_text": shared, "locator": {}},  # duplicate of seq 0
    ]
    mark_duplicates(units)
    assert units[0]["_dup_sequence"] is None, "First unit should not be a dup"
    assert units[1]["_dup_sequence"] is None, "Unique unit should not be a dup"
    assert units[2]["_dup_sequence"] == 0, f"Third unit should dup seq 0, got {units[2]['_dup_sequence']}"
    print("  seq 0: not a dup  ✓")
    print("  seq 1: not a dup  ✓")
    print(f"  seq 2: dup of seq {units[2]['_dup_sequence']}  ✓")
    print("\n  PASS")


def test_format_detection_autodetect():
    section("Format auto-detection")
    email_text = "From: Alice <a@b.com>\nMessage-ID: <123>\n\nHello"
    assert _detect_format(email_text, "txt") == "email_chain"
    print("  email auto-detect  PASS")

    ts_text = "[00:00] Speaker: Hello\n[00:30] Other: World"
    assert _detect_format(ts_text, "txt") == "transcript"
    print("  transcript auto-detect  PASS")

    plain = "Just a plain paragraph with no special markers."
    assert _detect_format(plain, "txt") == "plain_text"
    print("  plain_text auto-detect  PASS")


if __name__ == "__main__":
    failures = []
    for test in [test_format_detection_autodetect, test_email_chain, test_transcript, test_plain_text, test_dedup]:
        try:
            test()
        except Exception as e:
            print(f"\n  FAIL: {e}")
            failures.append(test.__name__)

    print(f"\n{'=' * 60}")
    if failures:
        print(f"  FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED")
