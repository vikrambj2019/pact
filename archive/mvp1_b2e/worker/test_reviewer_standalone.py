"""Standalone tests for Step 5 evidence review components.

Part A (no LLM): quote validation, char offset computation, evidence status aggregation.
Part B (LLM):    full review pipeline against sample source units.

Usage:
    python3 mvp1_b2e/worker/test_reviewer_standalone.py
    ANTHROPIC_API_KEY=sk-... python3 mvp1_b2e/worker/test_reviewer_standalone.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
os.environ.setdefault("LLM_MODEL", "claude-haiku-4-5-20251001")

from mvp1_b2e.worker.reviewer import (
    _aggregate_status,
    _validate_findings,
)


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DOC_ID = str(uuid.uuid4())
UNIT_A_ID = str(uuid.uuid4())
UNIT_B_ID = str(uuid.uuid4())

UNIT_MAP = {
    UNIT_A_ID: {
        "id": UNIT_A_ID,
        "document_id": DOC_ID,
        "original_text": "The Salesforce migration will take 14 weeks according to the IT team estimate.",
        "normalized_text": "The Salesforce migration will take 14 weeks according to the IT team estimate.",
        "locator": {},
        "author_label": "IT Team",
    },
    UNIT_B_ID: {
        "id": UNIT_B_ID,
        "document_id": DOC_ID,
        "original_text": "Salesforce confirmed a fast-track implementation in 10 weeks with a dedicated team.",
        "normalized_text": "Salesforce confirmed a fast-track implementation in 10 weeks with a dedicated team.",
        "locator": {},
        "author_label": "Sarah",
    },
}


# ---------------------------------------------------------------------------
# Part A: validation and aggregation
# ---------------------------------------------------------------------------

def test_validate_normalized_only_rejected():
    """Quote found only in normalized_text must be rejected — offsets must be into original_text."""
    section("validate — quote only in normalized_text → rejected")
    # Build a unit where original and normalized differ (e.g. quoted lines stripped)
    unit_id = str(uuid.uuid4())
    unit_map = {unit_id: {
        "id": unit_id,
        "document_id": DOC_ID,
        "original_text": "> Forwarded: the budget is fifty thousand dollars\nThanks for confirming.",
        "normalized_text": "Thanks for confirming.",
        "locator": {},
        "author_label": "Bob",
    }}
    # "the budget is fifty thousand" exists in original but NOT in normalized
    findings = [{"source_unit_id": unit_id, "quote": "the budget is fifty thousand dollars",
                 "verdict": "supports", "explanation": ""}]
    valid, warns = _validate_findings(findings, unit_map)
    # Should succeed — quote IS in original_text
    assert len(valid) == 1
    assert valid[0]["start_char"] == unit_map[unit_id]["original_text"].index("the budget is fifty thousand dollars")

    # A quote that is ONLY in normalized_text (not original) must be rejected
    findings2 = [{"source_unit_id": unit_id, "quote": "Thanks for confirming.",
                  "verdict": "supports", "explanation": ""}]
    valid2, warns2 = _validate_findings(findings2, unit_map)
    # "Thanks for confirming." IS in original_text too (last line) — so this should pass
    assert len(valid2) == 1

    # Now test a quote that exists ONLY in normalized and not in original
    unit_map2 = {unit_id: {**unit_map[unit_id],
                            "original_text": "> abc def ghi",
                            "normalized_text": "synthetic text not in original"}}
    findings3 = [{"source_unit_id": unit_id, "quote": "synthetic text not in original",
                  "verdict": "supports", "explanation": ""}]
    valid3, warns3 = _validate_findings(findings3, unit_map2)
    assert len(valid3) == 0, "Quote only in normalized_text must be rejected"
    assert len(warns3) == 1
    assert "original_text" in warns3[0]
    print("  normalized-only quote correctly rejected  PASS")

def test_validate_exact_quote():
    section("validate — exact quote found → correct offsets")
    findings = [{
        "source_unit_id": UNIT_A_ID,
        "quote": "14 weeks",
        "verdict": "supports",
        "explanation": "States 14 week duration directly",
    }]
    valid, warns = _validate_findings(findings, UNIT_MAP)
    assert len(valid) == 1, f"Expected 1 valid finding, got {len(valid)}"
    assert not warns
    f = valid[0]
    assert f["start_char"] == UNIT_MAP[UNIT_A_ID]["original_text"].index("14 weeks")
    assert f["end_char"] == f["start_char"] + len("14 weeks")
    assert f["source_document_id"] == DOC_ID
    print(f"  start={f['start_char']} end={f['end_char']}  PASS")


def test_validate_quote_not_found():
    section("validate — fabricated quote stripped with warning")
    findings = [{
        "source_unit_id": UNIT_A_ID,
        "quote": "This text does not appear anywhere in the source",
        "verdict": "supports",
        "explanation": "Made up",
    }]
    valid, warns = _validate_findings(findings, UNIT_MAP)
    assert len(valid) == 0
    assert len(warns) == 1
    assert "not found" in warns[0]
    print(f"  warning: {warns[0][:60]}  PASS")


def test_validate_unknown_unit_id():
    section("validate — unknown unit_id stripped with warning")
    findings = [{
        "source_unit_id": "nonexistent-uuid",
        "quote": "anything",
        "verdict": "supports",
        "explanation": "",
    }]
    valid, warns = _validate_findings(findings, UNIT_MAP)
    assert len(valid) == 0
    assert len(warns) == 1
    assert "not found in workspace" in warns[0]
    print(f"  warning: {warns[0][:60]}  PASS")


def test_validate_multiple_mixed():
    section("validate — mix of valid and invalid findings")
    findings = [
        {"source_unit_id": UNIT_A_ID, "quote": "14 weeks", "verdict": "supports", "explanation": ""},
        {"source_unit_id": UNIT_B_ID, "quote": "10 weeks", "verdict": "contradicts", "explanation": ""},
        {"source_unit_id": UNIT_A_ID, "quote": "invented quote xyz", "verdict": "supports", "explanation": ""},
    ]
    valid, warns = _validate_findings(findings, UNIT_MAP)
    assert len(valid) == 2, f"Expected 2 valid, got {len(valid)}"
    assert len(warns) == 1
    verdicts = {f["verdict"] for f in valid}
    assert verdicts == {"supports", "contradicts"}
    print(f"  valid={len(valid)} warns={len(warns)}  verdicts={verdicts}  PASS")


def test_aggregate_supported():
    section("aggregate — supports only → supported_within_scope")
    findings = [
        {"verdict": "supports"}, {"verdict": "limits"},
    ]
    assert _aggregate_status(findings) == "supported_within_scope"
    print("  PASS")


def test_aggregate_contradicted():
    section("aggregate — contradicts only → contradicted_within_scope")
    findings = [{"verdict": "contradicts"}]
    assert _aggregate_status(findings) == "contradicted_within_scope"
    print("  PASS")


def test_aggregate_conflicting():
    section("aggregate — supports + contradicts → conflicting")
    findings = [{"verdict": "supports"}, {"verdict": "contradicts"}]
    assert _aggregate_status(findings) == "conflicting"
    print("  PASS")


def test_aggregate_limits_only():
    section("aggregate — limits only → unresolved")
    findings = [{"verdict": "limits"}]
    assert _aggregate_status(findings) == "unresolved"
    print("  PASS")


def test_aggregate_empty():
    section("aggregate — no findings → unresolved")
    assert _aggregate_status([]) == "unresolved"
    print("  PASS")


# ---------------------------------------------------------------------------
# Part B: LLM end-to-end review
# ---------------------------------------------------------------------------

import pytest

_needs_api_key = pytest.mark.skipif(
    not (os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    reason="ANTHROPIC_API_KEY not set",
)


@_needs_api_key
async def test_llm_review():
    section("LLM review — Salesforce 14-week claim against two contradictory units")
    from mvp1_b2e.worker.reviewer import _review_chunk
    from mvp1_b2e.worker.llm import LLMClient
    import time

    claim_text = "Salesforce migration takes 14 weeks"
    units = list(UNIT_MAP.values())
    for u in units:
        u["author_label"] = u.get("author_label", "")

    llm = LLMClient()
    t0 = time.monotonic()
    findings = await _review_chunk(claim_text, units, llm)
    elapsed = time.monotonic() - t0

    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  raw findings: {len(findings)}")
    for f in findings:
        print(f"    [{f['verdict']}] {f['quote']!r} — {f['explanation'][:60]}")

    assert len(findings) >= 1, "Expected at least one finding"
    assert all(f.get("verdict") in ("supports", "contradicts", "limits") for f in findings)

    # Now validate
    valid, warns = _validate_findings(findings, UNIT_MAP)
    print(f"  valid after quote-check: {len(valid)}  warns: {len(warns)}")
    for w in warns:
        print(f"    ⚠ {w}")

    status = _aggregate_status(valid)
    print(f"  evidence_status: {status}")
    assert status in ("supported_within_scope", "conflicting", "contradicted_within_scope",
                      "unresolved"), f"Unexpected status: {status}"
    print("  PASS")


async def run_llm_tests():
    try:
        await test_llm_review()
        return True
    except Exception as e:
        import traceback
        print(f"\n  FAIL: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sync_tests = [
        test_validate_exact_quote,
        test_validate_quote_not_found,
        test_validate_unknown_unit_id,
        test_validate_multiple_mixed,
        test_aggregate_supported,
        test_aggregate_contradicted,
        test_aggregate_conflicting,
        test_aggregate_limits_only,
        test_aggregate_empty,
    ]

    failures = []
    for t in sync_tests:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"\n  FAIL: {e}")
            traceback.print_exc()
            failures.append(t.__name__)

    if os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        ok = asyncio.run(run_llm_tests())
        if not ok:
            failures.append("llm_review")
    else:
        print("\n  (Skipping LLM test — set ANTHROPIC_API_KEY to run)")

    print(f"\n{'=' * 60}")
    if failures:
        print(f"  FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        total = len(sync_tests) + (1 if os.environ.get("LLM_API_KEY") else 0)
        print(f"  ALL {total} TESTS PASSED")
