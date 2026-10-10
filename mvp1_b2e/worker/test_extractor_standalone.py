"""Standalone test for the extraction pipeline — uses real LLM, no DB.

Tests chunker + extractor + merger end-to-end on the vendor-selection email chain
and the product-roadmap transcript.

Usage:
    ANTHROPIC_API_KEY=sk-... python3 mvp1_b2e/worker/test_extractor_standalone.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Minimal settings patch so LLMClient can init without a .env file
os.environ.setdefault("LLM_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
os.environ.setdefault("LLM_MODEL", "claude-haiku-4-5-20251001")  # fast + cheap for tests

from mvp1_b2e.worker.parser import _parse_email_chain, _parse_transcript
from mvp1_b2e.worker.chunker import chunk_units
from mvp1_b2e.worker.extractor import extract_chunk
from mvp1_b2e.worker.merger import merge_extractions
from mvp1_b2e.worker.llm import LLMClient

INPUT = ROOT / "mvp1_b2e" / "input"


def fake_units(parsed: list[dict]) -> list[dict]:
    """Attach fake UUIDs to parsed unit dicts for testing."""
    import uuid
    return [
        {**u, "id": str(uuid.uuid4())}
        for u in parsed
    ]


def section(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


async def test_email_extraction():
    section("Email chain extraction — vendor-selection-01.eml")
    text = (INPUT / "emails" / "vendor-selection-01.eml").read_text()
    parsed = _parse_email_chain(text)
    units = fake_units(parsed)
    print(f"  units: {len(units)}")

    llm = LLMClient()
    chunks = chunk_units(units)
    print(f"  chunks: {len(chunks)}")

    t0 = time.monotonic()
    chunk_results = []
    for i, chunk in enumerate(chunks):
        print(f"  extracting chunk {i+1}/{len(chunks)} ({len(chunk)} units)...", flush=True)
        result = await extract_chunk(chunk, llm)
        chunk_results.append(result)

    merged = merge_extractions(chunk_results)
    elapsed = time.monotonic() - t0

    print(f"\n  elapsed: {elapsed:.1f}s")
    print(f"  candidate_questions : {len(merged['candidate_questions'])}")
    print(f"  options             : {len(merged['options'])}")
    print(f"  criteria            : {len(merged['criteria'])}")
    print(f"  constraints         : {len(merged['constraints'])}")
    print(f"  claims              : {len(merged['claims'])}")
    print(f"  positions           : {len(merged['positions'])}")
    print(f"  objections          : {len(merged['objections'])}")
    print(f"  blockers            : {len(merged['blockers'])}")
    print(f"  actions             : {len(merged['actions'])}")

    print("\n  --- Candidate questions ---")
    for q in merged["candidate_questions"]:
        print(f"    • {q['text']}")

    print("\n  --- Options ---")
    for o in merged["options"]:
        print(f"    • {o['title']}")

    print("\n  --- Blockers ---")
    for b in merged["blockers"]:
        print(f"    [{b['kind']}] {b['text']}")

    assert len(merged["options"]) >= 2, "Expected at least 2 options (Salesforce, HubSpot, Dynamics)"
    assert len(merged["candidate_questions"]) >= 1, "Expected at least 1 candidate question"
    print("\n  PASS")
    return merged


async def test_transcript_extraction():
    section("Transcript extraction — product-roadmap-q1-meeting.txt")
    text = (INPUT / "transcripts" / "product-roadmap-q1-meeting.txt").read_text()
    parsed = _parse_transcript(text)
    units = fake_units(parsed)
    print(f"  units: {len(units)}")

    llm = LLMClient()
    chunks = chunk_units(units)
    print(f"  chunks: {len(chunks)}")

    t0 = time.monotonic()
    chunk_results = []
    for i, chunk in enumerate(chunks):
        print(f"  extracting chunk {i+1}/{len(chunks)} ({len(chunk)} units)...", flush=True)
        result = await extract_chunk(chunk, llm)
        chunk_results.append(result)

    merged = merge_extractions(chunk_results)
    elapsed = time.monotonic() - t0

    print(f"\n  elapsed: {elapsed:.1f}s")
    print(f"  candidate_questions : {len(merged['candidate_questions'])}")
    print(f"  options             : {len(merged['options'])}")
    print(f"  positions           : {len(merged['positions'])}")
    print(f"  blockers            : {len(merged['blockers'])}")
    print(f"  actions             : {len(merged['actions'])}")

    print("\n  --- Options ---")
    for o in merged["options"]:
        print(f"    • {o['title']}")

    print("\n  --- Actions ---")
    for a in merged["actions"]:
        owner = f" [{a.get('owner_label', '')}]" if a.get("owner_label") else ""
        print(f"    •{owner} {a['text']}")

    assert len(merged["options"]) >= 2, "Expected at least 2 options (SSO, mobile rewrite, analytics)"
    print("\n  PASS")
    return merged


async def main():
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    failures = []
    for test in [test_email_extraction, test_transcript_extraction]:
        try:
            await test()
        except Exception as e:
            import traceback
            print(f"\n  FAIL: {e}")
            traceback.print_exc()
            failures.append(test.__name__)

    print(f"\n{'=' * 65}")
    if failures:
        print(f"  FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
