"""Standalone tests for Step 4 deliberation components.

Part A (no LLM): operations executor — runs without API key.
Part B (LLM):    chat engine end-to-end — requires ANTHROPIC_API_KEY.

Usage:
    python3 mvp1_b2e/worker/test_deliberation_standalone.py           # Part A only
    ANTHROPIC_API_KEY=sk-... python3 ...                               # Part A + B
"""
from __future__ import annotations

import asyncio
import copy
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
os.environ.setdefault("LLM_MODEL", "claude-haiku-4-5-20251001")

from mvp1_b2e.worker.operations import apply_operations, ALLOWED_TYPES


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Shared fixture: a minimal brief snapshot with known IDs
# ---------------------------------------------------------------------------

OPT_A_ID = str(uuid.uuid4())
OPT_B_ID = str(uuid.uuid4())
CONSTRAINT_ID = str(uuid.uuid4())
CLAIM_ID = str(uuid.uuid4())
OBJECTION_ID = str(uuid.uuid4())
BLOCKER_ID = str(uuid.uuid4())

BASE_SNAPSHOT = {
    "question": {"text": "Which CRM should we choose?", "scope": None},
    "options": [
        {"id": OPT_A_ID, "title": "Salesforce", "description": "Pilot done", "source_unit_ids": []},
        {"id": OPT_B_ID, "title": "HubSpot", "description": "API-friendly", "source_unit_ids": []},
    ],
    "criteria": [],
    "constraints": [
        {"id": CONSTRAINT_ID, "text": "Live before Jan 5", "kind": "negotiable", "source_unit_ids": []},
    ],
    "claims": [
        {"id": CLAIM_ID, "text": "Salesforce migration takes 14 weeks",
         "kind": "factual", "evidence_status": "unverified", "source_unit_ids": []},
    ],
    "positions": [],
    "objections": [
        {"id": OBJECTION_ID, "text": "Salesforce timeline is risky",
         "option_ids": [OPT_A_ID], "resolution_status": "open", "source_unit_ids": []},
    ],
    "blockers": [
        {"id": BLOCKER_ID, "text": "Legal EU data residency sign-off pending",
         "kind": "unresolved_dependency", "resolution_status": "open", "source_unit_ids": []},
    ],
    "separate_issues": [],
    "actions": [],
}


def fresh():
    return copy.deepcopy(BASE_SNAPSHOT)


# ---------------------------------------------------------------------------
# Part A: operation executor tests
# ---------------------------------------------------------------------------

def test_add_option():
    section("add_option")
    snap, warns = apply_operations(fresh(), [
        {"type": "add_option", "fields": {"title": "Dynamics 365", "description": "Azure native",
                                          "source_unit_ids": []}}
    ])
    titles = [o["title"] for o in snap["options"]]
    assert "Dynamics 365" in titles, f"Option not added: {titles}"
    assert len(snap["options"]) == 3
    new_opt = next(o for o in snap["options"] if o["title"] == "Dynamics 365")
    uuid.UUID(new_opt["id"])  # must be valid UUID
    assert not warns
    print(f"  new option id: {new_opt['id'][:8]}...  PASS")


def test_update_constraint():
    section("update_constraint — negotiable → mandatory")
    snap, warns = apply_operations(fresh(), [
        {"type": "update_constraint", "entity_id": CONSTRAINT_ID,
         "fields": {"kind": "mandatory"}}
    ])
    c = next(x for x in snap["constraints"] if x["id"] == CONSTRAINT_ID)
    assert c["kind"] == "mandatory", f"kind not updated: {c['kind']}"
    assert not warns
    print(f"  constraint kind: {c['kind']}  PASS")


def test_update_claim():
    section("update_claim — evidence_status to supported")
    snap, warns = apply_operations(fresh(), [
        {"type": "update_claim", "entity_id": CLAIM_ID,
         "fields": {"evidence_status": "supported_within_scope"}}
    ])
    c = next(x for x in snap["claims"] if x["id"] == CLAIM_ID)
    assert c["evidence_status"] == "supported_within_scope"
    assert not warns
    print(f"  claim evidence_status: {c['evidence_status']}  PASS")


def test_resolve_blocker():
    section("resolve_blocker")
    snap, warns = apply_operations(fresh(), [
        {"type": "resolve_blocker", "entity_id": BLOCKER_ID,
         "fields": {"resolution_status": "resolved", "response": "Legal confirmed Dynamics 365 OK"}}
    ])
    b = next(x for x in snap["blockers"] if x["id"] == BLOCKER_ID)
    assert b["resolution_status"] == "resolved"
    assert b.get("response") == "Legal confirmed Dynamics 365 OK"
    assert not warns
    print(f"  blocker status: {b['resolution_status']}  PASS")


def test_update_position_create():
    section("update_position — create new")
    snap, warns = apply_operations(fresh(), [
        {"type": "update_position",
         "fields": {"participant_label": "Sarah", "option_id": OPT_A_ID,
                    "stance": "support", "source_unit_ids": []}}
    ])
    pos = snap["positions"]
    assert len(pos) == 1
    assert pos[0]["participant_label"] == "Sarah"
    assert pos[0]["option_id"] == OPT_A_ID
    assert pos[0]["stance"] == "support"
    uuid.UUID(pos[0]["id"])
    assert not warns
    print(f"  position created  stance={pos[0]['stance']}  PASS")


def test_update_position_upsert():
    section("update_position — upsert existing")
    snap = fresh()
    snap["positions"] = [{"id": str(uuid.uuid4()), "participant_label": "Sarah",
                          "option_id": OPT_A_ID, "stance": "conditional",
                          "conditions": "if timeline ok", "source_unit_ids": []}]
    snap, warns = apply_operations(snap, [
        {"type": "update_position",
         "fields": {"participant_label": "Sarah", "option_id": OPT_A_ID,
                    "stance": "oppose", "source_unit_ids": []}}
    ])
    pos = snap["positions"]
    assert len(pos) == 1, "Should not duplicate"
    assert pos[0]["stance"] == "oppose"
    print(f"  position updated  stance={pos[0]['stance']}  PASS")


def test_unknown_operation_warns():
    section("unknown operation — skipped with warning")
    snap, warns = apply_operations(fresh(), [{"type": "delete_everything", "fields": {}}])
    assert len(warns) == 1
    assert "delete_everything" in warns[0]
    assert snap["options"] == BASE_SNAPSHOT["options"]  # unchanged
    print(f"  warning: {warns[0]}  PASS")


def test_invalid_entity_id_warns():
    section("update with bad entity_id — warns, no crash")
    snap, warns = apply_operations(fresh(), [
        {"type": "update_option", "entity_id": "nonexistent-id", "fields": {"title": "X"}}
    ])
    assert len(warns) == 1
    assert "not found" in warns[0]
    print(f"  warning: {warns[0]}  PASS")


def test_field_allowlist():
    section("field allowlist — id field rejected")
    snap, warns = apply_operations(fresh(), [
        {"type": "update_option", "entity_id": OPT_A_ID,
         "fields": {"id": "evil-injection", "title": "Safe Title"}}
    ])
    opt = next(o for o in snap["options"] if o["id"] == OPT_A_ID)
    assert opt["id"] == OPT_A_ID, "id must not be overwritten"
    assert opt["title"] == "Safe Title"
    print(f"  id preserved: {opt['id'][:8]}...  title: {opt['title']}  PASS")


def test_add_separate_issue():
    section("add_separate_issue")
    snap, warns = apply_operations(fresh(), [
        {"type": "add_separate_issue",
         "fields": {"title": "IT vendor onboarding timeline",
                    "rationale": "Tangential to CRM choice", "source_unit_ids": []}}
    ])
    assert len(snap["separate_issues"]) == 1
    assert snap["separate_issues"][0]["title"] == "IT vendor onboarding timeline"
    uuid.UUID(snap["separate_issues"][0]["id"])
    print("  separate issue added  PASS")


def test_update_emerging_preference():
    section("update_emerging_preference")
    snap, warns = apply_operations(fresh(), [
        {"type": "update_emerging_preference",
         "fields": {"option_id": OPT_B_ID, "summary": "Cost and timeline favor HubSpot",
                    "supporting_refs": [], "dissenting_refs": []}}
    ])
    ep = snap.get("emerging_preference")
    assert ep is not None
    assert ep["option_id"] == OPT_B_ID
    print(f"  emerging_preference set for option {ep['option_id'][:8]}...  PASS")


# ---------------------------------------------------------------------------
# Part B: chat LLM end-to-end
# ---------------------------------------------------------------------------

import pytest

_needs_api_key = pytest.mark.skipif(
    not (os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    reason="ANTHROPIC_API_KEY not set",
)


@_needs_api_key
async def test_chat_clarification():
    section("Chat LLM — clarification (no operations expected)")
    from mvp1_b2e.worker.chat import run_chat
    from mvp1_b2e.worker.llm import LLMClient

    llm = LLMClient()
    result = await run_chat(
        snapshot=BASE_SNAPSHOT,
        revision_number=1,
        history=[],
        author_label="Marcus",
        message_text="What are the arguments against Salesforce?",
        llm=llm,
    )
    print(f"  request_type: {result['request_type']}")
    print(f"  response (first 120 chars): {result['response_text'][:120]!r}")
    print(f"  operations proposed: {len(result.get('operations', []))}")
    assert result.get("response_text"), "No response text"
    assert result.get("request_type") in ("clarification", "comparison", "other")
    print("  PASS")


@_needs_api_key
async def test_chat_correction():
    section("Chat LLM — correction (operations expected)")
    from mvp1_b2e.worker.chat import run_chat
    from mvp1_b2e.worker.llm import LLMClient

    llm = LLMClient()
    result = await run_chat(
        snapshot=BASE_SNAPSHOT,
        revision_number=1,
        history=[],
        author_label="Priya",
        message_text="The Jan 5 deadline is actually mandatory — procurement confirmed it last week. Please update the constraint.",
        llm=llm,
    )
    print(f"  request_type: {result['request_type']}")
    ops = result.get("operations", [])
    print(f"  operations: {[o['type'] for o in ops]}")
    assert result.get("response_text"), "No response text"
    print("  PASS")


async def run_llm_tests():
    import time
    for test in [test_chat_clarification, test_chat_correction]:
        t0 = time.monotonic()
        try:
            await test()
            print(f"  elapsed: {time.monotonic() - t0:.1f}s")
        except Exception as e:
            import traceback
            print(f"\n  FAIL: {e}")
            traceback.print_exc()
            return False
    return True


if __name__ == "__main__":
    # Part A — sync tests, no API key needed
    sync_tests = [
        test_add_option, test_update_constraint, test_update_claim, test_resolve_blocker,
        test_update_position_create, test_update_position_upsert, test_unknown_operation_warns,
        test_invalid_entity_id_warns, test_field_allowlist, test_add_separate_issue,
        test_update_emerging_preference,
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

    # Part B — LLM tests, need API key
    if os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        ok = asyncio.run(run_llm_tests())
        if not ok:
            failures.append("llm_tests")
    else:
        print("\n  (Skipping LLM tests — set ANTHROPIC_API_KEY to run)")

    print(f"\n{'=' * 60}")
    if failures:
        print(f"  FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        total = len(sync_tests) + (2 if os.environ.get("LLM_API_KEY") else 0)
        print(f"  ALL {total} TESTS PASSED")
