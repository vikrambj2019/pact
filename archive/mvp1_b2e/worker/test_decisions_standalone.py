"""Standalone tests for Step 6 decision recording components.

Part A (no DB): snapshot assembly, field validation, option resolution.

Usage:
    python3 mvp1_b2e/worker/test_decisions_standalone.py
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mvp1_b2e.api.routes.decisions import _build_immutable_snapshot


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Shared fixtures — lightweight fakes (no DB needed)
# ---------------------------------------------------------------------------

WS_ID = uuid.uuid4()
REV_ID = uuid.uuid4()
OPT_ID_A = str(uuid.uuid4())
OPT_ID_B = str(uuid.uuid4())

_SNAPSHOT = {
    "summary": "Adopt cloud vendor X",
    "claims": [
        {"id": "c1", "text": "Migration takes 14 weeks", "evidence_status": "supported_within_scope"},
    ],
    "options": [
        {"id": OPT_ID_A, "title": "Cloud X", "description": "Preferred option"},
        {"id": OPT_ID_B, "title": "Cloud Y", "description": "Alternative"},
    ],
}

_ws = SimpleNamespace(
    id=WS_ID,
    title="Cloud Migration Decision",
    primary_question="Which cloud vendor should we adopt?",
    scope="IT infrastructure only",
    decision_rule="Consensus of steering committee",
    lifecycle_status="deliberating",
    decision_owner_label="CTO",
    current_revision_id=REV_ID,
)

_rev = SimpleNamespace(
    id=REV_ID,
    revision_number=5,
    structured_snapshot=_SNAPSHOT,
)


# ---------------------------------------------------------------------------
# Part A: snapshot assembly
# ---------------------------------------------------------------------------

def test_snapshot_has_all_required_keys():
    section("snapshot — all required keys present")
    snap = _build_immutable_snapshot(
        _ws, _rev,
        decision_owner_label="CTO",
        recorded_by_label="Jane Smith",
        rationale="Cloud X meets all criteria and has steering committee support.",
        notes="Approved unanimously",
        recorded_at=datetime.now(timezone.utc),
        selected_option_id=OPT_ID_A,
    )
    required = {
        "workspace_title", "primary_question", "scope", "decision_rule",
        "revision_number", "revision_id", "brief",
        "selected_option_id", "selected_option_title", "rationale",
        "notes", "decision_owner_label", "recorded_by_label", "recorded_at",
    }
    missing = required - snap.keys()
    assert not missing, f"Missing keys: {missing}"
    print("  all required keys present  PASS")


def test_snapshot_values_match_inputs():
    section("snapshot — values match workspace / revision / request inputs")
    recorded_at = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)
    snap = _build_immutable_snapshot(
        _ws, _rev,
        decision_owner_label="CTO",
        recorded_by_label="Jane Smith",
        rationale="Clear evidence favors Cloud X.",
        notes="Approved",
        recorded_at=recorded_at,
        selected_option_id=OPT_ID_A,
    )
    assert snap["workspace_title"] == "Cloud Migration Decision"
    assert snap["primary_question"] == "Which cloud vendor should we adopt?"
    assert snap["scope"] == "IT infrastructure only"
    assert snap["decision_rule"] == "Consensus of steering committee"
    assert snap["revision_number"] == 5
    assert snap["revision_id"] == str(REV_ID)
    assert snap["brief"] is _SNAPSHOT
    assert snap["rationale"] == "Clear evidence favors Cloud X."
    assert snap["notes"] == "Approved"
    assert snap["decision_owner_label"] == "CTO"
    assert snap["recorded_by_label"] == "Jane Smith"
    assert snap["recorded_at"] == recorded_at.isoformat()
    print("  all values correct  PASS")


def test_selected_option_id_resolves_title():
    section("snapshot — selected_option_id resolves to title")
    snap = _build_immutable_snapshot(
        _ws, _rev,
        decision_owner_label="CTO",
        recorded_by_label="Jane",
        rationale="Best option.",
        notes="",
        recorded_at=datetime.now(timezone.utc),
        selected_option_id=OPT_ID_A,
    )
    assert snap["selected_option_id"] == OPT_ID_A
    assert snap["selected_option_title"] == "Cloud X"
    print(f"  resolved: {snap['selected_option_id']} → '{snap['selected_option_title']}'  PASS")


def test_selected_option_id_none_leaves_title_none():
    section("snapshot — selected_option_id=None leaves title None (defer case)")
    snap = _build_immutable_snapshot(
        _ws, _rev,
        decision_owner_label="CTO",
        recorded_by_label="Jane",
        rationale="Deferred pending more data.",
        notes="",
        recorded_at=datetime.now(timezone.utc),
        selected_option_id=None,
    )
    assert snap["selected_option_id"] is None
    assert snap["selected_option_title"] is None
    print("  None option_id → None title  PASS")


def test_unknown_selected_option_id_title_is_none():
    section("snapshot — unknown selected_option_id → title is None (caller must validate before building)")
    snap = _build_immutable_snapshot(
        _ws, _rev,
        decision_owner_label="CTO",
        recorded_by_label="Jane",
        rationale="Some rationale.",
        notes="",
        recorded_at=datetime.now(timezone.utc),
        selected_option_id="nonexistent-id",
    )
    # The builder doesn't raise; validation happens in the route before this call
    assert snap["selected_option_id"] == "nonexistent-id"
    assert snap["selected_option_title"] is None
    print("  unknown option_id → title=None  PASS")


def test_snapshot_brief_is_full_structured_snapshot():
    section("snapshot — brief field contains the full structured_snapshot")
    snap = _build_immutable_snapshot(
        _ws, _rev,
        decision_owner_label="",
        recorded_by_label="",
        rationale="r",
        notes="",
        recorded_at=datetime.now(timezone.utc),
    )
    assert snap["brief"]["summary"] == "Adopt cloud vendor X"
    assert len(snap["brief"]["claims"]) == 1
    assert snap["brief"]["claims"][0]["evidence_status"] == "supported_within_scope"
    assert len(snap["brief"]["options"]) == 2
    print("  brief contains full snapshot  PASS")


def test_snapshot_revision_id_is_string():
    section("snapshot — revision_id is a string UUID (JSON-serializable)")
    snap = _build_immutable_snapshot(
        _ws, _rev,
        decision_owner_label="",
        recorded_by_label="",
        rationale="r",
        notes="",
        recorded_at=datetime.now(timezone.utc),
    )
    assert isinstance(snap["revision_id"], str)
    assert uuid.UUID(snap["revision_id"]) == REV_ID
    print("  revision_id is a string UUID  PASS")


def test_snapshot_recorded_at_is_iso_string():
    section("snapshot — recorded_at is ISO 8601 string")
    recorded_at = datetime(2026, 10, 9, 15, 30, 0, tzinfo=timezone.utc)
    snap = _build_immutable_snapshot(
        _ws, _rev,
        decision_owner_label="",
        recorded_by_label="",
        rationale="r",
        notes="",
        recorded_at=recorded_at,
    )
    assert isinstance(snap["recorded_at"], str)
    assert "2026-10-09" in snap["recorded_at"]
    assert "15:30:00" in snap["recorded_at"]
    print(f"  recorded_at = {snap['recorded_at']}  PASS")


def test_snapshot_independence_from_workspace_changes():
    """Mutating the workspace after snapshot creation must not affect the snapshot."""
    section("snapshot — independent of workspace object after construction")
    import copy
    ws_copy = copy.copy(_ws)
    snap = _build_immutable_snapshot(
        ws_copy, _rev,
        decision_owner_label="CTO",
        recorded_by_label="",
        rationale="r",
        notes="",
        recorded_at=datetime.now(timezone.utc),
    )
    ws_copy.title = "CHANGED TITLE"
    assert snap["workspace_title"] == "Cloud Migration Decision"
    print("  post-construction mutation does not affect snapshot  PASS")


# ---------------------------------------------------------------------------
# Part B: Pydantic schema validation
# ---------------------------------------------------------------------------

def test_request_schema_rejects_empty_strings():
    section("RecordDecisionRequest — empty string fields rejected by Pydantic")
    from pydantic import ValidationError
    from mvp1_b2e.api.routes.decisions import RecordDecisionRequest

    cases = [
        {"decision_owner_label": "", "recorded_by_label": "Jane", "expected_revision": 1, "rationale": "r"},
        {"decision_owner_label": "CTO", "recorded_by_label": "", "expected_revision": 1, "rationale": "r"},
        {"decision_owner_label": "CTO", "recorded_by_label": "Jane", "expected_revision": 1, "rationale": ""},
    ]
    for i, kwargs in enumerate(cases):
        try:
            RecordDecisionRequest(**kwargs)
            assert False, f"Case {i} should have raised ValidationError"
        except ValidationError:
            pass
    print(f"  all {len(cases)} empty-string cases rejected  PASS")


def test_request_schema_valid_with_all_fields():
    section("RecordDecisionRequest — valid request accepted")
    from mvp1_b2e.api.routes.decisions import RecordDecisionRequest

    req = RecordDecisionRequest(
        decision_owner_label="CTO",
        recorded_by_label="Jane Smith",
        expected_revision=5,
        rationale="Cloud X selected because it meets all criteria.",
        selected_option_id=OPT_ID_A,
        notes="Meeting notes attached",
    )
    assert req.decision_owner_label == "CTO"
    assert req.rationale == "Cloud X selected because it meets all criteria."
    assert req.selected_option_id == OPT_ID_A
    print("  valid request accepted  PASS")


def test_request_schema_selected_option_id_optional():
    section("RecordDecisionRequest — selected_option_id is optional (defer/custom)")
    from mvp1_b2e.api.routes.decisions import RecordDecisionRequest

    req = RecordDecisionRequest(
        decision_owner_label="CTO",
        recorded_by_label="Jane",
        expected_revision=3,
        rationale="Deferred — need more data.",
    )
    assert req.selected_option_id is None
    print("  selected_option_id defaults to None  PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_snapshot_has_all_required_keys,
        test_snapshot_values_match_inputs,
        test_selected_option_id_resolves_title,
        test_selected_option_id_none_leaves_title_none,
        test_unknown_selected_option_id_title_is_none,
        test_snapshot_brief_is_full_structured_snapshot,
        test_snapshot_revision_id_is_string,
        test_snapshot_recorded_at_is_iso_string,
        test_snapshot_independence_from_workspace_changes,
        test_request_schema_rejects_empty_strings,
        test_request_schema_valid_with_all_fields,
        test_request_schema_selected_option_id_optional,
    ]

    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"\n  FAIL: {e}")
            traceback.print_exc()
            failures.append(t.__name__)

    print(f"\n{'=' * 60}")
    if failures:
        print(f"  FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"  ALL {len(tests)} TESTS PASSED")
