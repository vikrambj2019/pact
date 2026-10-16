"""Apply typed change operations to a brief snapshot.

The server owns the brief. The LLM proposes operations; this module executes them.
All new entities receive server-assigned UUIDs. The LLM references existing entities
by their current entity_id (present in the brief it was given).
"""
from __future__ import annotations

import copy
import uuid

# Allowlist — no arbitrary JSON paths or SQL accepted
ALLOWED_TYPES = {
    "add_option", "update_option",
    "add_criterion", "update_criterion",
    "add_constraint", "update_constraint",
    "add_claim", "update_claim", "retract_claim",
    "add_objection", "resolve_objection",
    "add_blocker", "resolve_blocker",
    "add_separate_issue",
    "add_action", "complete_action",
    "update_position",
    "update_emerging_preference",
}

# Which brief sections each prefix maps to
_SECTION = {
    "option": "options",
    "criterion": "criteria",
    "constraint": "constraints",
    "claim": "claims",
    "objection": "objections",
    "blocker": "blockers",
    "separate_issue": "separate_issues",
    "action": "actions",
}

# Fields the LLM is permitted to set per section (allowlist prevents arbitrary mutation)
_MUTABLE_FIELDS = {
    "options":         {"title", "description", "arguments_for", "arguments_against", "source_unit_ids"},
    "criteria":        {"name", "description", "source_unit_ids"},
    "constraints":     {"text", "kind", "basis", "source_unit_ids"},
    "claims":          {"text", "kind", "author_label", "evidence_status", "source_unit_ids"},
    "objections":      {"text", "option_ids", "author_label", "resolution_status", "source_unit_ids"},
    "blockers":        {"text", "kind", "resolution_status", "source_unit_ids"},
    "separate_issues": {"title", "rationale", "source_unit_ids"},
    "actions":         {"text", "owner_label", "completion_status", "source_unit_ids"},
}


def _find_by_id(section: list[dict], entity_id: str) -> dict | None:
    return next((item for item in section if item.get("id") == entity_id), None)


def _safe_fields(section_name: str, fields: dict) -> dict:
    allowed = _MUTABLE_FIELDS.get(section_name, set())
    return {k: v for k, v in fields.items() if k in allowed}


def apply_operations(snapshot: dict, operations: list[dict]) -> tuple[dict, list[str]]:
    """Apply a list of operations to a deep copy of the snapshot.

    Returns (new_snapshot, warnings).  Warnings are non-fatal — the operation
    is skipped and a message is appended rather than raising.
    """
    new = copy.deepcopy(snapshot)
    warnings: list[str] = []

    for op in operations:
        op_type = op.get("type", "")
        if op_type not in ALLOWED_TYPES:
            warnings.append(f"Skipped unknown operation type '{op_type}'")
            continue
        try:
            _dispatch(new, op, warnings)
        except Exception as exc:
            warnings.append(f"Operation '{op_type}' error: {exc}")

    return new, warnings


def _dispatch(snap: dict, op: dict, warnings: list[str]) -> None:
    op_type: str = op["type"]
    fields: dict = op.get("fields", {})
    entity_id: str = op.get("entity_id", "")

    # ── add operations ───────────────────────────────────────────────────────
    if op_type.startswith("add_"):
        key = op_type[4:]                      # strip "add_"
        section_name = _SECTION.get(key)
        if section_name is None:
            warnings.append(f"No section mapping for '{op_type}'")
            return
        safe = _safe_fields(section_name, fields)
        safe["id"] = str(uuid.uuid4())         # server assigns
        snap.setdefault(section_name, []).append(safe)
        return

    # ── update_position (special: find-or-create by participant+option) ──────
    if op_type == "update_position":
        participant = fields.get("participant_label", "")
        option_id = fields.get("option_id", "")
        stance = fields.get("stance")
        if not participant or not option_id or not stance:
            warnings.append("update_position requires participant_label, option_id, stance")
            return
        positions = snap.setdefault("positions", [])
        existing = next(
            (p for p in positions
             if p.get("participant_label") == participant and p.get("option_id") == option_id),
            None
        )
        if existing:
            for k in ("stance", "conditions", "source_unit_ids"):
                if k in fields:
                    existing[k] = fields[k]
        else:
            positions.append({
                "id": str(uuid.uuid4()),
                "participant_label": participant,
                "option_id": option_id,
                "stance": stance,
                "conditions": fields.get("conditions", ""),
                "source_unit_ids": fields.get("source_unit_ids", []),
            })
        return

    # ── update_emerging_preference ───────────────────────────────────────────
    if op_type == "update_emerging_preference":
        _EP_FIELDS = {"option_id", "summary", "supporting_refs", "dissenting_refs"}
        safe = {k: v for k, v in fields.items() if k in _EP_FIELDS}
        snap["emerging_preference"] = safe if safe else None
        return

    # ── update / resolve / retract / complete operations ────────────────────
    # Derive section from the verb+noun: update_option → options, resolve_blocker → blockers
    for prefix in ("update_", "resolve_", "retract_", "complete_"):
        if op_type.startswith(prefix):
            key = op_type[len(prefix):]
            section_name = _SECTION.get(key)
            if section_name is None:
                warnings.append(f"No section mapping for '{op_type}'")
                return
            section = snap.get(section_name, [])
            item = _find_by_id(section, entity_id)
            if item is None:
                warnings.append(f"Entity '{entity_id}' not found in {section_name} — skipped")
                return

            if op_type == "retract_claim":
                item["evidence_status"] = "retracted"
                return
            if op_type == "complete_action":
                item["completion_status"] = "completed"
                return
            if op_type.startswith("resolve_"):
                item["resolution_status"] = fields.get("resolution_status", "resolved")
                if "response" in fields:
                    item["response"] = fields["response"]
                return

            # Generic update — apply allowed fields only
            safe = _safe_fields(section_name, fields)
            item.update(safe)
            return

    warnings.append(f"Unhandled operation '{op_type}'")
