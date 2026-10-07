"""The decision card: schema, public projection, and the rules every model-proposed change must pass.

Pure logic, no LLM calls. Pact does not know whether a participant is a real person or simulated;
it only knows who the admin is and what each participant has permitted.
"""
from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone

ADMIN_ID = "human_admin"
CARD_SECTIONS = {"proposals", "criteria", "claims", "assumptions", "arguments", "positions",
                 "relationships", "unresolved"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_copy(value):
    return copy.deepcopy(value)


def new_card(question: str, admin_name: str, members: list[dict], admin_mapping_allowed: bool = True) -> dict:
    """Create an empty card. `members` are the non-admin participants:
    {id, name, role?, participant_type?, mapping_allowed?, share_allowed?}."""
    admin = {"id": ADMIN_ID, "name": admin_name, "role": "admin and participant",
             "participant_type": "human", "mapping_allowed": admin_mapping_allowed, "share_allowed": True}
    people = [admin, *members]
    permissions = []
    for person in people:
        mapping_granted = bool(person.get("mapping_allowed", False))
        sharing_granted = bool(person.get("share_allowed", False))
        permissions.append({
            "participant_id": person["id"],
            "mapping": {"status": "granted" if mapping_granted else "not_requested",
                        "source_message_id": "session_setup" if mapping_granted else None},
            "share_in_group_card": {"status": "granted" if sharing_granted else "not_requested",
                                    "source_message_id": "session_setup" if sharing_granted else None},
        })
    return {
        "schema_version": "mvp1_c2c.2",
        "experiment_id": str(uuid.uuid4()),
        "card_version": 0,
        "topic": {"title": question, "problem": question, "decision_requested": question,
                  "scope": "", "status": "open", "source_message_ids": []},
        "participants": [{"id": p["id"], "name": p["name"], "role": p.get("role", "participant"),
                          "participant_type": p.get("participant_type", "human")}
                         for p in people],
        "decision_process": {"method": "admin_recorded", "approver_ids": [ADMIN_ID],
                             "unanimity_required": False, "silence_counts_as_agreement": False},
        "individual_mapping": {"mvp_version": 1,
                               "policy": {"require_opt_in": True,
                                          "record_clear_statements_with_provenance": True,
                                          "require_confirmation_for_ambiguous_interpretations": True,
                                          "silence_counts_as_confirmation": False},
                               "participant_permissions": permissions,
                               "pending_interpretations": []},
        "criteria": [], "proposals": [], "claims": [], "assumptions": [], "arguments": [],
        "positions": [], "relationships": [], "evidence": [], "claim_checks": [],
        "unresolved": [], "decision": {"status": "not_recorded"},
        "messages": [], "snapshots": [], "change_log": [], "audit": [], "facilitation": [],
    }


def public_card(card: dict) -> dict:
    """Project out admin bookkeeping while retaining group decision state."""
    private = {"audit", "snapshots", "facilitation", "run_settings"}
    result = {k: json_copy(v) for k, v in card.items() if k not in private}
    # Other participants see only the explicitly shared individual positions.
    permissions = {p["participant_id"]: p for p in card["individual_mapping"]["participant_permissions"]}
    visible = [pos for pos in card["positions"] if
               permissions.get(pos.get("participant_id"), {}).get("share_in_group_card", {}).get("status") == "granted"]
    result["positions"] = visible
    result["individual_mapping"] = {"policy": card["individual_mapping"]["policy"]}
    return result


def apply_change(card: dict, change: dict, message: dict) -> None:
    """Apply a single observer change to a card, or raise ValueError if it breaks a rule."""
    section = change.get("section")
    if section not in CARD_SECTIONS:
        raise ValueError("Observer proposed a forbidden card section.")
    quote = change.get("quote", "")
    if not quote or quote not in message["text"]:
        raise ValueError("Card update has no exact source quote.")
    owner = change.get("participant_id")
    try:
        value = json.loads(change.get("value_json", ""))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Card update value must contain valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("Card update value must be an object.")
    if not change.get("entity_id") or change.get("operation") not in {"add", "update"}:
        raise ValueError("Card update needs a stable ID and add/update operation.")
    is_personal = section == "positions"
    if is_personal and owner != message["speaker_id"]:
        raise ValueError("A personal position must be attributed to its speaker.")
    if section == "claims" and value.get("made_by") != message["speaker_id"]:
        raise ValueError("Claim attribution must match the source speaker.")
    if section == "claims" and change.get("operation") == "add" and not isinstance(value.get("checkable"), bool):
        raise ValueError("A new claim must say whether it is checkable.")
    if section == "proposals" and value.get("status", "proposed") not in {"proposed", "under_consideration"}:
        raise ValueError("Observer cannot approve, select, defer, or decide a proposal.")
    if section == "claims" and value.get("verification", {}).get("status", "not_checked") != "not_checked":
        raise ValueError("Observer cannot mark a claim as researched or verified.")
    if section == "positions":
        permissions = {p["participant_id"]: p for p in card["individual_mapping"]["participant_permissions"]}
        if permissions.get(owner, {}).get("mapping", {}).get("status") != "granted":
            card["audit"].append({"at": now_iso(), "event": "mapping_skipped",
                                   "detail": "Participant mapping permission is not granted.",
                                   "message_id": message["id"]})
            return
    if change.get("needs_confirmation"):
        confirmer = owner or message["speaker_id"]
        if confirmer != ADMIN_ID:
            card["audit"].append({"at": now_iso(), "event": "interpretation_skipped",
                                   "detail": "Ambiguous item for a non-admin participant not queued; no confirmer available.",
                                   "message_id": message["id"]})
            return
        proposed_text = (value.get("statement") or value.get("title") or
                         value.get("description") or value.get("question") or json.dumps(value))
        pending = {"id": change.get("entity_id", str(uuid.uuid4())), "participant_id": confirmer,
                   "proposed_type": section, "proposed_statement": proposed_text,
                   "source_message_ids": [message["id"]],
                   "change": json_copy(change), "base_card_version": card["card_version"] + 1,
                   "confirmation": {"status": "pending", "confirmed_by": None, "source_message_id": None},
                   "included_in_confirmed_state": False}
        card["individual_mapping"]["pending_interpretations"].append(pending)
        card["audit"].append({"at": now_iso(), "event": "interpretation_pending",
                               "detail": "Ambiguous item held for confirmation.", "message_id": message["id"]})
        return
    if is_personal:
        value = {**value, "id": change.get("entity_id"), "participant_id": owner,
                 "source_message_ids": [message["id"]]}
    value = {**value, "id": change.get("entity_id"),
             "source_message_ids": list(dict.fromkeys(value.get("source_message_ids", []) + [message["id"]]))}
    rows = card[section]
    existing = next((i for i, row in enumerate(rows) if row.get("id") == value["id"]), None)
    previous = json_copy(rows[existing]) if existing is not None else None
    if change.get("operation") == "update":
        if existing is None:
            raise ValueError("Observer cannot update an unknown entity.")
        if section == "positions" and rows[existing].get("participant_id") != owner:
            raise ValueError("Observer cannot revise another participant's position.")
        rows[existing] = {**rows[existing], **value, "revision": rows[existing].get("revision", 0) + 1}
    elif existing is None:
        rows.append({**value, "revision": 1})
    else:
        card["audit"].append({"at": now_iso(), "event": "duplicate",
                               "detail": "Repeated item did not create a new card entry.", "entity_id": value["id"]})
        return
    card["change_log"].append({"version": card["card_version"] + 1, "entity_id": value["id"],
                                "section": section, "participant_id": owner,
                                "previous_value": previous, "new_value": json_copy(value),
                                "source_message_id": message["id"], "quote": quote,
                                "reason": change.get("reason", ""), "recorded_at": now_iso()})
