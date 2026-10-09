"""The decision card: its shape, and the rules every observer action must pass before it is written.

Pure logic, no LLM calls. The observer proposes *actions* (add_option, record_affirmation, ...);
`apply_action` validates each one and writes it. Every change keeps who said it, when, and the exact
quote, so the history of each item can be read back.

Pact does not know whether a participant is a real person or simulated; it only knows who the admin
is and what each participant has permitted.
"""
from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone

ADMIN_ID = "human_admin"
SCHEMA_VERSION = "mvp1_c2c.3"

SECTIONS = ("options", "constraints", "criteria", "claims", "agreements", "preferences", "open_issues")
ID_PREFIX = {"options": "opt", "constraints": "con", "criteria": "crit", "claims": "claim",
             "agreements": "agr", "preferences": "pref", "open_issues": "issue"}
OPTION_STATUSES = {"considering", "dropped"}
CONSTRAINT_KINDS = {"budget", "dates", "people", "logistics", "other"}
CLAIM_KINDS = {"fact", "prediction", "opinion"}
CHECK_TYPES = ("flight", "price", "distance", "weather", "schedule", "rule", "availability", "other")
REASON_STANCES = {"for", "against"}

ACTIONS = {
    "add_option", "add_option_alias", "set_option_status", "add_reason",
    "add_constraint", "supersede_constraint",
    "add_criterion",
    "add_claim", "challenge_claim", "retract_claim", "correct_claim",
    "record_affirmation", "record_objection",
    "set_preference",
    "add_issue", "resolve_issue",
}


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
        "schema_version": SCHEMA_VERSION,
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
        **{section: [] for section in SECTIONS},
        "evidence": [], "claim_checks": [], "decision": {"status": "not_recorded"},
        "messages": [], "snapshots": [], "change_log": [], "audit": [], "facilitation": [],
        "observation": {"pending": [], "completed": []},
    }


def _permission(card: dict, participant_id: str, kind: str) -> bool:
    for p in card["individual_mapping"]["participant_permissions"]:
        if p["participant_id"] == participant_id:
            return p.get(kind, {}).get("status") == "granted"
    return False


def public_card(card: dict) -> dict:
    """What participants may see: admin bookkeeping removed, preferences only where sharing is granted."""
    private = {"audit", "snapshots", "facilitation", "run_settings", "change_log", "observation"}
    result = {k: json_copy(v) for k, v in card.items() if k not in private}
    result["preferences"] = [p for p in result["preferences"]
                             if _permission(card, p["participant_id"], "share_in_group_card")]
    result["individual_mapping"] = {"policy": card["individual_mapping"]["policy"]}
    return result


def current_constraints(card: dict) -> list[dict]:
    return [c for c in card["constraints"] if c["status"] != "superseded"]


def observer_view(card: dict) -> dict:
    """A compact card for the observer: current items, IDs and statuses; no messages or history."""
    def pick(item, *keys):
        return {k: item[k] for k in ("id", *keys) if item.get(k) not in (None, [], "")}
    return {
        "topic": card["topic"]["title"],
        "participants": [{"id": p["id"], "name": p["name"]} for p in card["participants"]],
        "options": [pick(o, "text", "aliases", "status", "proposed_by") for o in card["options"]],
        "constraints": [pick(c, "text", "kind", "status", "superseded_by") for c in card["constraints"]],
        "criteria": [pick(c, "text") for c in card["criteria"]],
        "claims": [pick(c, "statement", "made_by", "kind", "status", "corrected_to", "option_id")
                   for c in card["claims"]],
        "agreements": [{"id": a["id"], "subject_id": a["subject_id"], "status": a["status"],
                        "affirmed_by": [x["participant_id"] for x in a["affirmers"]],
                        "objected_by": [x["participant_id"] for x in a["objectors"]]}
                       for a in card["agreements"]],
        "preferences": [pick(p, "participant_id", "stance", "option_id", "leaning", "conditional")
                        for p in card["preferences"]],
        "open_issues": [pick(i, "text", "status") for i in card["open_issues"]],
    }


# ── applying observer actions ────────────────────────────────────────────────

def provenance(message: dict, quote: str) -> dict:
    """Who said it, when, and the exact words."""
    return {"message_id": message["id"], "speaker_id": message["speaker_id"],
            "at": message.get("recorded_at"), "quote": quote}


def _find(card: dict, sections: tuple[str, ...], item_id: str) -> tuple[str, dict]:
    for section in sections:
        for item in card[section]:
            if item["id"] == item_id:
                return section, item
    raise ValueError(f"{item_id!r} is not an existing item in {', '.join(sections)}.")


def _text(action: dict, field: str = "text") -> str:
    value = (action.get(field) or "").strip()
    if not value:
        raise ValueError(f"{action['action']} needs '{field}'.")
    return value


def _touch(item: dict, event: str, message: dict, quote: str, **extra) -> None:
    item["history"].append({"event": event, **extra, **provenance(message, quote)})
    if message["id"] not in item["source_message_ids"]:
        item["source_message_ids"].append(message["id"])


def _create(card: dict, section: str, fields: dict, message: dict, quote: str) -> dict:
    item = {"id": f"{ID_PREFIX[section]}_{len(card[section]) + 1}", **fields,
            "source_message_ids": [], "history": []}
    _touch(item, "added", message, quote)
    card[section].append(item)
    return item


def _agreement_status(card: dict, agreement: dict) -> None:
    everyone = {p["id"] for p in card["participants"]}
    affirmed = {a["participant_id"] for a in agreement["affirmers"]}
    agreement["affirmed_count"] = len(affirmed & everyone)
    agreement["participant_count"] = len(everyone)
    agreement["status"] = ("agreed_by_all" if affirmed >= everyone and not agreement["objectors"]
                           else "partial")
    subject_section, subject = _find(card, ("options", "constraints"), agreement["subject_id"])
    if subject_section == "constraints" and subject["status"] != "superseded":
        subject["status"] = "contested" if agreement["objectors"] else "stated"


def _stance_on(card: dict, action: dict, message: dict, quote: str, side: str) -> dict:
    """Record the speaker's own yes (side='affirmers') or no (side='objectors') on an option or constraint."""
    _, subject = _find(card, ("options", "constraints"), action["target_id"])
    agreement = next((a for a in card["agreements"] if a["subject_id"] == subject["id"]), None)
    if agreement is None:
        agreement = _create(card, "agreements", {"subject_id": subject["id"], "text": subject["text"],
                                                 "affirmers": [], "objectors": [], "status": "partial"},
                            message, quote)
    speaker = message["speaker_id"]
    other = "objectors" if side == "affirmers" else "affirmers"
    agreement[other] = [x for x in agreement[other] if x["participant_id"] != speaker]
    agreement[side] = [x for x in agreement[side] if x["participant_id"] != speaker]
    agreement[side].append({"participant_id": speaker, **provenance(message, quote)})
    _touch(agreement, "affirmed" if side == "affirmers" else "objected", message, quote)
    _agreement_status(card, agreement)
    return agreement


def apply_action(card: dict, action: dict, message: dict, refs: dict | None = None) -> str | None:
    """Validate one observer action against its source message and write it. Returns the item ID,
    or None when the action was held or skipped (see audit). Raises ValueError if it breaks a rule.

    `refs` maps temporary references (e.g. "new_banff") to IDs created earlier in the same observation."""
    refs = {} if refs is None else refs
    name = action.get("action")
    if name not in ACTIONS:
        raise ValueError(f"Unknown observer action {name!r}.")
    quote = action.get("quote", "")
    if not quote or quote not in message["text"]:
        raise ValueError("Action has no exact quote from its source message.")
    speaker = message["speaker_id"]
    if speaker not in {p["id"] for p in card["participants"]}:
        raise ValueError("Source message has an unknown speaker.")
    if action.get("target_id") in refs:
        action = {**action, "target_id": refs[action["target_id"]]}
    if action.get("option_id") in refs:
        action = {**action, "option_id": refs[action["option_id"]]}

    if action.get("ambiguous"):
        return _hold_ambiguous(card, action, message)
    if name == "set_preference" and not _permission(card, speaker, "mapping"):
        card["audit"].append({"at": now_iso(), "event": "mapping_skipped",
                              "detail": "Participant mapping permission is not granted.",
                              "message_id": message["id"]})
        return None

    before = {s: json_copy(card[s]) for s in SECTIONS}
    section, item = _apply(card, name, action, message, quote)
    if item is None:
        return None
    if action.get("ref"):
        refs[action["ref"]] = item["id"]
    previous = next((x for x in before[section] if x["id"] == item["id"]), None)
    card["change_log"].append({"version": card["card_version"] + 1, "entity_id": item["id"],
                               "section": section, "action": name,
                               "previous_value": previous, "new_value": json_copy(item),
                               **provenance(message, quote),
                               "reason": action.get("reason", ""), "recorded_at": now_iso()})
    return item["id"]


def _apply(card: dict, name: str, action: dict, message: dict, quote: str) -> tuple[str, dict | None]:
    speaker = message["speaker_id"]

    if name == "add_option":
        text = _text(action)
        key = text.lower()
        for option in card["options"]:
            if key == option["text"].lower() or key in (a.lower() for a in option["aliases"]):
                card["audit"].append({"at": now_iso(), "event": "duplicate",
                                      "detail": "Option already on the card.", "entity_id": option["id"]})
                return "options", None
        aliases = [a.strip() for a in action.get("aliases") or [] if a.strip()]
        return "options", _create(card, "options", {"text": text, "aliases": aliases, "proposed_by": speaker,
                                                    "status": "considering", "reasons": []}, message, quote)

    if name == "add_option_alias":
        _, option = _find(card, ("options",), action.get("target_id", ""))
        alias = _text(action)
        if alias.lower() not in (a.lower() for a in option["aliases"]):
            option["aliases"].append(alias)
        _touch(option, "alias_added", message, quote, alias=alias)
        return "options", option

    if name == "set_option_status":
        _, option = _find(card, ("options",), action.get("target_id", ""))
        status = action.get("status")
        if status not in OPTION_STATUSES:
            raise ValueError("Options can only be 'considering' or 'dropped'; only the admin records a decision.")
        event = ("revived" if option["status"] == "dropped" and status == "considering" else status)
        option["status"] = status
        _touch(option, event, message, quote)
        return "options", option

    if name == "add_reason":
        _, option = _find(card, ("options",), action.get("target_id", ""))
        stance = action.get("stance")
        if stance not in REASON_STANCES:
            raise ValueError("A reason must be 'for' or 'against' an option.")
        option["reasons"].append({"stance": stance, "text": _text(action), "participant_id": speaker,
                                  **provenance(message, quote)})
        _touch(option, f"reason_{stance}", message, quote)
        return "options", option

    if name == "add_constraint":
        kind = action.get("kind") or "other"
        if kind not in CONSTRAINT_KINDS:
            raise ValueError(f"Constraint kind must be one of {sorted(CONSTRAINT_KINDS)}.")
        return "constraints", _create(card, "constraints", {"text": _text(action), "kind": kind,
                                                            "stated_by": speaker, "status": "stated"},
                                      message, quote)

    if name == "supersede_constraint":
        _, old = _find(card, ("constraints",), action.get("target_id", ""))
        if old["status"] == "superseded":
            raise ValueError(f"{old['id']} is already superseded by {old.get('superseded_by')}.")
        kind = action.get("kind") or old["kind"]
        if kind not in CONSTRAINT_KINDS:
            raise ValueError(f"Constraint kind must be one of {sorted(CONSTRAINT_KINDS)}.")
        new = _create(card, "constraints", {"text": _text(action), "kind": kind, "stated_by": speaker,
                                            "status": "stated", "supersedes": old["id"]}, message, quote)
        old["status"], old["superseded_by"] = "superseded", new["id"]
        _touch(old, "superseded", message, quote, superseded_by=new["id"])
        return "constraints", new

    if name == "add_criterion":
        return "criteria", _create(card, "criteria", {"text": _text(action), "raised_by": speaker}, message, quote)

    if name == "add_claim":
        kind = action.get("kind")
        if kind not in CLAIM_KINDS:
            raise ValueError(f"Claim kind must be one of {sorted(CLAIM_KINDS)}.")
        if not isinstance(action.get("checkable"), bool):
            raise ValueError("A new claim must say whether it is checkable.")
        option_id = action.get("option_id") or None
        if option_id:
            _find(card, ("options",), option_id)
        checkable = action["checkable"] and kind != "opinion"
        check_type = (action.get("check_type") or "other") if checkable else None
        if check_type and check_type not in CHECK_TYPES:
            raise ValueError(f"check_type must be one of {list(CHECK_TYPES)}.")
        return "claims", _create(card, "claims", {
            "statement": _text(action), "made_by": speaker, "kind": kind, "option_id": option_id,
            "checkable": checkable, "check_type": check_type, "status": "unchallenged", "revision": 0,
            "challenges": [], "verification": {"status": "not_checked", "check_ids": []}}, message, quote)

    if name == "challenge_claim":
        _, claim = _find(card, ("claims",), action.get("target_id", ""))
        if claim["made_by"] == speaker:
            raise ValueError("A speaker cannot challenge their own claim; use correct_claim or retract_claim.")
        claim["challenges"].append({"text": _text(action), "participant_id": speaker, **provenance(message, quote)})
        if claim["status"] == "unchallenged":
            claim["status"] = "disputed"
        _touch(claim, "challenged", message, quote)
        return "claims", claim

    if name in {"retract_claim", "correct_claim"}:
        _, claim = _find(card, ("claims",), action.get("target_id", ""))
        if claim["made_by"] != speaker:
            raise ValueError("Only the person who made a claim can retract or correct it.")
        if name == "retract_claim":
            claim["status"] = "retracted"
            _touch(claim, "retracted", message, quote)
        else:
            claim["status"], claim["corrected_to"] = "corrected", _text(action)
            _touch(claim, "corrected", message, quote, corrected_to=claim["corrected_to"])
        claim["revision"] = claim.get("revision", 0) + 1
        previous = claim.get("verification", {})
        claim["verification"] = {"status": "not_checked", "check_ids": previous.get("check_ids", [])}
        for check in card["claim_checks"]:
            if check.get("claim_id") == claim["id"]:
                check["stale"] = True
        if previous.get("status") == "checked":
            _touch(claim, "verification_invalidated", message, quote,
                   previous_verdict=previous.get("verdict"))
        return "claims", claim

    if name == "record_affirmation":
        return "agreements", _stance_on(card, action, message, quote, "affirmers")

    if name == "record_objection":
        return "agreements", _stance_on(card, action, message, quote, "objectors")

    if name == "set_preference":
        stance = _text(action, "stance")
        option_id = action.get("option_id") or None
        leaning = None
        if option_id:
            _find(card, ("options",), option_id)
            leaning = action.get("leaning")
            if leaning not in REASON_STANCES:
                raise ValueError("A stance on an option must say whether it leans 'for' or 'against' it.")
        conditional = (action.get("conditional") or "").strip()
        current = next((p for p in card["preferences"]
                        if p["participant_id"] == speaker and p.get("option_id") == option_id), None)
        if current is None:
            return "preferences", _create(card, "preferences", {"participant_id": speaker, "stance": stance,
                                                                "option_id": option_id, "leaning": leaning,
                                                                "conditional": conditional},
                                          message, quote)
        _touch(current, "changed", message, quote, previous_stance=current["stance"],
               previous_leaning=current.get("leaning"))
        current["stance"], current["leaning"], current["conditional"] = stance, leaning, conditional
        return "preferences", current

    if name == "add_issue":
        known = {p["id"] for p in card["participants"]}
        involves = [p for p in action.get("participant_ids") or [] if p in known]
        return "open_issues", _create(card, "open_issues", {"text": _text(action), "involves": involves,
                                                            "raised_by": speaker, "status": "open"},
                                      message, quote)

    if name == "resolve_issue":
        _, issue = _find(card, ("open_issues",), action.get("target_id", ""))
        issue["status"] = "resolved"
        _touch(issue, "resolved", message, quote)
        return "open_issues", issue

    raise ValueError(f"Unhandled action {name!r}.")


def _hold_ambiguous(card: dict, action: dict, message: dict) -> None:
    """Ambiguous readings are never written to the card. The admin can confirm their own; others are skipped."""
    speaker = message["speaker_id"]
    if speaker != ADMIN_ID:
        card["audit"].append({"at": now_iso(), "event": "interpretation_skipped",
                              "detail": f"Ambiguous {action['action']} not recorded; no way to confirm it with the speaker.",
                              "message_id": message["id"], "quote": action.get("quote")})
        return None
    pending = card["individual_mapping"]["pending_interpretations"]
    summary = action.get("text") or action.get("stance") or json.dumps(
        {k: v for k, v in action.items() if k not in {"quote", "reason", "ambiguous", "message_id"} and v})
    pending.append({"id": f"interp_{len(pending) + 1}", "participant_id": speaker,
                    "proposed_type": action["action"], "proposed_statement": summary,
                    "source_message_ids": [message["id"]], "action": json_copy(action),
                    "base_card_version": card["card_version"] + 1,
                    "confirmation": {"status": "pending", "confirmed_by": None, "source_message_id": None},
                    "included_in_confirmed_state": False})
    card["audit"].append({"at": now_iso(), "event": "interpretation_pending",
                          "detail": "Ambiguous item held for confirmation.", "message_id": message["id"]})
    return None
