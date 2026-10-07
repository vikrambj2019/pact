"""A Pact session: one decision card and Pact's three functions on it.

  1. observe(message)      — runs on every message; fills the card with who said what, when, quoted.
  2. check_claims(...)     — on request; checks one claim, one person's claims, or all claims.
  3. interpret()           — on request; explains where the decision stands.

Plus the admin's own authority (confirm an interpretation, record the decision), which no model can do.
Pact accepts messages from any participant and never decides who speaks next; the simulation
harness (or, later, a real chat) does that.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from .card import (ADMIN_ID, SCHEMA_VERSION, SECTIONS, apply_action, json_copy, new_card, now_iso,
                   observer_view)
from .check import select_claims

CONTEXT_MESSAGES = 15
PACT_COMMAND = re.compile(r"^\s*pact[,\s]", re.IGNORECASE)


class PactSession:
    def __init__(self, card: dict, llm, path: Path | None = None, extras: dict | None = None):
        # `extras` is opaque data saved next to the card (e.g. a harness's own state); Pact never reads it.
        self.card, self.llm, self.path = card, llm, path
        self.extras = extras if extras is not None else {}
        self.lock = threading.RLock()  # observation can finish in the background while other requests run

    @classmethod
    def create(cls, question: str, admin_name: str, members: list[dict], llm, path: Path | None = None,
               admin_mapping_allowed: bool = True, extras: dict | None = None):
        if not question.strip() or not admin_name.strip():
            raise ValueError("A decision question and your participant name are required.")
        card = new_card(question.strip(), admin_name.strip(), members, admin_mapping_allowed)
        card["run_settings"] = {"pact_model": getattr(llm, "model", None)}
        session = cls(card, llm, path, extras)
        session.save()
        return session

    @classmethod
    def load(cls, path: Path, llm):
        data = json.loads(path.read_text())
        card = data.pop("card")
        if card.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"This session uses an older card format ({card.get('schema_version')}) "
                             f"and cannot be opened. Start a new session.")
        return cls(card, llm, path, extras=data)

    @property
    def session_id(self) -> str:
        return self.card["experiment_id"]

    def save(self):
        if not self.path:
            return
        with self.lock:
            text = json.dumps({"card": self.card, **self.extras}, ensure_ascii=False, indent=2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, self.path)

    def snapshot(self) -> dict:
        """A consistent copy of the card, safe to serialize while observation runs in the background."""
        with self.lock:
            return json_copy(self.card)

    def log(self, event, detail, **extra):
        self.card["audit"].append({"at": now_iso(), "event": event, "detail": detail, **extra})

    # ── 1. OBSERVE ───────────────────────────────────────────────────────────

    def add_message(self, speaker_id: str, text: str, source: str, observe: bool = True,
                    reply_to: str | None = None):
        text = text.strip()
        with self.lock:
            known = {p["id"] for p in self.card["participants"]}
            if speaker_id not in known:
                raise ValueError("Unknown participant.")
            if not text:
                raise ValueError("Message cannot be empty.")
            message = {"id": f"msg_{len(self.card['messages']) + 1}", "speaker_id": speaker_id,
                       "text": text, "source": source, "recorded_at": now_iso()}
            if reply_to:
                if reply_to not in {m["id"] for m in self.card["messages"]}:
                    raise ValueError("Reply target is not in this conversation.")
                message["reply_to"] = reply_to
            self.card["messages"].append(message)
            self.card["card_version"] += 1
            self.save()
        if observe:
            self.observe(message)
        return message

    def add_admin_message(self, text: str):
        return self.add_message(ADMIN_ID, text, "human")

    def observe(self, message: dict) -> dict:
        return self.observe_messages([message])

    def observe_messages(self, messages: list[dict]) -> dict:
        """Ask the observer for actions on messages already in the conversation, then apply each action on
        its own: a rejected action is logged with its reason and never blocks the others. The model call
        runs outside the lock; applying runs inside it. The card version goes up once per observation."""
        with self.lock:
            ids = [m["id"] for m in self.card["messages"]]
            start = ids.index(messages[0]["id"])
            context = json_copy(self.card["messages"][max(0, start - CONTEXT_MESSAGES):start])
            view = observer_view(self.card)
        try:
            actions = self.llm.observer.propose(view, messages, context)
        except Exception as exc:
            with self.lock:
                self.log("observation_error", f"{type(exc).__name__}: {exc}"[:300],
                         message_ids=[m["id"] for m in messages])
                self.card["card_version"] += 1
            self.save()
            return {"applied": 0, "rejected": 0, "error": type(exc).__name__}
        by_id = {m["id"]: m for m in messages}
        refs, applied, rejected = {}, 0, 0
        with self.lock:
            for action in actions:
                message = by_id.get(action.get("message_id"))
                try:
                    if message is None:
                        raise ValueError("Action cites a message that is not part of this observation.")
                    apply_action(self.card, action, message, refs)
                    applied += 1
                except ValueError as exc:
                    rejected += 1
                    self.log("observation_rejected", str(exc), action=action.get("action"),
                             message_id=action.get("message_id"), quote=action.get("quote"))
            self.log("observation", f"Applied {applied} action(s), rejected {rejected}.",
                     message_ids=list(by_id))
            self.card["card_version"] += 1
        self.save()
        return {"applied": applied, "rejected": rejected}

    # ── 2. CHECK ─────────────────────────────────────────────────────────────

    def check_claims(self, claim_ids: list[str] | None = None, made_by: str | None = None) -> dict:
        """Check claims by ID, every claim one participant made, or (neither given) every claim.
        Claims the observer marked not checkable are skipped and reported, never researched."""
        selected = select_claims(self.card, claim_ids, made_by)
        checkable = [c for c in selected if c.get("checkable")]
        skipped = [c["id"] for c in selected if not c.get("checkable")]
        if not checkable:
            raise ValueError("None of those claims can be checked against public sources "
                             "(opinions, predictions, and personal facts are not checked).")
        results = []
        for claim in checkable:
            try:
                results.append(self.check_claim(claim["id"]))
            except Exception as exc:
                results.append({"claim_id": claim["id"], "error": str(exc)})
        if skipped:
            self.log("claims_not_checkable", "Skipped claims that are not checkable.", claim_ids=skipped)
            self.save()
        return {"checks": results, "skipped_not_checkable": skipped}

    def check_claim(self, claim_id: str):
        claim = next((c for c in self.card["claims"] if c["id"] == claim_id), None)
        if not claim:
            raise ValueError("Select a claim recorded on the card.")
        if not claim.get("checkable"):
            raise ValueError("This claim is not checkable against public sources.")
        request_message = self.add_message(ADMIN_ID, f"Pact, check this claim: {claim.get('corrected_to') or claim['statement']}",
                                           "explicit_request", observe=False)
        try:
            current = {**claim, "statement": claim.get("corrected_to") or claim["statement"]}
            result = self.llm.checker.check(self.card, current, self.card["topic"]["title"])
        except Exception as exc:
            self.log("claim_check_error", type(exc).__name__, claim_id=claim_id,
                     request_message_id=request_message["id"])
            self.save()
            raise
        entry = {"id": f"check_{len(self.card['claim_checks']) + 1}", "claim_id": claim_id,
                 "requested_by": ADMIN_ID, "request_message_id": request_message["id"],
                 "requested_at": now_iso(), **result}
        self.card["claim_checks"].append(entry)
        self.card["evidence"].extend({"id": f"{entry['id']}_source_{i+1}", "claim_check_id": entry["id"], **src}
                                     for i, src in enumerate(result.get("sources", [])))
        verification = claim.setdefault("verification", {"status": "not_checked", "check_ids": []})
        verification["status"] = "checked"
        verification.setdefault("check_ids", []).append(entry["id"])
        self.log("claim_check", "Explicitly requested claim research completed.", claim_id=claim_id,
                 status=result.get("status"))
        self.card["card_version"] += 1
        self.save()
        return entry

    def check_statement(self, statement: str):
        """Check a fact the admin names that is not recorded as a claim on the card."""
        statement = statement.strip()
        if not statement:
            raise ValueError("Could not extract a verifiable statement from your request.")
        adhoc_claim = {"statement": statement, "made_by": ADMIN_ID, "source_message_ids": []}
        result = self.llm.checker.check(self.card, adhoc_claim, self.card["topic"]["title"])
        entry = {"id": f"check_{len(self.card['claim_checks']) + 1}",
                 "claim_id": None, "requested_by": ADMIN_ID,
                 "request_message_id": None, "requested_at": now_iso(), **result,
                 "ad_hoc_statement": statement}
        self.card["claim_checks"].append(entry)
        self.card["card_version"] += 1
        self.add_message(ADMIN_ID, f"Pact checked (ad-hoc): {statement}", "explicit_request", observe=False)
        return entry

    def add_manual_claim(self, statement: str):
        statement = statement.strip()
        if not statement:
            raise ValueError("Claim statement cannot be empty.")
        claim = {"id": f"claim_{len(self.card['claims']) + 1}", "statement": statement,
                 "made_by": ADMIN_ID, "kind": "fact", "checkable": True, "status": "unchallenged",
                 "challenges": [], "verification": {"status": "not_checked", "check_ids": []},
                 "source_message_ids": [], "history": [{"event": "added_by_admin", "at": now_iso()}]}
        self.card["claims"].append(claim)
        self.card["card_version"] += 1
        self.save()
        return claim

    # ── 3. INTERPRET ─────────────────────────────────────────────────────────

    def interpret(self):
        request_message = self.add_message(ADMIN_ID, "Pact, help us understand where this decision stands.",
                                           "explicit_request", observe=False)
        try:
            result = self.llm.interpreter.interpret(self.card)
        except Exception as exc:
            self.log("help_error", type(exc).__name__, request_message_id=request_message["id"])
            self.save()
            raise
        if isinstance(result, dict):
            valid_ids = {m["id"] for m in self.card["messages"]}
            if not set(result.get("source_message_ids", [])) <= valid_ids:
                raise ValueError("Pact help cited a message ID that is not in this conversation.")
        item = {"id": f"help_{len(self.card['facilitation']) + 1}", "requested_by": ADMIN_ID,
                "request_message_id": request_message["id"], "requested_at": now_iso(),
                "output": result, "status": "Pact interpretation"}
        self.card["facilitation"].append(item)
        self.log("help", "Pact help explicitly requested.", help_id=item["id"])
        self.save()
        return item

    # ── 'pact, ...' commands route to check or interpret ────────────────────

    def handle_command(self, text: str) -> bool:
        """Handle a 'pact, ...' message. Returns False when it is not a command Pact acts on."""
        if not PACT_COMMAND.match(text):
            return False
        intent = self.llm.commands.parse(text, self.card)
        action = intent.get("action", "unknown")
        if action == "check_claims":
            claim_ids = [] if intent.get("check_all") else intent.get("claim_ids", [])
            made_by = None if intent.get("check_all") else (intent.get("made_by") or None)
            if not (claim_ids or made_by or intent.get("check_all")):
                raise ValueError("Say which claims to check: a claim, a person's claims, or all claims.")
            outcome = self.check_claims(claim_ids or None, made_by)
            checked = len([r for r in outcome["checks"] if "error" not in r])
            skipped = len(outcome["skipped_not_checkable"])
            self.add_message(ADMIN_ID, f"Pact checked {checked} claim(s); skipped {skipped} not checkable.",
                             "explicit_request", observe=False)
            return True
        if action == "check_statement":
            self.check_statement(intent.get("statement", ""))
            return True
        if action == "interpret":
            self.interpret()
            return True
        return False

    # ── admin authority ──────────────────────────────────────────────────────

    def confirm_interpretation(self, interpretation_id: str, correction_or_confirmation: str):
        pending = next((p for p in self.card["individual_mapping"]["pending_interpretations"]
                        if p["id"] == interpretation_id), None)
        if not pending or pending["confirmation"]["status"] != "pending":
            raise ValueError("No pending interpretation with that ID.")
        if pending["participant_id"] != ADMIN_ID:
            raise ValueError("Only you can confirm your own interpretation in this MVP.")
        if pending["base_card_version"] != self.card["card_version"]:
            raise ValueError("The card changed since this interpretation; review the latest conversation first.")
        text = correction_or_confirmation.strip()
        if not text:
            raise ValueError("Enter a confirmation or correction message.")
        is_confirmation = text.lower().startswith("confirm")
        confirmation = self.add_message(ADMIN_ID, text,
                                        "human_confirmation" if is_confirmation else "human_correction",
                                        observe=not is_confirmation)
        pending = next(p for p in self.card["individual_mapping"]["pending_interpretations"]
                       if p["id"] == interpretation_id)
        if is_confirmation:
            original_id = pending["source_message_ids"][0]
            original = next(m for m in self.card["messages"] if m["id"] == original_id)
            action = {**json_copy(pending["action"]), "ambiguous": False}
            with self.lock:
                apply_action(self.card, action, original)
            pending["confirmation"] = {"status": "confirmed", "confirmed_by": ADMIN_ID,
                                        "source_message_id": confirmation["id"]}
            pending["included_in_confirmed_state"] = True
        else:
            pending["confirmation"] = {"status": "corrected", "confirmed_by": ADMIN_ID,
                                        "source_message_id": confirmation["id"]}
            pending["correction"] = text
        self.log("interpretation_review", pending["confirmation"]["status"], interpretation_id=interpretation_id)
        self.card["card_version"] += 1
        self.save()
        return pending

    def record_decision(self, option_id: str, rationale: str):
        approvers = self.card["decision_process"]["approver_ids"]
        if ADMIN_ID not in approvers:
            raise ValueError("The human participant is not an authorized approver in this session.")
        if not any(o["id"] == option_id for o in self.card["options"]):
            raise ValueError("Choose an option on the card.")
        if not rationale.strip():
            raise ValueError("Add the rationale for the decision.")
        authorization = self.add_message(ADMIN_ID,
                                         f"I record the decision to select {option_id}. Rationale: {rationale.strip()}",
                                         "decision_record", observe=False)
        decision = {"status": "recorded", "selected_option_id": option_id,
                    "decided_at": now_iso(), "authorized_by": [ADMIN_ID],
                    "rationale": {"statement": rationale.strip(),
                                  "source_message_ids": [authorization["id"]]}}
        snapshot = {"id": f"snapshot_{len(self.card['snapshots']) + 1}", "recorded_at": now_iso(),
                    "decision": json_copy(decision), "topic": json_copy(self.card["topic"]),
                    **{section: json_copy(self.card[section]) for section in SECTIONS},
                    "source_message_ids": [m["id"] for m in self.card["messages"]]}
        self.card["snapshots"].append(snapshot)
        decision["decided_at"] = snapshot["recorded_at"]
        decision["recorded_snapshot_id"] = snapshot["id"]
        self.card["decision"] = decision
        self.card["card_version"] += 1
        self.log("decision_recorded", "Human admin recorded an authorized decision.", option_id=option_id)
        self.save()
