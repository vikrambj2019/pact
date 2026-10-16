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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .card import (ADMIN_ID, SCHEMA_VERSION, SECTIONS, apply_action, json_copy, new_card, now_iso,
                   observer_view, provenance)
from .check import select_claims
from .interpret import render_interpretation

CONTEXT_MESSAGES = 15
MAX_PARALLEL_CHECKS = 4
PACT_COMMAND = re.compile(r"^\s*pact[,\s]", re.IGNORECASE)


class PactSession:
    def __init__(self, card: dict, llm, path: Path | None = None, extras: dict | None = None):
        # `extras` is opaque data saved next to the card (e.g. a harness's own state); Pact never reads it.
        self.card, self.llm, self.path = card, llm, path
        self.extras = extras if extras is not None else {}
        self.lock = threading.RLock()  # guards card state and persistence
        self.observation_lock = threading.RLock()  # one ordered observer worker per session
        # Older v3 sessions already have observation audit records; do not replay those messages.
        if "observation" not in self.card:
            completed = list(dict.fromkeys(mid for event in self.card.get("audit", [])
                                           if event.get("event") == "observation"
                                           for mid in event.get("message_ids", [])))
            self.card["observation"] = {"completed": completed, "pending": [
                m["id"] for m in self.card["messages"] if m["id"] not in completed
                and m.get("source") not in {"pact_command", "explicit_request", "decision_record",
                                            "human_confirmation"}]}

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
            if source not in {"pact_command", "explicit_request", "decision_record", "human_confirmation"}:
                self.card["observation"]["pending"].append(message["id"])
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
        """Drain pending messages through the requested message in conversation order.

        Background tasks may arrive out of order. A later task also processes earlier pending
        messages; completed messages are never replayed. Calls run outside the card lock, but one
        observer worker runs at a time. If the observer's card view changed during a call, discard
        its proposals and retry against the new view. Failed work stays pending for a later retry.
        """
        if not messages:
            raise ValueError("Observation needs at least one message.")
        with self.observation_lock:
            with self.lock:
                ids = [m["id"] for m in self.card["messages"]]
                requested = [m["id"] for m in messages]
                if any(mid not in ids for mid in requested):
                    raise ValueError("Observation message is not in this conversation.")
                end = max(ids.index(mid) for mid in requested)
                pending = set(self.card["observation"]["pending"])
                work = json_copy([m for m in self.card["messages"][:end + 1] if m["id"] in pending])
                if not work:
                    return {"applied": 0, "rejected": 0}
                start = ids.index(work[0]["id"])
                context = json_copy(self.card["messages"][max(0, start - CONTEXT_MESSAGES):start])
            for attempt in range(3):
                with self.lock:
                    view = observer_view(self.card)
                try:
                    actions = self.llm.observer.propose(view, work, context)
                except Exception as exc:
                    with self.lock:
                        self.log("observation_error", f"{type(exc).__name__}: {exc}"[:300],
                                 message_ids=[m["id"] for m in work])
                        self.card["card_version"] += 1
                        self.save()
                    return {"applied": 0, "rejected": 0, "error": type(exc).__name__}
                with self.lock:
                    if observer_view(self.card) != view:
                        self.log("observation_stale", "Card changed; discarded observer proposals.",
                                 message_ids=[m["id"] for m in work], attempt=attempt + 1)
                        continue
                    by_id = {m["id"]: m for m in work}
                    order = {m["id"]: i for i, m in enumerate(work)}
                    refs, applied, rejected = {}, 0, 0
                    # Preserve action order within a message, even when a model groups messages backwards.
                    for action in sorted(actions, key=lambda a: order.get(a.get("message_id"), len(work))):
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
                    state = self.card["observation"]
                    state["completed"].extend(by_id)
                    state["pending"] = [mid for mid in state["pending"] if mid not in by_id]
                    self.log("observation", f"Applied {applied} action(s), rejected {rejected}.",
                             message_ids=list(by_id))
                    self.card["card_version"] += 1
                    self.save()
                    return {"applied": applied, "rejected": rejected}
            with self.lock:
                self.log("observation_error", "Card kept changing; messages remain pending.",
                         message_ids=[m["id"] for m in work])
                self.save()
            return {"applied": 0, "rejected": 0, "error": "StaleObservation"}

    # ── 2. CHECK ─────────────────────────────────────────────────────────────

    def check_claims(self, claim_ids: list[str] | None = None, made_by: str | None = None,
                     request_message: dict | None = None) -> dict:
        """Check claims by ID, every claim one participant made, or (neither given) every claim.
        Claims the observer marked not checkable are skipped and reported, never researched.
        Research runs in parallel; results are recorded in the order the claims appear on the card."""
        with self.lock:
            selected = select_claims(self.card, claim_ids, made_by)
            checkable = [json_copy(c) for c in selected if c.get("checkable") and c["status"] != "retracted"]
            skipped = [c["id"] for c in selected if not c.get("checkable") or c["status"] == "retracted"]
        if not checkable:
            raise ValueError("None of those claims can be checked against public sources "
                             "(opinions, predictions, and personal facts are not checked).")
        if request_message is None:
            request_message = self.add_message(
                ADMIN_ID, "Pact, check " + "; ".join(self._claim_text(c) for c in checkable),
                "explicit_request", observe=False)
        card = self.snapshot()
        question = card["topic"]["title"]

        def research(claim):
            try:
                return self.llm.checker.check(card, {**claim, "statement": self._claim_text(claim)}, question)
            except Exception as exc:  # recorded per claim below
                return exc

        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_CHECKS, len(checkable))) as pool:
            outcomes = list(pool.map(research, checkable))
        checks = []
        with self.lock:
            for claim, outcome in zip(checkable, outcomes):
                if isinstance(outcome, Exception):
                    self.log("claim_check_error", f"{type(outcome).__name__}: {outcome}"[:300],
                             claim_id=claim["id"], request_message_id=request_message["id"])
                    checks.append({"claim_id": claim["id"], "error": outcome})
                else:
                    checks.append(self._record_check(claim["id"], outcome, request_message, claim.get("revision", 0)))
            if skipped:
                self.log("claims_not_checkable", "Skipped claims that are not checkable.", claim_ids=skipped)
        self.save()
        return {"checks": checks, "skipped_not_checkable": skipped}

    def check_claim(self, claim_id: str, request_message: dict | None = None) -> dict:
        claim = next((c for c in self.card["claims"] if c["id"] == claim_id), None)
        if not claim:
            raise ValueError("Select a claim recorded on the card.")
        if not claim.get("checkable"):
            raise ValueError("This claim is not checkable against public sources.")
        [entry] = self.check_claims([claim_id], request_message=request_message)["checks"]
        if "error" in entry:
            raise entry["error"]
        return entry

    def check_statement(self, statement: str, request_message: dict | None = None) -> dict:
        """Check a fact the admin names: it becomes the admin's claim on the card, then is checked like any other."""
        claim = self.add_admin_claim(statement, request_message)
        return self.check_claim(claim["id"], request_message)

    def add_admin_claim(self, statement: str, source_message: dict | None = None) -> dict:
        statement = statement.strip()
        if not statement:
            raise ValueError("Claim statement cannot be empty.")
        with self.lock:
            origin = (provenance(source_message, source_message["text"]) if source_message
                      else {"speaker_id": ADMIN_ID, "at": now_iso()})
            claim = {"id": f"claim_{len(self.card['claims']) + 1}", "statement": statement,
                     "made_by": ADMIN_ID, "kind": "fact", "option_id": None, "checkable": True,
                     "check_type": "other", "status": "unchallenged", "revision": 0, "challenges": [],
                     "verification": {"status": "not_checked", "check_ids": []},
                     "source_message_ids": [source_message["id"]] if source_message else [],
                     "history": [{"event": "added_by_admin", **origin}]}
            self.card["claims"].append(claim)
            self.card["card_version"] += 1
        self.save()
        return claim

    add_manual_claim = add_admin_claim

    @staticmethod
    def _claim_text(claim: dict) -> str:
        return claim.get("corrected_to") or claim["statement"]

    def _record_check(self, claim_id: str, result: dict, request_message: dict, checked_revision: int) -> dict:
        """Write a check result: the check entry, its evidence, and the verdict on the claim. Caller holds the lock."""
        claim = next(c for c in self.card["claims"] if c["id"] == claim_id)
        stale = claim.get("revision", 0) != checked_revision or claim["status"] == "retracted"
        entry = {"id": f"check_{len(self.card['claim_checks']) + 1}", "claim_id": claim_id,
                 "claim_revision": checked_revision, "stale": stale,
                 "requested_by": ADMIN_ID, "request_message_id": request_message["id"],
                 "requested_at": request_message.get("recorded_at"), "completed_at": now_iso(), **result}
        self.card["claim_checks"].append(entry)
        self.card["evidence"].extend({"id": f"{entry['id']}_ev_{i + 1}", "claim_check_id": entry["id"],
                                      "claim_id": claim_id, **ev} for i, ev in enumerate(result.get("evidence", [])))
        claim = next(c for c in self.card["claims"] if c["id"] == claim_id)
        check_ids = claim.get("verification", {}).get("check_ids", []) + [entry["id"]]
        if stale:
            # Preserve the historical report, but never endorse a newer statement with old evidence.
            claim["verification"]["check_ids"] = check_ids
            self.log("claim_check_stale", "Claim changed while research ran; result kept as history only.",
                     claim_id=claim_id, check_id=entry["id"])
            claim.setdefault("history", []).append({"event": "stale_check_completed", "check_id": entry["id"],
                                                    "message_id": request_message["id"],
                                                    "speaker_id": request_message["speaker_id"],
                                                    "at": request_message.get("recorded_at")})
            self.card["card_version"] += 1
            return entry
        claim["verification"] = {"status": "checked", "verdict": result.get("verdict"),
                                 "summary": result.get("summary", ""), "as_of": result.get("as_of", ""),
                                 "checked_at": entry["completed_at"], "check_ids": check_ids}
        claim.setdefault("history", []).append({"event": "checked", "verdict": result.get("verdict"),
                                                "check_id": entry["id"], "message_id": request_message["id"],
                                                "speaker_id": request_message["speaker_id"],
                                                "at": request_message.get("recorded_at")})
        self.log("claim_check", "Explicitly requested claim check completed.", claim_id=claim_id,
                 verdict=result.get("verdict"))
        self.card["card_version"] += 1
        return entry

    # ── 3. INTERPRET ─────────────────────────────────────────────────────────

    def interpret(self, request_message: dict | None = None):
        if request_message is None:
            request_message = self.add_message(ADMIN_ID, "Pact, help us understand where this decision stands.",
                                               "explicit_request", observe=False)
        # Interpret one immutable snapshot and save the version it describes.
        card = self.snapshot()
        try:
            selection = self.llm.interpreter.interpret(card)
            result = render_interpretation(card, selection)
        except Exception as exc:
            with self.lock:
                self.log("help_rejected" if isinstance(exc, ValueError) else "help_error", str(exc)[:300],
                         request_message_id=request_message["id"])
                self.save()
            raise
        result["summary"] = "\n".join([result["headline"], *(f"• {x}" for x in result["open_items"]),
                                       f"Next: {result['next_step']}"])
        with self.lock:
            if card["card_version"] != self.card["card_version"]:
                self.log("help_rejected", "Card changed during interpretation; ask again.",
                         request_message_id=request_message["id"])
                self.save()
                raise ValueError("The card changed during interpretation; ask again.")
            item = {"id": f"help_{len(self.card['facilitation']) + 1}", "requested_by": ADMIN_ID,
                    "request_message_id": request_message["id"], "requested_at": now_iso(),
                    "card_version": card["card_version"], "output": result, "status": "Pact interpretation"}
            self.card["facilitation"].append(item)
            self.log("help", "Pact help explicitly requested.", help_id=item["id"])
            self.save()
        return item

    # ── 'pact, ...' commands route to check or interpret ────────────────────

    def handle_command(self, text: str) -> bool:
        """Handle a 'pact, ...' message. The admin's own words are recorded as the request.
        Returns False when it is not a command Pact acts on (the caller records it as a normal message)."""
        if not PACT_COMMAND.match(text):
            return False
        intent = self.llm.commands.parse(text, self.card)
        action = intent.get("action", "unknown")
        if action == "check_claims" and not (intent.get("check_all") or intent.get("claim_ids")
                                             or intent.get("made_by")):
            raise ValueError("Say which claims to check: a claim, a person's claims, or all claims.")
        if action == "check_statement" and not (intent.get("statement") or "").strip():
            raise ValueError("Could not tell which fact to check.")
        if action not in {"check_claims", "check_statement", "interpret"}:
            return False
        request = self.add_message(ADMIN_ID, text, "pact_command", observe=False)
        if action == "check_claims":
            if intent.get("check_all"):
                self.check_claims(request_message=request)
            else:
                self.check_claims(intent.get("claim_ids") or None, intent.get("made_by") or None, request)
        elif action == "check_statement":
            self.check_statement(intent["statement"], request)
        else:
            self.interpret(request)
        return True

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
