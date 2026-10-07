"""Run with: uvicorn mvp1_c2c.server:app --port 8505 --reload
Or from repo root: python -m uvicorn mvp1_c2c.server:app --port 8505
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from mvp1_c2c.core import AnthropicBackend, Simulation, now_iso, public_card
except ImportError:
    from core import AnthropicBackend, Simulation, now_iso, public_card

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
STATIC = ROOT / "static"

app = FastAPI(title="Pact MVP1 C2C")

# In-memory session store: experiment_id -> Simulation
_sessions: dict[str, Simulation] = {}


# ── helpers ──────────────────────────────────────────────────────────────────

def _sim(sid: str) -> Simulation:
    if sid not in _sessions:
        # Try to auto-reload from disk (e.g. after server restart)
        path = RUNS / f"{sid}.json"
        if not path.exists():
            raise HTTPException(404, f"Session {sid!r} not found.")
        try:
            data = json.loads(path.read_text())
            settings = data["card"].get("run_settings", {})
            pm = settings.get("participant_model")
            rm = settings.get("research_model")
            backend = AnthropicBackend(
                model=pm if pm and pm.startswith("claude-") else None,
                research_model=rm if rm and rm.startswith("claude-") else None,
            )
            _sessions[sid] = Simulation.load(path, backend)
        except Exception as exc:
            raise HTTPException(500, f"Could not reload session: {exc}")
    return _sessions[sid]


def _card_response(sim: Simulation) -> dict:
    return {"card": sim.card, "calls": getattr(sim.backend, "calls", 0)}


# ── session lifecycle ─────────────────────────────────────────────────────────

class NewSessionRequest(BaseModel):
    question: str
    human_name: str
    bot_profiles: list[dict]
    max_bot_turns: int = 18
    mapping_allowed: bool = True


@app.post("/api/sessions")
def create_session(req: NewSessionRequest):
    try:
        backend = AnthropicBackend()
        sim = Simulation.create(
            req.question, req.human_name, req.bot_profiles,
            req.max_bot_turns, backend,
            path=RUNS / "pending.json", mapping_allowed=req.mapping_allowed,
        )
        sid = sim.card["experiment_id"]
        sim.path = RUNS / f"{sid}.json"
        sim.save()
        (RUNS / "pending.json").unlink(missing_ok=True)
        _sessions[sid] = sim
        return _card_response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/runs")
def list_runs():
    files = sorted(RUNS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"files": [p.name for p in files]}


class LoadRequest(BaseModel):
    filename: str


@app.post("/api/sessions/load")
def load_session(req: LoadRequest):
    path = RUNS / req.filename
    if not path.exists():
        raise HTTPException(404, f"{req.filename} not found")
    try:
        data = json.loads(path.read_text())
        settings = data["card"].get("run_settings", {})
        pm = settings.get("participant_model")
        rm = settings.get("research_model")
        backend = AnthropicBackend(
            model=pm if pm and pm.startswith("claude-") else None,
            research_model=rm if rm and rm.startswith("claude-") else None,
        )
        sim = Simulation.load(path, backend)
        sid = sim.card["experiment_id"]
        _sessions[sid] = sim
        return _card_response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    return _card_response(_sim(sid))


# ── conversation actions ──────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    text: str


def _parse_pact_command(text: str, sim: Simulation) -> dict | None:
    """If the message is a 'pact, ...' command, interpret via LLM and execute."""
    if not re.match(r"^\s*pact[,\s]", text, re.IGNORECASE):
        return None

    intent = sim.backend.parse_command(text, sim.card)
    action = intent.get("action", "unknown")

    if action == "check_claims":
        claim_ids = intent.get("claim_ids", [])
        all_claims = sim.card.get("claims", [])
        if not claim_ids:
            participant_map = {p["id"]: p["name"] for p in sim.card["participants"]}
            if not all_claims:
                raise ValueError(
                    "No claims on the card yet. Claims are verifiable factual statements "
                    "(e.g. 'September temperatures at Arches average 95°F'). "
                    "Keep chatting — the observer will extract them as they appear."
                )
            available = "; ".join(
                f"{c['id']} by {participant_map.get(c.get('made_by',''), '?')}: {c['statement'][:60]}"
                for c in all_claims
            )
            raise ValueError(
                f"No claims matched that description. Claims on the card: {available}"
            )
        # Filter to only valid IDs that exist on the card
        valid_ids = {c["id"] for c in all_claims}
        claim_ids = [cid for cid in claim_ids if cid in valid_ids]
        if not claim_ids:
            raise ValueError("LLM returned claim IDs that don't exist on the card. Try again.")
        results = []
        for cid in claim_ids:
            try:
                results.append(sim.check_claim(cid))
            except Exception as exc:
                results.append({"claim_id": cid, "error": str(exc)})
        checked = len([r for r in results if "error" not in r])
        sim.add_message("human_admin",
                        f"Pact checked {checked}/{len(claim_ids)} claim(s). Reasoning: {intent.get('reasoning', '')}",
                        "explicit_request", observe=False)
        return _card_response(sim)

    if action == "check_ad_hoc":
        statement = intent.get("ad_hoc_statement", "").strip()
        if not statement:
            raise ValueError("Could not extract a verifiable statement from your request.")
        # Run check directly without requiring a card claim entry
        adhoc_claim = {"statement": statement, "made_by": "human_admin", "source_message_ids": []}
        result = sim.backend.check_claim(sim.card, adhoc_claim, sim.card["topic"]["title"])
        entry = {"id": f"check_{len(sim.card['claim_checks']) + 1}",
                 "claim_id": None, "requested_by": "human_admin",
                 "request_message_id": None, "requested_at": now_iso(), **result,
                 "ad_hoc_statement": statement}
        sim.card["claim_checks"].append(entry)
        sim.card["card_version"] += 1
        sim.add_message("human_admin", f"Pact checked (ad-hoc): {statement}", "explicit_request", observe=False)
        sim.save()
        return _card_response(sim)

    if action == "help":
        sim.ask_for_help()
        return _card_response(sim)

    return None  # unknown — fall through as normal message


@app.post("/api/sessions/{sid}/message")
def human_message(sid: str, req: MessageRequest):
    sim = _sim(sid)
    try:
        result = _parse_pact_command(req.text, sim)
        if result is not None:
            return result
        sim.add_human_message(req.text)
        return _card_response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


def _bg_observe(sim: Simulation, msg: dict):
    """Run observe in the background after a bot turn."""
    try:
        sim.observe(msg)
        sim.save()
    except Exception:
        pass  # observation errors are non-fatal; card stays at current version


@app.post("/api/sessions/{sid}/bot-turn")
def bot_turn(sid: str, background_tasks: BackgroundTasks):
    sim = _sim(sid)
    try:
        msg = sim.bot_turn(observe=False)
        background_tasks.add_task(_bg_observe, sim, msg)
        return _card_response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/sessions/{sid}/round")
def next_round(sid: str, background_tasks: BackgroundTasks):
    sim = _sim(sid)
    try:
        msgs = sim.next_round(observe=False)
        for msg in msgs:
            background_tasks.add_task(_bg_observe, sim, msg)
        return _card_response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ── Pact actions ──────────────────────────────────────────────────────────────

class AddClaimRequest(BaseModel):
    statement: str


@app.post("/api/sessions/{sid}/claims")
def add_claim(sid: str, req: AddClaimRequest):
    sim = _sim(sid)
    statement = req.statement.strip()
    if not statement:
        raise HTTPException(400, "Claim statement cannot be empty.")
    claim_id = f"claim_{len(sim.card['claims']) + 1}"
    sim.card["claims"].append({
        "id": claim_id,
        "statement": statement,
        "made_by": "human_admin",
        "source_message_ids": [],
        "verification": {"status": "not_checked", "check_ids": []},
    })
    sim.card["card_version"] += 1
    sim.save()
    return _card_response(sim)


@app.post("/api/sessions/{sid}/check-claim/{claim_id}")
def check_claim(sid: str, claim_id: str):
    sim = _sim(sid)
    try:
        result = sim.check_claim(claim_id)
        return _card_response(sim)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/sessions/{sid}/help")
def ask_help(sid: str):
    sim = _sim(sid)
    try:
        sim.ask_for_help()
        return _card_response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class DecisionRequest(BaseModel):
    proposal_id: str
    rationale: str = ""


@app.post("/api/sessions/{sid}/decision")
def record_decision(sid: str, req: DecisionRequest):
    sim = _sim(sid)
    try:
        sim.record_decision(req.proposal_id, req.rationale)
        return _card_response(sim)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc))


class ConfirmRequest(BaseModel):
    text: str


@app.post("/api/sessions/{sid}/confirm/{iid}")
def confirm_interpretation(sid: str, iid: str, req: ConfirmRequest):
    sim = _sim(sid)
    try:
        sim.confirm_interpretation(iid, req.text)
        return _card_response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ── static files ──────────────────────────────────────────────────────────────

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    html = STATIC / "index.html"
    if html.exists():
        return FileResponse(html)
    return JSONResponse({"status": "ok", "api": "/docs"})
