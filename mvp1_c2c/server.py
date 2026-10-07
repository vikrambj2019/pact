"""Run from the repo root: python -m uvicorn mvp1_c2c.server:app --port 8505 --reload

Routes are split by who acts:
  * Pact routes      — the product: messages, claim checks, help, confirmations, decisions.
  * Simulation routes — the test harness: ask a simulated participant (or a round of them) to speak.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .pact import AnthropicClient, PactLLM
from .simulation import ParticipantLLM, Simulation, saved_models

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)
STATIC = ROOT / "static"

app = FastAPI(title="Pact MVP1 C2C")

# In-memory session store: experiment_id -> Simulation (which wraps the PactSession)
_sessions: dict[str, Simulation] = {}


# ── helpers ──────────────────────────────────────────────────────────────────

def _open(path: Path) -> Simulation:
    pact_model, participant_model = saved_models(path)
    client = AnthropicClient()
    return Simulation.load(path, PactLLM(client, pact_model), ParticipantLLM(client, participant_model))


def _sim(sid: str) -> Simulation:
    if sid not in _sessions:
        # Try to auto-reload from disk (e.g. after server restart)
        path = RUNS / f"{sid}.json"
        if not path.exists():
            raise HTTPException(404, f"Session {sid!r} not found.")
        try:
            _sessions[sid] = _open(path)
        except Exception as exc:
            raise HTTPException(500, f"Could not reload session: {exc}")
    return _sessions[sid]


def _response(sim: Simulation) -> dict:
    state = sim.state
    return {"card": sim.pact.card,
            "simulation": {"bot_turns": state["bot_turns"], "max_bot_turns": state["max_bot_turns"]},
            "calls": getattr(sim.pact.llm.client, "calls", 0)}


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
        client = AnthropicClient()
        sim = Simulation.create(
            req.question, req.human_name, req.bot_profiles, req.max_bot_turns,
            PactLLM(client), ParticipantLLM(client),
            path=RUNS / "pending.json", mapping_allowed=req.mapping_allowed,
        )
        sid = sim.pact.session_id
        sim.pact.path = RUNS / f"{sid}.json"
        sim.pact.save()
        (RUNS / "pending.json").unlink(missing_ok=True)
        _sessions[sid] = sim
        return _response(sim)
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
        sim = _open(path)
        _sessions[sim.pact.session_id] = sim
        return _response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    return _response(_sim(sid))


# ── Pact routes (the product) ─────────────────────────────────────────────────

class MessageRequest(BaseModel):
    text: str


@app.post("/api/sessions/{sid}/message")
def human_message(sid: str, req: MessageRequest):
    sim = _sim(sid)
    try:
        if not sim.pact.handle_command(req.text):
            sim.pact.add_admin_message(req.text)
        return _response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class AddClaimRequest(BaseModel):
    statement: str


@app.post("/api/sessions/{sid}/claims")
def add_claim(sid: str, req: AddClaimRequest):
    sim = _sim(sid)
    try:
        sim.pact.add_manual_claim(req.statement)
        return _response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/sessions/{sid}/check-claim/{claim_id}")
def check_claim(sid: str, claim_id: str):
    sim = _sim(sid)
    try:
        sim.pact.check_claim(claim_id)
        return _response(sim)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/sessions/{sid}/help")
def ask_help(sid: str):
    sim = _sim(sid)
    try:
        sim.pact.interpret()
        return _response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class DecisionRequest(BaseModel):
    proposal_id: str
    rationale: str = ""


@app.post("/api/sessions/{sid}/decision")
def record_decision(sid: str, req: DecisionRequest):
    sim = _sim(sid)
    try:
        sim.pact.record_decision(req.proposal_id, req.rationale)
        return _response(sim)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc))


class ConfirmRequest(BaseModel):
    text: str


@app.post("/api/sessions/{sid}/confirm/{iid}")
def confirm_interpretation(sid: str, iid: str, req: ConfirmRequest):
    sim = _sim(sid)
    try:
        sim.pact.confirm_interpretation(iid, req.text)
        return _response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ── Simulation routes (the test harness) ──────────────────────────────────────

def _bg_observe(sim: Simulation, msg: dict):
    """Let Pact observe a simulated message after the response is sent."""
    try:
        sim.pact.observe(msg)
        sim.pact.save()
    except Exception:
        pass  # observation errors are non-fatal; card stays at current version


@app.post("/api/sessions/{sid}/bot-turn")
def bot_turn(sid: str, background_tasks: BackgroundTasks):
    sim = _sim(sid)
    try:
        msg = sim.bot_turn(observe=False)
        background_tasks.add_task(_bg_observe, sim, msg)
        return _response(sim)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/sessions/{sid}/round")
def next_round(sid: str, background_tasks: BackgroundTasks):
    sim = _sim(sid)
    try:
        msgs = sim.next_round(observe=False)
        for msg in msgs:
            background_tasks.add_task(_bg_observe, sim, msg)
        return _response(sim)
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
