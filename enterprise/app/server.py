"""
Pact enterprise API server.

    cd /path/to/pact
    uvicorn enterprise.app.server:app --port 8507 --reload

Or run directly:
    python enterprise/app/server.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add enterprise/ and social/ to path (run_fast.py also does this, but be explicit)
_ENTERPRISE = Path(__file__).resolve().parent.parent
_SOCIAL = _ENTERPRISE.parent / "social"
for _p in (_ENTERPRISE, _SOCIAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_fast import run as _run_pipeline  # noqa: E402

app = FastAPI(title="Pact")

# In-memory job store: job_id -> {status, result?, error?, duration_seconds?}
_jobs: dict[str, dict] = {}

_STATIC = Path(__file__).parent / "static"
_SESSIONS = Path(__file__).parent / "sessions"
_SESSIONS.mkdir(exist_ok=True)


# ── models ────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str
    source_type: str = "auto"  # "auto" | "email" | "transcript"


# ── background worker ─────────────────────────────────────────────────────────

def _run_job(job_id: str, text: str, source_type: str) -> None:
    _jobs[job_id]["status"] = "running"
    tmp = None
    t0 = time.time()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            tmp = Path(f.name)

        result = _run_pipeline(tmp, log=lambda *_: None)
        duration = round(time.time() - t0, 1)

        _jobs[job_id] = {"status": "done", "result": result, "duration_seconds": duration}

        # Persist session
        session = {
            "session_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_type": source_type,
            "duration_seconds": duration,
            "char_count": len(text),
            "input_text": text,
            "result": result,
        }
        (_SESSIONS / f"{job_id}.json").write_text(
            json.dumps(session, indent=2), encoding="utf-8"
        )

    except Exception as exc:
        duration = round(time.time() - t0, 1)
        _jobs[job_id] = {"status": "failed", "error": str(exc), "duration_seconds": duration}
    finally:
        if tmp and tmp.exists():
            tmp.unlink()


# ── routes ────────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(body: AnalyzeRequest):
    if not body.text.strip():
        return JSONResponse({"error": "text is required"}, status_code=400)
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued"}
    t = threading.Thread(
        target=_run_job, args=(job_id, body.text, body.source_type), daemon=True
    )
    t.start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return job


@app.get("/")
async def root():
    return FileResponse(str(_STATIC / "index.html"))


app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


# ── dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("enterprise.app.server:app", host="0.0.0.0", port=8507, reload=True,
                reload_dirs=[str(_ENTERPRISE)])
