"""run_brief.py: text file in, one JSON object out, with the executive summary as a field."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from run_brief import brief_from_text

from .scripted_llm import ScriptedLLM
from .test_anniversary_chat import CHAT, chunk0, chunk1, chunk2, synthesized_with_traps
from .test_eval_pipeline import GOOD_SUMMARY

ROOT = Path(__file__).resolve().parent.parent


def test_brief_json_has_summary_as_a_field():
    llm = ScriptedLLM([chunk0(), chunk1(), chunk2()], synthesized_with_traps, GOOD_SUMMARY)
    result = asyncio.run(brief_from_text(CHAT, llm, source="anniversary_chat.txt", stable_ids=True))
    out = json.loads(json.dumps(result, default=str))                 # serialisable
    assert set(out) == {"id", "created_at", "source", "format", "question", "executive_summary",
                        "status", "emerging_preference", "brief", "report"}
    assert out["format"] == "whatsapp" and out["source"] == "anniversary_chat.txt"
    assert out["question"] == "Which date and venue for the parents' 40th anniversary dinner?"
    assert "COMMITTED" in out["executive_summary"] and "[Nadia, 10/6/26 1:31 PM]" in out["executive_summary"]
    assert out["status"].startswith("9 options proposed (4 dropped")
    assert "executive_summary" not in out["brief"] and out["brief"]["options"]
    assert out["report"]["unit_count"] == 87 and out["report"]["chunk_count"] == 3
    assert out["report"]["summary"]["omitted_items"] == 0


def test_cli_help():
    r = subprocess.run([sys.executable, "run_brief.py", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "--summary-only" in r.stdout
