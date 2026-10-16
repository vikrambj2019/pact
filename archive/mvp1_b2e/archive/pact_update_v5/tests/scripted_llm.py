"""A stand-in for LLMClient that answers each tool call from a script, plus a helper that
runs the DB-free pipeline (extract_and_merge → build_brief) the way eval/run_eval.py does."""
from __future__ import annotations

import asyncio
import copy
import json

from pact.worker.synthesizer import _unique, build_brief, extract_and_merge


class ScriptedLLM:
    def __init__(self, chunks: list[dict], synth, summary: dict):
        self._chunks = [copy.deepcopy(c) for c in chunks]
        self._synth = synth
        self._summary = summary
        self.calls: list[str] = []

    async def extract(self, system, user, tool, timeout=None, max_tokens=None):
        name = tool["name"]
        self.calls.append(name)
        if name == "extract_candidates":
            return self._chunks.pop(0)
        if name == "produce_brief":
            cands = json.loads(user.split("Extracted candidates from the full discussion:\n", 1)[1]
                               .rsplit("\n\nSynthesize", 1)[0])
            return self._synth(cands)
        if name == "produce_exec_summary":
            return copy.deepcopy(self._summary)
        if name == "check_support":            # the entailment check passes everything
            return {"verdicts": [{"id": i["id"], "supported": True} for i in json.loads(user)]}
        raise AssertionError(name)


def run_pipeline(units: list[dict], llm: ScriptedLLM, question: str) -> dict:
    async def go():
        merged, _ = await extract_and_merge(units, llm)
        text = {u["id"]: u["normalized_text"] for u in units}
        meta = {u["id"]: {"author": u["author_label"], "ts": u["locator"]["timestamp_str"]} for u in units}
        brief, *_ = await build_brief(merged, question, None, text, meta,
                                      _unique(u["author_label"] for u in units), llm)
        return brief
    return asyncio.run(go())


def cited(L, text: str, line: int, quote: str) -> dict:
    return {"text": text, "verbatim_quote": quote, "source_unit_id": L(line)}
