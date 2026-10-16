"""Thin async LLM wrapper for the enterprise eval harness.

Matches the interface expected by extractor.extract_chunk and
synthesizer._generate_exec_summary: llm.extract(system, user, tool, timeout).
"""
from __future__ import annotations

import asyncio
import os
from functools import partial
from pathlib import Path

import anthropic

SONNET = "claude-sonnet-4-6"


def _load_api_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for directory in [Path.cwd(), *Path.cwd().parents]:
        env_file = directory / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() in ("ANTHROPIC_API_KEY", "LLM_API_KEY"):
                        return v.strip().strip("\"'")
    return None


class LLMClient:
    def __init__(self, model: str = SONNET):
        api_key = _load_api_key()
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    async def extract(self, system: str, user: str, tool: dict, timeout: int = 150) -> dict:
        loop = asyncio.get_event_loop()
        fn = partial(self._call, system, user, tool)
        return await asyncio.wait_for(loop.run_in_executor(None, fn), timeout=timeout)

    def _call(self, system: str, user: str, tool: dict) -> dict:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": user}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return block.input
        raise RuntimeError(f"no tool_use block in response: {resp.content}")
