"""Lightweight Anthropic client for structured extraction."""
from __future__ import annotations

from anthropic import AsyncAnthropic
from ..api.config import settings


class LLMClient:
    def __init__(self):
        self._client = AsyncAnthropic(api_key=settings.llm_api_key)

    async def extract(self, system: str, user: str, tool_schema: dict, timeout: int | None = None) -> dict:
        """Call Claude with a single forced tool; return the tool_use input dict."""
        tool_name = tool_schema["name"]
        resp = await self._client.messages.create(
            model=settings.llm_model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": tool_name},
            timeout=timeout or settings.llm_timeout_seconds,
        )
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("Structured response truncated; reduce chunk size or increase output limit")
        for block in resp.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input
        raise RuntimeError(f"LLM returned no tool_use block for tool '{tool_name}'")

    async def complete(self, system: str, user: str, max_tokens: int = 1024, timeout: int | None = None) -> str:
        """Call Claude for a plain-text response; return the text content."""
        resp = await self._client.messages.create(
            model=settings.llm_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=timeout or settings.llm_timeout_seconds,
        )
        for block in resp.content:
            if block.type == "text":
                return block.text
        raise RuntimeError("LLM returned no text block")
