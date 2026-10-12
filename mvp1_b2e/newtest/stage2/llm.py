"""Thin LLM interface so stages can run on Sonnet in prod and on fakes in evals."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-4-6"


def _load_api_key() -> str:
    for directory in [Path.cwd(), *Path.cwd().parents]:
        env_file = directory / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() in ("ANTHROPIC_API_KEY", "LLM_API_KEY"):
                        return v.strip().strip("\"'")
    raise RuntimeError("No ANTHROPIC_API_KEY found in any .env file up from CWD")


class LLM(Protocol):
    def structured(self, *, system: str, user: str, tool_name: str,
                   schema: dict[str, Any], max_tokens: int = 2048) -> dict[str, Any]:
        """Return a dict matching `schema`."""
        ...


class AnthropicLLM:
    """Forces a single tool call so output is always schema-shaped JSON."""

    def __init__(self, model: str = SONNET, client=None):
        import anthropic
        self.client = client or anthropic.Anthropic(api_key=_load_api_key())
        self.model = model

    def structured(self, *, system, user, tool_name, schema, max_tokens=2048):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            tools=[{"name": tool_name, "description": f"Return the {tool_name} result.",
                    "input_schema": schema}],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return block.input
        raise RuntimeError(f"No tool_use block returned for {tool_name}")
