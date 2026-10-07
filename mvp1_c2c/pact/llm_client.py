"""Anthropic Messages API transport, shared by Pact and the simulation harness."""
from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import certifi

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_local_env(path: Path = REPO_ROOT / ".env") -> None:
    """Load simple KEY=value entries without printing or overriding shell env."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()


class AnthropicClient:
    """Small Anthropic Messages API adapter. Holds the key and counts calls; callers pick the model."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Live mode needs ANTHROPIC_API_KEY in the environment or root .env.")
        self.calls = 0

    def call(self, body: dict, model: str, timeout: int = 60, extended_output: bool = False) -> dict:
        payload = {"model": model, "max_tokens": 2500, **body}  # body may override max_tokens
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}
        if extended_output:
            headers["anthropic-beta"] = "output-128k-2025-02-19"
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context(cafile=certifi.where())) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Anthropic request failed ({exc.code}): {detail}") from exc
        self.calls += 1
        return data

    @staticmethod
    def output_text(response: dict) -> str:
        chunks = []
        for content in response.get("content", []):
            if content.get("type") == "text":
                chunks.append(content.get("text", ""))
        return "\n".join(chunks)

    def structured(self, instructions: str, payload: dict, name: str, schema: dict, model: str,
                   max_tokens: int | None = None, timeout: int = 60, extended_output: bool = False) -> dict:
        body: dict = {"system": instructions,
                      "messages": [{"role": "user", "content": json.dumps(payload)}],
                      "tools": [{"name": name, "description": "Return the requested structured result.",
                                 "input_schema": schema}],
                      "tool_choice": {"type": "tool", "name": name}}
        if max_tokens:
            body["max_tokens"] = max_tokens
        response = self.call(body, model=model, timeout=timeout, extended_output=extended_output)
        for block in response.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == name:
                inp = block.get("input", {})
                if not inp and response.get("stop_reason") == "max_tokens":
                    raise RuntimeError("structured: output truncated at max_tokens — increase max_tokens")
                return inp
        raise RuntimeError(f"Anthropic did not return the required structured result "
                           f"(stop_reason={response.get('stop_reason')}, "
                           f"content types={[b.get('type') for b in response.get('content',[])]})")
