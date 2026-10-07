"""The models behind Pact's three functions, sharing one client and one model setting."""
from __future__ import annotations

import os

from .check import ClaimChecker
from .commands import CommandParser
from .interpret import Interpreter
from .llm_client import AnthropicClient
from .observe import Observer


class PactLLM:
    def __init__(self, client: AnthropicClient, model: str | None = None):
        self.client = client
        self.model = model or os.getenv("ANTHROPIC_RESEARCH_MODEL",
                                        os.getenv("PACT_RESEARCH_MODEL", "claude-sonnet-4-6"))
        self.observer = Observer(client, self.model)        # 1. observe
        self.checker = ClaimChecker(client, self.model)     # 2. check
        self.interpreter = Interpreter(client, self.model)  # 3. interpret
        self.commands = CommandParser(client, self.model)
