"""Pact: shared decision intelligence. The product under test.

Three functions, one module each:
  1. observe   (observe.py)   — fill the decision card from what people say, with exact quotes.
  2. check     (check.py)     — on request, research claims against public web sources.
  3. interpret (interpret.py) — on request, explain where the decision stands.

card.py holds the card and the rules every change must pass; session.py ties it together.
Nothing in this package knows about simulated participants.
"""
from .card import ADMIN_ID, apply_change, new_card, now_iso, public_card
from .check import ALLOWED_DOMAINS, CHECKABLE_DEFINITION, select_claims
from .llm import PactLLM
from .llm_client import AnthropicClient
from .session import PactSession

__all__ = ["ADMIN_ID", "ALLOWED_DOMAINS", "AnthropicClient", "CHECKABLE_DEFINITION", "PactLLM",
           "PactSession", "apply_change", "new_card", "now_iso", "public_card", "select_claims"]
