"""Test harness: LLM-played participants that converse so Pact has something to observe.

Not part of Pact. Depends on `pact`; `pact` never imports from here.
"""
from .participants import ParticipantLLM
from .runner import Simulation, saved_models

__all__ = ["ParticipantLLM", "Simulation", "saved_models"]
