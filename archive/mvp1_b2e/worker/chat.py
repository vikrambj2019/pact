"""Deliberation chat engine — classify request, generate response, propose operations."""
from __future__ import annotations

import json

_SYSTEM = """\
You are a decision brief assistant. A team is deliberating on a structured decision brief. \
You help them refine it by answering questions and proposing typed changes.

You receive the current brief (with stable entity IDs), recent conversation history, and a new message.

Your responsibilities:
- Answer questions clearly, grounding responses in the brief content and conversation history.
- If the message requests a correction, addition, reclassification, or scope change, propose typed operations.
- Always show proposed changes before they are applied — never apply them yourself.
- Do not record decisions, assign authority, or infer positions from silence.
- Keep responses concise. Use follow_up_question only when genuinely needed for clarification.

Request types:
  clarification   — question answered without changing the brief
  correction      — fix attribution, position, claim, or constraint
  challenge       — add an objection or counter-argument
  comparison      — compare options against criteria
  scope_change    — move an item in or out of scope / to a separate issue
  readiness       — assess what is blocking the decision
  other           — anything that doesn't fit the above

Operation rules:
- add_* operations: omit entity_id (server assigns it)
- update_*/resolve_*/retract_*/complete_* operations: include the exact entity_id from the brief
- Only use the operation types listed in the schema
- source_unit_ids must be real IDs from the brief's entities; do not invent them
- If nothing in the brief needs to change, return an empty operations list
"""

_CHAT_TOOL = {
    "name": "deliberation_response",
    "description": "Respond to a user message and optionally propose typed changes to the brief.",
    "input_schema": {
        "type": "object",
        "required": ["response_text", "request_type", "operations"],
        "properties": {
            "response_text": {
                "type": "string",
                "description": "The response shown to the user."
            },
            "request_type": {
                "type": "string",
                "enum": ["clarification", "correction", "challenge", "comparison",
                         "scope_change", "readiness", "other"]
            },
            "follow_up_question": {
                "type": "string",
                "description": "Optional clarifying question if the request is ambiguous."
            },
            "operations": {
                "type": "array",
                "description": "Typed operations proposed for human review. Empty if no changes needed.",
                "items": {
                    "type": "object",
                    "required": ["type", "fields"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "add_option", "update_option",
                                "add_criterion", "update_criterion",
                                "add_constraint", "update_constraint",
                                "add_claim", "update_claim", "retract_claim",
                                "add_objection", "resolve_objection",
                                "add_blocker", "resolve_blocker",
                                "add_separate_issue",
                                "add_action", "complete_action",
                                "update_position",
                                "update_emerging_preference"
                            ]
                        },
                        "entity_id": {
                            "type": "string",
                            "description": "ID of existing entity to update/resolve (omit for add_* operations)"
                        },
                        "fields": {
                            "type": "object",
                            "description": "Fields to set on the entity."
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this change is proposed."
                        }
                    }
                }
            }
        }
    }
}


def _format_brief(snapshot: dict) -> str:
    """Compact brief representation for the prompt — include IDs so LLM can reference them."""
    sections = {}
    for key in ("question", "options", "criteria", "constraints", "claims",
                 "positions", "objections", "blockers", "separate_issues", "actions",
                 "emerging_preference"):
        val = snapshot.get(key)
        if val:
            sections[key] = val
    return json.dumps(sections, indent=2)


def _format_history(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = "User" if m["role"] == "human" else "Assistant"
        lines.append(f"{role} ({m['author_label']}): {m['text']}")
    return "\n".join(lines) if lines else "(no prior messages)"


async def run_chat(
    snapshot: dict,
    revision_number: int,
    history: list[dict],
    author_label: str,
    message_text: str,
    llm,
) -> dict:
    """Run a chat turn. Returns {response_text, request_type, follow_up_question, operations}."""
    user_prompt = (
        f"Current brief (revision {revision_number}):\n"
        f"{_format_brief(snapshot)}\n\n"
        f"Recent conversation:\n{_format_history(history)}\n\n"
        f"New message from {author_label}:\n{message_text}"
    )
    result = await llm.extract(_SYSTEM, user_prompt, _CHAT_TOOL, timeout=60)
    return result
