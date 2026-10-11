"""Run a chat/transcript file through Pact and print the brief as JSON.

    python run_brief.py chat.txt                      # JSON to stdout
    python run_brief.py chat.txt -o brief.json        # JSON to a file
    python run_brief.py chat.txt --question "Which venue and date?"
    python run_brief.py chat.txt --summary-only       # just the executive summary text

No database is needed. Uses the worker's LLMClient, so the model and API key come from
the same settings as the worker (pact.api.config: LLM_API_KEY, llm_model).

Output (one JSON object):
    id                  run id
    source              input file name
    format              detected format (whatsapp/teams chat, transcript, email_chain, plain_text)
    question            the decision question used
    executive_summary   the rendered card (plain text)
    status              the computed status line
    emerging_preference computed from positions
    brief               the full structured brief (options, positions, actions, ...)
    report              verification report: omitted items, restore notes, warnings
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pact.worker.parser import _detect_format, _parse_email_chain, _parse_plain_text, _parse_transcript, _parse_whatsapp
from pact.worker.synthesizer import _compute_status_line, _unique, build_brief, extract_and_merge

_PARSERS = {"whatsapp": _parse_whatsapp, "transcript": _parse_transcript,
            "email_chain": _parse_email_chain, "plain_text": _parse_plain_text}


def parse_text(text: str, declared_type: str = "", stable_ids: bool = False) -> tuple[str, list[dict]]:
    """Text → (format, units with ids). Email chains also get re-quote de-duplication."""
    fmt = _detect_format(text, declared_type)
    units = _PARSERS[fmt](text)
    if fmt == "email_chain":
        from pact.worker.dedup import mark_duplicates
        units = [u for u in mark_duplicates(units) if u.pop("_dup_sequence", None) is None]
    for u in units:
        u["id"] = f"u{u['sequence']}" if stable_ids else str(uuid.uuid4())
    return fmt, units


async def brief_from_text(text: str, llm, question: str | None = None, declared_type: str = "",
                          source: str = "", stable_ids: bool = False) -> dict:
    fmt, units = parse_text(text, declared_type, stable_ids)
    if not units:
        raise ValueError("no messages found in the input")

    merged, chunk_count = await extract_and_merge(units, llm)
    question = question or ((merged.get("candidate_questions") or [{}])[0].get("text"))
    unit_text = {u["id"]: u["normalized_text"] for u in units}
    unit_meta = {u["id"]: {"author": u.get("author_label") or "", "ts": u["locator"].get("timestamp_str", "")}
                 for u in units}
    participants = _unique(u["author_label"] for u in units if u.get("author_label"))

    brief, restore_notes, ref_warnings, summary_report = await build_brief(
        merged, question, None, unit_text, unit_meta, participants, llm,
    )
    return {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "format": fmt,
        "question": question,
        "executive_summary": brief.get("executive_summary", ""),
        "status": _compute_status_line(brief),
        "emerging_preference": brief.get("emerging_preference"),
        "brief": {k: v for k, v in brief.items() if k not in ("executive_summary", "executive_summary_report")},
        "report": {
            "unit_count": len(units),
            "chunk_count": chunk_count,
            "summary": summary_report,
            "restored_or_corrected": restore_notes,
            "ref_warnings": ref_warnings,
            "dropped_at_extraction": merged.get("dropped_items", []),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a chat file through Pact and print the brief as JSON.")
    ap.add_argument("file", help="text file: WhatsApp/Teams export, [MM:SS] transcript, email chain or plain text")
    ap.add_argument("-o", "--out", help="write JSON here instead of stdout")
    ap.add_argument("-q", "--question", help="decision question (default: the top one found in the chat)")
    ap.add_argument("--type", default="", choices=("", "email_chain", "transcript"),
                    help="force a format; auto-detected otherwise")
    ap.add_argument("--summary-only", action="store_true", help="print only the executive summary text")
    args = ap.parse_args()

    from pact.worker.llm import LLMClient

    path = Path(args.file)
    text = path.read_text(encoding="utf-8", errors="replace")
    result = asyncio.run(brief_from_text(text, LLMClient(), args.question, args.type, source=path.name))

    if args.summary_only:
        output = result["executive_summary"] + "\n"
    else:
        output = json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n"
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"wrote {args.out}  (id {result['id']})", file=sys.stderr)
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
