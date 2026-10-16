"""Stage 1 — normalize: transcript -> atomic spans.

Two steps:
  1. coalesce (deterministic code): group same-author runs into bursts.
  2. segment  (Haiku, one call per burst): split each burst into atomic spans and
     mark self-corrections. Pure structure — no element classification here.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from llm import LLM
from models import Burst, RawMessage, Span, SpanStatus, Transcript

COALESCE_GAP = timedelta(seconds=120)  # exported chats have minute resolution; email -> never coalesce
CONTEXT_BURSTS = 4                     # preceding bursts shown read-only for references


# ── 1. Coalesce ─────────────────────────────────────────────────────────────

def coalesce(messages: list[RawMessage], gap: timedelta = COALESCE_GAP) -> list[Burst]:
    """Same author + no intervening speaker + gap <= threshold -> one burst.
    Deterministic on purpose: cheap, testable, no model judgment needed."""
    msgs = sorted(messages, key=lambda m: m.ts)
    bursts: list[Burst] = []
    for m in msgs:
        if not m.text.strip():
            continue
        last = bursts[-1] if bursts else None
        if last and last.author == m.author and m.ts - last.end_ts <= gap:
            last.message_ids.append(m.id)
            last.texts.append(m.text)
            last.end_ts = m.ts
        else:
            bursts.append(Burst(id=f"b{len(bursts) + 1}", author=m.author,
                                start_ts=m.ts, end_ts=m.ts,
                                message_ids=[m.id], texts=[m.text]))
    return bursts


# ── 2. Segment ──────────────────────────────────────────────────────────────

SEGMENT_SYSTEM = """You segment group-chat messages into ATOMIC SPANS for a decision-tracking system.

A span is the smallest verbatim piece of text that carries ONE extractable idea: an option,
a stance, a constraint, a claim, a question, a blocker, a task, a concern, etc.
You do NOT classify spans. You only decide boundaries and self-corrections.

Rules
1. SPLIT when one message carries independent ideas.
   "I can't do the 24th but I can do the 5th" -> ["I can't do the 24th", "I can do the 5th"]
   "It is still unsigned because the retention clause is unresolved" -> keep whole if the
   cause and effect are one assertion; split only if each half stands alone.
   A COMMITMENT or TASK ("I'll ask", "I'll push procurement to...", "I can send it by Fri")
   is ALWAYS its own span — never merged with a fact, stance or question in the same message:
   "Don't know. Their quote says X. I'll ask." -> ["Don't know. Their quote says X", "I'll ask"]
   "I'll push procurement on the fee. I can support Relay if we land under $200k."
     -> ["I'll push procurement on the fee", "I can support Relay if we land under $200k"]
   Likewise a message that does two DIFFERENT jobs (re-ranks criteria AND declares an outcome,
   states a fact AND takes a stance) has at least one span per job.
   ONE SPAN PER OPTION / QUESTION: a message that takes stances on, or settles, two
   different things is split by thing, even with no punctuation between them:
   "yes to the lake + saturday" -> ["yes to the lake", "saturday"]
   "settled: the lake, saturday the 9th" -> ["settled: the lake", "saturday the 9th"]
   Casual chat often has no sentence boundaries: split at ideas, not at punctuation.
   A REASON given for a stance is its own span (it is a claim or criterion, not a stance):
   "I'd propose Nov 14-15. UAT should be done by then" -> ["I'd propose Nov 14-15", "UAT should be done by then"]
2. DON'T over-split. Emphasis, hedging, filler and restatement stay with their idea.
   "I really, really think we should go with Saturday" -> one span.
   A clause that only continues the SAME constraint or the SAME task stays attached:
   "our freeze starts Nov 23. After that I can't add weekend cover" -> one span
   "I'll brief the auditors on Nov 10 and confirm back here" -> one span
   Test: each fragment must be classifiable on its own with its own truth value.
3. Spans must be VERBATIM substrings of exactly one message (copy characters exactly;
   you may drop leading/trailing filler like "ok so" or "lol"). Only greetings, thanks,
   emojis and pure chit-chat may be omitted. NEVER drop a sentence that gives a reason,
   fact, estimate, date or commitment — e.g. "Close is well behind us and I can brief
   the auditors the week before" must be kept. If unsure, keep it.
4. SELF-CORRECTION inside the burst: if the author quickly retracts or fixes something they
   said earlier in the SAME burst ("oh wait", "sorry I meant", "actually no"), still emit the
   earlier span but mark it status="self_corrected" and set corrects_index on the later span.
   A genuine reasoned change of mind is NOT a self-correction — leave both active.
5. context_note is ONLY for resolving a reference that is unclear on its own: a pronoun,
   "that", "then", "either date", "none", "same for me". Write it as a bare mapping using
   words that appear in the conversation: "'that' = Nov 25-30 month-end close window",
   "'None' = no objection to Nov 14-15". Do NOT describe intent, motive or purpose, and do
   NOT name what an option is for unless the conversation literally says so.
   If the span is understandable on its own, context_note must be null.

Burst messages are numbered [0], [1], ... Return spans in reading order."""

SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "message_index": {"type": "integer",
                                      "description": "index of the burst message the text comes from"},
                    "text": {"type": "string", "description": "verbatim substring"},
                    "status": {"type": "string", "enum": ["active", "self_corrected"]},
                    "corrects_index": {"type": ["integer", "null"],
                                       "description": "index (in this spans list) of the span this one corrects"},
                    "context_note": {"type": ["string", "null"]},
                },
                "required": ["message_index", "text", "status"],
            },
        }
    },
    "required": ["spans"],
}


def _render_context(prev: list[Burst]) -> str:
    if not prev:
        return "(start of conversation)"
    return "\n".join(f"{b.author}: {' / '.join(b.texts)}" for b in prev)


def _segment_burst(llm: LLM, burst: Burst, prev: list[Burst]) -> list[Span]:
    user = (
        "PRECEDING CONTEXT (read-only, do not segment):\n"
        f"{_render_context(prev)}\n\n"
        f"BURST TO SEGMENT — author: {burst.author}\n"
        + "\n".join(f"[{i}] {t}" for i, t in enumerate(burst.texts))
    )
    out = llm.structured(system=SEGMENT_SYSTEM, user=user,
                         tool_name="segment", schema=SEGMENT_SCHEMA)

    raw = [s for s in out.get("spans", []) if isinstance(s, dict) and (s.get("text") or "").strip()]
    items = []                                   # (mi, pos, raw_index, Span)
    for j, s in enumerate(raw):
        mi = s.get("message_index", 0)
        mi = mi if isinstance(mi, int) and 0 <= mi < len(burst.texts) else 0
        text = s["text"].strip()
        source = burst.texts[mi]
        pos = source.find(text)
        if pos < 0:
            pos = source.lower().find(text.lower())
        verified = pos >= 0 or " ".join(text.lower().split()) in " ".join(source.lower().split())
        items.append((mi, pos if pos >= 0 else 10**6, j, Span(
            id="", burst_id=burst.id, author=burst.author, ts=burst.start_ts, text=text,
            source_message_ids=[burst.message_ids[mi]],
            context_note=s.get("context_note"), verified=verified)))

    items += _recover_uncovered(burst, items)    # nothing the author wrote silently vanishes
    items.sort(key=lambda t: (t[0], t[1]))
    spans = []
    for n, (_, _, _, sp) in enumerate(items, 1):
        sp.id = f"{burst.id}.s{n}"
        spans.append(sp)
    by_raw = {j: sp for (_, _, j, sp) in items if j is not None}
    msg_of = {id(sp): mi for (mi, _, _, sp) in items}

    # Self-corrections. A link (later span -> earlier span, corrects_index) is accepted.
    # A bare status="self_corrected" is accepted only when the same author sent a LATER
    # message in this burst; on the burst's last message it is ignored (it misfired on
    # deliberate multi-clause messages like "withdrawing my pick ... sunol is my pick now").
    links: dict[int, int] = {}
    for j, s in enumerate(raw):
        k = s.get("corrects_index")
        if isinstance(k, int) and 0 <= k < j:
            links[k] = j
    last = len(burst.texts) - 1
    for j, s in enumerate(raw):
        if j in links or s.get("status") != "self_corrected" or j not in by_raw:
            continue
        mi = msg_of[id(by_raw[j])]
        if mi >= last:
            continue
        nxt = next((sp for sp in spans if msg_of[id(sp)] > mi), None)
        if nxt is not None:
            by_raw[j].status = SpanStatus.self_corrected
            by_raw[j].superseded_by = nxt.id
    for k, j in links.items():
        if k in by_raw and j in by_raw:
            by_raw[k].status = SpanStatus.self_corrected
            by_raw[k].superseded_by = by_raw[j].id
    return spans


_WORD = re.compile(r"[A-Za-z0-9$][A-Za-z0-9$'.\-]*")
MIN_RECOVER_WORDS = 3     # an uncovered stretch this long is content, not filler


def _recover_uncovered(burst: Burst, items: list) -> list:
    """Coverage guard. The segmenter sometimes returns nothing for a message, or drops a
    clause. Any message with no spans, or any uncovered stretch of >= MIN_RECOVER_WORDS
    words, comes back as a verbatim span marked recovered=True."""
    extra = []
    for mi, text in enumerate(burst.texts):
        mine = [(pos, sp) for (m, pos, _, sp) in items if m == mi]
        if not mine:
            if len(_WORD.findall(text)) >= 2:
                extra.append((mi, 0, None, _mk(burst, mi, text.strip())))
            continue
        covered = [False] * len(text)
        for pos, sp in mine:
            if 0 <= pos < len(text):
                for c in range(pos, min(len(text), pos + len(sp.text))):
                    covered[c] = True
        start = None
        for c in range(len(text) + 1):
            free = c < len(text) and not covered[c]
            if free and start is None:
                start = c
            elif not free and start is not None:
                chunk = text[start:c].strip(" ,.;:-–—!?\n\t")
                if len(_WORD.findall(chunk)) >= MIN_RECOVER_WORDS:
                    extra.append((mi, text.find(chunk), None, _mk(burst, mi, chunk)))
                start = None
    return extra


def _mk(burst: Burst, mi: int, text: str) -> Span:
    return Span(id="", burst_id=burst.id, author=burst.author, ts=burst.start_ts, text=text,
                source_message_ids=[burst.message_ids[mi]], verified=True, recovered=True)


def _fix_timestamps(spans: list[Span], messages: list[RawMessage]) -> None:
    ts_by_id = {m.id: m.ts for m in messages}
    for sp in spans:
        sp.ts = ts_by_id.get(sp.source_message_ids[0], sp.ts)


def normalize(transcript: Transcript, llm: LLM, *, gap: timedelta = COALESCE_GAP,
              max_workers: int = 8) -> list[Span]:
    """Transcript -> flat, ordered list of atomic spans."""
    msgs = transcript.messages
    if transcript.channel == "email":
        gap = timedelta(0)  # each email is already a deliberate unit
    bursts = coalesce(msgs, gap)

    def work(i: int) -> list[Span]:
        return _segment_burst(llm, bursts[i], bursts[max(0, i - CONTEXT_BURSTS):i])

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        per_burst = list(pool.map(work, range(len(bursts))))

    spans = [sp for group in per_burst for sp in group]
    _fix_timestamps(spans, msgs)
    return spans


if __name__ == "__main__":
    import sys
    from llm import AnthropicLLM
    t = Transcript.model_validate_json(open(sys.argv[1]).read())
    for sp in normalize(t, AnthropicLLM()):
        print(sp.model_dump_json())
