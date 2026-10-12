"""Run normalize + classify on a chat export and write <name>.output.json.

    python run.py transcripts/ent-01-easy-cutover.txt [--model claude-haiku-5-5]

Accepts WhatsApp-style exports ("10/19/26, 9:05 AM - Name: text") or a Transcript JSON.
Output format matches the existing eval outputs (source, model, messages, spans, classified).
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from classify import classify
from llm import HAIKU, AnthropicLLM
from models import RawMessage, Transcript
from normalize import normalize

LINE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s?[AP]M) - ([^:]+): (.*)$")


def load_whatsapp(path: Path) -> Transcript:
    msgs: list[RawMessage] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(raw.strip())
        if m:
            d, t, author, text = m.groups()
            fmt = "%m/%d/%y %I:%M %p" if len(d.split("/")[-1]) == 2 else "%m/%d/%Y %I:%M %p"
            t = re.sub(r"\s+", " ", t.replace(" ", " ").replace("\xa0", " "))
            ts = datetime.strptime(f"{d} {t}", fmt)
            msgs.append(RawMessage(id=f"m{len(msgs) + 1}", author=author.strip(), ts=ts, text=text))
        elif msgs and raw.strip():            # continuation of a multi-line message
            msgs[-1].text += "\n" + raw.strip()
    return Transcript(channel="whatsapp", messages=msgs)


def load(path: Path) -> Transcript:
    if path.suffix == ".json":
        return Transcript.model_validate_json(path.read_text())
    return load_whatsapp(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--model", default=HAIKU)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    t = load(a.transcript)
    llm = AnthropicLLM(model=a.model)
    spans = normalize(t, llm)
    classified = classify(spans, llm)

    out = {
        "source": str(a.transcript),
        "model": a.model,
        "messages": len(t.messages),
        "spans": [json.loads(s.model_dump_json()) for s in spans],
        "classified": [{
            "span_id": c.span.id, "author": c.span.author, "text": c.span.text,
            "status": c.span.status.value, "context_note": c.span.context_note,
            "labels": [json.loads(l.model_dump_json()) for l in c.labels],
            "rationale": c.rationale,
        } for c in classified],
    }
    dest = a.out or a.transcript.with_suffix(".output.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}  ({len(spans)} spans from {len(t.messages)} messages)")


if __name__ == "__main__":
    main()
