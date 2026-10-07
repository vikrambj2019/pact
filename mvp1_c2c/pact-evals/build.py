#!/usr/bin/env python3
"""Build WhatsApp-style .txt and .json transcripts from src/*.src, and validate gold labels.

Usage: python3 build.py
Reads  src/<name>.src and gold/<name>.gold.json
Writes out/<name>.txt (WhatsApp Android export style) and out/<name>.json (structured, with ids and reply links)
"""
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
SRC, GOLD, OUT = ROOT / "src", ROOT / "gold", ROOT / "out"
MSG_ID = re.compile(r"^m\d{3}$")
REQUIRED_GOLD_KEYS = [
    "decision_question", "options", "constraints", "criteria", "claims", "agreements",
    "open_issues", "outcome", "next_steps", "traps", "card_must_include", "card_must_not_include",
]
ENCRYPTION_NOTICE = (
    "Messages and calls are end-to-end encrypted. No one outside of this chat, "
    "not even WhatsApp, can read or listen to them. Tap to learn more."
)


def parse(path):
    meta, msgs = {}, []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        if raw.startswith("#"):
            key, _, val = raw[1:].partition(":")
            meta[key.strip()] = val.strip()
            continue
        parts = raw.split("|")
        reply = parts.pop()[1:] if parts[-1].startswith(">") else None
        if len(parts) != 4:
            sys.exit(f"{path.name}:{n}: expected id|delta|sender|text (no '|' inside text)")
        mid, delta, sender, text = parts
        msgs.append({"id": mid, "delta": int(delta), "sender": sender, "text": text, "reply_to": reply})
    return meta, msgs


def stamp(ts):
    return f"{ts.month}/{ts.day}/{ts:%y}, {ts.hour % 12 or 12}:{ts:%M} {'AM' if ts.hour < 12 else 'PM'}"


def build(path):
    meta, msgs = parse(path)
    name = path.stem
    seen = set()
    for i, m in enumerate(msgs, 1):
        if m["id"] != f"m{i:03d}":
            sys.exit(f"{name}: id {m['id']} out of sequence, expected m{i:03d}")
        if m["reply_to"] and m["reply_to"] not in seen:
            sys.exit(f"{name}: {m['id']} replies to unknown or later message {m['reply_to']}")
        seen.add(m["id"])
    start = datetime.strptime(meta["start"], "%Y-%m-%d %H:%M")
    ts = start
    out_msgs = []
    for m in msgs:
        ts += timedelta(minutes=m["delta"])
        rec = {"id": m["id"], "ts": ts.strftime("%Y-%m-%dT%H:%M:00"), "sender": m["sender"], "text": m["text"]}
        if m["reply_to"]:
            rec["reply_to"] = m["reply_to"]
        out_msgs.append((ts, rec))
    counts = Counter(m["sender"] for m in msgs)
    if "expect_each" in meta:
        want = int(meta["expect_each"])
        bad = {k: v for k, v in counts.items() if v != want}
        if bad or len(counts) != 6:
            sys.exit(f"{name}: expected 6 senders x {want}, got {dict(counts)}")
    elif len(counts) != 6:
        sys.exit(f"{name}: expected 6 senders, got {dict(counts)}")

    lines = [f"{stamp(start - timedelta(minutes=1))} - {ENCRYPTION_NOTICE}"]
    lines += [f"{stamp(t)} - {r['sender']}: {r['text']}" for t, r in out_msgs]
    (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    participants = list(dict.fromkeys(m["sender"] for m in msgs))
    doc = {
        "id": name, "title": meta.get("title", name), "group": meta.get("group", ""),
        "level": int(meta.get("level", 0)), "timezone": "America/Los_Angeles",
        "participants": participants, "nicknames": meta.get("nicknames", ""),
        "messages": [r for _, r in out_msgs],
    }
    (OUT / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return name, len(msgs), dict(counts), {m["id"] for m in msgs}


def gold_ids(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from gold_ids(v)
    elif isinstance(node, list):
        for v in node:
            yield from gold_ids(v)
    elif isinstance(node, str) and MSG_ID.match(node):
        yield node


def check_gold(name, ids):
    path = GOLD / f"{name}.gold.json"
    if not path.exists():
        return "no gold file"
    gold = json.loads(path.read_text(encoding="utf-8"))
    missing_keys = [k for k in REQUIRED_GOLD_KEYS if k not in gold]
    unknown = sorted({i for i in gold_ids(gold) if i not in ids})
    if missing_keys or unknown:
        sys.exit(f"{name}: gold problems: missing keys {missing_keys}, unknown message ids {unknown}")
    return f"gold ok ({sum(1 for _ in gold_ids(gold))} id refs)"


def main():
    OUT.mkdir(exist_ok=True)
    for path in sorted(SRC.glob("*.src")):
        name, n, counts, ids = build(path)
        print(f"{name}: {n} messages, per sender {counts}; {check_gold(name, ids)}")


if __name__ == "__main__":
    main()
