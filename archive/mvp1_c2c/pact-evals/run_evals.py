#!/usr/bin/env python3
"""Replay observer evals through the product's current PactSession API.

Default: incremental six-message batches. --batch-size 0 uses a whole-transcript batch.
Matching of free-text gold labels is heuristic; inspect output cards alongside scores.
No claim research is performed: these transcripts contain invented external facts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO))

from mvp1_c2c.pact import AnthropicClient, PactLLM, PactSession, new_card, now_iso  # noqa: E402

OUT_DIR = ROOT / "out"
GOLD_DIR = ROOT / "gold"
RESULTS_DIR = ROOT / "results"
STEMS = ["hiking-01-simple", "hiking-02-one-dispute", "hiking-03-shifting-constraints",
         "hiking-04-interleaved-threads", "hiking-05-chaos"]


def build_card(transcript: dict) -> tuple[dict, dict]:
    names = transcript["participants"]
    members = [{"id": f"p_{i}", "name": name, "role": "participant",
                "mapping_allowed": True, "share_allowed": True}
               for i, name in enumerate(names[1:], 1)]
    card = new_card(transcript.get("title", transcript["id"]), names[0], members)
    return card, {p["name"]: p["id"] for p in card["participants"]}


def run_transcript(stem: str, pact: PactLLM, batch_size: int = 6) -> dict:
    if batch_size < 0:
        raise ValueError("batch_size must be nonnegative")
    transcript = json.loads((OUT_DIR / f"{stem}.json").read_text())
    gold = json.loads((GOLD_DIR / f"{stem}.gold.json").read_text())
    card, name_to_id = build_card(transcript)
    session = PactSession(card, pact)
    records = []
    errors = []
    for msg in transcript["messages"]:
        if msg["sender"] not in name_to_id:
            raise ValueError(f"Unknown transcript sender: {msg['sender']}")
        record = {"id": msg["id"], "speaker_id": name_to_id[msg["sender"]],
                  "text": msg["text"], "source": "simulated",
                  "recorded_at": msg.get("ts") or now_iso()}
        if msg.get("reply_to"):
            record["reply_to"] = msg["reply_to"]
        records.append(record)
    size = batch_size or max(1, len(records))
    for start in range(0, len(records), size):
        batch = records[start:start + size]
        # Append only the messages available at this point: no look-ahead to later discussion.
        card["messages"].extend(batch)
        card["observation"]["pending"].extend(m["id"] for m in batch)
        card["card_version"] += len(batch)
        observation = session.observe_messages(batch)
        if observation.get("error"):
            errors.append({"message_ids": [m["id"] for m in batch], "error": observation["error"]})
            break  # Later batches cannot fairly bypass failed earlier work.
    errors += [{"msg": event.get("message_id"), "error": event["detail"]}
               for event in card["audit"] if event["event"] == "observation_rejected"]
    result = {"stem": stem, "card": card, "gold": gold, "errors": errors, "batch_size": batch_size}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


STOPWORDS = {"a", "an", "the", "is", "are", "in", "of", "to", "for", "and", "or", "that",
             "it", "its", "by", "with", "as", "per", "person", "all", "six"}


def tokens(text: str) -> set[str]:
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return set(re.findall(r"[a-z0-9]+", text)) - STOPWORDS


def similar(a: str, b: str) -> bool:
    """Conservative lexical overlap for approximate discovery scores, never semantic proof."""
    left, right = tokens(a), tokens(b)
    return bool(left and right) and len(left & right) / min(len(left), len(right)) >= 0.6


def score(result: dict) -> dict:
    card, gold = result["card"], result["gold"]
    scores = {"matching": "heuristic; manually review cards and gold"}
    scores["options"] = f"{sum(any(similar(g['name'], o['text']) for o in card['options']) for g in gold.get('options', []))}/{len(gold.get('options', []))}"
    gold_claims = [c for c in gold.get("claims", []) if c.get("checkable")]
    claims = [c for c in card["claims"] if c["status"] != "retracted"]
    matched = sum(any(similar(g["text"], c.get("corrected_to") or c["statement"]) for c in claims)
                  for g in gold_claims)
    scores["claims"] = f"{matched}/{len(gold_claims)}"
    criteria = gold.get("criteria", [])
    scores["criteria"] = f"{sum(any(similar(g['text'], c['text']) for c in card['criteria']) for g in criteria)}/{len(criteria)}"
    names = {p["id"]: p["name"] for p in card["participants"]}
    false = []
    for agreement in card["agreements"]:
        sources = set(agreement["source_message_ids"])
        candidates = [g for g in gold.get("agreements", [])
                      if sources & set(g.get("msgs", [])) and similar(agreement["text"], g["text"])]
        for affirmation in agreement["affirmers"]:
            name = names[affirmation["participant_id"]]
            if not any(name in g.get("explicit_affirmers", []) and
                       affirmation["message_id"] in g.get("msgs", []) for g in candidates):
                false.append({"subject_id": agreement["subject_id"], "who": name,
                              "message_id": affirmation["message_id"]})
    scores["false_affirmations"] = len(false)
    scores["false_affirmation_detail"] = false
    scores["false_agreements"] = len({f["subject_id"] for f in false})
    messages = {m["id"]: m for m in card["messages"]}
    bad_provenance = []
    for section in ("options", "constraints", "criteria", "claims", "agreements", "preferences", "open_issues"):
        for item in card[section]:
            for event in item["history"]:
                source = messages.get(event.get("message_id"))
                if (not source or source["speaker_id"] != event.get("speaker_id")
                        or not event.get("quote") or event["quote"] not in source["text"]):
                    bad_provenance.append({"item_id": item["id"], "message_id": event.get("message_id")})
    scores["provenance_violations"] = len(bad_provenance)
    # Test only current visible field values. Historical quotes are expected to retain superseded values.
    visible = {
        "options": [{"text": o["text"], "status": o["status"]} for o in card["options"]],
        "constraints": [{"text": c["text"], "status": c["status"]} for c in card["constraints"] if c["status"] != "superseded"],
        "claims": [{"statement": c.get("corrected_to") or c["statement"], "status": c["status"]} for c in claims],
        "preferences": [{k: p.get(k) for k in ("stance", "leaning", "conditional")} for p in card["preferences"]],
        "open_issues": [i["text"] for i in card["open_issues"] if i["status"] == "open"],
    }
    current_text = json.dumps(visible)
    scores["must_include"] = f"{sum(similar(g, current_text) for g in gold.get('card_must_include', []))}/{len(gold.get('card_must_include', []))}"
    violations = [g for g in gold.get("card_must_not_include", []) if similar(g, current_text)]
    scores["must_not_violations"] = len(violations)
    scores["must_not_detail"] = violations
    scores["observer_errors"] = len(result["errors"])
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems", nargs="*", choices=None)
    parser.add_argument("--batch-size", type=int, default=6, help="Messages per batch; 0 means entire transcript")
    args = parser.parse_args()
    stems = args.stems or STEMS
    if args.batch_size < 0 or any(stem not in STEMS for stem in stems):
        parser.error("Use known hiking stems and a nonnegative batch size.")
    pact = PactLLM(AnthropicClient())
    failed = False
    for stem in stems:
        start = time.monotonic()
        result = run_transcript(stem, pact, args.batch_size)
        scores = score(result)
        failed |= bool(result["errors"])
        print(json.dumps({"stem": stem, "scores": scores, "seconds": round(time.monotonic() - start, 1)}, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
