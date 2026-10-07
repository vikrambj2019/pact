#!/usr/bin/env python3
"""Run Pact's observe function against each eval transcript and score against gold.

Uses the pact package only; no simulated participants are involved.

Usage (from repo root):
    python3 mvp1_c2c/pact-evals/run_evals.py [stem ...]

Examples:
    python3 mvp1_c2c/pact-evals/run_evals.py                   # all five
    python3 mvp1_c2c/pact-evals/run_evals.py hiking-01-simple  # one
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO))

from mvp1_c2c.pact import AnthropicClient, PactLLM, apply_change, new_card, now_iso  # noqa: E402

OUT_DIR = ROOT / "out"
GOLD_DIR = ROOT / "gold"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── card builder ──────────────────────────────────────────────────────────────

def build_card(transcript: dict) -> tuple[dict, dict]:
    """Create a card from a transcript using initial_card(). Returns (card, name→id map)."""
    names = transcript["participants"]
    members = [
        {"id": f"p_{i}", "name": n, "role": "participant",
         "mapping_allowed": True, "share_allowed": True}
        for i, n in enumerate(names[1:], 1)
    ]
    card = new_card(transcript.get("title", transcript["id"]), names[0], members)
    name_to_id = {p["name"]: p["id"] for p in card["participants"]}
    return card, name_to_id


# ── observer runner ───────────────────────────────────────────────────────────

def run_transcript(stem: str, pact: PactLLM) -> dict:
    transcript = json.loads((OUT_DIR / f"{stem}.json").read_text())
    gold = json.loads((GOLD_DIR / f"{stem}.gold.json").read_text())

    card, name_to_id = build_card(transcript)
    errors = []
    n = len(transcript["messages"])

    print(f"\n{'='*60}")
    print(f"  {stem}  ({n} messages — single API call)")
    print(f"{'='*60}")

    # Build message records using real participant IDs
    records = []
    for msg in transcript["messages"]:
        speaker_id = name_to_id.get(msg["sender"])
        if not speaker_id:
            continue
        record = {"id": msg["id"], "speaker_id": speaker_id,
                  "text": msg["text"], "source": "simulated",
                  "recorded_at": msg.get("ts", now_iso())}
        records.append(record)
        card["messages"].append(record)

    # One API call for the whole conversation
    try:
        batch_result = pact.observer.observe_batch(card, records)
        total_applied = 0
        for record in records:
            changes = batch_result.get(record["id"], [])
            for change in changes:
                try:
                    apply_change(card, change, record)
                    total_applied += 1
                except ValueError as e:
                    errors.append({"msg": record["id"], "error": str(e)})
        card["card_version"] += 1
        print(f"  → {total_applied} changes applied"
              f"  (proposals={len(card['proposals'])}"
              f" claims={len(card['claims'])}"
              f" criteria={len(card['criteria'])}"
              f" unresolved={len(card['unresolved'])})")
    except Exception as exc:
        import traceback
        errors.append({"error": f"observe_batch failed: {exc}"})
        print(f"  ERROR: {exc}")
        traceback.print_exc()

    result = {"stem": stem, "card": card, "gold": gold, "errors": errors}
    (RESULTS_DIR / f"{stem}.json").write_text(
        json.dumps({"card": card, "gold": gold, "errors": errors}, indent=2)
    )
    return result


# ── scorer ────────────────────────────────────────────────────────────────────

def score(result: dict) -> dict:
    card = result["card"]
    gold = result["gold"]

    scores = {}

    # 1. Proposals vs gold options
    gold_opts = {o["name"].lower() for o in gold.get("options", [])}
    card_props = {(p.get("title") or "").lower() for p in card["proposals"]}
    matched_opts = sum(1 for g in gold_opts
                       if any(word in g for word in (next(iter(p.split()), "") for p in card_props))
                       or any(g in p or p in g for p in card_props))
    scores["options"] = f"{matched_opts}/{len(gold_opts)}"

    # 2. Claims vs gold claims (checkable ones)
    gold_claims = [c for c in gold.get("claims", []) if c.get("checkable")]
    card_claim_texts = [
        (c.get("statement") or c.get("description") or c.get("text") or "").lower()
        for c in card["claims"]
    ]
    matched_claims = 0
    for gc in gold_claims:
        stopwords = {"a","an","the","is","are","in","of","to","for","and","or","that","it","its","by"}
        words = [w for w in gc["text"].lower().split() if w not in stopwords]
        key = words[:5]
        if any(sum(1 for w in key if w in ct) >= 3 for ct in card_claim_texts):
            matched_claims += 1
    scores["claims"] = f"{matched_claims}/{len(gold_claims)}"

    # 3. False agreement rate
    gold_agreements = gold.get("agreements", [])
    gold_agreed_texts = [a["text"].lower() for a in gold_agreements]
    card_agreements = [p for p in card["proposals"] if p.get("status") == "agreed"]
    false_agreements = 0
    for ca in card_agreements:
        ct = (ca.get("title") or "").lower()
        if not any(ct in ga or ga in ct for ga in gold_agreed_texts):
            false_agreements += 1
    scores["false_agreements"] = false_agreements

    # 4. Criteria
    gold_criteria = gold.get("criteria", [])
    card_criteria_texts = [
        (c.get("description") or c.get("title") or c.get("text") or "").lower()
        for c in card["criteria"]
    ]
    stopwords_crit = {"a","an","the","is","are","in","of","to","for","and","or","that","it","its","by","with","as"}
    matched_crit = 0
    for gc in gold_criteria:
        words = [w.strip(".,;:()") for w in gc["text"].lower().split() if w not in stopwords_crit and len(w) > 2]
        key = words[:5]
        if any(sum(1 for w in key if w in ct) >= min(2, len(key)) for ct in card_criteria_texts):
            matched_crit += 1
    scores["criteria"] = f"{matched_crit}/{len(gold_criteria)}"

    # 5. card_must_include / card_must_not_include (keyword heuristic)
    # Search only structured fields — exclude raw messages so we don't match things the observer correctly ignored
    stopwords = {"a","an","the","is","are","in","of","to","for","and","or","that","it","its","by","with","as"}
    structured_card = {k: v for k, v in card.items()
                       if k not in ("messages", "audit", "change_log", "snapshots")}
    full_card_text = json.dumps(structured_card).lower()

    def content_words(s):
        return [w.strip(".,;:()") for w in s.lower().split() if w not in stopwords and len(w) > 2]

    must_hits = 0
    for m in gold.get("card_must_include", []):
        words = content_words(m)
        if sum(1 for w in words[:6] if w in full_card_text) >= min(3, len(words)):
            must_hits += 1
    must_not_hits = 0
    must_not_detail = []
    for m in gold.get("card_must_not_include", []):
        words = content_words(m)
        if sum(1 for w in words[:6] if w in full_card_text) >= min(4, len(words)):
            must_not_hits += 1
            must_not_detail.append(m)
    scores["must_include"] = f"{must_hits}/{len(gold.get('card_must_include', []))}"
    scores["must_not_violations"] = must_not_hits
    scores["must_not_detail"] = must_not_detail

    # 6. Observer errors
    scores["observer_errors"] = len(result["errors"])

    return scores


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    stems = sys.argv[1:] or [
        "hiking-01-simple",
        "hiking-02-one-dispute",
        "hiking-03-shifting-constraints",
        "hiking-04-interleaved-threads",
        "hiking-05-chaos",
    ]

    pact = PactLLM(AnthropicClient())
    all_scores = {}

    for stem in stems:
        t0 = time.time()
        result = run_transcript(stem, pact)
        elapsed = time.time() - t0
        s = score(result)
        all_scores[stem] = s

        print(f"\n  ── scores ({stem}) ──")
        for k, v in s.items():
            if k == "must_not_detail":
                continue
            print(f"     {k:30s} {v}")
        for m in s.get("must_not_detail", []):
            print(f"     {'':30s} ⚠ LEAKED: {m}")
        print(f"  elapsed: {elapsed:.1f}s  |  API calls: {pact.client.calls}")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    header = f"  {'stem':40s}  opts  claims  crit  !agree  errors"
    print(header)
    for stem, s in all_scores.items():
        print(f"  {stem:40s}  {s['options']:5s}  {s['claims']:6s}  "
              f"{s['criteria']:4s}  {s['false_agreements']:6}  {s['observer_errors']}")


if __name__ == "__main__":
    main()
