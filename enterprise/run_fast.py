"""Enterprise pipeline with 3-pass parallel reconcile.

    python run_fast.py evals/sources/real-02-email-benefit-expense-chain.txt
    python run_fast.py evals/sources/long-01-leadership-clinical-intelligence.txt

Pipeline: ingest -> normalize -> classify -> reconcile_fast (parallel) -> summarize
Output: evals/outputs/<name>.fast_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SOCIAL = _HERE.parent / "social"
for _p in (_HERE, _SOCIAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ingest import parse
from classify import classify
from llm import HAIKU, SONNET, AnthropicLLM
from normalize import normalize
from reconcile import dump
from reconcile_fast import reconcile_fast
from summarize import make_summary


def _to_rows(classified, spans) -> list[dict]:
    span_meta = {s.id: s for s in spans}
    rows = []
    for c in classified:
        s = span_meta[c.span.id]
        rows.append({
            "span_id":      c.span.id,
            "author":       c.span.author,
            "text":         c.span.text,
            "status":       c.span.status.value,
            "labels":       [{"type": l.type.value, "confidence": l.confidence, "gist": l.gist}
                             for l in c.labels],
            "context_note": c.span.context_note,
            "recovered":    c.span.recovered,
            "msg":          s.source_message_ids[0],
            "ts":           s.ts.isoformat(),
        })
    return rows


def run(path: Path, haiku_model: str = HAIKU, sonnet_model: str = SONNET,
        batch: int = 6, workers: int = 12, consolidate: str = "sonnet", log=print) -> dict:
    """consolidate: which model merges duplicate questions/options ("sonnet", "haiku", or "off" for code-only matching)."""
    text = path.read_text(encoding="utf-8")

    # 1. Ingest
    src = parse(text)
    log(f"  source_type : {src.source_type}")
    if src.subject:
        log(f"  subject     : {src.subject!r}")
    log(f"  messages    : {len(src.transcript.messages)}")
    log(f"  speakers    : {src.speakers}")

    # 2. Normalize (Haiku)
    haiku = AnthropicLLM(model=haiku_model)
    sonnet = AnthropicLLM(model=sonnet_model)
    spans = normalize(src.transcript, haiku)
    log(f"  spans       : {len(spans)}")

    # 3. Classify (Haiku)
    classified = classify(spans, haiku)

    # 4. Reconcile fast (3-pass parallel, Haiku)
    rows = _to_rows(classified, spans)
    reg = reconcile_fast(rows, src.speakers, haiku, batch=batch, workers=workers,
                         consolidate=(consolidate != "off"),
                         consolidate_llm=(sonnet if consolidate == "sonnet" else None), log=log)

    # 5. Assemble result
    result = dump(reg, src.speakers, str(path), haiku_model)
    result["_enterprise"] = {
        "source_type":  src.source_type,
        "subject":      src.subject,
        "speakers":     src.speakers,
        "observers":    src.observers,
        "msg_meta":     src.msg_meta,
        "spans":        [json.loads(s.model_dump_json()) for s in spans],
        "classified":   [{
            "span_id":      c.span.id,
            "author":       c.span.author,
            "text":         c.span.text,
            "status":       c.span.status.value,
            "context_note": c.span.context_note,
            "labels":       [{"type": l.type.value, "confidence": l.confidence, "gist": l.gist}
                             for l in c.labels],
        } for c in classified],
    }

    # 6. Summarize (five Sonnet calls: main point, positions, summary, tighten, audit)
    log(f"\n[summarize] Sonnet passes ({sonnet_model})...")

    def _call_llm(system, user):
        resp = sonnet.client.messages.create(
            model=sonnet.model, max_tokens=2048,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    result["summary"] = make_summary(result, _call_llm)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--haiku-model", default=HAIKU)
    ap.add_argument("--sonnet-model", default=SONNET)
    ap.add_argument("--consolidate", choices=["sonnet", "haiku", "off"], default="sonnet",
                    help="model that merges duplicate questions/options (one call); off = code-only matching")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    out = a.out or (_HERE / "evals" / "outputs" / a.source.stem).with_suffix(".fast_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[{a.source.name}]")
    t0 = time.time()
    result = run(a.source, haiku_model=a.haiku_model, sonnet_model=a.sonnet_model,
                 batch=a.batch, workers=a.workers, consolidate=a.consolidate)
    elapsed = time.time() - t0

    out.write_text(json.dumps(result, indent=2))

    ent = result["_enterprise"]
    n_el = len(result.get("elements", []))
    n_ev = len(result.get("events", []))
    n_warn = len(result.get("warnings", []))
    print(f"\n  -> {out}")
    print(f"  time={elapsed:.1f}s  spans={len(ent['spans'])}  elements={n_el}  "
          f"events={n_ev}  warnings={n_warn}")

    if result.get("summary"):
        print()
        print("── Summary ──────────────────────────────────────────────────────")
        print(result["summary"])
        print("─────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
