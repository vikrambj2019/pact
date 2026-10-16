"""Run the full enterprise pipeline on one source file.

    python run.py evals/sources/ent-08-email-vendor-private-note.txt
    python run.py evals/sources/ent-10-meeting-golive-decision.txt --out out.json

Pipeline: ingest -> normalize -> classify -> reconcile -> summarize
Output: evals/outputs/<name>.result.json  (reconcile dump + enterprise metadata)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── path bootstrap ──────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SOCIAL = _HERE.parent / "social"
for _p in (_HERE, _SOCIAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ingest import parse                                     # noqa: E402
from classify import classify                                # noqa: E402
from llm import HAIKU, SONNET, AnthropicLLM                 # noqa: E402
from normalize import normalize                              # noqa: E402
from reconcile import dump, matrix, reconcile               # noqa: E402
from summarize import make_summary                           # noqa: E402


# ---------------------------------------------------------------------------
# Bridge: ClassifiedSpan list -> reconcile row dicts (no intermediate file)
# ---------------------------------------------------------------------------

def _to_rows(classified, spans) -> list[dict]:
    """Convert classify() output directly to the row format reconcile() expects."""
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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(path: Path, haiku_model: str = HAIKU, sonnet_model: str = SONNET,
        log=print) -> dict:
    text = path.read_text(encoding="utf-8")

    # 1. Ingest
    src = parse(text)
    log(f"  source_type : {src.source_type}")
    log(f"  subject     : {src.subject!r}" if src.subject else "  subject     : (transcript)")
    log(f"  messages    : {len(src.transcript.messages)}")
    log(f"  speakers    : {src.speakers}")
    if src.observers:
        log(f"  observers   : {src.observers}")

    # 2. Normalize (Haiku)
    haiku = AnthropicLLM(model=haiku_model)
    spans = normalize(src.transcript, haiku)
    log(f"  spans       : {len(spans)}")

    # 3. Classify (Haiku)
    classified = classify(spans, haiku)

    # 4. Reconcile (Sonnet)
    rows = _to_rows(classified, spans)
    sonnet = AnthropicLLM(model=sonnet_model)
    reg = reconcile(rows, src.speakers, sonnet, log=log)

    # 5. Assemble result
    result = dump(reg, src.speakers, str(path), sonnet_model)

    # _enterprise must be set before summarize so Ctx(d) can access spans + classified
    result["_enterprise"] = {
        "source_type":  src.source_type,
        "subject":      src.subject,
        "speakers":     src.speakers,
        "observers":    src.observers,
        "msg_meta":     src.msg_meta,
        "spans":       [json.loads(s.model_dump_json()) for s in spans],
        "classified":  [{
            "span_id":      c.span.id,
            "author":       c.span.author,
            "text":         c.span.text,
            "status":       c.span.status.value,
            "context_note": c.span.context_note,
            "labels":       [{"type": l.type.value, "confidence": l.confidence, "gist": l.gist}
                             for l in c.labels],
        } for c in classified],
    }

    # 6. Summarize (Sonnet) — two passes: main point from full conversation, then structured summary
    def _call_llm(system, user):
        resp = sonnet.client.messages.create(
            model=sonnet.model,
            max_tokens=2048,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    result["summary"] = make_summary(result, _call_llm)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--haiku-model", default=HAIKU)
    ap.add_argument("--sonnet-model", default=SONNET)
    a = ap.parse_args()

    out = a.out or (_HERE / "evals" / "outputs" / a.source.stem).with_suffix(".result.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[{a.source.name}]")
    result = run(a.source, haiku_model=a.haiku_model, sonnet_model=a.sonnet_model)

    out.write_text(json.dumps(result, indent=2))
    print(f"\n  -> {out}")

    ent = result["_enterprise"]
    n_el = len(result.get("elements", []))
    n_ev = len(result.get("events", []))
    n_warn = len(result.get("warnings", []))
    n_spans = len(ent["spans"])
    print(f"  spans={n_spans}  elements={n_el}  events={n_ev}  warnings={n_warn}")

    # Print decision matrix
    from reconcile import matrix as _matrix, Registry, Element, Event
    from dataclasses import dataclass
    reg_check = _rebuild_registry(result, ent["spans"])
    if reg_check:
        print()
        print(matrix(reg_check, ent["speakers"]))

    # Print summary
    if result.get("summary"):
        print()
        print("── Summary ──────────────────────────────────────────────────────")
        print(result["summary"])
        print("─────────────────────────────────────────────────────────────────")


def _rebuild_registry(result: dict, spans_meta: list[dict]):
    """Lightweight registry reconstruction for matrix display — reads the dump."""
    try:
        from reconcile import Registry, Element, Event
        from dataclasses import fields
        import dataclasses

        span_map = {s["id"]: {"author": s["author"], "ts": s["ts"],
                               "msg": s["source_message_ids"][0], "text": s["text"],
                               "labels": [], "status": "active"}
                    for s in spans_meta}
        reg = Registry(span_map)
        for e in result.get("elements", []):
            el = Element(**{f.name: e.get(f.name, f.default if f.default is not dataclasses.MISSING
                            else (f.default_factory() if f.default_factory is not dataclasses.MISSING else None))
                            for f in fields(Element)})
            reg.el[el.id] = el
        for ev in result.get("events", []):
            reg.events.append(Event(**ev))
        return reg
    except Exception:
        return None


if __name__ == "__main__":
    main()
