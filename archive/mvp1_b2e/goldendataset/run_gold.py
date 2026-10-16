"""Run the fixed pipeline against the golden datasets and save outputs.

Usage (from repo root):
    python mvp1_b2e/goldendataset/run_gold.py --credentials /path/to/.env --model claude-sonnet-4-6
    python mvp1_b2e/goldendataset/run_gold.py --credentials /path/to/.env --model claude-sonnet-4-6 \\
        --dataset ent-01-easy-cutover --runs 3
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import types
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXED = HERE.parent / "worker"
INPUT_DIR = HERE / "sample_input"
GOLD_DIR = HERE / "output"
RUNS_DIR = HERE / "runs"


# ---------------------------------------------------------------------------
# Load fixed pipeline without the app's SQLAlchemy / settings dependencies
# ---------------------------------------------------------------------------

def _load(name: str, extra: dict | None = None):
    tree = ast.parse((FIXED / (name + ".py")).read_text())
    tree.body = [
        node for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and (
            node.level or (node.module or "").startswith(("sqlalchemy", "..api", "..worker"))
        ))
    ]
    module = types.ModuleType(name)
    module.__dict__.update({"AsyncSession": object, **(extra or {})})
    exec(compile(tree, str(FIXED / (name + ".py")), "exec"), module.__dict__)
    return module


async def _noop_review(*args, **kwargs):
    return {"status": "skipped", "stage": kwargs.get("stage", ""), "reviewed_element_count": 0,
            "coverage_complete": False, "verdicts": [], "verification_scope": "skipped for evaluation"}

async def _noop_review_summary(*args, **kwargs):
    return {"status": "skipped", "stage": "executive_summary", "reviewed_element_count": 0,
            "coverage_complete": False, "verdicts": [], "verification_scope": "skipped for evaluation"}

reviewer = _load("extraction_reviewer")
extractor = _load("extractor", {"review_extraction": reviewer.review_extraction})
merger = _load("merger")
chunker = _load("chunker")
parser_mod = _load("parser")
synthesizer = _load("synthesizer", {
    "review_extraction": reviewer.review_extraction,
    "review_summary": reviewer.review_summary,
    "quote_in_text": extractor.quote_in_text,
    "_norm_text": merger._norm_text,
    "_same_option": merger._same_option,
    "_similar": merger._similar,
})


# ---------------------------------------------------------------------------
# LLM client — uses the Anthropic SDK (already a project dependency)
# ---------------------------------------------------------------------------

class _LLM:
    def __init__(self, key: str, model: str):
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic(api_key=key)
        self._model = model

    async def extract(self, system: str, user: str, tool_schema: dict, timeout=None) -> dict:
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": tool_schema["name"]},
            timeout=timeout or 180,
        )
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("Structured response truncated; reduce chunk size or increase max_tokens")
        for block in resp.content:
            if block.type == "tool_use" and block.name == tool_schema["name"]:
                return block.input
        raise RuntimeError("No tool_use block in response")

    async def complete(self, system: str, user: str, max_tokens: int = 1024, timeout=None) -> str:
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=timeout or 60,
        )
        for block in resp.content:
            if block.type == "text":
                return block.text
        raise RuntimeError("No text block in response")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

async def run_dataset(dataset_id: str, llm: _LLM, runs: int, output_dir: Path,
                      skip_reviewer: bool = False, coverage_llm: _LLM | None = None) -> None:
    txt_path = INPUT_DIR / f"{dataset_id}.txt"
    gold_path = GOLD_DIR / f"{dataset_id}.gold.json"

    if not txt_path.exists():
        raise FileNotFoundError(f"Transcript not found: {txt_path}")
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    gold = json.loads(gold_path.read_text())
    question_text = gold.get("decision_question", "")

    raw_text = txt_path.read_text()
    units = parser_mod._parse_whatsapp(raw_text)
    if not units:
        # Fall back to plain text if WhatsApp parsing yields nothing
        units = parser_mod._parse_plain_text(raw_text)
    for i, u in enumerate(units):
        u["id"] = f"{dataset_id}-{i:03d}"

    unit_seq = {u["id"]: u["sequence"] for u in units}

    if skip_reviewer:
        extractor.review_extraction = _noop_review
        synthesizer.review_extraction = _noop_review
        synthesizer.review_summary = _noop_review_summary

    output_dir.mkdir(parents=True, exist_ok=True)

    for run_idx in range(runs):
        print(json.dumps({"dataset": dataset_id, "run": run_idx + 1, "units": len(units),
                           "main_model": llm._model,
                           "coverage_model": (coverage_llm or llm)._model}), flush=True)

        results: list[dict] = []
        known_options: list[dict] = []
        known_decisions: list[str] = []
        open_items: dict[str, dict] = {}
        known_actions: list[dict] = []
        known_agreements: list[str] = []
        known_state: dict = {}

        for chunk_idx, chunk in enumerate(chunker.chunk_units(units)):
            extracted = await extractor.extract_chunk(
                chunk, llm, chunk_idx,
                known_options, known_decisions,
                list(open_items.values()),
                known_actions, known_agreements, known_state,
                coverage_llm=coverage_llm,
            )
            results.append(extracted)
            interim = merger.merge_extractions(results, unit_seq)

            known_options = [{"title": o["title"], "decision": o.get("decision", "")}
                             for o in interim["options"]]
            known_decisions = list(dict.fromkeys(
                o["decision"] for o in known_options if o["decision"]
            ))
            known_actions = [
                {k: a[k] for k in ("text", "owner_label", "owner_kind", "status", "due", "external_due") if k in a}
                for a in interim.get("actions", [])
            ]
            known_agreements = list(dict.fromkeys(
                a["subject"] for a in interim.get("agreements", [])
            ))
            known_state = {
                s: [{k: v for k, v in item.items() if k not in ("quote", "source_unit_ids", "id")}
                    for item in interim.get(s, [])]
                for s in ("options", "positions", "claims", "actions", "agreements")
            }
            open_items = {
                merger._norm_text(item["text"]): {"kind": kind, "text": item["text"]}
                for section, kind in (("blockers", "blocker"), ("objections", "objection"))
                for item in interim.get(section, [])
                if item.get("resolution_status", "open") == "open"
            }
            print(json.dumps({
                "dataset": dataset_id, "run": run_idx + 1,
                "chunk": chunk_idx + 1,
                "dropped": len(extracted.get("_dropped", [])),
            }), flush=True)

        merged = merger.merge_extractions(results, unit_seq)
        merged = merger.reconcile_explicit_replies(merged, units)

        # Backfill proposer: for options with no proposer-marked position, the author
        # of the option's earliest cited unit is the proposer.  This is deterministic
        # and handles cases where the LLM emits only a support/conditional stance.
        uid_to_author = {u["id"]: u.get("author_label", "") for u in units}
        for opt in merged.get("options", []):
            opt_positions = [p for p in merged.get("positions", [])
                             if p.get("option_title") == opt["title"]]
            if not opt_positions or any(p.get("proposer") for p in opt_positions):
                continue
            opt_sources = opt.get("source_unit_ids", [])
            earliest_uid = min(opt_sources, key=lambda uid: unit_seq.get(uid, 99999), default=None)
            if not earliest_uid:
                continue
            author = uid_to_author.get(earliest_uid, "")
            if author:
                match = next((p for p in opt_positions
                              if p.get("participant_label", "").lower() == author.lower()), None)
                if match:
                    match["proposer"] = True

        body = synthesizer._assign_server_ids(synthesizer._build_brief(merged))
        body["question"] = {"text": question_text}

        report: dict = {}
        body["executive_summary"] = synthesizer._render_decision_card(
            body,
            {u["id"]: u["normalized_text"] for u in units},
            {u["id"]: {"author": u.get("author_label", ""), "ts": u.get("locator", {}).get("timestamp_str", "")}
             for u in units},
            report,
        )

        out = {
            "meta": {
                "dataset": dataset_id,
                "model": llm._model,
                "run": run_idx + 1,
                "units": len(units),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "brief": body,
            "report": report,
            "units": [{"id": u["id"], "author_label": u.get("author_label"), "normalized_text": u["normalized_text"]}
                      for u in units],
        }
        out_path = output_dir / f"run-{run_idx + 1}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(json.dumps({"dataset": dataset_id, "run": run_idx + 1, "saved": str(out_path)}), flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args) -> None:
    creds: dict[str, str] = {}
    for line in Path(args.credentials).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip().strip("\"'")

    key = creds.get("ANTHROPIC_API_KEY") or creds.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("No Anthropic key found in credentials file (ANTHROPIC_API_KEY or LLM_API_KEY)")

    llm = _LLM(key, args.model)
    # coverage_llm is the model for the focused speech-act pass; defaults to main model
    coverage_llm = _LLM(key, args.coverage_model) if args.coverage_model else None

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted(p.stem.replace(".gold", "") for p in GOLD_DIR.glob("*.gold.json"))

    for dataset_id in datasets:
        out_dir = RUNS_DIR / dataset_id
        await run_dataset(dataset_id, llm, args.runs, out_dir,
                          skip_reviewer=args.skip_reviewer, coverage_llm=coverage_llm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run fixed pipeline against golden datasets")
    ap.add_argument("--credentials", required=True, help="Path to .env file with ANTHROPIC_API_KEY")
    ap.add_argument("--model", required=True,
                    help="Model for main extraction pass, e.g. claude-haiku-4-5-20251001")
    ap.add_argument("--coverage-model", default=None,
                    help="Model for focused coverage pass (positions/speech acts). "
                         "Defaults to --model. Use claude-sonnet-4-6 for best accuracy.")
    ap.add_argument("--dataset", default=None, help="Single dataset ID to run (default: all)")
    ap.add_argument("--runs", type=int, default=1, help="Number of runs per dataset")
    ap.add_argument("--skip-reviewer", action="store_true",
                    help="Bypass the LLM reviewer (for evaluation; production always reviews)")
    args = ap.parse_args()
    try:
        asyncio.run(main(args))
    except Exception as exc:
        print(json.dumps({"error": str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__}))
        raise SystemExit(1)
