"""LLM-based comparison of pipeline output against gold expected state.

Usage (from repo root):
    # Compare a specific run against its gold:
    python mvp1_b2e/goldendataset/compare_gold.py --credentials /path/to/.env --model claude-sonnet-4-6 \\
        --dataset ent-01-easy-cutover --run 1

    # Compare all available runs across all datasets:
    python mvp1_b2e/goldendataset/compare_gold.py --credentials /path/to/.env --model claude-sonnet-4-6

The LLM evaluates each claim derived from the gold expected state against the
pipeline output and returns a pass / fail / partial verdict with a brief reason.
Results are printed as JSON lines and saved alongside the run file.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLD_DIR = HERE / "output"
RUNS_DIR = HERE / "runs"


# ---------------------------------------------------------------------------
# Claim extraction — converts gold pact_state into evaluable claim objects
# ---------------------------------------------------------------------------

def _gold_claims(gold_state: dict) -> list[dict]:
    """Derive a flat list of semantic claims from the gold expected state."""
    claims: list[dict] = []

    for opt in gold_state.get("options", []):
        # Strip gold-specific annotation suffixes like "(buy)", "(build)", "(lease)" etc.
        import re as _re
        bare_title = _re.sub(r"\s*\((buy|build|lease|rent|hire|vendor)\)\s*$", "", opt["title"], flags=_re.I).strip()
        claims.append({
            "id": f"option_status::{opt['title']}",
            "section": "options",
            "claim": (
                f"The pipeline output contains an option semantically equivalent to '{bare_title}' "
                f"with status '{opt['status']}' for the '{opt.get('decision', '?')}' decision "
                f"(decision label may vary: 'approach', 'provider', 'vendor' etc. are all acceptable)."
            ),
        })
        if opt.get("proposed_by"):
            claims.append({
                "id": f"option_proposer::{opt['title']}",
                "section": "positions",
                "claim": (
                    f"'{opt['proposed_by']}' is recorded as the proposer of option '{opt['title']}'."
                ),
            })

    # Keep only the FINAL stance per participant/option (gold may list historical progression).
    _seen_pos: dict[str, dict] = {}
    for p in gold_state.get("positions", []):
        key = f"{p['participant']}::{p['option']}"
        _seen_pos[key] = p  # last entry wins
    for p in _seen_pos.values():
        claims.append({
            "id": f"position::{p['participant']}::{p['option']}",
            "section": "positions",
            "claim": (
                f"'{p['participant']}' has stance '{p['stance']}' on option '{p['option']}'."
                + (f" Conditions: {p['conditions']}." if p.get("conditions") else "")
            ),
        })

    for p in gold_state.get("superseded_positions", []):
        was = p.get("was") or p.get("stance", "?")
        now = p.get("now", "")
        now_str = f" and now has '{now}'" if now else ""
        claims.append({
            "id": f"superseded::{p['participant']}::{p['option']}",
            "section": "superseded_positions",
            "claim": (
                f"'{p['participant']}' previously held stance '{was}' on '{p['option']}'"
                f"{now_str}. The earlier stance should be superseded/updated."
            ),
        })

    for a in gold_state.get("actions", []):
        due_str = f", due {a['due']}" if a.get("due") else ""
        status_str = f" with status '{a['status']}'" if a.get("status") else ""
        claims.append({
            "id": f"action::{a.get('owner', 'unassigned')}::{a['text'][:40]}",
            "section": "actions",
            "claim": (
                f"There is an action '{a['text']}' owned by '{a.get('owner', 'unassigned')}'"
                f"{status_str}{due_str}."
            ),
        })

    for b in gold_state.get("blockers", []):
        resolution = b.get("status", b.get("resolution_status", "open"))
        claims.append({
            "id": f"blocker::{b['text'][:50]}",
            "section": "blockers",
            "claim": (
                f"There is a blocker '{b['text']}' with resolution status '{resolution}'."
            ),
        })

    for obj in gold_state.get("objections", []):
        resolution = obj.get("status", obj.get("resolution_status", "open"))
        claims.append({
            "id": f"objection::{obj['text'][:50]}",
            "section": "objections",
            "claim": (
                f"There is an objection '{obj['text']}' with resolution status '{resolution}'."
            ),
        })

    for s in gold_state.get("separate_issues", []):
        title = s if isinstance(s, str) else s.get("title", str(s))
        claims.append({
            "id": f"separate::{title[:50]}",
            "section": "separate_issues",
            "claim": f"'{title}' is classified as a separate issue (not part of the main decision).",
        })

    for c in gold_state.get("constraints", []):
        text = c if isinstance(c, str) else c.get("text", str(c))
        claims.append({
            "id": f"constraint::{text[:50]}",
            "section": "constraints",
            "claim": f"The constraint '{text}' is present in the pipeline output.",
        })

    return claims


# ---------------------------------------------------------------------------
# LLM evaluator
# ---------------------------------------------------------------------------

_EVAL_SYSTEM = """\
You are evaluating whether a decision-brief pipeline correctly extracted information
from a discussion transcript. You will be given:
1. A list of claims derived from a human-authored gold standard.
2. The pipeline's output (a structured JSON brief).

For each claim, return a verdict:
- "pass": the pipeline output clearly satisfies the claim.
- "fail": the pipeline output clearly contradicts or omits what the claim requires.
- "partial": the pipeline output partially satisfies the claim (e.g. right item, wrong status or owner).

Be lenient with exact wording: if the pipeline uses semantically equivalent text
(e.g. "Nov 14–15" vs "Nov 14-15", "send customer notice" vs "customer notice sent"),
treat it as a match. Be strict about stance, status, owner, and due-date fields.

Additional leniency rules:
- Decision labels: "provider", "approach", "buy-vs-build", "vendor" etc. for the same
  buy/build decision are semantically equivalent — pass if the options clearly match.
- "withdrawn" stance: if the pipeline records "oppose" for a participant who was the
  proposer of that option (proposer=true or proposed_by matches), treat it as
  equivalent to "withdrawn" and pass.
- "acknowledged" and "undecided" stances are semantically equivalent for comparison
  purposes — pass either when the gold expects the other.
- A blocker and a constraint may describe the same gate — if the pipeline captures a
  requirement as a blocker rather than a constraint (or vice versa) but the substance
  is present, give "partial" rather than "fail".
"""

_EVAL_TOOL = {
    "name": "evaluate_claims",
    "description": "Return a verdict for each claim.",
    "input_schema": {
        "type": "object",
        "required": ["evaluations"],
        "properties": {
            "evaluations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["claim_id", "verdict", "reason"],
                    "properties": {
                        "claim_id": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["pass", "fail", "partial"]},
                        "reason": {"type": "string", "description": "One sentence explaining the verdict."},
                    },
                },
            }
        },
    },
}

# Sections that matter for comparison (skip internal pipeline fields)
_COMPARE_SECTIONS = (
    "options", "positions", "superseded_positions", "objections",
    "blockers", "separate_issues", "actions", "constraints", "agreements",
)


def _pipeline_summary(brief: dict) -> dict:
    """Extract only the fields relevant for comparison from the pipeline brief."""
    # Build option_id -> title map so positions remain readable after _assign_server_ids
    # strips option_title and replaces it with a UUID.
    id_to_title = {o["id"]: o["title"] for o in brief.get("options", []) if "id" in o}

    # Derive proposed_by for each option from positions with proposer=True
    option_proposers: dict[str, str] = {}
    for pos in brief.get("positions", []) + brief.get("superseded_positions", []):
        if pos.get("proposer"):
            opt_title = id_to_title.get(pos.get("option_id", ""), "")
            if opt_title and opt_title not in option_proposers:
                option_proposers[opt_title] = pos.get("participant_label", "")

    out: dict = {}
    for section in _COMPARE_SECTIONS:
        items = brief.get(section, [])
        if not items:
            continue
        cleaned = []
        for item in items:
            # Drop IDs and source refs; keep semantically meaningful fields
            keep = {k: v for k, v in item.items()
                    if k not in ("id", "source_unit_ids", "restored", "_chunk_index", "_seq",
                                 "citation_repaired", "first_unit_sequence")}
            # Re-attach option_title for positions so the evaluator knows which option is referenced
            if "option_id" in keep:
                title = id_to_title.get(keep["option_id"])
                if title:
                    keep["option_title"] = title
                del keep["option_id"]
            # Add proposed_by to options so the evaluator can check proposers directly
            if section == "options" and "title" in keep:
                proposer = option_proposers.get(keep["title"])
                if proposer:
                    keep["proposed_by"] = proposer
            cleaned.append(keep)
        out[section] = cleaned
    return out


async def evaluate(claims: list[dict], brief: dict, llm_key: str, model: str) -> list[dict]:
    """Send claims + pipeline output to the LLM; return per-claim verdicts."""
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=llm_key)

    pipeline_summary = _pipeline_summary(brief)

    # Batch into groups of 20 to stay within context limits
    batch_size = 20
    all_verdicts: list[dict] = []

    for batch_start in range(0, len(claims), batch_size):
        batch = claims[batch_start: batch_start + batch_size]
        user = (
            "Pipeline output (JSON):\n"
            + json.dumps(pipeline_summary, indent=2)
            + "\n\nClaims to evaluate:\n"
            + json.dumps([{"id": c["id"], "claim": c["claim"]} for c in batch], indent=2)
        )

        resp = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=_EVAL_SYSTEM,
            messages=[{"role": "user", "content": user}],
            tools=[_EVAL_TOOL],
            tool_choice={"type": "tool", "name": "evaluate_claims"},
            timeout=120,
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == "evaluate_claims":
                all_verdicts.extend(block.input.get("evaluations", []))
                break

    return all_verdicts


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

def _score(claims: list[dict], verdicts: list[dict]) -> dict:
    verdict_by_id = {v["claim_id"]: v for v in verdicts}
    by_section: dict[str, dict] = {}
    total_pass = total_fail = total_partial = total_missing = 0

    for c in claims:
        section = c["section"]
        if section not in by_section:
            by_section[section] = {"pass": 0, "fail": 0, "partial": 0, "missing": 0, "details": []}

        v = verdict_by_id.get(c["id"])
        if v is None:
            verdict = "missing"
            reason = "LLM did not return a verdict for this claim"
            total_missing += 1
            by_section[section]["missing"] += 1
        else:
            verdict = v["verdict"]
            reason = v.get("reason", "")
            if verdict == "pass":
                total_pass += 1
                by_section[section]["pass"] += 1
            elif verdict == "fail":
                total_fail += 1
                by_section[section]["fail"] += 1
            else:
                total_partial += 1
                by_section[section]["partial"] += 1

        by_section[section]["details"].append({
            "claim_id": c["id"],
            "claim": c["claim"],
            "verdict": verdict,
            "reason": reason,
        })

    total = len(claims)
    return {
        "passed": total_pass,
        "partial": total_partial,
        "failed": total_fail,
        "missing": total_missing,
        "total": total,
        "score_pct": round(100 * (total_pass + 0.5 * total_partial) / max(total, 1), 1),
        "by_section": by_section,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def compare_one(dataset_id: str, run_num: int, llm_key: str, model: str) -> dict:
    gold_path = GOLD_DIR / f"{dataset_id}.gold.json"
    run_path = RUNS_DIR / dataset_id / f"run-{run_num}.json"

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")
    if not run_path.exists():
        raise FileNotFoundError(f"Run file not found: {run_path}")

    gold = json.loads(gold_path.read_text())
    run = json.loads(run_path.read_text())
    brief = run["brief"]

    claims = _gold_claims(gold["pact_state"])
    verdicts = await evaluate(claims, brief, llm_key, model)
    score = _score(claims, verdicts)

    result = {
        "dataset": dataset_id,
        "run": run_num,
        "model": model,
        "level": gold.get("level"),
        "score": score,
        "verdicts": verdicts,
    }

    # Save comparison result alongside the run file
    cmp_path = RUNS_DIR / dataset_id / f"run-{run_num}-comparison.json"
    cmp_path.write_text(json.dumps(result, indent=2))

    return result


async def main(args) -> None:
    creds: dict[str, str] = {}
    for line in Path(args.credentials).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip().strip("\"'")

    key = creds.get("ANTHROPIC_API_KEY") or creds.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("No Anthropic key found in credentials file")

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted(p.stem.replace(".gold", "") for p in GOLD_DIR.glob("*.gold.json"))

    summary_rows = []
    for dataset_id in datasets:
        run_dir = RUNS_DIR / dataset_id
        if not run_dir.exists():
            print(json.dumps({"skip": dataset_id, "reason": "no runs found"}), flush=True)
            continue

        run_files = sorted(f for f in run_dir.glob("run-*.json")
                           if f.stem.split("-")[-1].isdigit())
        runs_to_check = (
            [run_dir / f"run-{args.run}.json"] if args.run else run_files
        )

        for run_path in runs_to_check:
            run_num = int(run_path.stem.split("-")[1])
            print(json.dumps({"comparing": dataset_id, "run": run_num}), flush=True)
            result = await compare_one(dataset_id, run_num, key, args.model)

            score = result["score"]
            row = {
                "dataset": dataset_id,
                "level": result["level"],
                "run": run_num,
                "score_pct": score["score_pct"],
                "passed": score["passed"],
                "partial": score["partial"],
                "failed": score["failed"],
                "total": score["total"],
            }
            print(json.dumps(row), flush=True)
            summary_rows.append(row)

            # Print per-section failures for visibility
            for section, sec_data in score["by_section"].items():
                for d in sec_data["details"]:
                    if d["verdict"] != "pass":
                        print(json.dumps({
                            "dataset": dataset_id, "run": run_num,
                            "section": section, "verdict": d["verdict"],
                            "claim_id": d["claim_id"], "reason": d["reason"],
                        }), flush=True)

    if summary_rows:
        print("\n=== SUMMARY ===", flush=True)
        for row in summary_rows:
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM-based comparison of pipeline output vs gold standard")
    ap.add_argument("--credentials", required=True, help="Path to .env file with ANTHROPIC_API_KEY")
    ap.add_argument("--model", required=True, help="Model ID for the evaluator, e.g. claude-sonnet-4-6")
    ap.add_argument("--dataset", default=None, help="Single dataset ID to compare (default: all)")
    ap.add_argument("--run", type=int, default=None, help="Specific run number to compare (default: all runs)")
    args = ap.parse_args()
    try:
        asyncio.run(main(args))
    except Exception as exc:
        print(json.dumps({"error": str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__}))
        raise SystemExit(1)
