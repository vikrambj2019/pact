"""Score a decision brief against an answer key.

Works on both Pact's ``structured_snapshot`` (options carry ids, positions carry
``option_id``, the card is ``executive_summary``) and a single-pass baseline's output
(positions carry ``option_title``, the card is ``card``).

Every check is pass/fail and belongs to a category, so runs can be compared on:
false agreements, status labels, actions, separate issues, card safety and citations.

    python -m eval.score_brief eval/answer_key.json path/to/brief.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_STRONG = ("support", "proposed")
_CARD_HEADINGS = ("NEEDS ATTENTION", "PROPOSED", "COMMITTED", "UNVERIFIED", "PREFERENCE")


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def _hits(pattern: str, text: str, exclude: str | None = None) -> bool:
    return bool(_rx(pattern).search(text or "")) and not (exclude and _rx(exclude).search(text or ""))


def _norm(name: str) -> str:
    return (name or "").strip().lower()


def _positions(brief: dict) -> list[dict]:
    titles = {o.get("id"): o.get("title", "") for o in brief.get("options", [])}
    out = []
    for p in brief.get("positions", []) or []:
        title = p.get("option_title") or titles.get(p.get("option_id"), "")
        out.append({**p, "option_title": title})
    return out


def _card_sections(card: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"_head": []}
    current = "_head"
    for line in (card or "").splitlines():
        if line.strip().upper() in _CARD_HEADINGS:
            current = line.strip().upper()
            sections.setdefault(current, [])
            continue
        if line.strip():
            sections.setdefault(current, []).append(line.strip())
    return sections


class _Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, category: str, name: str, passed: bool, detail: str = ""):
        self.checks.append({"category": category, "name": name, "passed": bool(passed), "detail": detail})


def score_brief(brief: dict, key: dict) -> dict:
    r = _Report()
    options = brief.get("options", []) or []
    positions = _positions(brief)
    active = [o for o in options if o.get("status") != "dropped"]

    # ---- agreements: exact affirmer sets, and no false agreement
    false_agreements = 0
    for ag in key["agreements"]:
        opts = [o for o in active if _hits(ag["pattern"], o.get("title", ""), ag.get("exclude"))]
        titles = {o.get("title", "") for o in opts}
        got = {_norm(p["participant_label"]) for p in positions
               if p["option_title"] in titles and p.get("stance") in _STRONG}
        want = {_norm(n) for n in ag["affirmers"]}
        extra = got - want
        false_agreements += len(extra)
        r.add("agreement", f"{ag['label']}: option present and active", bool(opts),
              f"titles={sorted(titles)}")
        r.add("agreement", f"{ag['label']}: affirmers exactly {sorted(ag['affirmers'])}", got == want,
              f"got={sorted(got)} missing={sorted(want - got)} extra={sorted(extra)}")
        for n in ag.get("not_affirmers", []):
            r.add("false_agreement", f"{ag['label']}: {n} is not counted as agreeing", _norm(n) not in got)
        for stance in ("conditional", "acknowledged"):
            for n in ag.get(stance, []):
                has = any(_norm(p["participant_label"]) == _norm(n) and p["option_title"] in titles
                          and p.get("stance") == stance for p in positions)
                r.add("agreement", f"{ag['label']}: {n} is {stance}", has)

    # ---- dropped options
    for d in key["dropped"]:
        matching = [o for o in options if _hits(d["pattern"], o.get("title", ""))]
        r.add("status", f"{d['label']} is dropped, not live",
              bool(matching) and all(o.get("status") == "dropped" for o in matching),
              f"statuses={[o.get('status') for o in matching]}")

    # ---- fallback options
    for f in key.get("fallback", []):
        matching = [o for o in options if _hits(f["pattern"], o.get("title", ""))]
        r.add("status", f"{f['label']} is a fallback",
              bool(matching) and all(o.get("status") == "fallback" for o in matching),
              f"statuses={[o.get('status') for o in matching]}")

    # ---- constraints
    for c in key.get("constraints", []):
        found = [x for x in brief.get("constraints", []) or [] if _hits(c["pattern"], x.get("text", ""))]
        r.add("status", f"constraint: {c['label']}", bool(found))
        if c.get("sensitive"):
            r.add("privacy", f"constraint '{c['label']}' marked sensitive",
                  bool(found) and all(x.get("sensitive") for x in found))

    # ---- open blockers
    ob = key["open_blockers"]
    open_b = [b for b in brief.get("blockers", []) or [] if b.get("resolution_status", "open") == "open"]
    texts = " | ".join(b.get("text", "") for b in open_b)
    r.add("status", f"at most {ob['max_count']} open blockers", len(open_b) <= ob["max_count"], texts)
    for p in ob.get("must_mention_all", []):
        r.add("status", f"open blockers include /{p}/", _hits(p, texts), texts)
    if ob.get("must_mention_any"):
        r.add("status", f"open blockers include one of {ob['must_mention_any']}",
              any(_hits(p, texts) for p in ob["must_mention_any"]), texts)
    for bad in ob.get("must_not_mention", []):
        r.add("status", f"'{bad}' is not an open blocker", not _hits(re.escape(bad), texts), texts)

    # ---- open objections
    oo = key.get("open_objections")
    if oo:
        titles = {o.get("id"): o.get("title", "") for o in options}
        live = [o for o in brief.get("objections", []) or []
                if o.get("resolution_status", "open") == "open" and not o.get("moot")]
        targets = [[titles.get(i, "") for i in o.get("option_ids", [])] or o.get("option_titles", []) for o in live]
        r.add("status", f"{oo['count']} open objections", len(live) == oo["count"], f"{[o.get('text') for o in live]}")
        if oo.get("all_on"):
            r.add("status", f"open objections all target /{oo['all_on']}/",
                  bool(live) and all(t and all(_hits(oo["all_on"], x) for x in t) for t in targets), f"{targets}")

    # ---- actions
    actions = brief.get("actions", []) or []
    for a in key["actions"]:
        matches = [x for x in actions if _hits(a["pattern"], x.get("text", ""), a.get("exclude"))]
        if a.get("unowned"):
            hit = [x for x in matches if x.get("status", "open") == "open" and not x.get("owner_label")]
            r.add("action", f"{a['label']}: open with no named owner", bool(hit),
                  f"{[(x.get('text'), x.get('owner_label')) for x in matches]}")
            if hit and "deadline_contains" in a:
                dl = f"{hit[0].get('deadline') or ''} {hit[0].get('due') or ''}".lower()
                r.add("action", f"{a['label']}: deadline {a['deadline_contains']}", a["deadline_contains"] in dl, dl)
            continue
        if "owner_not" in a:
            bad = [x for x in matches if x.get("status", "open") == "open"
                   and _norm(x.get("owner_label")) == _norm(a["owner_not"])]
            r.add("action", a["label"], not bad, f"{[(x.get('text'), x.get('owner_label'), x.get('status')) for x in matches]}")
            continue
        ok_owner = [x for x in matches if _norm(x.get("owner_label")) == _norm(a["owner"])]
        r.add("action", f"{a['label']}: present with owner", bool(ok_owner),
              f"{[(x.get('text'), x.get('owner_label')) for x in matches]}")
        if not ok_owner:
            continue
        x = ok_owner[0]
        if "status" in a:
            r.add("action", f"{a['label']}: status {a['status']}", x.get("status", "open") == a["status"], str(x.get("status")))
        if "acceptance" in a:
            r.add("action", f"{a['label']}: {a['acceptance']}", x.get("acceptance", "committed") == a["acceptance"],
                  str(x.get("acceptance")))
        if "due_contains" in a:
            r.add("action", f"{a['label']}: due {a['due_contains']}", a["due_contains"] in (x.get("due") or "").lower(),
                  str(x.get("due")))
        if "deadline_contains" in a:
            dl = f"{x.get('deadline') or ''} {x.get('due') or ''}".lower()
            r.add("action", f"{a['label']}: deadline {a['deadline_contains']}", a["deadline_contains"] in dl, dl)

    # ---- separate issues
    sep = " | ".join(f"{i.get('title', '')} {i.get('rationale', '')}" for i in brief.get("separate_issues", []) or [])
    for s in key["separate_issues"]:
        r.add("separate", f"separate issue: {s['label']}", _hits(s["pattern"], sep), sep)

    # ---- the card
    card = brief.get("executive_summary") or brief.get("card") or ""
    ck = key["card"]
    sections = _card_sections(card)
    low = card.lower()
    for term in ck.get("privacy_terms", []):
        r.add("privacy", f"card does not mention '{term}'", term.lower() not in low)
    for term in ck.get("must_not_contain", []):
        r.add("card", f"card does not mention '{term}'", term.lower() not in low)
    committed = " | ".join(sections.get("COMMITTED", []))
    for pat in ck.get("committed_must_not_match", []):
        r.add("card", f"committed does not match /{pat}/", not _hits(pat, committed), committed)
    for rq in ck.get("requested_must_be_qualified", []):
        lines = [ln for ln in card.splitlines() if _hits(rq["pattern"], ln)]
        r.add("card", rq["label"], all(_hits(rq["qualifier"], ln) for ln in lines), " | ".join(lines))
    attention = " | ".join(sections.get("NEEDS ATTENTION", []))
    for pat in ck.get("attention_must_not_match", []):
        r.add("card", f"needs-attention does not raise /{pat}/", not _hits(pat, attention), attention)
    for pat in ck.get("attention_should_mention", []):
        r.add("card", f"needs-attention raises /{pat}/", _hits(pat, attention), attention)
    outside = " | ".join(ln for k, v in sections.items() if k != "UNVERIFIED" for ln in v)
    for pat in ck.get("outside_unverified_must_not_match", []):
        r.add("card", f"/{pat}/ is not stated as fact", not _hits(pat, outside))
    unverified = " | ".join(sections.get("UNVERIFIED", []))
    for pat in ck.get("unverified_should_mention", []):
        r.add("card", f"unverified section flags /{pat}/", _hits(pat, unverified), unverified)
    for name in ck.get("dropped_must_be_marked", []):
        lines = [ln for ln in card.splitlines() if _hits(rf"\b{name}\b", ln)]
        r.add("card", f"{name} appears only as dropped", all(_hits("dropp|ruled out|out\\b|removed", ln) for ln in lines),
              " | ".join(lines))
    if ck.get("every_line_cited"):
        bullets = [ln for k, v in sections.items() if k not in ("_head", "PREFERENCE") for ln in v]
        uncited = [ln for ln in bullets if not re.search(r"\[[^\]]+\]\s*$", ln)]
        r.add("citation", "every card line cites its source message", bool(bullets) and not uncited,
              f"{len(uncited)} of {len(bullets)} uncited")

    passed = sum(c["passed"] for c in r.checks)
    by_cat: dict[str, list[int]] = {}
    for c in r.checks:
        by_cat.setdefault(c["category"], [0, 0])
        by_cat[c["category"]][0] += c["passed"]
        by_cat[c["category"]][1] += 1
    return {
        "score": passed / len(r.checks) if r.checks else 0.0,
        "passed": passed,
        "total": len(r.checks),
        "false_agreements": false_agreements,
        "by_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())},
        "checks": r.checks,
    }


def main(argv: list[str]) -> int:
    key = json.loads(Path(argv[1]).read_text())
    brief = json.loads(Path(argv[2]).read_text())
    result = score_brief(brief, key)
    for c in result["checks"]:
        if not c["passed"]:
            print(f"FAIL [{c['category']}] {c['name']}  {c['detail'][:160]}")
    print(f"\n{result['passed']}/{result['total']} checks passed; false agreements: {result['false_agreements']}")
    print(result["by_category"])
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
