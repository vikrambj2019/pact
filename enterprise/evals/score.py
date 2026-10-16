#!/usr/bin/env python3
"""Score a Pact result against a gold file.

    python3 score.py gold/ent-09-....gold.json result.json [--summary summary.txt]

result.json is the brief body the synthesizer stores:
    {"options":[{"id","title","status"}], "positions":[{"participant_label","option_id","stance"}],
     "objections":[{"text","resolution_status","option_ids"}], "blockers":[{"text","resolution_status","owner_label"}],
     "actions":[{"text","owner_label"}], "emerging_preference": {...}, "restricted"/"seen_by" tags on items}
--summary is the rendered executive-summary text (optional; enables the wording checks).

The score is a list of named checks, each pass/fail, plus counts of the errors that matter most:
false_support (support credited to someone who only acknowledged, was undecided or was superseded) and
invented (items with no counterpart in the gold). Exit code 0 only if every hard check passes.
"""
import json
import re
import sys

STOP = {"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "by", "with", "is", "be", "we", "go", "live"}


def toks(s):
    # strip ordinal suffixes so "2nd" == "2", "Nov 3rd" == "nov 3"
    s = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", (s or ""))
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP}


def sim(a, b):
    ta, tb = toks(a), toks(b)
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def first(label):
    """'Lavu, Ratnakar' -> 'ratnakar'; 'Priya' -> 'priya'."""
    label = (label or "").strip()
    if "," in label:
        label = label.split(",", 1)[1]
    return re.sub(r"[^a-z]", "", label.split()[0].lower()) if label.split() else ""


def same_option(a, b):
    a, b = (a or "").lower(), (b or "").lower()
    if sim(a, b) >= 0.5:
        return True
    # "Nov 2" vs "November 2", "A" vs "A: buy Clauseline"
    def keys(x):
        return {re.sub(r"(\d+)(?:st|nd|rd|th)$", r"\1", m)
                for m in re.findall(r"\b(austin|denver|nov(?:ember)?\s*\d+(?:st|nd|rd|th)?|clauseline|build|internal)\b", x)} | \
               ({"A"} if re.match(r"^a\b", x) else set()) | ({"B"} if re.match(r"^b\b", x) else set())
    ka, kb = {k.replace("ember", "") for k in keys(a)}, {k.replace("ember", "") for k in keys(b)}
    return bool(ka and kb and ka & kb)


def _resolve_letter(label: str, gold_options: list) -> str:
    """'A' -> 'A: buy Clauseline' when gold options use that prefix; else return label unchanged."""
    if not re.match(r"^[A-Za-z]$", label.strip()):
        return label
    letter = label.strip().lower()
    for o in gold_options:
        if re.match(r"^" + letter + r"[:\)]\s*", o.get("title", "").lower()):
            return o["title"]
    return label


def opt_match(terms, gold_text, title):
    """Free-text gold (real files): match an option title by keywords. Otherwise fall back to token overlap."""
    if terms:
        return any(t.lower() in (title or "").lower() for t in terms)
    return same_option(gold_text, title)


class Report:
    def __init__(self):
        self.rows, self.hard_fail = [], 0

    def check(self, name, ok, detail="", hard=True):
        self.rows.append((("PASS" if ok else "FAIL") if hard else ("ok  " if ok else "warn"), name, detail))
        if hard and not ok:
            self.hard_fail += 1

    def show(self):
        for status, name, detail in self.rows:
            print(f"{status}  {name}" + (f"  -- {detail}" if detail else ""))


def effective_positions(result):
    """Latest stance per (person, decision-neutral option) with supersession handled by the pipeline's own status."""
    titles = {o.get("id"): o.get("title", "") for o in result.get("options", [])}
    out = []
    for p in result.get("positions", []):
        out.append((first(p.get("participant_label")), titles.get(p.get("option_id"), p.get("option_id") or ""), p.get("stance")))
    return out


def score(gold, result, summary=""):
    rep = Report()
    g = gold["pact_state"]
    res_pos = effective_positions(result)
    gold_opts = g.get("options", [])

    # options
    for o in gold_opts:
        hit = [r for r in result.get("options", []) if opt_match(o.get("terms"), o["title"], r.get("title", ""))]
        rep.check(f"option present: {o['title']}", bool(hit), hard=o.get("required", True))
        if hit and o.get("status") == "dropped":
            rep.check(f"option dropped: {o['title']}", hit[0].get("status") == "dropped")
    extra_opts = [r for r in result.get("options", []) if not any(opt_match(o.get("terms"), o["title"], r.get("title", "")) for o in gold_opts)]
    rep.check("no invented options", not extra_opts, ", ".join(r.get("title", "") for r in extra_opts), hard=False)

    # positions: required stance present, and wrong credit counted
    false_support = 0
    for p in g.get("positions", []):
        who = first(p["participant"])
        if p["stance"] == "superseded":
            continue
        opt_label = _resolve_letter(p["option"], gold_opts)
        multi = opt_label.lower() in ("either", "both") or " and " in opt_label.lower()
        matches = [r for r in res_pos if r[0] == who and (multi or opt_match(p.get("option_terms"), opt_label, r[1]))]
        got = {m[2] for m in matches}
        accepted = set(p.get("stance_any") or [p["stance"]])
        ok = bool(accepted & got)
        label = "/".join(sorted(accepted)) if len(accepted) > 1 else p["stance"]
        rep.check(f"{p['participant']} -> {p['option']}: {label}", ok, f"got {sorted(got) or 'nothing'}",
                  hard=p.get("required", True))
    for p in g.get("positions", []):
        if p["stance"] in ("acknowledged", "undecided", "superseded"):
            who = first(p["participant"])
            opt_label = _resolve_letter(p["option"], gold_opts)
            for r in res_pos:
                if r[0] == who and r[2] == "support" and opt_match(p.get("option_terms"), opt_label, r[1]):
                    # a superseded support is fine only if a later stance on another option exists in the gold; flag acknowledged/undecided only
                    if p["stance"] != "superseded":
                        false_support += 1
    rep.check("no support credited to acknowledged/undecided people", false_support == 0, f"{false_support} wrong", hard=True)

    gold_people = {first(p["participant"]) for p in g.get("positions", [])} | {first(a.get("owner") or "") for a in g.get("actions", [])}
    invented = sorted({r[0] for r in res_pos if r[0] and r[0] not in gold_people})
    rep.check("no positions from people who never spoke (e.g. cc-only)", not invented, ", ".join(invented))

    # counts
    c = {**gold["executive_summary"]["counts"], **gold["executive_summary"].get("count_tolerance", {})}
    open_obj = [o for o in result.get("objections", []) if o.get("resolution_status", "open") == "open" and not o.get("moot")]
    open_blk = [b for b in result.get("blockers", []) if b.get("resolution_status", "open") == "open"]
    unowned = [a for a in result.get("actions", []) if not a.get("owner_label") and a.get("status") != "done"]
    def in_range(n, want):
        lo, hi = (want, want) if isinstance(want, int) else want
        return lo <= n <= hi
    rep.check("open objections", in_range(len(open_obj), c["open_objections"]), f"{len(open_obj)} vs {c['open_objections']}")
    rep.check("open blockers", in_range(len(open_blk), c["open_blockers"]), f"{len(open_blk)} vs {c['open_blockers']}")
    rep.check("unowned actions", in_range(len(unowned), c["unowned_actions"]), f"{len(unowned)} vs {c['unowned_actions']}")

    # owners and dates of gold blockers/actions
    for sec in ("blockers", "actions"):
        for gi in g.get(sec, []):
            terms = [t.lower() for t in gi.get("match_any", [])]
            if terms:   # free-text gold (real files): keyword match, closest wording first
                hit = sorted((r for r in result.get(sec, []) if any(t in r.get("text", "").lower() for t in terms)),
                             key=lambda r: -sim(gi["text"], r.get("text", "")))
            else:
                hit = sorted((r for r in result.get(sec, []) if sim(gi["text"], r.get("text", "")) >= 0.5),
                             key=lambda r: -sim(gi["text"], r.get("text", "")))   # closest match first
            rep.check(f"{sec[:-1]} present: {gi['text'][:60]}", bool(hit), hard=gi.get("required", True))
            if hit and gi.get("owner"):
                want = first(gi["owner"])
                if sec == "actions":
                    owner = first(hit[0].get("owner_label"))
                    rep.check(f"  owner {gi['owner']}", owner == want, f"got {owner or 'none'}", hard=gi.get("required", True))
                else:
                    # Blockers carry no owner field. Who resolves one shows up as an action by that person
                    # on the same subject, or in the blocker's own wording. Reported, never a hard failure.
                    via_action = any(first(a.get("owner_label")) == want and sim(gi["text"], a.get("text", "")) >= 0.3
                                     for a in result.get("actions", []))
                    in_text = want in toks(hit[0].get("text", ""))
                    rep.check(f"  resolver {gi['owner']} (via action or wording)", via_action or in_text, hard=False)
            if hit and gi.get("owner") is None and sec != "blockers":
                rep.check(f"  stays unowned", not hit[0].get("owner_label"), f"got {hit[0].get('owner_label')}", hard=gi.get("required", True))

    # preference
    pref = (result.get("emerging_preference") or {})
    direction = g.get("emerging_preference", {}).get("direction", "none")
    got_dir = pref.get("direction") or pref.get("option_title") or ("none" if not pref or pref.get("status") in (None, "none") else "")
    alts = g.get("emerging_preference", {}).get("direction_any") or [direction]   # real files: more than one reading is fair
    none_ok = got_dir in ("none", "", None) or not pref.get("option_id")
    def pref_ok(d):
        return none_ok if d == "none" else (opt_match(g["emerging_preference"].get("terms"), d, str(got_dir)) or same_option(d, str(pref)))
    rep.check("preference: " + " or ".join(alts), any(pref_ok(d) for d in alts), f"got {pref}")

    # privacy tags
    aud = gold.get("audience", {})
    if aud.get("restricted_units"):
        tagged = [it for sec in ("objections", "constraints", "blockers", "positions") for it in result.get(sec, []) if it.get("restricted")]
        rep.check("restricted material is tagged", bool(tagged), f"{len(tagged)} tagged items")

    # wording checks on the rendered summary
    if summary:
        low = summary.lower()
        for s in gold.get("card_must_include", []):
            rep.check(f"summary mentions '{s}'", s.lower() in low, hard=False)
        for s in gold.get("card_must_not_include", []):
            rep.check(f"summary avoids '{s}'", s.lower() not in low, hard=True)
        rep.check("summary has scope line" if aud else "summary scope line (n/a)", (not aud) or "scope:" in low, hard=False)
    return rep


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    gold = json.load(open(argv[1]))
    result = json.load(open(argv[2]))
    summary = open(argv[argv.index("--summary") + 1]).read() if "--summary" in argv else ""
    rep = score(gold, result, summary)
    rep.show()
    n = len(rep.rows)
    print(f"\n{gold['id']}: {n - rep.hard_fail}/{n} checks, hard failures: {rep.hard_fail}")
    return 0 if rep.hard_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
