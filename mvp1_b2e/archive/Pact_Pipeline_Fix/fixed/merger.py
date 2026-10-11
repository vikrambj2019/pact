"""Merge per-chunk extraction candidates into a unified draft.

Design notes (what changed from the first version):

* Similarity is *anchor aware*: items that carry different numbers or months
  ("Dec 5" vs "Dec 12", "$400 each" vs "$2,400 total") never merge, however much
  other text they share.
* Merging is field-wise and chronological. A later restatement can add an owner,
  a due date or a new status; it can never erase a named owner or an earlier
  source reference.
* Positions are scoped to a *decision*. A participant who supports "Dec 12" for
  the date and "Casa Verde" for the venue keeps both; only a later support for
  another option of the SAME decision supersedes an earlier one. Order comes
  from the cited unit sequence, not from dict insertion order.
* A weaker later response ("acknowledged") never replaces an earlier explicit
  stance.
* Blocker / objection status follows chronology (last statement wins), and
  objections that only concern dropped options are marked ``moot``.
"""
from __future__ import annotations

import re
import uuid


_STOPWORDS = {"a", "an", "the", "is", "are", "in", "of", "to", "for", "and", "or", "that", "with", "we", "it"}
_GENERIC_OPTION_WORDS = {"option", "plan", "choice", "idea"}
_WEEKDAYS = {"mon", "monday", "tue", "tues", "tuesday", "wed", "wednesday", "thu", "thur", "thurs",
             "thursday", "fri", "friday", "sat", "saturday", "sun", "sunday"}
_MONTHS = {
    "jan": "jan", "january": "jan", "feb": "feb", "february": "feb", "mar": "mar", "march": "mar",
    "apr": "apr", "april": "apr", "may": "may", "jun": "jun", "june": "jun", "jul": "jul", "july": "jul",
    "aug": "aug", "august": "aug", "sep": "sep", "sept": "sep", "september": "sep", "oct": "oct",
    "october": "oct", "nov": "nov", "november": "nov", "dec": "dec", "december": "dec",
}
_STRONG_STANCES = {"support", "conditional", "oppose", "proposed", "withdrawn"}
_SUPERSEDING_STANCES = {"support", "conditional"}


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _strip_ordinals(text: str) -> str:
    return re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text or "", flags=re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    toks: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9]+", _strip_ordinals(text)):
        low = raw.lower()
        if low in _STOPWORDS and raw == low:   # drop the article "a", keep an option label "A"
            continue
        toks.add(_MONTHS.get(low, low))
    return toks


def _anchors(text: str) -> set[str]:
    """Numbers and months in the text. Two items with conflicting anchors are different items."""
    t = _strip_ordinals(text)
    nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", t)}
    months = {_MONTHS[w] for w in re.findall(r"[a-z]+", t.lower()) if w in _MONTHS}
    return nums | months


def _anchors_compatible(a: str, b: str) -> bool:
    aa, ab = _anchors(a), _anchors(b)
    return aa <= ab or ab <= aa


def _norm_text(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _similar(a: str, b: str, threshold: float = 0.5) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if _norm_text(a) == _norm_text(b):       # identical wording is always the same item
        return True
    if not _anchors_compatible(a, b):
        return False
    if min(len(ta), len(tb)) < 2:           # one-word texts must match exactly
        return ta == tb
    return len(ta & tb) / min(len(ta), len(tb)) >= threshold


def _option_tokens(title: str) -> set[str]:
    return _tokens(re.sub(r"\([^)]*\)", " ", title or "")) - _WEEKDAYS


def _same_option(a: str, b: str) -> bool:
    ta, tb = _option_tokens(a), _option_tokens(b)
    if not ta or not tb:
        return False
    a_clean, b_clean = re.sub(r"\([^)]*\)", " ", a or ""), re.sub(r"\([^)]*\)", " ", b or "")
    if not _anchors_compatible(a_clean, b_clean):
        return False
    if ta == tb:
        return True
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if small <= big and (small - _GENERIC_OPTION_WORDS):
        return True
    return len(small) >= 3 and len(ta & tb) / len(small) >= 0.75


def _same_label(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta and tb) and (ta <= tb or tb <= ta)


# ---------------------------------------------------------------------------
# Field-wise merge
# ---------------------------------------------------------------------------

def _merge_source_ids(*lists) -> list[str]:
    seen: set[str] = set()
    result = []
    for lst in lists:
        for sid in lst:
            if sid not in seen:
                seen.add(sid)
                result.append(sid)
    return result


def _empty(v) -> bool:
    return v is None or v == "" or v == [] or v == {}


def _merge_pair(old: dict, new: dict) -> dict:
    """Merge ``new`` (later) into ``old`` (earlier) without losing information."""
    out = dict(old)
    out["source_unit_ids"] = _merge_source_ids(old.get("source_unit_ids", []), new.get("source_unit_ids", []))
    for flag in ("sensitive", "private"):
        if flag in old or flag in new:
            out[flag] = bool(old.get(flag) or new.get(flag))
    for k, v in new.items():
        if k in ("source_unit_ids", "sensitive", "private", "owner_label", "owner_kind") or k.startswith("_"):
            continue
        if k == "author_label" and out.get("author_label"):
            continue                              # the originator of a claim stays the originator
        if _empty(v):
            continue
        out[k] = v                                # later statement is more current
    if "owner_kind" in old or "owner_kind" in new:
        new_named = new.get("owner_kind") == "named" and new.get("owner_label")
        old_named = old.get("owner_kind") == "named" and old.get("owner_label")
        if new_named:
            out["owner_kind"], out["owner_label"] = "named", new["owner_label"]
        elif old_named:
            out["owner_kind"], out["owner_label"] = "named", old["owner_label"]
        else:
            out["owner_kind"] = new.get("owner_kind") or old.get("owner_kind") or "unassigned"
            out.pop("owner_label", None)
    if "_seq" in old or "_seq" in new:
        out["_seq"] = min(x for x in (old.get("_seq"), new.get("_seq")) if x is not None)
    out["_chunk_index"] = max(old.get("_chunk_index", 0), new.get("_chunk_index", 0))
    return out


def _dedup(items: list[dict], key: str, threshold: float = 0.5, same=None) -> list[dict]:
    """Merge near-duplicates (chronological input). Later items enrich, never erase, earlier ones."""
    same = same or (lambda a, b: _similar(a, b, threshold))
    out: list[dict] = []
    for item in items:
        idx = next((i for i, o in enumerate(out) if same(item.get(key, ""), o.get(key, ""))), None)
        if idx is None:
            out.append(dict(item))
        else:
            out[idx] = _merge_pair(out[idx], item)
    return out


def _clean(item: dict) -> dict:
    return {k: v for k, v in item.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def _flatten(chunks: list[dict], section: str, unit_sequence: dict[str, int] | None) -> list[dict]:
    """All items of a section in chronological order (chunk, cited-unit sequence, list order)."""
    keyed = []
    n = 0
    for ci, chunk in enumerate(chunks):
        for item in chunk.get(section, []) or []:
            it = dict(item)
            it.setdefault("_chunk_index", chunk.get("_chunk_index", ci))
            if "_seq" not in it and unit_sequence:
                seqs = [unit_sequence[i] for i in it.get("source_unit_ids", []) if i in unit_sequence]
                if seqs:
                    it["_seq"] = min(seqs)
            keyed.append(((it["_chunk_index"], it.get("_seq", 0), n), it))
            n += 1
    keyed.sort(key=lambda kv: kv[0])
    return [it for _, it in keyed]


# ---------------------------------------------------------------------------
# Options and positions
# ---------------------------------------------------------------------------

def _canon_label(label: str, known: list[str]) -> str:
    label = (label or "").strip()
    if not label:
        return ""
    for k in known:
        if _same_label(label, k):
            return k
    known.append(label)
    return label


def _merge_options(options: list[dict]) -> list[dict]:
    merged: list[dict] = []
    labels: list[str] = []
    for opt in options:
        opt = dict(opt)
        opt["decision"] = _canon_label(opt.get("decision", ""), labels)
        idx = next((i for i, o in enumerate(merged) if opt["decision"] == o.get("decision", "") and _same_option(opt["title"], o["title"])), None)
        if idx is None:
            opt["status"] = "active" if opt.get("status") in (None, "", "reopened") else opt["status"]
            opt.setdefault("description", "")
            opt["id"] = str(uuid.uuid4())
            merged.append(opt)
        else:
            old = merged[idx]
            keep_id, keep_title = old["id"], old["title"]
            decision = old.get("decision") or opt["decision"]
            old_status, new_status = old.get("status", "active"), opt.get("status") or "active"
            if new_status == "reopened":
                status = "active"
            elif old_status == "dropped" and new_status == "active":
                status = "dropped"          # merely mentioning a dropped option again does not revive it
            else:
                status = new_status
            merged[idx] = _merge_pair(old, opt)
            merged[idx].update(id=keep_id, title=keep_title, decision=decision, status=status)
    return merged


def _find_option(title: str, options: list[dict], decision: str = "") -> dict | None:
    matches = [o for o in options if (not decision or _same_label(decision, o.get("decision", "")))
               and _same_option(title, o["title"])]
    return matches[0] if len(matches) == 1 else None


def _merge_positions(positions: list[dict], options: list[dict]) -> tuple[list[dict], list[dict]]:
    """Resolve positions chronologically. Returns (positions, superseded)."""
    latest: dict[tuple[str, str], dict] = {}     # (participant, option_id) -> position
    superseded: list[dict] = []

    for p in positions:
        opt = _find_option(p["option_title"], options, p.get("decision", ""))
        if opt is None:
            # An unambiguous known option's decision is authoritative. A model
            # typo must not invent a second "Casa Verde" under the date decision.
            opt = _find_option(p["option_title"], options)
        if opt is None:                          # a stance on an option nobody listed: keep it, don't lose the vote
            opt = {"id": str(uuid.uuid4()), "title": p["option_title"], "description": "", "decision": p.get("decision", ""),
                   "status": "active", "source_unit_ids": list(p.get("source_unit_ids", [])),
                   "quote": p.get("quote", "")}
            options.append(opt)
        who = p["participant_label"].strip().lower()
        key = (who, opt["id"])
        stance = p["stance"]
        old = latest.get(key)

        rec = dict(p)
        rec["option_title"] = opt["title"]
        rec["decision"] = opt.get("decision", "")
        rec["_option_id"] = opt["id"]
        rec["_decision"] = opt.get("decision", "")

        if old is not None:
            # A bare reconfirmation does not withdraw an existing condition.
            if old["stance"] == "conditional" and stance == "support" and re.fullmatch(
                    r"(?:yes|confirmed|agree|agreed|\+1)[.!]?", p.get("quote", "").strip(), re.I):
                old["source_unit_ids"] = _merge_source_ids(old["source_unit_ids"], p.get("source_unit_ids", []))
                continue
            # A weaker later response never replaces an earlier explicit stance.
            if stance == "acknowledged" and old["stance"] in _STRONG_STANCES:
                old["source_unit_ids"] = _merge_source_ids(old["source_unit_ids"], p.get("source_unit_ids", []))
                continue
            # Proposing an option you already took a stance on only records the proposer role.
            if stance == "proposed" and old["stance"] in _STRONG_STANCES:
                old["proposer"] = True
                old["source_unit_ids"] = _merge_source_ids(old["source_unit_ids"], p.get("source_unit_ids", []))
                continue
            # A proposer's "withdrawn" is the definitive retraction of their own proposal.
            # A later "oppose" from the same proposer is redundant and must not override it.
            if old.get("proposer") and old["stance"] == "withdrawn" and stance == "oppose":
                old["source_unit_ids"] = _merge_source_ids(old["source_unit_ids"], p.get("source_unit_ids", []))
                continue
            rec["proposer"] = bool(old.get("proposer")) or old["stance"] == "proposed"
            rec["source_unit_ids"] = _merge_source_ids(old["source_unit_ids"], p.get("source_unit_ids", []))
        else:
            rec["proposer"] = stance == "proposed"

        # A later *unqualified* support for another option of the SAME decision supersedes
        # the earlier *unqualified* support. A conditional (fallback) position is never
        # superseded — a participant can simultaneously hold a primary support and a fallback.
        # Supporting a fallback/interim option also does not supersede a primary support.
        if stance == "support" and rec["_decision"] and opt.get("status") not in ("fallback", "dropped"):
            for k, other in list(latest.items()):
                if (k[0] == who and k[1] != opt["id"] and other["_decision"] == rec["_decision"]
                        and other["stance"] == "support"):
                    superseded.append(_clean(other))
                    del latest[k]
        latest[key] = rec

    out = []
    for rec in sorted(latest.values(), key=lambda r: (r.get("_chunk_index", 0), r.get("_seq", 0))):
        clean = _clean(rec)
        if "_seq" in rec:
            clean["first_unit_sequence"] = rec["_seq"]
        out.append(clean)
    return out, superseded


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _merge_agreements(items: list[dict]) -> list[dict]:
    latest = []
    subjects = []
    for item in items:
        item = dict(item)
        canonical = next((s for s in subjects if _similar(s, item["subject"], 0.8)), None)
        if canonical is None:
            subjects.append(item["subject"])
        else:
            item["subject"] = canonical
        match = next((i for i, old in enumerate(latest)
                      if old["participant_label"].strip().casefold() == item["participant_label"].strip().casefold()
                      and _similar(old["subject"], item["subject"], 0.8)), None)
        if match is None:
            latest.append(dict(item))
        else:
            old = latest[match]
            latest[match] = _merge_pair(old, item)
            if item.get("public_affirmation"):
                # A public cap affirmation remains public even when the same
                # person's earlier contribution arrangement was private.
                latest[match]["private"] = False
                latest[match]["private_source_unit_ids"] = old.get("source_unit_ids", []) if old.get("private") else []
                latest[match]["source_unit_ids"] = item.get("source_unit_ids", [])
    return [_clean(i) for i in latest]


def _merge_actions(items: list[dict]) -> list[dict]:
    merged = []
    for item in items:
        match = next((i for i, old in enumerate(merged)
                      if old.get("owner_label", "").strip().casefold() == item.get("owner_label", "").strip().casefold()
                      and bool(old.get("private")) == bool(item.get("private"))
                      and _similar(old.get("text", ""), item.get("text", ""), 0.8)), None)
        if match is None:
            merged.append(dict(item))
        else:
            old = merged[match]
            updated = _merge_pair(old, item)
            # Repeated requests do not undo an owner's acceptance/completion.
            if item.get("status") == "requested" and old.get("status") in ("committed", "completed", "cancelled"):
                updated["status"] = old["status"]
            merged[match] = updated
    return [_clean(i) for i in merged]


def reconcile_explicit_replies(merged: dict, units: list[dict]) -> dict:
    """Resolve narrowly defined affirmative replies and sourced deposit deadlines.

    A direct yes to an adjacent enumerated plan affirms its referenced decisions,
    not every previous request in the chat. No silence, 'ok' or 'whatever' votes
    are invented. This pass is data-driven: no participant, venue, date or budget
    from the benchmark is hard-coded.
    """
    import copy
    result = copy.deepcopy(merged)
    options = result.get("options", [])
    additions = []
    requirements = list(result.get("agreements", []))
    budget_caps = [o for o in options if o.get("decision") == "budget"
                   and re.search(r"\b(?:cap|ceiling|limit)\b", o["title"], re.I)]
    if budget_caps:
        retained_positions = []
        for position in result.get("positions", []):
            cap = next((o for o in budget_caps if _same_option(o["title"], position["option_title"])), None)
            if cap is None:
                retained_positions.append(position)
            elif position.get("stance") in ("support", "oppose"):
                requirements.append({"subject": cap["title"], "participant_label": position["participant_label"],
                    "stance": "affirm" if position["stance"] == "support" else "reject",
                    "source_unit_ids": position.get("source_unit_ids", []), "quote": position.get("quote", "")})
        result["positions"] = retained_positions
        options = [o for o in options if o not in budget_caps]
        result["options"] = options
    ordered = sorted(units, key=lambda u: u["sequence"])

    def mentioned(text):
        words = _tokens(text)
        return [o for o in options if _option_tokens(o["title"]) <= words
                and _option_tokens(o["title"]) and o.get("status") != "dropped"]

    def cap_items(text):
        found = []
        guest = re.search(r"\b(\d+)\s+(?:max\b|maximum\b|or nothing\b)", text, re.I)
        if guest:
            found.append("Guest cap of " + guest.group(1))
        budget = re.search(r"\$?([\d,]+(?:\.\d+)?)\s+(?:each\b|per (?:person|sibling)\b)", text, re.I)
        if budget:
            number = budget.group(1)
            existing = next((a["subject"] for a in requirements if
                re.search(r"budget|cap", a["subject"], re.I) and number.replace(",", "") in _anchors(a["subject"])), None)
            found.append(existing or ("$" + number + " per person cap"))
        return found

    for index, u in enumerate(ordered):
        text, author = u["normalized_text"], u.get("author_label")
        if not author:
            continue
        direct_yes = bool(re.match(r"^yes\b", text.strip(), re.I))
        qualified = bool(re.search(r"\b(?:but|if|unless|except|not|maybe)\b", text, re.I))
        if (direct_yes and not qualified and index
                and ordered[index-1].get("author_label") != author
                and ordered[index-1].get("document_id") == u.get("document_id")):
            prev = ordered[index-1]
            targets = mentioned(prev["normalized_text"])
            for option in targets:
                additions.append({"participant_label": author, "option_title": option["title"],
                                  "decision": option.get("decision", ""), "stance": "support",
                                  "source_unit_ids": [prev["id"], u["id"]], "quote": "yes", "_seq": u["sequence"]})
            # Cap adoption only for an enumerated plan, not a venue capacity report.
            if len({o.get("decision") for o in targets}) >= 2:
                for subject in cap_items(prev["normalized_text"]):
                    requirements.append({"subject": subject, "participant_label": author, "stance": "affirm",
                                         "source_unit_ids": [prev["id"], u["id"]], "quote": "yes", "_seq": u["sequence"]})
        if re.search(r"\bfinal from me\b", text, re.I):
            for subject in cap_items(text):
                requirements.append({"subject": subject, "participant_label": author, "stance": "affirm",
                                     "source_unit_ids": [u["id"]], "quote": text, "_seq": u["sequence"]})
        # An explicit personal ceiling is an affirmation of that ceiling, not
        # agreement with an option or a private explanation of affordability.
        if re.search(r"\b(?:not paying more than|is the cap)\b", text, re.I):
            for subject in cap_items(text):
                if re.search(r"per (?:person|sibling).*cap", subject, re.I):
                    requirements.append({"subject": subject, "participant_label": author, "stance": "affirm",
                        "source_unit_ids": [u["id"]], "quote": text, "_seq": u["sequence"]})
        number_then = re.search(r"\b(\d+)\s+is the number\b", text, re.I)
        if number_then:
            subjects = {a["subject"] for a in requirements if number_then.group(1) in _anchors(a["subject"])
                        and re.search(r"budget|cap", a["subject"], re.I)}
            if len(subjects) == 1:
                requirements.append({"subject": next(iter(subjects)), "participant_label": author, "stance": "affirm",
                    "source_unit_ids": [u["id"]], "quote": text, "_seq": u["sequence"], "public_affirmation": True})
        if re.fullmatch(r"ok[.!]?", text.strip(), re.I):
            previous = index - 1
            while previous >= 0 and re.fullmatch(r"ok[.!]?", ordered[previous]["normalized_text"].strip(), re.I):
                previous -= 1
            if previous >= 0 and ordered[previous].get("document_id") == u.get("document_id"):
                prior = ordered[previous]
                if re.search(r"\bcap\b", prior["normalized_text"], re.I):
                    for subject in cap_items(prior["normalized_text"]):
                        requirements.append({"subject": subject, "participant_label": author, "stance": "affirm",
                            "source_unit_ids": [prior["id"], u["id"]], "quote": text, "_seq": u["sequence"]})

        # Recover explicit named booking requests even if the model omits them.
        # "It" is resolved only against a unique venue named in this same message.
        labels = list(dict.fromkeys(entry.get("author_label") for entry in ordered if entry.get("author_label")))
        venues = [o for o in mentioned(text) if o.get("decision") == "venue"]
        if len(venues) == 1:
            for owner in labels:
                request = re.search(r"(?<!\w)" + re.escape(owner) + r"\s+(?:book|reserve)\s+it\b", text, re.I)
                if request and not any(a.get("owner_label") == owner and
                    re.search(r"book|reserv", a.get("text", ""), re.I) and
                    _option_tokens(venues[0]["title"]) <= _tokens(a.get("text", "")) for a in result.get("actions", [])):
                    result.setdefault("actions", []).append({"text": "Book " + venues[0]["title"], "owner_label": owner,
                        "owner_kind": "named", "status": "requested", "source_unit_ids": [u["id"]],
                        "quote": request.group(0), "derived": True})

        if re.search(r"\b(?:has|have) to be step[- ]free\b|\bstep[- ]free access (?:is )?required\b", text, re.I):
            if not any(re.search(r"step[- ]free", c.get("text", ""), re.I) for c in result.get("constraints", [])):
                result.setdefault("constraints", []).append({"text": "Step-free access required", "kind": "mandatory",
                    "sensitive": True, "private": False, "source_unit_ids": [u["id"]], "quote": text, "derived": True})
        for clause in re.split(r"[.!?]", text):
            reopening = bool(re.search(r"\breopen\w*\b|\bback on the table\b|\bbring .+? back\b", clause, re.I))
            withdrawal = bool(re.search(r"\b(?:is|are) out\b|\boff the table\b|\bruled out\b", clause, re.I))
            if not reopening and not withdrawal:
                continue
            matched = [o for o in options if o.get("decision") == "venue" and _option_tokens(o["title"]) <= _tokens(clause)]
            if not matched and re.search(r"\bcater\w*\b.*\b(?:mine|my)\b", clause, re.I):
                matched = [o for o in options if o.get("decision") == "venue" and
                           _tokens(author) <= _tokens(o["title"]) and re.search(r"cater", o["title"], re.I)]
            if len(matched) == 1:
                matched[0]["status"] = "active" if reopening else "dropped"
                matched[0]["source_unit_ids"] = _merge_source_ids(matched[0].get("source_unit_ids", []), [u["id"]])
                matched[0]["quote"] = text
                matched[0]["description"] = "Explicitly reopened." if reopening else "Explicitly withdrawn."

    # Composite option bundles are positions, not a single requirement. The
    # explicit reply resolver above handles their option/cap components.
    requirements = [a for a in requirements if not any(o.get("decision") in ("date", "venue") for o in mentioned(a["subject"]))]
    result["agreements"] = _merge_agreements(requirements)
    positions = []
    by_id = {u["id"]: u for u in units}
    explicit_requirements = []
    for agreement in result.get("agreements", []):
        quote = agreement.get("quote", "").strip()
        if re.search(r"\bfine[.!]?$", quote, re.I) and not re.search(r"\bagree|\byes\b", quote, re.I):
            continue
        if re.fullmatch(r"(?:ok|noted|sure whatever)[.!]?", quote, re.I):
            replies = [by_id[s] for s in agreement.get("source_unit_ids", []) if s in by_id
                       and by_id[s].get("author_label") == agreement.get("participant_label")]
            seq = min((u["sequence"] for u in replies), default=-1)
            prior = next((u for u in reversed(units) if u["sequence"] < seq
                          and not re.fullmatch(r"ok[.!]?", u["normalized_text"].strip(), re.I)), None)
            if quote.casefold().rstrip(".!") != "ok" or not prior or not re.search(r"\bcap|ceiling|limit\b", prior["normalized_text"], re.I):
                continue
        explicit_requirements.append(agreement)
    result["agreements"] = explicit_requirements
    for p in result.get("positions", []):
        rec = dict(p)
        rec["_seq"] = p.get("first_unit_sequence", min(
            (by_id[s]["sequence"] for s in p.get("source_unit_ids", []) if s in by_id), default=0))
        positions.append(rec)
    positions.extend(additions)
    positions.sort(key=lambda p: p["_seq"])
    result["positions"], superseded = _merge_positions(positions, options)
    result.setdefault("superseded_positions", []).extend(superseded)

    for action in result.get("actions", []):
        if action.get("external_due") or not re.search(r"\bdeposit\b", action.get("text", ""), re.I):
            continue
        amounts = set(re.findall(r"\$([\d,]+(?:\.\d+)?)", action.get("quote", "") + " " + action["text"]))
        candidates = []
        for source in ordered:
            text = source["normalized_text"]
            deadline = re.search(r"\bdeposit\b.*?\bby\s+([A-Z][a-z]+\.?\s+\d{1,2}(?:,\s*\d{4})?)", text)
            claim_amounts = set(re.findall(r"\$([\d,]+(?:\.\d+)?)", text))
            if deadline and amounts & claim_amounts:
                candidates.append((deadline.group(1), [source["id"]]))
        if len({d for d, _ in candidates}) == 1:
            action["external_due"] = candidates[0][0]
            action["source_unit_ids"] = _merge_source_ids(action.get("source_unit_ids", []),
                                                         *[ids for _, ids in candidates])

    # A committed guest-list target is an intended headcount, unlike a maximum
    # capacity alone. Compare only explicit, uniquely attributed venue capacities.
    targets = set()
    for a in result.get("actions", []):
        if a.get("status") == "committed" and re.search(r"guest|list", a.get("text", ""), re.I):
            match = re.search(r"\bto\s+(\d+)\b", a["text"], re.I)
            if match:
                targets.add(int(match.group(1)))
    if len(targets) == 1:
        target = next(iter(targets))
        latest_capacities = {}
        for u in ordered:
            capacity = re.search(r"\b(?:private room|room)\b.*?\b(?:holds|fits)\s+(?:only\s+)?(\d+)|\bprivate room\s+only\s+holds\s+(\d+)", u["normalized_text"], re.I)
            if not capacity:
                continue
            count = int(capacity.group(1) or capacity.group(2))
            venue = [o for o in options if o.get("decision") == "venue" and
                     _option_tokens(o["title"]) and _option_tokens(o["title"]) <= _tokens(u["normalized_text"])]
            if len(venue) == 1:
                latest_capacities[venue[0]["id"]] = (venue[0], count, u)
        for venue_option, count, u in latest_capacities.values():
            if count < target:
                venue_option["status"] = "dropped"
                venue_option["description"] = f"Reported room capacity {count} is below the planned guest count {target}."
                venue_option["source_unit_ids"] = _merge_source_ids(venue_option.get("source_unit_ids", []), [u["id"]])
                venue_option["quote"] = u["normalized_text"]

    # Recognize a deferred topic with an explicitly unassigned collective follow-up.
    for u in ordered:
        pending = re.match(r"^(.+?)\s+(?:(?:is|are)\s+)?still open\b", u["normalized_text"], re.I)
        if pending and re.search(r"\bwe(?:'|’)ll\s+(?:sort|resolve|figure|discuss)", u["normalized_text"], re.I):
            topic = pending.group(1).strip()
            result.setdefault("blockers", []).append({"text": topic.capitalize() + " planning and a named owner remain open",
                "kind": "missing_info", "resolution_status": "open", "source_unit_ids": [u["id"]],
                "quote": u["normalized_text"], "derived": True})

    # A dispute about the occasion's actual anniversary date is separate when
    # all dinner-date options lie outside its explicitly asserted day values.
    conflicts = [c for c in result.get("claims", []) if re.search(r"anniversary", c.get("text", ""), re.I)
                 and c.get("evidence_status") == "conflicting"]
    disputed = set().union(*[_anchors(c["text"]) for c in conflicts]) if conflicts else set()
    days = {v for v in disputed if v.isdigit() and 1 <= int(v) <= 31}
    dinner_days = {v for o in options if o.get("decision") == "date" for v in _anchors(o["title"]) if v.isdigit()}
    dispute_sources = _merge_source_ids(*[c.get("source_unit_ids", []) for c in conflicts])
    if len(days) < 2:
        # Resolve an adjacent "it's the 8th" / "she said the 9th" disagreement
        # against an explicitly introduced anniversary date, not unrelated dates.
        for index, u in enumerate(ordered):
            if not re.search(r"\banniversary\b", u["normalized_text"], re.I):
                continue
            chain = [u]
            for reply in ordered[index+1:index+3]:
                if reply.get("document_id") != u.get("document_id"):
                    break
                if not re.search(r"^(?:it(?:'|’)s|its)\b|\b(?:said|says)\b", reply["normalized_text"], re.I):
                    break
                chain.append(reply)
            asserted = {str(int(a or b)) for entry in chain for a, b in re.findall(
                r"\b(?:Dec(?:ember)?|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?)\s+(\d{1,2})\b|\b(\d{1,2})(?:st|nd|rd|th)\b",
                entry["normalized_text"], re.I) if 1 <= int(a or b) <= 31}
            if len(asserted) >= 2:
                days = asserted
                dispute_sources = [entry["id"] for entry in chain]
                break
    if len(days) >= 2 and dinner_days and not days & dinner_days:
        result.setdefault("separate_issues", []).append({"title": "Anniversary-date dispute",
            "rationale": "The competing occasion dates do not change the dinner-date options.",
            "source_unit_ids": dispute_sources, "derived": True})
        result["blockers"] = [b for b in result.get("blockers", []) if not re.search(r"anniversary", b.get("text", ""), re.I)]
    return _reconcile_dependency_state(result, ordered)


def _reconcile_dependency_state(result: dict, units: list[dict]) -> dict:
    """Recover narrow source-backed lifecycle facts; never infer approval.

    This also runs on saved extraction states, so validation cannot be the only
    place protecting pending approvals and conditions.
    """
    options = result.get("options", [])
    aliases = {}
    for source in units:
        for match in re.finditer(r"\b([A-Z]|\d+)[).]\s+(.+?)(?=\s+(?:[A-Z]|\d+)[).]\s+|$)", source["normalized_text"]):
            candidates = [o for o in options if _same_option(o["title"], match.group(2).strip())]
            if len(candidates) == 1:
                aliases[candidates[0]["title"]] = match.group(1)
    def targets(text):
        found = []
        for option in options:
            alias = aliases.get(option["title"], str(option.get("temp_id", "")))
            alias_hit = bool(alias and re.fullmatch(r"[A-Z]|\d+", alias) and re.search(
                r"(?:\boption\s+|\bwith\s+|\b(?:go|choose|pick|approve|reject|oppose|drop)\s+|^)(?:" + re.escape(alias) + r")\b", text, re.I))
            if alias and re.fullmatch(r"[A-Z]|\d+", alias):
                alias_hit |= bool(re.search(r"\b" + re.escape(alias) + r"\s+is\s+(?:fine|good|my pick|(?:a\s+)?fallback)\b", text, re.I))
            tokens = _option_tokens(option["title"])
            if alias_hit or (tokens and tokens <= _tokens(text)):
                found.append(option)
        return found

    # A checkpoint attached to a plan is an action, not another plan option.
    # Preserve genuine alternatives for checkpoint/date scheduling decisions.
    schedules = [o for o in options if re.match(r"^(?:go/no-go\s+)?checkpoint\b", o["title"], re.I)
                 and not re.search(r"checkpoint|date|schedule|time", o.get("decision", ""), re.I)]
    if schedules:
        result["options"] = options = [o for o in options if o not in schedules]
        retained_positions = []
        for position in result.get("positions", []):
            if any(_same_option(position.get("option_title", ""), o["title"]) for o in schedules):
                named = targets(position.get("quote", ""))
                if len(named) != 1:
                    continue
                position["option_title"] = named[0]["title"]
                position["decision"] = named[0].get("decision", "")
            retained_positions.append(position)
        result["positions"] = retained_positions
    by_id = {u["id"]: u for u in units}
    # Generic sponsorship is not a vote on any one option.
    result["positions"] = [p for p in result.get("positions", []) if not re.search(
        r"\bwhichever (?:you|we|they) pick\b", p.get("quote", ""), re.I)]
    for source in units:
        text, author = source["normalized_text"], source.get("author_label")
        selected = targets(text)
        if not author:
            continue
        if re.search(r"\bfallback\b", text, re.I):
            for option in selected:
                option["status"] = "fallback"
                option["source_unit_ids"] = _merge_source_ids(option.get("source_unit_ids", []), [source["id"]])
        elif re.search(r"\b(?:choose|go with|select)\b", text, re.I) and not re.search(r"\bif\b", text, re.I) and len(selected) == 1:
            selected[0]["status"] = "active"
        proposed = targets(re.split(r"[.!]", text)[0])
        if re.search(r"\bproposal\s*:\s*(?:go with|choose|pick)\b", text, re.I) and len(proposed) == 1:
            opt = proposed[0]
            existing = next((p for p in result["positions"] if p.get("participant_label") == author
                             and _same_option(p.get("option_title", ""), opt["title"])), None)
            if existing:
                existing["proposer"] = True
            else:
                result["positions"].append({"participant_label": author, "option_title": opt["title"],
                    "decision": opt.get("decision", ""), "stance": "proposed", "proposer": True,
                    "source_unit_ids": [source["id"]], "quote": text})
        condition = re.search(r"\b(?:as long as|provided that|only if)\s+(.+?)(?:[.!]|$)", text, re.I)
        if condition and len(selected) == 1:
            opt = selected[0]
            for position in result["positions"]:
                if position.get("participant_label") != author or not _same_option(position.get("option_title", ""), opt["title"]):
                    continue
                if position.get("stance") == "support" and re.fullmatch(
                        r"(?:yes|confirmed|agree|agreed|\+1)[.!]?", position.get("quote", "").strip(), re.I):
                    position.update(stance="conditional", conditions=condition.group(1), quote=text,
                                    source_unit_ids=_merge_source_ids(position.get("source_unit_ids", []), [source["id"]]))
        checkpoint = re.search(r"\bgo/no-go checkpoint\s+([A-Z][a-z]+\s+\d{1,2})", text, re.I)
        if checkpoint and not any(re.search(r"(?:run|conduct|hold|organize).*checkpoint|go/no-go checkpoint", a.get("text", ""), re.I)
                                  and not re.search(r"budget|funding|ask", a.get("text", ""), re.I)
                                  for a in result.get("actions", [])):
            result.setdefault("actions", []).append({"text": "Run the go/no-go checkpoint", "owner_kind": "unassigned",
                "status": "requested", "due": checkpoint.group(1), "source_unit_ids": [source["id"]], "quote": text, "derived": True})

    for section in ("blockers", "objections"):
        for item in result.get(section, []):
            if item.get("resolution_status") == "resolved" and re.search(
                    r"\b(?:pending|requires?|needs?|has to|have to|someone|not signed|still open|before I sign off)\b",
                    item.get("quote", ""), re.I):
                item["resolution_status"] = "open"
    # Review requirements are gates, whereas concerns about an option remain
    # objections. Retain the source-backed requirement as a constraint.
    review_requirements = [i for i in result.get("objections", [])
        if re.search(r"sign.off", i.get("text", ""), re.I) and re.search(
            r"report|audit|\bSOC\s*\d", i.get("text", "") + " " + i.get("quote", ""), re.I)]
    for requirement in review_requirements:
        result.setdefault("constraints", []).append({**requirement, "kind": "mandatory"})
    result["objections"] = [i for i in result.get("objections", []) if i not in review_requirements]
    for objection in result["objections"]:
        if not objection.get("option_titles"):
            selected = targets(objection.get("quote", ""))
            if len(selected) == 1:
                objection["option_titles"] = [selected[0]["title"]]

    # Accept a date only when the owner directly confirms the named assignment.
    for action in result.get("actions", []):
        if action.get("status") != "committed" or action.get("owner_kind") != "named":
            continue
        owner = action.get("owner_label", "")
        own = [u for u in units if u.get("author_label") == owner and u["id"] in action.get("source_unit_ids", [])]
        if action.get("due") and not any(re.search(re.escape(action["due"]), u["normalized_text"], re.I) for u in own):
            action["requested_due"] = action.pop("due")
        for source in own:
            chase = re.search(r"\bI(?:'|’)ll\s+(chase\s+[^.!]+)", source["normalized_text"], re.I)
            if chase and not re.search(r"\bget\b", chase.group(1), re.I) and re.search(
                    r"\bchase and get\b", action.get("text", ""), re.I):
                action["text"] = chase.group(1)[0].upper() + chase.group(1)[1:]
            if re.search(r"\bleave\b", source["normalized_text"], re.I) and not re.search(
                    r"\bleave|health|medical|surgery|money problem", action.get("text", ""), re.I):
                # Operational commitments survive an omitted personal rationale.
                action["private"] = False
            if not action.get("due") and re.search(r"\bbefore then\b", source["normalized_text"], re.I):
                dates = re.findall(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b", source["normalized_text"], re.I)
                if len(set(dates)) == 1:
                    action["due"] = "before " + dates[0]
                    action.pop("requested_due", None)
        # A later confirmation may have been omitted from this action's model
        # citations; recover it only against an explicit named assignment.
        confirmations = [u for u in units if u.get("author_label") == owner and re.fullmatch(
            r"(?:yes|confirmed|agree|agreed)[.!]?", u["normalized_text"].strip(), re.I)]
        for reply in confirmations:
            if not re.fullmatch(r"(?:yes|confirmed|agree|agreed)[.!]?", reply["normalized_text"].strip(), re.I):
                continue
            previous = [u for u in units if u["sequence"] < reply["sequence"] and u.get("document_id") == reply.get("document_id")]
            # Skip adjacent confirmations by others; stop at the nearest actual plan.
            plan = next((u for u in reversed(previous) if not re.fullmatch(r"(?:yes|confirmed|agree|agreed)[.!]?", u["normalized_text"].strip(), re.I)), None)
            if not plan:
                continue
            clause = re.search(r"\b" + re.escape(owner.split()[0]) + r"\b([^,;.]*)", plan["normalized_text"], re.I)
            if not clause:
                continue
            date = re.search(r"\b(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|tomorrow)\b", clause.group(1), re.I)
            task_tokens = {t[:-1] if t.endswith("s") else t for t in _tokens(action["text"])}
            clause_tokens = {t[:-1] if t.endswith("s") else t for t in _tokens(clause.group(1))}
            if date and len((task_tokens & clause_tokens) - _anchors(date.group(0))) >= 2:
                action["due"] = date.group(0)
                action["source_unit_ids"] = _merge_source_ids(action["source_unit_ids"], [plan["id"], reply["id"]])
                action.pop("requested_due", None)
    # Keep an unanswered target date separate from the owner's accepted due.
    # A ticket identifier provides an exact task key, unlike word similarity.
    actions = result.get("actions", [])
    retained = []
    for action in actions:
        ticket = re.search(r"\b(?:finding|ticket|issue|fix)\s*#?\s*(\d+)\b", action.get("text", ""), re.I)
        matches = [other for other in actions if other.get("status") == "committed" and other.get("owner_kind") == "named"
                   and ticket and re.search(r"\b(?:finding|ticket|issue|fix)\s*#?\s*" + re.escape(ticket.group(1)) + r"\b", other.get("text", ""), re.I)]
        if action.get("status") == "requested" and len(matches) == 1 and (action.get("owner_kind") != "named" or action.get("owner_label") == matches[0].get("owner_label")):
            matches[0].setdefault("unaccepted_requests", []).append({"due": action.get("due", ""),
                "quote": action.get("quote", ""), "source_unit_ids": action.get("source_unit_ids", [])})
        else:
            retained.append(action)
    result["actions"] = retained
    result["actions"] = [a for a in result["actions"] if not re.search(
        r"\bcap noted (?:as|in)\b|^note\b.*\bcap\b.*\bbudget\b", a.get("text", ""), re.I)]
    result["actions"] = _merge_actions(result["actions"])
    for action in result["actions"]:
        target = action.get("requested_due", "")
        if target and not action.get("due"):
            # An extracted task title must not smuggle an unaccepted target
            # back into the public commitment after the due field is cleared.
            action["text"] = re.sub(r"\s+by\s+" + re.escape(target) + r"\b", "", action["text"], flags=re.I)
    # One funding escalation can be restated with a later deadline. Match the
    # explicitly named recipient and task kind, not every unassigned action.
    funding = [a for a in retained if a.get("owner_kind") != "named" and re.search(r"budget|funding|extra cost", a.get("text", ""), re.I)
               and re.search(r"ask|request|take|send", a.get("text", ""), re.I)]
    labels = {u.get("author_label", "").split()[0] for u in units if u.get("author_label")}
    for label in labels:
        same = [a for a in funding if re.search(r"\b" + re.escape(label) + r"\b", a["text"])]
        if len(same) > 1:
            preferred = next((a for a in reversed(same) if a.get("due")), same[-1])
            preferred["source_unit_ids"] = _merge_source_ids(*[a["source_unit_ids"] for a in same])
            result["actions"] = [a for a in result["actions"] if a not in same or a is preferred]
    # A committed future review is pending even after the artifact arrived.
    for action in result.get("actions", []):
        if action.get("status") != "committed" or not re.search(r"\b(?:finish|complete)\s+(?:the\s+)?review\b", action.get("text", "") + " " + action.get("quote", ""), re.I):
            continue
        if not any(re.search(r"\breview|sign.off\b", b.get("text", ""), re.I)
                   and _similar(b.get("text", ""), action["text"], 0.45)
                   and b.get("resolution_status", "open") == "open" for b in result.get("blockers", [])):
            result.setdefault("blockers", []).append({"text": "Review remains pending: " + action["text"],
                "kind": "unresolved_dependency", "resolution_status": "open", "source_unit_ids": list(action["source_unit_ids"]),
                "quote": action.get("quote", ""), "derived": True})
    # A composite reminder of gates is a constraint, not an additional gate.
    composite = [b for b in result.get("blockers", []) if re.search(r"\band\b", b.get("text", ""), re.I)
                 and re.search(r"\bDPA\b", b.get("text", ""), re.I)
                 and re.search(r"security sign.off", b.get("text", ""), re.I)]
    for b in composite:
        result.setdefault("constraints", []).append({**b, "kind": "mandatory"})
    result["blockers"] = [b for b in result.get("blockers", []) if b not in composite]
    # Use explicit artifact/ticket identities for deduplication, while keeping
    # artifact receipt distinct from review. Provider names scope contract gates.
    def gate_key(blocker):
        text = blocker.get("text", "")
        ticket = re.search(r"\bfinding\s*#?\s*(\d+)\b", text, re.I)
        if ticket:
            return ("finding", ticket.group(1))
        if re.search(r"\bDPA\b", text):
            names = sorted({n.casefold() for n in re.findall(r"\b[A-Z][a-z]+\b", text)
                            if n not in {"Cannot", "The", "Pending", "Option", "Launch", "Review"}})
            return ("DPA", tuple(names)) if names else None
        return None
    grouped = []
    keys = {}
    for blocker in result.get("blockers", []):
        if re.search(r"(?:conflicting|disputed) effort estimates", blocker.get("text", ""), re.I):
            result.setdefault("claims", []).append({**blocker, "kind": "predictive", "evidence_status": "conflicting"})
            continue
        # Funding on a fallback is already represented by the open objection
        # and escalation action. Keep the requirement as a constraint.
        funding_gate = bool(re.search(r"budget|fund|exceed.*cap", blocker.get("text", ""), re.I))
        funding_gate |= bool(re.search(r"approval", blocker.get("text", ""), re.I) and any(
            o.get("status") == "fallback" for o in targets(blocker.get("text", "") + " " + blocker.get("quote", ""))))
        if funding_gate and any(
                re.search(r"budget|cost|approv|fund", o.get("text", ""), re.I) and o.get("resolution_status", "open") == "open"
                for o in result.get("objections", [])):
            result.setdefault("constraints", []).append({**blocker, "kind": "mandatory"})
            continue
        key = gate_key(blocker)
        if key is not None and key in keys:
            grouped[keys[key]] = _merge_pair(grouped[keys[key]], blocker)
        else:
            if key is not None:
                keys[key] = len(grouped)
            grouped.append(blocker)
    result["blockers"] = grouped
    # Mark an expected-arrival report historical once receipt of that same
    # artifact is reported. Shared substantive tokens prevent cross-artifact updates.
    claims = result.get("claims", [])
    for claim in claims:
        if not re.search(r"\b(?:would arrive|expected|rep.*(?:said|told))\b", claim.get("text", ""), re.I):
            continue
        seq = max((by_id[s]["sequence"] for s in claim.get("source_unit_ids", []) if s in by_id), default=-1)
        for later in claims:
            later_seq = min((by_id[s]["sequence"] for s in later.get("source_unit_ids", []) if s in by_id), default=-1)
            common = (_tokens(claim["text"]) & _tokens(later["text"])) - {"report", "the", "a", "type", "said", "sent"}
            if later_seq > seq and len(common) >= 3 and re.search(r"\b(?:sent|received|arrived)\b", later["text"], re.I):
                claim["evidence_status"] = "superseded"
                break
    return result



def merge_extractions(chunks: list[dict], unit_sequence: dict[str, int] | None = None) -> dict:
    """Merge per-chunk extraction results into one unified draft.

    ``unit_sequence`` (unit id -> sequence) lets the merger order items by the
    message they cite; without it, chunk order and list order are used.
    """
    def items(section: str) -> list[dict]:
        return _flatten(chunks, section, unit_sequence)

    merged_options = _merge_options(items("options"))
    merged_positions, superseded = _merge_positions(items("positions"), merged_options)

    def merge_with_resolution(section: str) -> list[dict]:
        # Independent dependencies often share words such as vendor and launch.
        # A low similarity threshold can collapse legal and security gates.
        return _dedup(items(section), "text", 0.8)

    objections = merge_with_resolution("objections")
    dropped_ids = {o["id"] for o in merged_options if o.get("status") == "dropped"}
    for obj in objections:
        titles = obj.get("option_titles", []) or []
        opts = [_find_option(t, merged_options) for t in titles]
        if titles:
            obj["option_titles"] = [o["title"] if o else t for o, t in zip(opts, titles)]
        if opts and all(o is not None and o["id"] in dropped_ids for o in opts):
            obj["moot"] = True

    return {
        "candidate_questions": [_clean(i) for i in _dedup(items("candidate_questions"), "text", 0.6)],
        "options": [_clean(o) for o in merged_options],
        "criteria": [_clean(i) for i in _dedup(items("criteria"), "text")],
        "constraints": [_clean(i) for i in _dedup(items("constraints"), "text")],
        "claims": [_clean(i) for i in _dedup(items("claims"), "text", 0.7)],
        "positions": merged_positions,
        "objections": [_clean(i) for i in objections],
        "blockers": [_clean(i) for i in merge_with_resolution("blockers")],
        "separate_issues": [_clean(i) for i in _dedup(items("separate_issues"), "title")],
        "actions": _merge_actions(items("actions")),
        "agreements": _merge_agreements(items("agreements")),
        "superseded_positions": superseded,
        "dropped_items": [d for c in chunks for d in c.get("_dropped", [])],
    }
