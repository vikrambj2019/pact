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
_STRONG_STANCES = {"support", "conditional", "oppose", "proposed"}
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
        toks.add(low)
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
    if "sensitive" in old or "sensitive" in new:
        out["sensitive"] = bool(old.get("sensitive") or new.get("sensitive"))
    for k, v in new.items():
        if k in ("source_unit_ids", "sensitive", "owner_label", "owner_kind") or k.startswith("_"):
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


# Wording that marks "the part of a task nobody took" rather than a repeat of it.
_LEFTOVER = re.compile(r"\b(week of|except|other than|instead|remaining|rest of|cover(?:s|ing)?|back ?up|fill(?:s|ing)? in|uncovered)\b", re.I)


def _same_action(new: dict, old: dict) -> bool:
    """Same task? Wording must match, and an owner-less restatement that adds a qualifier the owned task lacks
    ("cover the week of the 9th") is the part nobody took, not a repeat of the owned task. The merge would
    otherwise hand the leftover part to the person who declined it."""
    if not _similar(new.get("text", ""), old.get("text", "")):
        return False
    new_owned = new.get("owner_kind") == "named" and bool(new.get("owner_label"))
    old_owned = old.get("owner_kind") == "named" and bool(old.get("owner_label"))
    if old_owned and not new_owned and _LEFTOVER.search(new.get("text", "")) \
            and _anchors(new.get("text", "")) != _anchors(old.get("text", "")):
        return False
    return True


def _dedup(items: list[dict], key: str, threshold: float = 0.5, same=None, same_item=None) -> list[dict]:
    """Merge near-duplicates (chronological input). Later items enrich, never erase, earlier ones."""
    same = same or (lambda a, b: _similar(a, b, threshold))
    out: list[dict] = []
    for item in items:
        if same_item:
            idx = next((i for i, o in enumerate(out) if same_item(item, o)), None)
        else:
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
        idx = next((i for i, o in enumerate(merged) if _same_option(opt["title"], o["title"])), None)
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


def _find_option(title: str, options: list[dict]) -> dict | None:
    return next((o for o in options if _same_option(title, o["title"])), None)


def _merge_positions(positions: list[dict], options: list[dict]) -> tuple[list[dict], list[dict]]:
    """Resolve positions chronologically. Returns (positions, superseded)."""
    latest: dict[tuple[str, str], dict] = {}     # (participant, option_id) -> position
    superseded: list[dict] = []

    for p in positions:
        opt = _find_option(p["option_title"], options)
        if opt is None:                          # a stance on an option nobody listed: keep it, don't lose the vote
            opt = {"id": str(uuid.uuid4()), "title": p["option_title"], "description": "", "decision": "",
                   "status": "active", "source_unit_ids": list(p.get("source_unit_ids", [])),
                   "quote": p.get("quote", "")}
            options.append(opt)
        who = p["participant_label"].strip().lower()
        key = (who, opt["id"])
        stance = p["stance"]
        old = latest.get(key)

        rec = dict(p)
        rec["option_title"] = opt["title"]
        rec["_option_id"] = opt["id"]
        rec["_decision"] = opt.get("decision", "")

        if old is not None:
            # A weaker later response never replaces an earlier explicit stance.
            if stance == "acknowledged" and old["stance"] in _STRONG_STANCES:
                old["source_unit_ids"] = _merge_source_ids(old["source_unit_ids"], p.get("source_unit_ids", []))
                continue
            # Proposing an option you already took a stance on only records the proposer role.
            if stance == "proposed" and old["stance"] in _STRONG_STANCES:
                old["proposer"] = True
                old["source_unit_ids"] = _merge_source_ids(old["source_unit_ids"], p.get("source_unit_ids", []))
                continue
            rec["proposer"] = bool(old.get("proposer")) or old["stance"] == "proposed"
            rec["source_unit_ids"] = _merge_source_ids(old["source_unit_ids"], p.get("source_unit_ids", []))
        else:
            rec["proposer"] = stance == "proposed"

        # A later support for another option of the SAME decision supersedes this person's earlier support.
        if stance in _SUPERSEDING_STANCES and rec["_decision"]:
            for k, other in list(latest.items()):
                if (k[0] == who and k[1] != opt["id"] and other["_decision"] == rec["_decision"]
                        and other["stance"] in _SUPERSEDING_STANCES):
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
        return _dedup(items(section), "text")           # last status wins, owners/sources are kept

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
        "actions": [_clean(i) for i in _dedup(items("actions"), "text", same_item=_same_action)],
        "superseded_positions": superseded,
        "dropped_items": [d for c in chunks for d in c.get("_dropped", [])],
    }
