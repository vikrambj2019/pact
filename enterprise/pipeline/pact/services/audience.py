"""Who could see what. Shared by extraction (step 1) and synthesis (step 2).

Email units carry audience metadata in ``SourceUnit.locator`` (no schema change):

    locator = {
        "source_type": "email",
        "to": [...], "cc": [...],            # people addressed / copied
        "audience": [...],                    # to + cc
        "effective_audience": [...],          # audience + sender + anyone who later received a message quoting this one
        "audience_unknown": bool,             # headers missing (e.g. a cover note)
        "restricted": bool,                   # sender asked for it to stay private (own note or a note above it)
        "restricted_reason": str | None,
    }

Chat and meeting units have no ``effective_audience``: everyone in the conversation saw them.
"""
from __future__ import annotations

import copy
import re

_SECTIONS = ("criteria", "constraints", "options", "claims", "positions",
             "objections", "blockers", "separate_issues", "actions")


def _tokens(label: str) -> set[str]:
    label = re.sub(r"<[^>]*>", " ", label or "")
    return set(re.findall(r"[a-z0-9]+", label.lower()))


def same_person(a: str, b: str) -> bool:
    """'Lavu, Ratnakar' == 'Ratnakar Lavu' == 'ratnakar.lavu'. Token-subset match."""
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


def label_in(label: str, people) -> bool:
    return any(same_person(label, p) for p in people or [])


def _meta_audience(meta: dict | None):
    """Return (known, people). known=False when the unit has no audience restriction information."""
    if not meta:
        return False, []
    if meta.get("audience_unknown"):
        return True, []
    eff = meta.get("effective_audience")
    if eff is None:
        return False, []
    return True, list(eff)


def unit_visible_to(meta: dict | None, viewer: str) -> bool:
    """Chat/meeting units are visible to everyone in the conversation. Email units only to their
    effective audience. A unit whose recipients are unknown is visible only to its own author."""
    known, people = _meta_audience(meta)
    if not known:
        return True
    if (meta or {}).get("audience_unknown"):
        return same_person(viewer, (meta or {}).get("author", ""))
    return label_in(viewer, people)


def readers(unit_meta: dict[str, dict], unit_ids) -> tuple[list[str], int]:
    """People who saw *every* cited message that has audience information.

    Returns (names, unknown_count). ``unknown_count`` counts cited messages with no recipient
    information; they are left out of the intersection and reported separately.
    """
    sets: list[list[str]] = []
    unknown = 0
    for uid in unit_ids:
        meta = unit_meta.get(uid)
        known, people = _meta_audience(meta)
        if not known:
            continue
        if (meta or {}).get("audience_unknown"):
            unknown += 1
            continue
        sets.append(people)
    if not sets:
        return [], unknown
    first, rest = sets[0], sets[1:]
    common = [p for p in first if all(label_in(p, s) for s in rest)]
    return common, unknown


def cited_unit_ids(brief_body: dict) -> list[str]:
    seen: dict[str, None] = {}
    for section in _SECTIONS:
        for item in brief_body.get(section, []) or []:
            for uid in item.get("source_unit_ids", []) or []:
                seen.setdefault(uid, None)
    return list(seen)


def attach_audience(brief_body: dict, unit_meta: dict[str, dict]) -> None:
    """Tag every item with ``seen_by`` (people who saw all of its sources) and ``restricted``.

    Computed in code from the item's cited units, never by the model. Items sourced only from
    chat or meeting units get no ``seen_by`` (everyone present saw them).
    """
    for section in _SECTIONS:
        for item in brief_body.get(section, []) or []:
            ids = item.get("source_unit_ids", []) or []
            metas = [unit_meta.get(i) for i in ids]
            if any(m and m.get("restricted") for m in metas):
                item["restricted"] = True
            names, unknown = readers(unit_meta, ids)
            if any(_meta_audience(m)[0] for m in metas):
                item["seen_by"] = names
                if unknown:
                    item["audience_unknown"] = True


def filter_for_viewer(brief_body: dict, unit_meta: dict[str, dict], viewer: str) -> tuple[dict, dict]:
    """A copy of the state containing only what ``viewer`` could see.

    * an item whose sources are all hidden from the viewer is removed
    * an item with some hidden sources is kept, its hidden sources removed, and marked ``partial``
    * positions and objections that point at a removed option are removed with it
    Returns (filtered_state, {"removed": n, "partial": n}).
    """
    out = copy.deepcopy(brief_body)
    removed = partial = 0
    for section in _SECTIONS:
        kept = []
        for item in out.get(section, []) or []:
            ids = item.get("source_unit_ids", []) or []
            visible = [i for i in ids if unit_visible_to(unit_meta.get(i), viewer)]
            if ids and not visible:
                removed += 1
                continue
            if len(visible) != len(ids):
                item["source_unit_ids"] = visible
                item["partial"] = True
                partial += 1
            kept.append(item)
        out[section] = kept
    option_ids = {o.get("id") for o in out.get("options", []) if o.get("id")}
    if option_ids or not brief_body.get("options"):
        before = len(out.get("positions", []))
        out["positions"] = [p for p in out.get("positions", [])
                            if not p.get("option_id") or p["option_id"] in option_ids]
        removed += before - len(out["positions"])
        for ob in out.get("objections", []):
            if ob.get("option_ids"):
                ob["option_ids"] = [i for i in ob["option_ids"] if i in option_ids]
    return out, {"removed": removed, "partial": partial}


def scope_line(unit_meta: dict[str, dict], unit_ids) -> str:
    """One deterministic line saying who saw the cited material. Empty for chats and meetings."""
    ids = [i for i in unit_ids if i in unit_meta]
    audience_ids = [i for i in ids if _meta_audience(unit_meta[i])[0]]
    if not audience_ids:
        return ""
    names, unknown = readers(unit_meta, audience_ids)
    restricted = sum(1 for i in ids if unit_meta[i].get("restricted"))
    parts = [f"Scope: email, {len(audience_ids)} cited message{'s' if len(audience_ids) != 1 else ''}."]
    if names:
        parts.append("Seen in full by " + "; ".join(names) + ".")
    else:
        parts.append("No one is on every cited message.")
    if restricted:
        parts.append(f"{restricted} cited message{'s are' if restricted != 1 else ' is'} marked private by the sender.")
    if unknown:
        parts.append(f"{unknown} cited message{'s have' if unknown != 1 else ' has'} no recipient information.")
    return " ".join(parts)
