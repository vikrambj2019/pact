"""Stage 3 — reconcile: classified spans -> element registry + append-only event log.

    python reconcile.py outputs/ent-03-hard-data-platform.output.json [--model claude-sonnet-5-5]

Input is the *.output.json written by run.py. Writes *.reconcile.json next to it and
prints the current decision matrix.

Design
- The model walks the spans in order, a small batch at a time. For each batch it sees a
  compact snapshot of the registry and returns OPERATIONS (create, stance, status,
  supersede, decide, alias, merge, attach, skip).
- Code validates and applies every op. The model never writes state directly.
- Hard rules live in code, not the prompt:
    * a stance's participant is ALWAYS the author of the cited span (the op has no
      participant field), so "Raj is fine with it" can never move Raj's cell;
    * a claim that asserts other people's stances is checked against the ledger and
      flagged as consensus_conflict when it doesn't match;
    * creating an option that already exists (same title/alias) is redirected.
- The matrix and "what changed" views are pure functions over the log.

v1 scope: questions, options, positions, decisions, constraints, criteria, claims
(incl. objections), blockers. Actions / agreements / separate issues are not reconciled yet.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llm import LLM, SONNET

PREFIX = {"question": "q", "option": "opt", "criterion": "crit", "constraint": "con",
          "claim": "clm", "blocker": "blk"}
STATUS = {
    "question": {"open", "decided", "conditional", "reopened"},
    "option": {"active", "dropped", "fallback", "reopened", "chosen"},
    "claim": {"unverified", "conflicting", "superseded", "confirmed"},
    "constraint": {"active", "superseded", "satisfied"},
    "blocker": {"open", "resolved"},
    "criterion": {"active", "superseded"},
}
DEFAULT_STATUS = {"question": "open", "option": "active", "claim": "unverified",
                  "constraint": "active", "blocker": "open", "criterion": "active"}
STANCES = ["proposed", "support", "conditional", "oppose", "withdrawn", "undecided", "acknowledged"]
STANCE_CLASS = {"proposed": "support", "support": "support", "conditional": "conditional",
                "oppose": "oppose", "withdrawn": "withdrawn", "undecided": "undecided",
                "acknowledged": "undecided"}
# spans carrying only these labels are context, not reconciled (v1)
SKIP_ONLY = {"irrelevant", "action", "agreement", "separate_issue"}
BATCH = 6
CONTEXT_SPANS = 8


# ── State ──────────────────────────────────────────────────────────────────

@dataclass
class Element:
    id: str
    kind: str
    title: str
    author: str
    created_span: str
    status: str
    question_id: str | None = None
    aliases: list[str] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    merged_into: str | None = None
    superseded_by: str | None = None
    spans: list[str] = field(default_factory=list)


@dataclass
class Event:
    seq: int
    ts: str
    span_id: str
    kind: str                  # create|stance|status|supersede|decide|alias|merge|attach|flag
    element_id: str
    participant: str | None
    payload: dict
    because: list[str]


STANCE_LABELS = {"position", "option", "objection", "decision"}
SHORT_REPLY = 4                      # "+1", "A", "None, go.", "That works for BI."
_WORDS = re.compile(r"[a-z0-9]+")
_GENERIC = {"the", "a", "an", "of", "on", "at", "to", "for", "and", "or", "in", "with", "via",
            "trail", "loop", "route", "option", "plan", "hike", "end", "state", "version",
            "weekend", "day", "date", "window", "cutover", "platform", "vendor", "only",
            "as", "interim", "full", "new", "old"}


def _names(e: "Element") -> set[str]:
    toks = set()
    for n in [e.title, *e.aliases]:
        w = _WORDS.findall(n.lower())
        if len(w) == 1 and len(w[0]) == 1:      # poll letter alias: "A"
            toks.add(w[0])
        toks |= {t for t in w if t not in _GENERIC and (len(t) > 1 or t.isdigit())}
    return toks


def _prefix_hit(name_tok: str, hay: set[str]) -> bool:
    """Abbreviations only: sat~saturday, sun~sunday, dec~december. The shorter token must
    be 3-4 chars and not a generic word, so "trail" never matches "trailhead"."""
    for h in hay:
        if h in _GENERIC or name_tok in _GENERIC:
            continue
        short, long_ = sorted((h, name_tok), key=len)
        if 3 <= len(short) <= 4 and len(long_) > len(short) and long_.startswith(short):
            return True
    return False


def _core(title: str) -> set[str]:
    """Distinctive tokens of a name, minus numbers/units that vary between variants."""
    return {t for t in _WORDS.findall(title.lower())
            if t not in _GENERIC and t not in _VARIANT_NOISE and not t.isdigit() and len(t) > 1}


_VARIANT_NOISE = {"mile", "miles", "mi", "km", "hour", "hours", "lot", "summit", "version",
                  "short", "long", "cheap", "cheaper", "tier", "basic", "pro", "enterprise",
                  "start", "starting", "from", "climb", "with", "without", "nov", "dec", "jan",
                  "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct"}


def _mentions(e: "Element", hay: set[str]) -> bool:
    names = _names(e)
    return bool(names & hay) or any(_prefix_hit(n, hay) for n in names)


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


class Registry:
    def __init__(self, spans: dict[str, dict]):
        self.spans = spans                     # span_id -> {author, ts, msg, text}
        self.el: dict[str, Element] = {}
        self.events: list[Event] = []
        self.warnings: list[str] = []
        self.followups = 0
        self._n = Counter()

    # lookups
    def canon(self, eid: str | None) -> str | None:
        seen = set()
        while eid in self.el and self.el[eid].merged_into and eid not in seen:
            seen.add(eid)
            eid = self.el[eid].merged_into
        return eid

    def get(self, eid: str | None) -> Element | None:
        return self.el.get(self.canon(eid)) if eid else None

    def live(self, kind: str | None = None) -> list[Element]:
        return [e for e in self.el.values() if not e.merged_into and (kind is None or e.kind == kind)]

    def stance_now(self, opt_id: str, who: str) -> str | None:
        opt_id = self.canon(opt_id)
        for ev in reversed(self.events):
            if (ev.kind == "stance" and self.canon(ev.element_id) == opt_id
                    and ev.participant == who and not ev.payload.get("inferred")):
                return ev.payload["stance"]
        return None

    def stance_evidence(self, span_id: str, opt: "Element") -> str | None:
        """Why a stance would be INFERRED rather than stated, or None if the span supports it.
        A stance must come from a span that (a) the classifier read as stance-bearing and
        (b) names this option -- by title, alias, poll letter or the span's context note.
        A span that names a DIFFERENT option never evidences this one ("sunday yes" is not a
        stance on Saturday). Short replies naming nothing ("+1", "None, go", "same") are
        exempt from (b): they answer a poll or objection check."""
        sp = self.spans.get(span_id, {})
        labels = {l["type"] for l in sp.get("labels", [])}
        if not labels & STANCE_LABELS and not sp.get("recovered"):
            return f"span is {sorted(labels) or 'unlabelled'}, not a stance"
        text = set(_WORDS.findall(sp.get("text", "").lower()))
        note = set(_WORDS.findall((sp.get("context_note") or "").lower()))
        if _mentions(opt, text | note):
            return None
        others = [o for o in self.live("option") if o.id != opt.id and _mentions(o, text)]
        if others:
            return f"span names '{others[0].title}', not '{opt.title}'"
        if len(_WORDS.findall(sp.get("text", "").lower())) <= SHORT_REPLY:
            return None
        return f"span never names '{opt.title}'"

    def _event(self, kind, eid, span_id, payload=None, participant=None, because=()):
        sp = self.spans.get(span_id, {})
        ev = Event(seq=len(self.events) + 1, ts=sp.get("ts", ""), span_id=span_id, kind=kind,
                   element_id=eid, participant=participant, payload=payload or {},
                   because=[b for b in (self.canon(x) for x in because) if b in self.el])
        self.events.append(ev)
        e = self.el.get(eid)
        if e and span_id and span_id not in e.spans:
            e.spans.append(span_id)
        return ev

    def _find_option_by_name(self, title: str, aliases: list[str],
                             question: str | None = None) -> tuple[Element | None, str]:
        """Exact duplicate (same normalised name/alias), or a VARIANT: one name's distinctive
        tokens contain the other's ("Mt. Diablo 5-mile summit lot route" vs "Mt. Diablo
        summit trail" share {diablo, summit, mt}; "Relay enterprise tier" vs "Relay").
        A variant is only redirected when exactly one existing option qualifies and the new
        name doesn't also name a second option ("keep both Databricks and Snowflake")."""
        names = {_norm(title), *(_norm(a) for a in aliases)} - {""}
        live = [e for e in self.live("option")
                if not (question and e.question_id and e.question_id != question)]
        for e in live:
            if names & ({_norm(e.title)} | {_norm(a) for a in e.aliases}):
                return e, "exact"
        new = _core(title)
        if not new:
            return None, ""
        hits = [e for e in live if (c := _core(e.title)) and (c <= new or new <= c)]
        named = [e for e in live if _core(e.title) & new]
        if len(hits) == 1 and len(named) == 1:
            return hits[0], "variant"
        return None, ""

    # ── apply one batch of ops ────────────────────────────────────────────
    def apply(self, ops: list[dict], allowed_spans: set[str], participants: list[str]) -> None:
        refs: dict[str, str] = {}

        def resolve(x):
            if x is None:
                return None
            x = refs.get(x, x)
            return self.canon(x) if x in self.el else None

        def warn(msg, op):
            self.warnings.append(f"{msg}: {json.dumps(op)[:200]}")

        for op in ops:
            if not isinstance(op, dict):
                continue
            kind = op.get("op")
            span = op.get("span")
            if kind != "merge" and span not in allowed_spans:
                warn("span not in batch", op)
                continue
            author = self.spans.get(span, {}).get("author")
            because = [refs.get(b, b) for b in op.get("because", []) or []]

            if kind == "create":
                ek = op.get("kind")
                if ek not in PREFIX:
                    warn("bad kind", op); continue
                title = (op.get("title") or "").strip()
                if not title:
                    warn("empty title", op); continue
                aliases = [a for a in op.get("aliases", []) or [] if a]
                if ek == "option":
                    dup, how = self._find_option_by_name(title, aliases, resolve(op.get("question")))
                    if dup:
                        refs[op.get("ref", "")] = dup.id
                        if how == "variant" and title not in dup.aliases and title != dup.title:
                            dup.aliases.append(title)
                            self._event("alias", dup.id, span, {"alias": title, "variant": True}, author)
                        else:
                            self._event("attach", dup.id, span, because=because)
                        warn(f"{how} duplicate option redirected to {dup.id}", op)
                        continue
                self._n[ek] += 1
                eid = f"{PREFIX[ek]}{self._n[ek]}"
                q = resolve(op.get("question"))
                if ek == "option" and not q:
                    qs = self.live("question")
                    q = qs[-1].id if len(qs) == 1 else None
                attrs = {k: op[k] for k in ("claim_kind", "constraint_kind", "blocker_kind",
                                             "conditions", "about") if op.get(k)}
                if "about" in attrs:
                    attrs["about"] = [a for a in (resolve(x) for x in attrs["about"]) if a]
                e = Element(id=eid, kind=ek, title=title, author=author or "", created_span=span,
                            status=DEFAULT_STATUS[ek], question_id=q, aliases=aliases, attrs=attrs)
                self.el[eid] = e
                if op.get("ref"):
                    refs[op["ref"]] = eid
                self._event("create", eid, span, {"title": title, "kind": ek}, author, because)
                if ek == "claim" and op.get("asserts_stances"):
                    self._check_consensus(e, op["asserts_stances"], span, participants, resolve)

            elif kind == "stance":
                opt = self.get(resolve(op.get("option")))
                st = op.get("stance")
                if not opt or opt.kind != "option":
                    warn("stance on non-option", op); continue
                if st not in STANCES:
                    warn("bad stance", op); continue
                if not author:
                    warn("unknown author", op); continue
                prev = self.stance_now(opt.id, author)
                payload = {"stance": st, "prev": prev}
                if op.get("conditions"):
                    payload["conditions"] = op["conditions"]
                why = self.stance_evidence(span, opt)
                if why:                        # kept for review, never counted in the matrix
                    payload["inferred"] = True
                    payload["inferred_reason"] = why
                    warn(f"inferred stance ({why})", op)
                self._event("stance", opt.id, span, payload, author, because)

            elif kind == "status":
                e = self.get(resolve(op.get("element")))
                st = op.get("status")
                if not e:
                    warn("unknown element", op); continue
                if st not in STATUS.get(e.kind, set()):
                    warn("bad status for kind", op); continue
                if e.kind == "option" and st == "chosen":
                    warn("use decide to choose", op); continue
                if st == e.status:
                    self._event("attach", e.id, span, because=because); continue
                self._event("status", e.id, span, {"from": e.status, "to": st}, author, because)
                e.status = st

            elif kind == "supersede":
                old, new = self.get(resolve(op.get("old"))), self.get(resolve(op.get("new")))
                if not old or not new or old.id == new.id:
                    warn("bad supersede", op); continue
                self._event("supersede", old.id, span,
                            {"from": old.status, "to": "superseded", "by": new.id}, author,
                            because or [new.id])
                old.status, old.superseded_by = "superseded", new.id

            elif kind == "decide":
                q = self.get(resolve(op.get("question")))
                if not q or q.kind != "question":
                    warn("decide without question", op); continue
                chosen = [o for o in (self.get(resolve(x)) for x in op.get("chosen", []) or [])
                          if o and o.kind == "option"]
                ruled = [o for o in (self.get(resolve(x)) for x in op.get("ruled_out", []) or [])
                         if o and o.kind == "option"]
                if not chosen and not ruled:
                    warn("decide with no options", op); continue
                new_q = "conditional" if op.get("conditional") else "decided"
                self._event("decide", q.id, span,
                            {"chosen": [o.id for o in chosen], "ruled_out": [o.id for o in ruled],
                             "conditional": bool(op.get("conditional")),
                             "conditions": op.get("conditions", ""), "from": q.status, "to": new_q},
                            author, because)
                if chosen:
                    q.status = new_q
                for o in chosen:
                    if o.status != "chosen":
                        self._event("status", o.id, span, {"from": o.status, "to": "chosen"}, author, [q.id])
                        o.status = "chosen"
                for o in ruled:
                    if o.status != "dropped":
                        self._event("status", o.id, span, {"from": o.status, "to": "dropped"}, author, [q.id])
                        o.status = "dropped"

            elif kind == "alias":
                e = self.get(resolve(op.get("element")))
                a = (op.get("alias") or "").strip()
                if not e or not a:
                    warn("bad alias", op); continue
                if a not in e.aliases:
                    e.aliases.append(a)
                self._event("alias", e.id, span, {"alias": a}, author)

            elif kind == "merge":
                keep, drop = self.get(resolve(op.get("keep"))), self.get(resolve(op.get("drop")))
                if not keep or not drop or keep.id == drop.id or keep.kind != drop.kind:
                    warn("bad merge", op); continue
                drop.merged_into = keep.id
                keep.aliases += [x for x in [drop.title, *drop.aliases] if x not in keep.aliases]
                self._event("merge", keep.id, span or "", {"dropped": drop.id}, None)

            elif kind == "attach":
                e = self.get(resolve(op.get("element")))
                if not e:
                    warn("unknown element", op); continue
                self._event("attach", e.id, span, because=because)

            elif kind == "skip":
                continue
            else:
                warn("unknown op", op)

    def _check_consensus(self, claim: Element, asserted: list, span: str,
                         participants: list[str], resolve) -> None:
        """A claim like 'all seven of us are on board' asserts stances for others.
        Compare with the ledger; record a flag on mismatch."""
        mismatches = []
        for a in asserted:
            who, opt, st = a.get("participant"), resolve(a.get("option")), a.get("stance")
            if who in ("everyone", "all", "*"):
                targets = participants
            else:
                targets = [who] if who in participants else []
            for p in targets:
                if p == claim.author or not opt:
                    continue
                actual = self.stance_now(opt, p)
                if STANCE_CLASS.get(actual or "", "blank") != STANCE_CLASS.get(st or "", st):
                    mismatches.append({"participant": p, "option": opt,
                                       "asserted": st, "actual": actual or "blank"})
        if mismatches:
            claim.status = "conflicting"
            self._event("flag", claim.id, span,
                        {"type": "consensus_conflict", "mismatches": mismatches}, claim.author)


# ── Prompt ─────────────────────────────────────────────────────────────────

SYSTEM = """You maintain the decision ledger for a group chat. You receive the current REGISTRY
(every element tracked so far) and a few NEW SPANS (already labelled by a classifier).
Return the operations that update the registry for exactly those new spans.

Element kinds
- question   a decision the group is making ("which platform", "which weekend")
- option     an alternative for a question ("Databricks", "Nov 14-15", "Q2")
- criterion  a factor for judging options
- constraint a hard or soft limit that filters options
- claim      an assertion by a participant (facts, estimates, reasons, objections,
             claims about what others think). Use claim_kind: factual|predictive|causal|
             priority|objection|consensus|other. For objections set about=[option ids].
- blocker    an open question / missing info / pending dependency stopping a decision.
             blocker_kind: open_question|missing_info|unresolved_dependency.

Operations (each must cite "span" from NEW SPANS; "because" = element ids that caused it)
- {"op":"create","ref":"n1","kind":...,"title":"short canonical name","question":"q1",
   "aliases":[...],"claim_kind":...,"constraint_kind":"mandatory|negotiable|unknown",
   "blocker_kind":...,"about":[...],"asserts_stances":[...],"span":...,"because":[...]}
   "ref" lets later ops in this batch refer to the new element ("n1").
- {"op":"stance","option":"opt2","stance":"proposed|support|conditional|oppose|withdrawn|
   undecided|acknowledged","conditions":"...","span":...,"because":[...]}
   The participant is ALWAYS the span's author — you cannot set it.
- {"op":"status","element":"opt3","status":...,"span":...,"because":[...]}
   option: active|dropped|fallback|reopened   blocker: open|resolved
   claim: unverified|conflicting|confirmed    constraint: active|superseded|satisfied
   question: open|reopened
- {"op":"supersede","old":"clm2","new":"clm7","span":...}   a later claim/constraint replaces an earlier one
- {"op":"decide","question":"q1","chosen":["opt1"],"ruled_out":["opt3"],"conditional":false,
   "conditions":"...","span":...}
- {"op":"alias","element":"opt1","alias":"A","span":...}
- {"op":"merge","keep":"opt1","drop":"opt4","span":...}     two elements are the same thing
- {"op":"attach","element":"con2","span":...}               span restates an existing element
- {"op":"skip","span":...,"reason":"..."}

Rules
1. Every NEW SPAN gets at least one op (skip is fine for filler or self-corrected spans).
2. Before creating anything, check the REGISTRY. Same option under another name -> alias or
   attach, never a second option. A restated constraint -> attach.
3. Each option belongs to a question. If the group is clearly deciding a second thing
   (e.g. the date as well as the platform), create that question when its first option appears.
4. Stances: only for the span author's OWN stance on an option. "+1", "same", "None"
   (to "anyone against X?"), "A" (in a poll where A = Databricks) are stances. Resolve
   poll letters and nicknames with alias ops. A conditional stance carries "conditions".
   Withdrawing your own option = stance withdrawn + status dropped.
   A stance goes ONLY on the option the span talks about. "sunday yes" is not a stance on
   Saturday; offering to check a trail's distance is not support for it. Stances the span
   doesn't evidence are recorded as "inferred" and excluded from the matrix.
5. A statement about what OTHER people think ("Raj is fine with it", "all seven of us are on
   board") is NEVER a stance. Create a claim with claim_kind "consensus" and
   "asserts_stances":[{"participant":"Raj" or "everyone","option":"opt1","stance":"support"}].
6. A later statement that corrects an earlier claim or constraint (15% -> 10%, "it says
   Type I, not Type II") -> create the new element, then supersede the old one. A
   contradiction that isn't settled yet -> status conflicting on the earlier claim, because=[new].
7. When a span answers a blocker, set the blocker resolved with because=[the answering element].
8. decide only when someone closes a question for the group ("then it's decided", "direction:
   X"). Use conditional:true when it is explicitly contingent. Re-deciding later is fine.
9. Use "because" whenever a change has a visible cause in the chat: a stance changes after a
   constraint or claim, an option is dropped because of a cost claim, etc.
10. Classifier labels are hints. Trust the text. Titles are short noun phrases."""

OPS_SCHEMA = {
    "type": "object",
    "properties": {"ops": {"type": "array", "items": {"type": "object"}},
                   "notes": {"type": "string"}},
    "required": ["ops"],
}


def render_registry(reg: Registry) -> str:
    out = []
    for kind in ("question", "option", "constraint", "criterion", "claim", "blocker"):
        els = reg.live(kind)
        if not els:
            continue
        out.append(kind.upper() + "S")
        for e in els:
            aka = f" aka {e.aliases}" if e.aliases else ""
            q = f" ({e.question_id})" if e.question_id else ""
            line = f"  {e.id}{q} [{e.status}] {e.title}{aka} — by {e.author}"
            if kind == "option":
                who = {}
                for ev in reg.events:
                    if (ev.kind == "stance" and reg.canon(ev.element_id) == e.id
                            and not ev.payload.get("inferred")):
                        c = ev.payload.get("conditions")
                        who[ev.participant] = ev.payload["stance"] + (f"({c})" if c else "")
                if who:
                    line += " | " + ", ".join(f"{p}:{s}" for p, s in who.items())
            if e.attrs.get("claim_kind"):
                line += f" <{e.attrs['claim_kind']}>"
            out.append(line)
    return "\n".join(out) or "(empty)"


def render_span(s: dict) -> str:
    labs = ", ".join(f"{l['type']}" + (f" ({l['gist']})" if l.get("gist") else "")
                     for l in s["labels"])
    flag = " [SELF_CORRECTED]" if s.get("status") == "self_corrected" else ""
    return f"[{s['span_id']}] {s['author']} ({s['msg']}, {s['ts'][11:16]}): {s['text']}{flag}\n    labels: {labs}"


# ── Driver ─────────────────────────────────────────────────────────────────

def load_classified(path: str) -> tuple[list[dict], list[str]]:
    d = json.load(open(path))
    meta = {s["id"]: s for s in d["spans"]}
    rows = []
    for c in d["classified"]:
        m = meta[c["span_id"]]
        rows.append({"span_id": c["span_id"], "author": c["author"], "text": c["text"],
                     "status": c.get("status", "active"), "labels": c["labels"],
                     "context_note": c.get("context_note") or m.get("context_note"),
                     "recovered": m.get("recovered", False),
                     "msg": m["source_message_ids"][0], "ts": m["ts"]})
    participants = list(dict.fromkeys(r["author"] for r in rows))
    return rows, participants


def reconcile(rows: list[dict], participants: list[str], llm: LLM, *,
              batch: int = BATCH, log=print) -> Registry:
    reg = Registry({r["span_id"]: r for r in rows})
    work = [i for i, r in enumerate(rows)
            if r.get("status") != "self_corrected"
            and not {l["type"] for l in r["labels"]} <= SKIP_ONLY]
    for k in range(0, len(work), batch):
        idx = work[k:k + batch]
        lo = idx[0]
        ctx = rows[max(0, lo - CONTEXT_SPANS):lo]
        user = (
            f"PARTICIPANTS: {', '.join(participants)}\n\n"
            f"REGISTRY:\n{render_registry(reg)}\n\n"
            "RECENT CHAT (already processed):\n"
            + ("\n".join(f"{r['author']}: {r['text']}" for r in ctx) or "(start)") + "\n\n"
            "NEW SPANS:\n" + "\n".join(render_span(rows[i]) for i in idx)
        )
        out = llm.structured(system=SYSTEM, user=user, tool_name="ledger_ops",
                             schema=OPS_SCHEMA, max_tokens=4096)
        ops = out.get("ops", [])
        reg.apply(ops, {rows[i]["span_id"] for i in idx}, participants)

        # Coverage guard: a position-labelled span that produced no stance op and no
        # explicit skip gets one focused follow-up call. (The stage-2 analogue of
        # normalize._recover_uncovered: nothing a participant said silently vanishes.)
        missed = _missed_stances(reg, ops, [rows[i] for i in idx])
        if missed:
            follow = (
                f"PARTICIPANTS: {', '.join(participants)}\n\n"
                f"REGISTRY:\n{render_registry(reg)}\n\n"
                "MISSED STANCES: these spans are labelled as positions but you emitted no "
                "stance op and no skip for them. For EACH one emit a stance op on the option it "
                "is about (with conditions if conditional), or a skip op with a reason.\n\n"
                "NEW SPANS:\n" + "\n".join(render_span(r) for r in missed)
            )
            out2 = llm.structured(system=SYSTEM, user=follow, tool_name="ledger_ops",
                                  schema=OPS_SCHEMA, max_tokens=2048)
            reg.apply(out2.get("ops", []), {r["span_id"] for r in missed}, participants)
            reg.followups += 1
        if log:
            log(f"  batch {k // batch + 1}/{(len(work) + batch - 1) // batch}: "
                f"{len(ops)} ops, {len(reg.live())} elements"
                + (f", follow-up for {len(missed)} missed stance(s)" if missed else ""))
    return reg


def _missed_stances(reg: "Registry", ops: list[dict], batch_rows: list[dict]) -> list[dict]:
    stanced = {ev.span_id for ev in reg.events if ev.kind == "stance"}
    skipped = {op.get("span") for op in ops if isinstance(op, dict) and op.get("op") == "skip"}
    return [r for r in batch_rows
            if "position" in {l["type"] for l in r["labels"]}
            and r.get("status") != "self_corrected"
            and r["span_id"] not in stanced and r["span_id"] not in skipped]


# ── Projection (pure functions over the registry/log) ─────────────────────

def matrix(reg: Registry, participants: list[str]) -> str:
    lines = []
    qs = reg.live("question") or [None]
    for q in qs:
        opts = [o for o in reg.live("option") if (q is None or o.question_id == q.id)]
        if not opts:
            continue
        head = f"{q.id} [{q.status}] {q.title}" if q else "(no question)"
        lines.append(f"\n{head}")
        lines.append("| option | status | " + " | ".join(participants) + " |")
        lines.append("|---|---|" + "---|" * len(participants))
        for o in opts:
            cells = [(reg.stance_now(o.id, p) or "") for p in participants]
            lines.append(f"| {o.title} | {o.status} | " + " | ".join(cells) + " |")
    orphans = [o for o in reg.live("option") if o.question_id and not reg.get(o.question_id)]
    for o in orphans:
        lines.append(f"(orphan option {o.id} {o.title})")
    return "\n".join(lines)


def changes(reg: Registry, since_seq: int = 0) -> list[str]:
    """Human-readable 'what shifted and why' since a point in the log."""
    out = []
    for ev in reg.events:
        if ev.seq <= since_seq or ev.kind in ("attach", "create", "alias"):
            continue
        e = reg.get(ev.element_id)
        name = e.title if e else ev.element_id
        why = ""
        if ev.because:
            why = " — because: " + "; ".join(
                f"{reg.get(b).title}" for b in ev.because if reg.get(b))
        t = ev.ts[5:16].replace("T", " ")
        if ev.kind == "stance" and ev.payload.get("inferred"):
            out.append(f"{t} ? {ev.participant}: {name} {ev.payload['stance']} "
                       f"(inferred, not counted: {ev.payload.get('inferred_reason')})")
            continue
        if ev.kind == "stance":
            prev, now = ev.payload.get("prev"), ev.payload["stance"]
            if prev == now:                       # restatement, not a shift
                continue
            cond = f" [{ev.payload['conditions']}]" if ev.payload.get("conditions") else ""
            out.append(f"{t} {ev.participant}: {name} {prev or '—'} → {now}{cond}{why}")
        elif ev.kind in ("status", "supersede"):
            out.append(f"{t} {name}: {ev.payload.get('from')} → {ev.payload.get('to')}{why}")
        elif ev.kind == "decide":
            ch = ", ".join(reg.get(x).title for x in ev.payload["chosen"] if reg.get(x))
            out.append(f"{t} DECIDED {name}: {ch}"
                       + (f" (conditional: {ev.payload['conditions']})" if ev.payload["conditional"] else ""))
        elif ev.kind == "flag":
            mm = ", ".join(f"{m['participant']} is {m['actual']}" for m in ev.payload["mismatches"])
            out.append(f"{t} ⚠ {ev.participant}'s claim '{name}' doesn't match: {mm}")
    return out


def dump(reg: Registry, participants: list[str], source: str, model: str) -> dict:
    return {
        "source": source, "model": model, "participants": participants,
        "span_msgs": {sid: s["msg"] for sid, s in reg.spans.items()},
        "elements": [asdict(e) for e in reg.el.values()],
        "events": [asdict(e) for e in reg.events],
        "warnings": reg.warnings,
        "followups": reg.followups,
    }


def main():
    from llm import AnthropicLLM
    ap = argparse.ArgumentParser()
    ap.add_argument("classified", help="*.output.json from run.py")
    ap.add_argument("--model", default=SONNET)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows, participants = load_classified(a.classified)
    reg = reconcile(rows, participants, AnthropicLLM(model=a.model))
    out = a.out or re.sub(r"\.output\.json$", "", a.classified) + ".reconcile.json"
    Path(out).write_text(json.dumps(dump(reg, participants, a.classified, a.model), indent=1))
    print(matrix(reg, participants))
    print("\nWHAT SHIFTED")
    print("\n".join(changes(reg)))
    print(f"\nwrote {out}  ({len(reg.live())} elements, {len(reg.events)} events, "
          f"{len(reg.warnings)} warnings)")


if __name__ == "__main__":
    main()
