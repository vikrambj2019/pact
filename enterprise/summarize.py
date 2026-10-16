"""Executive summary from a reconcile output (registry + event log + classified spans).

    from pact_engine.summarize import build_input, SUMMARY_SYSTEM, status_line, make_summary
    user = build_input(result_dict)                      # compact, filtered view; code decides every count/status/stance
    text = make_summary(result_dict, call_llm)           # call_llm(system: str, user: str) -> str   (Sonnet is plenty)

CLI check (no model call):  python summarize.py outputs/long-01.reconcile.json [--no-private]
    prints the model input.

Design
- Code decides everything that is a count, a status, a stance, an owner or a date. The model only writes sentences.
- Inferred stances never reach the model. "Not listed" therefore means "no explicit position recorded", never "no view".
- Privacy: messages the engine marked restricted, and messages sent to a narrower audience, are tagged in the input.
  build_input(d, include_private=False) leaves private items out entirely.
- Next steps, unanswered questions and the decision's after-effects are built from the CLASSIFIED spans' own words,
  not from reconcile's registry (reconcile skips actions/agreements in v1) and not from the gists (a gist can be reversed).
"""
from __future__ import annotations

import json
import re
import sys

SUMMARY_SYSTEM = """You write the executive summary of a group discussion for someone who was not in it.
You are given the discussion's decision state as structured data. You report that state. You do not decide anything,
recommend anything, or take a side.

INPUT (computed from the transcript; the only source of truth)
- CONVERSATION: the exchange itself, message by message, in the speakers' own words (private and irrelevant lines removed,
  long sources thinned to the lines that matter). This is what you read to understand what happened. The structured blocks
  below it are bookkeeping; when a stance or a claim conflicts with the words in CONVERSATION, trust CONVERSATION and leave
  the stance out.
- STATUS: computed by code. Copy it verbatim as the first line. Do not restate or contradict it.
- SCOPE: who could see what. Obey it.
- DECISION: who announced it, who replied afterwards and what they said, who has no reply recorded.
- QUESTIONS, OPTIONS: alternatives, with the EXPLICIT stances people took (name: stance, condition, and their own words).
  OPTIONS PROPOSED ONLY have a proposer and no recorded reaction.
- OTHER WORDS ABOUT THE OPTIONS: what people with no explicit stance on file said. Report their words; never turn them into
  "supports" or "opposes".
- POSITIONS DRAFT: each person's view in one or two lines, written from a full read of the exchange.
- BLOCKERS, CONSTRAINTS, CRITERIA, PRESENTED, CLAIMS, FLAGS.
- NEXT STEPS: commitments and requests, in the speakers' own words, in order.
- AWAITING A REPLY: questions put to a named person who has not spoken since, and open calls for a volunteer with the replies that followed.
- PRIVATE: items the sender asked to keep private. Only present if the reader may see them.

WRITE, in this order, using exactly these headings. Skip a heading whose section is absent from the input. Never write "None".
1. Status -- the STATUS line, verbatim, as the first line of your answer, with no label such as "Status:" in front of it.
2. The main point -- three to five plain sentences, written the way you would explain it to a colleague who missed it:
   who wants what and why; what was asked of whom, and whether the person asked actually gave it (a reply that proposes something
   else is not a yes); where two people disagree, what each says; what is still unresolved. Use only what
   CONVERSATION shows, name every person, and say "asked ... and has not confirmed" rather than guessing at agreement.
   Put the figure, date or deadline at stake in this section if CONVERSATION states one. If a DECISION block exists, replace this with "Decision" and write what was decided,
   "announced by <Name>", then who replied with agreement or an objection (their words, briefly), and "No reply recorded from <Names>".
   Never write "the group chose/selected/agreed". Silence is not agreement. If the DECISION block says the announcement went
   only to some people, say so in the same sentence and do not describe anyone else as told or as having replied.
3. Presented, no challenge recorded -- up to four bullets. Shared ground, never "agreed" or "aligned". Leave this section out
   when the source is an exchange of feedback or a back-and-forth, and whenever a DECISION block exists; use it only for
   explanations nobody answered.
4. Open questions -- up to five bullets, merge near-duplicates. Say who raised each. If a DECISION block exists, do not list
   the question it answers as open.
5. Where people stand -- one bullet per person who expressed a view, from POSITIONS DRAFT: "<Name>: <what they want, propose
   or object to, in their own terms>". OPTIONS stances only confirm a line; never add a stance that POSITIONS DRAFT lacks, and never
   put a dollar figure or a condition on a person's view unless their own words did. Someone who moved: "<Name> first said <X>,
   then <Y>". Options with no stance anywhere: one line "Proposed, no reaction recorded: ..." (at most six, then "and N more"). Say "no explicit position recorded", never "no position" or "no view".
   People under OTHER WORDS: one line quoting what they said, e.g. "Erik Lindqvist said 'I'm fine with the second then'."
6. What is blocking progress -- up to five bullets, merged, each with who raised it. "Deferred to <follow-up>" gets its own bullet.
7. Next steps -- up to six bullets: "<Name> will <do X> by <date>" only when the words are a commitment; "<Name> asked <Name> to <X>"
   when they are a request; a short acceptance ("Will do") belongs to the ask before it. Keep dates and times exactly as written.
8. Awaiting a reply -- only items that are NOT already a line under Next steps: "<Name> asked <Name> ...; no reply recorded."
   For open calls say what the replies show and no more. If every item is already in Next steps, leave this section out.
9. Ground rules people set -- up to five lines, with the speaker: limits, thresholds ("more than ten percent, come back to me"),
   time limits ("until March, not forever") and confidentiality requests. Skip anything that is only a date for a follow-up.
   If a decision carries a condition or limit, say it in the decision sentence too.
10. Things people asserted that others may want to check -- at most four, "<Name> said ...".
11. Flags -- what the speaker said and what the record shows, one sentence each.
Never write "(recipient unspecified)" or any note about a missing addressee. If the words do not name who was asked, write the line as "<Name> asked for ..." with no addressee. In an email chain, do not add a "Presented" section.
12. Marked private -- only if PRIVATE is present. One short bullet per item, each starting "Marked private by <Name>:".
    Nothing private appears in any other section.

RULES
- Attribute every position, claim, blocker and step to a name, spelled as in the input. Never write "Unknown"; use the input's wording.
- Stance words come only from the given stances. Do not upgrade a conditional to a support, or a question to a position.
  The speaker's own words are given with each stance: if they speak about capacity, availability or logistics rather than about
  the option itself, say that in plain words ("Tom said he has no capacity to help build") instead of "opposes".
- Every line under Next steps and Where people stand must be traceable to words in CONVERSATION or NEXT STEPS. Never name
  someone as owner, lead, convener or supporter unless their words (or the words directed at them and accepted) say so.
  A condition on a stance ("only if ...") must be one the speaker said, not one a different speaker raised.
- Do not recommend, rank, or predict. Do not use: consensus, alignment, aligned, clearly, obviously, key takeaway,
  going forward, leverage, synergy, robust.
- Keep the speakers' vocabulary for products and teams. A name that looks garbled in the input is copied as given.
- Do not mention the data structure, ids, spans, models or confidence.
- Scope notes: an item tagged with a narrower audience is written with "(seen only by <Names>)".
- Plain sentences. No bold except the headings. Say each fact once: if the main point already says it, a later section repeats
  it only to add a name, a condition, a number, a date or the speaker's own words. Cut repeated words and filler, never a
  distinct point: every stance, condition, figure, date, caveat and objection survives, even if it looks minor, because the
  reader decides what matters. Drop only pleasantries and talk about nothing in the discussion. If the input has more items
  than a section allows, keep the ones with the most named people attached and end the section with "and N more".
- If STATUS says no decision was reached, never imply one."""

_JUNK = re.compile(r"(sent from my|get outlook|confidential|disclaimer|mailto:|unsubscribe|^from:|^sent:|^to:|^cc:|"
                   r"\b\d{3}[-.)\s]+\d{3}[-.\s]+\d{4}\b|^thanks[,!. ]*$|^best[,!. ]*$|^regards[,!. ]*$)", re.I)
_TALLY = re.compile(r"\bvotes?\b|\btall(y|ies)\b|\bmajority\b|\bout of \d+\b|\b\d+\s*(to|-|:)\s*\d+\b|\b\w+ \d+\s*[,;]\s*\w+ \d+\b", re.I)
_PREF = re.compile(r"\b(i('d| would| will| 'll)?\s+(pick|choose|go with|vote|prefer|take|opt)|my (vote|pick|preference)|i'?m (for|leaning)|i prefer)\b", re.I)
_LOGISTICS = re.compile(r"\b(attendance|attend|joining the (call|meeting)|dial[- ]?in|audio|screen[- ]?shar|unknown item|unidentified)\b", re.I)
_OPEN_CALL = re.compile(r"(^\W*(anyone|anybody|someone)\b.*\?|\b(can|could|will|does|is) (anyone|anybody)\b|^\W*who (can|will|wants|is able)\b|"
                       r"\bsomeone (else )?(needs to|should|must|can|could)\b|^\W*anyone\W*$)", re.I)


# ----------------------------------------------------------------------------- names


def disp(n):
    """'Last, First (Nick)' -> 'First Last'. Unknown -> 'an unidentified sender'."""
    if not n or n.strip().lower() == "unknown":
        return "an unidentified sender"
    if "," in n:
        last, first = (x.strip() for x in n.split(",", 1))
        nick = re.search(r"\((.*?)\)", first)
        first = re.sub(r"\s*\(.*?\)", "", first).strip()
        return f"{first} {last}".strip() + (f" ({nick.group(1)})" if nick else "")   # keep the nickname: others write "Walle"
    return n.strip()


def _first(n):
    return None if (not n or n.strip().lower() == "unknown") else disp(n).split()[0]


def _n(k, one, many=None):
    return f"{k} {one if k == 1 else (many or one + 's')}"


def _clip(t, k=120):
    t = re.sub(r"\s+", " ", t or "").strip().replace('"', "'")
    return t if len(t) <= k else t[: k - 1].rstrip() + "..."


def _toks(t):
    return {w for w in re.findall(r"[a-z0-9']+", (t or "").lower()) if len(w) > 3}


# ----------------------------------------------------------------------------- context


class Ctx:
    def __init__(self, d):
        ent = d.get("_enterprise") or {}
        self.spans = ent.get("spans") or []
        self.order = {s["id"]: i for i, s in enumerate(self.spans)}
        self.text = {s["id"]: s.get("text", "") for s in self.spans}
        self.author = {s["id"]: s.get("author") for s in self.spans}
        self.msgs = {m["id"]: m for m in ent.get("msg_meta") or []}
        self.span_msg = dict(d.get("span_msgs") or {})
        for s in self.spans:
            self.span_msg.setdefault(s["id"], (s.get("source_message_ids") or [None])[0])
        self.cls = {c["span_id"]: c for c in ent.get("classified") or []}
        self.speakers = [s for s in ent.get("speakers") or d.get("participants") or [] if s.lower() != "unknown"]
        aud = set()
        for m in self.msgs.values():
            aud.update(m.get("audience") or [])
        self.full_audience = aud
        restricted = [m for m in self.msgs.values() if m.get("restricted")]
        self.n_restricted = len(restricted)
        self.all_restricted = bool(self.msgs) and len(restricted) >= 0.9 * len(self.msgs)
        self.restricted_reason = next((m.get("restricted_reason") for m in restricted), None)

    def msgs_of(self, e):
        ids = list(e.get("spans") or []) + ([e["created_span"]] if e.get("created_span") else [])
        return {self.span_msg.get(s) for s in ids if self.span_msg.get(s)}

    def _scope_msgs(self, ms):
        """('private', reason) | ('narrow', [names]) | None for a set of message ids."""
        metas = [self.msgs[m] for m in ms if m in self.msgs]
        if not metas or self.all_restricted:
            return None
        if all(m.get("restricted") for m in metas):
            return ("private", metas[0].get("restricted_reason"))
        aud = set()
        for m in metas:
            aud.update(m.get("audience") or [])
        if aud and self.full_audience and aud < self.full_audience:
            return ("narrow", sorted(disp(a) for a in aud))
        return None

    def scope(self, e):
        return self._scope_msgs(self.msgs_of(e))

    def span_scope(self, sid):
        m = self.span_msg.get(sid)
        return self._scope_msgs({m}) if m else None


def _live(d):
    return [e for e in d["elements"] if not e.get("merged_into")]


# ----------------------------------------------------------------------------- stances


def _stance_records(d, cx):
    """Explicit stances only, cleaned. Returns {(option_id, person): dict(stance, cond, span, text, moved_to)}."""
    els = {e["id"]: e for e in d["elements"]}
    rec, hist = {}, {}
    for ev in d["events"]:
        if ev["kind"] != "stance" or ev["payload"].get("inferred"):
            continue
        opt = els.get(ev["element_id"])
        if not opt or opt["kind"] != "option":
            continue
        who, st, cond, sid = ev["participant"], ev["payload"]["stance"], ev["payload"].get("conditions"), ev["span_id"]
        text = cx.text.get(sid, "")
        if st == "oppose" and len(text.split()) <= 4 and not (_toks(text) & _toks(opt["title"])):
            continue                                   # "Not me." is an answer to a different question, not an objection
        if st == "proposed":
            if who == opt["author"]:
                continue                               # the proposer is already shown as "proposed by"
            st = "support" if _PREF.search(text) else "suggested"
        rec[(opt["id"], who)] = dict(stance=st, cond=cond, span=sid, text=text, moved_to=None)
        hist.setdefault((opt.get("question_id"), who), []).append((opt["id"], st, sid))
    # a person who supports a second option of the same question has moved
    for (qid, who), h in hist.items():
        sup = [(o, s) for o, st, s in h if st in ("support", "conditional")]
        if len({o for o, _ in sup}) < 2:
            continue
        last_o, last_s = sup[-1]
        later = els.get(last_o, {})
        words = rec[(last_o, who)]["text"]
        said_switch = bool(_toks(words) & _toks(later.get("title"))) or re.search(r"\b(switch|instead|rather|chang|mov|actually)", words, re.I)
        for o, s in sup[:-1]:
            if o != last_o and s != last_s and said_switch and els.get(o, {}).get("author") != who \
                    and (o, who) in rec and rec[(o, who)]["stance"] in ("support", "conditional"):
                rec[(o, who)]["moved_to"] = last_o
    return rec


def _count_people(rec):
    return len({p for (_, p), r in rec.items() if r["stance"] in ("support", "oppose", "conditional", "undecided")})


# ----------------------------------------------------------------------------- decision


def _decision(d, cx, els, rec):
    evs = [ev for ev in d["events"] if ev["kind"] == "decide"]
    if not evs:
        return None
    ev = evs[-1]
    chosen = [e for e in els if e["kind"] == "option" and e["status"] == "chosen"]
    qs = [e for e in els if e["kind"] == "question" and e["status"] == "decided"]
    titles = [e["title"] for e in chosen] or [e["title"] for e in qs]
    if not titles:
        return None
    idx = cx.order.get(ev["span_id"], -1)
    who = ev["participant"]
    aud = set((cx.msgs.get(cx.span_msg.get(ev["span_id"]), {}) or {}).get("audience") or [])
    narrow = bool(aud and cx.full_audience and aud < cx.full_audience)      # announced to fewer people than the discussion had
    after, replied = [], set()
    for s in cx.spans[idx + 1:]:
        a = s["author"]
        sc = cx.span_scope(s["id"])
        if a == who or (sc and sc[0] == "private"):
            continue
        replied.add(a)
        labs = [l["type"] for l in (cx.cls.get(s["id"], {}).get("labels") or [])]
        if set(labs) & {"agreement", "objection", "position", "blocker"}:
            after.append((a, s["text"]))
    silent = [] if narrow else [p for p in cx.speakers if p != who and p not in replied]
    before = sorted({p for (o, p), r in rec.items() if o in {e["id"] for e in chosen} and r["stance"] == "support" and p != who})
    return dict(titles=titles, who=who, span=ev["span_id"], text=cx.text.get(ev["span_id"], ""), after=after, silent=silent,
                supported_before=before, narrow=narrow,
                sent_to=sorted(disp(a) for a in aud if a != who) if narrow else [],
                not_sent_to=sorted(disp(a) for a in cx.full_audience - aud) if narrow else [])


# ----------------------------------------------------------------------------- status


def _announced(dec, limit=230):
    """The announcement in its own words, minus leading filler ("Back to the vendor. Here is where I land."). '' if unusable."""
    t = re.sub(r"\s+", " ", (dec.get("text") or "")).strip()
    parts = re.split(r"(?<=[.!?])\s+", t)
    while len(parts) > 1 and len(parts[0].split()) <= 6:
        parts.pop(0)
    while len(parts) > 1 and parts[-1].endswith("?"):
        parts.pop()                        # "Any objection?" is a question to the room, not part of the decision
    t = " ".join(parts).strip().rstrip(".")
    t = t.replace('"', "'")
    return t if 3 <= len(t.split()) and len(t) <= limit else ""


def status_line(d, cx=None, rec=None, dec=None) -> str:
    cx = cx or Ctx(d)
    els = _live(d)
    if rec is None:
        rec = _stance_records(d, cx)
        dec = _decision(d, cx, els, rec)
    n = lambda k, st: sum(1 for e in els if e["kind"] == k and e["status"] in st)
    people = _count_people(rec)
    tail = [_n(n("question", {"open", "reopened", "conditional"}), "open question"),
            _n(n("blocker", {"open"}), "open blocker"),
            (_n(people, "person", "people") + " took an explicit position on an option") if people else "no one took an explicit position on an option"]
    if dec:
        to = f" in a message sent only to {', '.join(dec['sent_to'])}" if dec["narrow"] else ""
        head = f"Decision announced by {disp(dec['who'])}{to}: {_announced(dec) or '; '.join(dec['titles'])}."
    else:
        head = "No decision was reached."
    tail = [t for t in tail if not t.startswith("0 ")]
    return head + " " + ", ".join(tail).replace(", no one", ", and no one") + "."


# ----------------------------------------------------------------------------- claims


def _is_action_span(cx, sid):
    return any(l["type"] == "action" for l in (cx.cls.get(sid, {}).get("labels") or []))


def _claims_worth_showing(d, els, skip=frozenset(), cap=12, cx=None):
    cited = set()
    for ev in d["events"]:
        for b in ev.get("because") or []:
            cited.add(b)
        if ev["kind"] == "flag":
            cited.add(ev["element_id"])
    for e in els:
        cited.update((e.get("attrs") or {}).get("about", []))
    claims = [e for e in els if e["kind"] == "claim" and e["id"] not in skip and not _TALLY.search(e["title"])
              and not _JUNK.search(e["title"]) and not (cx and _is_action_span(cx, e.get("created_span")))]
    keep = [c for c in claims if c["status"] != "unverified" or c["id"] in cited]
    rest = sorted((c for c in claims if c not in keep), key=lambda c: -len(c["spans"]))
    out, per = [], {}
    for c in keep + rest:
        if per.get(c["author"], 0) >= 3:
            continue
        per[c["author"]] = per.get(c["author"], 0) + 1
        out.append(c)
    return out[:cap]


def _presented(d, els, per_msg=2, cap=8):
    """Claims from presentation turns (a message that produced 4+ claims) that nothing disputes, plus anything a
    second speaker confirmed. Returns (presented, ids of all 'lecture' claims)."""
    sm = d["span_msgs"]
    contested = {e["id"] for e in els if e["kind"] == "claim" and e["status"] in ("conflicting", "superseded")}
    for e in els:
        if e["kind"] in ("blocker", "question"):
            contested.update((e.get("attrs") or {}).get("about", []))
    by_msg = {}
    for e in els:
        if e["kind"] == "claim" and not _TALLY.search(e["title"]) and not _JUNK.search(e["title"]):
            by_msg.setdefault(sm.get(e["created_span"]), []).append(e)
    out = [c for cs in by_msg.values() for c in cs if c["status"] == "confirmed"]
    out += [c for m, cs in by_msg.items() if m and len(cs) >= 4 for c in cs[:per_msg] if c["id"] not in contested and c not in out]
    talk = {c["id"] for m, cs in by_msg.items() if m and len(cs) >= 4 for c in cs}
    return out[:cap], talk


# ----------------------------------------------------------------------------- next steps / awaiting reply


def _actions(cx, cap=20):
    out = []
    for i, s in enumerate(cx.spans):
        c = cx.cls.get(s["id"])
        if not c or not any(l["type"] == "action" and l.get("confidence", 1) >= 0.5 for l in c["labels"]):
            continue
        t = s["text"]
        if _JUNK.search(t) or len(t.split()) < 2:
            continue
        ctx = None
        if len(t.split()) <= 6 and i:
            p = cx.spans[i - 1]
            if not (cx.span_scope(p["id"]) or (None,))[0] == "private":
                ctx = f'after {disp(p["author"])}: "{_clip(p["text"], 100)}"'
        out.append(dict(span=s["id"], author=s["author"], text=_clip(t, 200), ctx=ctx, scope=cx.span_scope(s["id"])))
    return out


_DATEY = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? ?\d|\b(mon|tues?|wed(nes)?|thu(rs)?|fri|sat|sun)[a-z]*\b|"
                    r"\bby\b|\btomorrow\b|\bnext (week|month)\b|\b\d{1,2}(:\d\d)?\s?(am|pm)\b|\b(est|et|pst)\b|\b\d{1,2}(st|nd|rd|th)\b", re.I)


def _rank_actions(cx, acts, cap=12):
    """Keep the ones with a date, an addressee or an explicit commitment; fall back to order. Output stays in meeting order."""
    names = {_first(p) for p in cx.speakers} - {None}

    def score(a):
        t = a["text"]
        sc = 2 * bool(_DATEY.search(t)) + 2 * bool(re.match(r"^\W*@?(%s)\b" % "|".join(map(re.escape, names)), t, re.I) if names else 0)
        sc += bool(re.match(r"^(i('ll| will| can)|we('ll| will)|please)\b", t, re.I))
        return sc

    if len(acts) <= cap:
        return acts
    top = sorted(range(len(acts)), key=lambda i: (-score(acts[i]), i))[:cap]
    return [acts[i] for i in sorted(top)]


def _awaiting(cx, cap=6):
    firsts = {}
    for p in cx.speakers:
        firsts.setdefault(_first(p), []).append(p)
    firsts = {f: ps[0] for f, ps in firsts.items() if len(ps) == 1}       # ambiguous first names are skipped
    out, last_call_burst = [], None
    pub = lambda x: (cx.span_scope(x["id"]) or (None,))[0] != "private"
    for i, s in enumerate(cx.spans):
        t, a = s["text"].strip(), s["author"]
        sc = cx.span_scope(s["id"])
        if _JUNK.search(t) or (sc and sc[0] == "private"):
            continue
        for f, p in firsts.items():
            if p != a and re.match(rf"^\W*@?{re.escape(f)}\b", t, re.I) and "?" in t:
                if not any(x["author"] == p for x in cx.spans[i + 1:]):
                    out.append(dict(kind="ask", asker=a, to=p, text=_clip(t, 160), span=s["id"], scope=sc))
                break
        else:
            burst = s.get("burst_id")
            if _OPEN_CALL.search(t) and burst != last_call_burst:
                last_call_burst = burst
                nxt = [x for x in cx.spans[i + 1:i + 7] if pub(x)]
                same = [(x["author"], _clip(x["text"], 80)) for x in nxt if x.get("burst_id") == burst][:3]
                rep = [(x["author"], _clip(x["text"], 80)) for x in nxt if x.get("burst_id") != burst][:3]
                out.append(dict(kind="call", asker=a, text=_clip(t, 160), span=s["id"], replies=same + rep, scope=sc))
    return out[:cap]


def _reactions(cx, rec, cap_people=8, per=2):
    """People with no explicit stance on file, and the words of theirs labelled position/agreement/objection."""
    have = {p for (_, p) in rec}
    by = {}
    for s in cx.spans:
        a = s["author"]
        sc = cx.span_scope(s["id"])
        if a in have or a.lower() == "unknown" or (sc and sc[0] == "private") or _JUNK.search(s["text"]):
            continue
        labs = {l["type"] for l in (cx.cls.get(s["id"], {}).get("labels") or [])}
        if labs & {"position", "agreement", "objection"} and len(s["text"].split()) >= 2:
            by.setdefault(a, []).append(_clip(s["text"], 90))
    return [(a, v[-per:]) for a, v in list(by.items())[:cap_people]]


_PRIORITY = {"objection": 4, "decision": 4, "blocker": 3, "candidate_question": 3, "action": 3, "constraint": 3, "position": 2,
             "option": 2, "claim": 1, "agreement": 2, "process": 1, "criterion": 2, "separate_issue": 0, "irrelevant": -1}


def _conversation(cx, include_private=True, budget=1100):
    """The exchange in order, grouped by message, thinned to the lines that carry weight when it is too long."""
    keep = []
    first_msg = cx.span_msg.get(min(cx.spans, key=lambda x: int(re.sub(r"\D", "", str(cx.span_msg.get(x["id"]) or "0")) or 0))["id"]) if cx.spans else None
    for s in cx.spans:
        t = s["text"].strip()
        sc = cx.span_scope(s["id"])
        if len(t.split()) < 3 or _JUNK.search(t) or (sc and sc[0] == "private" and not include_private):
            continue
        labs = [l["type"] for l in (cx.cls.get(s["id"], {}).get("labels") or [])]
        pr = max([_PRIORITY.get(l, 1) for l in labs] or [1])
        if cx.span_msg.get(s["id"]) == first_msg:
            pr = max(pr, 2)                   # whoever opens the exchange (e.g. sends the memo) is always shown
        if pr < 0:
            continue
        keep.append((s, pr, len(_clip(t, 220).split())))
    total = sum(w for _, _, w in keep)
    floor = 0
    while total > budget and floor < 4:
        floor += 1
        keep = [k for k in keep if k[1] >= floor]
        total = sum(w for _, _, w in keep)
    out, last, line = [], None, None
    mnum = lambda sid: int(re.sub(r"\D", "", str(cx.span_msg.get(sid) or "0")) or 0)
    keep.sort(key=lambda k: (mnum(k[0]["id"]), cx.order.get(k[0]["id"], 0)))      # message order, not engine order
    for s, pr, _ in keep:
        m = cx.span_msg.get(s["id"])
        if m != last:
            meta = cx.msgs.get(m, {}) or {}
            subj = meta.get("subject") or ""
            aud = meta.get("audience") or []
            note = f" [sent to {len(aud)} people]" if aud and cx.full_audience and len(aud) < len(cx.full_audience) else ""
            out.append(f"{disp(s['author'])}{note}:")
            last = m
        out.append(f'  - "{_clip(s["text"], 220)}"')
    return out


# ----------------------------------------------------------------------------- dedupe


def _dedupe(items, key=lambda x: x, thr=0.7):
    kept = []
    for it in items:
        a = _toks(key(it))
        if any(a and _toks(key(k)) and len(a & _toks(key(k))) / len(a | _toks(key(k))) >= thr for k in kept):
            continue
        kept.append(it)
    return kept


# ----------------------------------------------------------------------------- input


def build_input(d, include_private: bool = True, budget: int = 6000, main_point: str = "", positions: str = "") -> str:
    cx = Ctx(d)
    els_all = _live(d)
    sc = {e["id"]: cx.scope(e) for e in els_all}
    hidden = lambda e: (sc.get(e["id"]) or (None,))[0] == "private" and not include_private
    els = [e for e in els_all if not hidden(e)]
    private = [e for e in els_all if (sc.get(e["id"]) or (None,))[0] == "private" and include_private]
    public_ids = {e["id"] for e in els if e not in private}
    rec = _stance_records(d, cx)
    dec = _decision(d, cx, els, rec)
    ids = {e["id"]: e for e in els}

    def tag(e):
        s = sc.get(e["id"])
        if not s:
            return ""
        return " [PRIVATE]" if s[0] == "private" else f" [seen only by {', '.join(s[1])}]"

    main = lambda kind, pred=lambda e: True: [e for e in els if e["kind"] == kind and e["id"] in public_ids and pred(e)]
    L = ["STATUS: " + status_line(d, cx, rec, dec), "", "SCOPE"]
    if cx.all_restricted:
        L.append(f"- The whole source was marked private ({cx.restricted_reason}). Right after the status line write: "
                 "'Marked private: this summary is for people on the original distribution only.'")
    elif cx.n_restricted:
        L.append(f"- {_n(cx.n_restricted, 'message')} marked private by the sender's request; "
                 + ("their items are in PRIVATE only." if include_private else "their items were withheld."))
    n_inf = sum(1 for ev in d["events"] if ev["kind"] == "stance" and ev["payload"].get("inferred"))
    if n_inf:
        L.append(f"- {_n(n_inf, 'further stance')} inferred from context, not listed. Absence below means no explicit position recorded.")
    if cx.full_audience:
        L.append(f"- Full audience: {', '.join(sorted(disp(a) for a in cx.full_audience))}.")

    conv = _conversation(cx, include_private, budget)
    if conv:
        L += ["", "CONVERSATION (in order; each line is the speaker's own words)"] + conv
    if dec:
        L += ["", "DECISION", f"- Decided: {'; '.join(dec['titles'])}", f'- Announced by {disp(dec["who"])}: "{_clip(dec["text"], 160)}"']
        if dec["narrow"]:
            L.append(f"- This announcement went only to {', '.join(dec['sent_to'])}. NOT sent to: {', '.join(dec['not_sent_to'])}. "
                     "State this plainly; do not say the others were told or have replied.")
        if dec["supported_before"]:
            L.append("- Explicit support recorded before the announcement: " + ", ".join(disp(p) for p in dec["supported_before"]))
        for a, t in dec["after"][:6]:
            L.append(f'- After the announcement, {disp(a)}: "{_clip(t, 110)}"')
        if dec["silent"]:
            L.append("- No reply recorded after the announcement from: " + ", ".join(disp(p) for p in dec["silent"]))

    qs = main("question")
    qs_sorted = sorted(qs, key=lambda e: -len(e.get("spans") or []))[:12]
    if qs:
        L += ["", "QUESTIONS"]
        for e in (x for x in qs if x in qs_sorted):
            L.append(f"- [{e['status']}] {e['title']} (raised by {disp(e['author'])}){tag(e)}")
        if len(qs) > len(qs_sorted):
            L.append(f"(+{len(qs) - len(qs_sorted)} more not shown)")

    opts = main("option")
    by_opt = {}
    for (oid, p), r in rec.items():
        by_opt.setdefault(oid, []).append((p, r))
    with_pos = [o for o in opts if by_opt.get(o["id"])]
    only = [o for o in opts if not by_opt.get(o["id"])]
    if with_pos:
        L += ["", "OPTIONS"]
        for o in with_pos:
            parts = []
            for p, r in by_opt[o["id"]]:
                w = r["stance"]
                if r["moved_to"] and r["moved_to"] in ids:
                    parts.append(f"{disp(p)}: supported this earlier, then moved to \"{ids[r['moved_to']]['title']}\"")
                    continue
                parts.append(f"{disp(p)}: {w}" + (f" if {r['cond']}" if r["cond"] else "") + f' -- said: "{_clip(r["text"], 100)}"')
            L.append(f"- [{o['status']}] {o['title']} (proposed by {disp(o['author'])}){tag(o)} | " + " | ".join(parts))
    if only:
        L += ["", "OPTIONS PROPOSED ONLY (no reaction recorded)"]
        for o in only[:8]:
            L.append(f"- {o['title']} (proposed by {disp(o['author'])}){tag(o)}")
        if len(only) > 8:
            L.append(f"(+{len(only) - 8} more not shown)")

    rx = _reactions(cx, rec)
    if rx:
        L += ["", "OTHER WORDS ABOUT THE OPTIONS (people with no explicit stance on file; their own words, not tied to one option)"]
        for a, v in rx:
            L.append(f"- {disp(a)}: " + " / ".join(f'"{x}"' for x in v))

    bl_all = main("blocker", lambda e: e["status"] == "open" and not _LOGISTICS.search(e["title"]))
    bl = sorted(bl_all, key=lambda e: -len(e.get("spans") or []))[:8]
    if bl:
        L += ["", "BLOCKERS"]
        for e in (x for x in bl_all if x in bl):
            L.append(f"- {e['title']} (raised by {disp(e['author'])}){tag(e)}")
        if len(bl_all) > len(bl):
            L.append(f"(+{len(bl_all) - len(bl)} more minor ones not shown)")

    rules = _dedupe(main("constraint") + main("criterion")[:5], key=lambda e: e["title"])[:8]
    if rules:
        L += ["", "CONSTRAINTS AND CRITERIA"]
        for e in rules:
            L.append(f"- {'[met] ' if e['status'] == 'satisfied' else ''}{e['title']} ({disp(e['author'])}, {e['kind']}){tag(e)}")

    shown, used = _presented(d, [e for e in els if e["id"] in public_ids])
    if (d.get("_enterprise") or {}).get("source_type") == "email":
        shown = []                            # an email chain is replies and feedback, not a talk nobody answered
    if shown:
        L += ["", "PRESENTED (explained to the group; nothing recorded disputes it; not voted on)"]
        for e in shown:
            L.append(f"- {disp(e['author'])}" + (" (confirmed by another speaker)" if e["status"] == "confirmed" else "") + f": {e['title']}{tag(e)}")

    cl = [c for c in _claims_worth_showing(d, [e for e in els if e["id"] in public_ids], skip=used, cx=cx) if c not in shown]
    if cl:
        L += ["", "CLAIMS (speaker's statements, unverified unless marked)"]
        for e in cl:
            L.append(f"- [{e['status']}] {disp(e['author'])}: {e['title']}{tag(e)}")

    all_acts = _actions(cx)
    priv_acts = [x for x in all_acts if x["scope"] and x["scope"][0] == "private"]
    acts = [x for x in all_acts if x not in priv_acts]
    if acts:
        shown_acts = _rank_actions(cx, acts)
        L += ["", "NEXT STEPS (the speakers' own words, in order; the words decide who does what)"]
        for x in shown_acts:
            sn = f" [seen only by {', '.join(x['scope'][1])}]" if x["scope"] else ""
            L.append(f'- {disp(x["author"])}: "{x["text"]}"' + (f" ({x['ctx']})" if x["ctx"] else "") + sn)
        if len(acts) > len(shown_acts):
            L.append(f"(+{len(acts) - len(shown_acts)} more not shown)")

    aw = _awaiting(cx)
    if aw:
        L += ["", "AWAITING A REPLY"]
        for a in aw:
            if a["kind"] == "ask":
                sn = f" [seen only by {', '.join(a['scope'][1])}]" if a["scope"] else ""
                L.append(f'- {disp(a["asker"])} asked {disp(a["to"])}: "{a["text"]}" -- {disp(a["to"])} has not spoken since{sn}')
            else:
                rep = "; ".join(f'{disp(x)}: "{t}"' for x, t in a["replies"]) or "no replies"
                sn = f" [seen only by {', '.join(a['scope'][1])}]" if a["scope"] else ""
                L.append(f'- Open call by {disp(a["asker"])}: "{a["text"]}" -- what followed: {rep}{sn}')

    flags = []
    for ev in d["events"]:
        if ev["kind"] != "flag":
            continue
        c = ids.get(ev["element_id"])
        for m in ev["payload"].get("mismatches", []):
            if ids.get(m.get("option"), {}).get("kind") != "option":
                continue            # a "consensus" flag about a blocker/claim id is a reconcile slip, not a real mismatch
            flags.append(f"- {disp(ev['participant'])} described {disp(m['participant'])}'s view as '{m['asserted']}'; "
                         f"the record shows '{m['actual']}'" + (f" (statement: {c['title']})" if c else ""))
    if flags:
        L += ["", "FLAGS"] + flags

    if private or (include_private and priv_acts):
        L += ["", "PRIVATE (each item was sent with a request to keep it private; write them only under 'Marked private')"]
        for e in private:
            L.append(f"- {disp(e['author'])} ({e['kind']}; {sc[e['id']][1]}): {e['title']}")
        for x in (priv_acts if include_private else []):
            L.append(f'- {disp(x["author"])} (step; {x["scope"][1]}): "{x["text"]}"')
    if positions:
        L += ["", "POSITIONS DRAFT (each person's view, written from a full read of CONVERSATION; section 5 is written from this)", positions.strip()]
    if main_point:
        L += ["", "MAIN POINT DRAFT (written from a full read of CONVERSATION; keep its substance in section 2, "
                  "change it only where CONVERSATION contradicts it)", main_point.strip()]
    return "\n".join(L)


MAIN_POINT_SYSTEM = """You are given a group discussion (an email chain, a meeting transcript or a chat) as the speakers' own words.
Write the main point of it in six to eight plain sentences, for a colleague who missed it.
Cover: who wants what and why; what was asked of whom and whether the person asked actually gave it (a reply that proposes
something else is not a yes); EVERY distinct point on which two people disagree, each with what each person says (for example
what the thing is, whether a problem exists, who supplies the people, what is enough); what is still unresolved; any figure,
date or deadline at stake.
Use only what the words show. Name every person. A name in brackets after a person is the name others use for them: one person.
Do not recommend or predict. Do not say anyone agreed, disagreed or refused unless their words say so; describe what they
proposed or asked instead ("did not confirm, and recommended ..."). Do not say what is holding something up unless a speaker
says so; say what the thread shows instead ("the thread ends without ..."). Quote the figure exactly as written.
Do not use: consensus, alignment, aligned, clearly, obviously, leverage, synergy, robust. No headings, no bullets."""


POSITIONS_SYSTEM = """You are given a group discussion as the speakers' own words.
For each person who expressed a view on how the matter should be handled, write one or two lines:
"<Name>: <what they want, propose, ask for or object to>; <what they said about another person's proposal, if anything>".
Use only what their words show, in their own terms. Do not turn a question into a position. Do not attach a figure or a condition
to a person unless their words did. If a person partly agreed with someone ("I agree with your point of ..."), say so. Include the
person's proposals even when nobody reacted to them. A name in brackets after a person is the name others use for them: one person.
Skip people who only passed information along or scheduled things. No headings, no commentary, one line per person."""


AUDIT_SYSTEM = """You check a summary of a group discussion against the discussion itself.
You get CONVERSATION (the speakers' own words) and a SUMMARY. For every statement in the SUMMARY about what a named person said,
wants, supports, opposes, will do, asked or owns, check that the words in CONVERSATION support it, including who said it
and any condition attached to it.
Output the SUMMARY again, unchanged except:
- a statement the words do not support is deleted, or rewritten to say only what the words show;
- a statement credited to the wrong person is corrected to the right person, or deleted if unclear;
- a next step or role nobody stated is deleted.
Keep the headings, the first line and everything supported exactly as written. Output only the summary, nothing else."""


TIGHTEN_SYSTEM = """You tighten a summary of a group discussion. Do not shorten it by dropping points.
Output the SUMMARY again with the same headings, in the same order, changed only by:
- a fact that is stated more than once is stated once, in the section where it is most useful; the other places keep only what
  they add (a name, a condition, a number, a date, a quote);
- wording is cut: repeated words, filler, "it was noted that", restating who the person is;
- two bullets that say the same thing become one bullet that keeps both names;
- a heading whose items are now all elsewhere is deleted.
Never delete or change a number, a date, a quoted phrase, a name, a condition ("only if", "until", "unless"), an objection or a
caveat, even a minor one. Never add anything. Output only the summary."""


_NUM = re.compile(r"\$?\d[\d,.]*\s?(?:[kKmM]\b|%|bp\b)?")
_QUO = re.compile(r'"([^"]{3,})"|\u201c([^\u201d]{3,})\u201d')
_CAP = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _facts(text):
    body = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("#"))
    nums = {m.group(0).strip().rstrip(".,") for m in _NUM.finditer(body)}
    quotes = {(a or b).strip().lower() for a, b in _QUO.findall(body)}
    caps = set(_CAP.findall(body))
    return nums, quotes, caps


def _tighten(call_llm, text):
    """Keep the tightened text only if it is shorter and lost no number, quoted phrase or capitalised name."""
    try:
        new = call_llm(TIGHTEN_SYSTEM, "SUMMARY\n" + text).strip()
    except Exception:
        return text
    if not new or len(new) >= len(text):
        return text
    a, b = _facts(text), _facts(new)
    if any(x - y for x, y in zip(a, b)):
        return text
    return new


def _norm_first(line):
    return re.sub(r"^[\s*#>]*status\s*:\s*", "", line.strip(), flags=re.I).strip("* ")


def make_summary(d, call_llm, include_private: bool = True, audit: bool = True) -> str:
    """call_llm(system, user) -> str. Calls: (1) full-read main point, (2) full-read positions, (3) structured summary, (3b) tighten (kept only if no fact is lost), (4) audit against the
    conversation (set audit=False to skip). The first line is forced to match the computed status, once."""
    conv = "\n".join(_conversation(Ctx(d), include_private, budget=10**6))
    draft = call_llm(MAIN_POINT_SYSTEM, conv).strip()
    positions = call_llm(POSITIONS_SYSTEM, conv).strip()
    text = call_llm(SUMMARY_SYSTEM, build_input(d, include_private, main_point=draft, positions=positions)).strip()
    text = _tighten(call_llm, text)
    if audit:
        text = call_llm(AUDIT_SYSTEM, "CONVERSATION\n" + conv + "\n\nSUMMARY\n" + text).strip()
    want = status_line(d)
    lines = text.split("\n")
    while lines and (not lines[0].strip() or _norm_first(lines[0]) == want):
        lines.pop(0)                       # drop every copy the model wrote (labelled or not), then put back exactly one
    return want + "\n\n" + "\n".join(lines)


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    print(build_input(data, include_private="--no-private" not in sys.argv))
