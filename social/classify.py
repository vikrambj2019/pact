"""Stage 2 — classify: atomic spans -> element labels (multi-label).

Input is exactly the output of normalize(). One Haiku call per span, with a small
window of neighbouring spans for context. No attribute extraction and no
reconciliation against the element log — those are later stages.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from llm import LLM
from models import NON_CONTENT, ClassifiedSpan, ElementType, Label, Span, SpanStatus

WINDOW_BEFORE = 8       # shown with context notes, right before the span
MAX_HISTORY = 150       # older spans shown compactly so novelty/consensus can be checked
WINDOW_AFTER = 2
MIN_PRIMARY = 0.35      # the strongest label survives down to this
MIN_SECONDARY = 0.5     # every additional label must clear this

CLASSIFY_SYSTEM = """You label ONE span from a group chat where people are making a decision together.
Assign every element type the span genuinely expresses (usually 1, sometimes 2), or a
non-content type ("process" or "irrelevant").

Element types
- candidate_question: the decision being deliberated ("which weekend for the cutover?").
  Only the question the group must decide — not information requests or calls for input.
- option: an alternative being INTRODUCED for the first time in the whole conversation, or
  an existing option whose STATUS CHANGES (dropped, withdrawn, made the fallback/backup,
  reopened). Check EARLIER CONVERSATION: if anyone already named it, then pricing it,
  comparing it, evaluating it, polling on it or listing it again is NOT option.
  Every option label must set option_role.
- criterion: an evaluative factor for judging options, not a hard limit ("ease of migration").
- constraint: a hard or soft limit that filters options and persists regardless of the
  decision ("must fit 20 people", "DPA must be signed before data flows", "budget max $500").
- claim: a factual, predictive or causal assertion by a participant, unverified
  ("it's still unsigned because the retention clause is unresolved", "trail closes at 6").
- position: the speaker's explicit stance on a specific option — propose, support,
  conditional, oppose, withdraw, undecided ("I'm in for Saturday", "I can do the 5th").
- agreement: the speaker affirms/rejects an explicit REQUIREMENT (budget cap, guest cap,
  access need) — not a vote on an option ("yes, $500 cap works for me").
- objection: an explicit concern against an option or direction ("Diablo parking is a nightmare").
- blocker: an open question, missing info or unresolved dependency preventing the decision
  NOW ("still waiting on legal", "we don't know the capacity yet"). Resolvable; a constraint is not.
- action: a concrete task someone committed to, or was asked to do, outside this chat
  ("I'll book the campsite by Friday", "Dev, can you write the runbook by Nov 9?",
  "I can hold that date" when it refers to a deliverable they own).
- separate_issue: a topic genuinely unrelated to the main decision (side thread).
- decision: the speaker, on behalf of the group, declares a question settled or an option
  chosen or ruled out ("Then it's decided: Nov 14-15", "OK, we're going with Provider B",
  "March is off"). Usually the organiser. Every decision label must set decision_kind.
  NOT decision: a personal commitment ("I'll commit two engineers" -> action); a report of
  what others think ("all seven of us are on board" -> claim, and it may be false); a
  proposal ("then we serve notice by Nov 30" -> option/position); a tentative lean
  ("so it's Relay, assuming it checks out" -> position).
- process: facilitation of the DISCUSSION itself — agenda steps, calls for input, asking
  a specific person to weigh in, objection checks ("Let's go through constraints first",
  "Marcus, anything from Finance?", "Anything else?", "Anyone against Nov 14-15?",
  "Let's settle it in this thread"). Non-content.
  NOT process: anything about the plan being made. "let's take the train", "or we meet
  there?" propose an alternative (option + position); "how are we getting there?" opens a
  question (candidate_question); "plan is X, if someone drives" settles it (decision).
- irrelevant: chit-chat, greetings, thanks, jokes. Non-content.

Hard rules
1. A REPLY TO AN OBJECTION CHECK ("None", "No objection", "None here", "go") is a POSITION
   (support for the option being checked). It is NEVER agreement.
2. AGREEMENT is only for affirming/rejecting a stated REQUIREMENT (budget cap, guest cap,
   access need, notice period). Saying an option works, or withdrawing one, is POSITION.
3. A REASON or JUSTIFICATION ("UAT should be done by then", "one more week of buffer",
   "close is well behind us") is a CLAIM and, if it names a factor for judging options,
   also a CRITERION. It is NEVER a position — the stance lives in its own span.
4. A statement that a constraint is or isn't met by options ("all three avoid that",
   "that's easy for either date") is a CLAIM, not option and not agreement.
5. A fixed date or dependency that gates feasibility is a CONSTRAINT even when stated as a
   fact ("UAT sign-off is scheduled for Friday Nov 6", "freeze starts Nov 23").
6. Withdrawing your own option = position (withdrawn) + option (status dropped).
   Naming an option as the backup/fallback = option (status change), plus position if the
   speaker is stating their own preference.
7. A condition attached to a task ("post the schedule once the runbook is done") is part of
   the ACTION, and a conditional fallback ("Backup is Nov 21-22 if UAT slips") is an OPTION.
   Neither is a blocker. Blockers are open questions, missing facts or pending
   dependencies that stop the group deciding NOW.
8. Saying an EXISTING option is acceptable ("X works too", "X is fine", "X works for me")
   is POSITION only — not option, even if it is the first time this speaker names it.
9. Only label what is EXPLICIT. Do not infer actions nobody took on.
10. POSITION is only the SPEAKER'S OWN stance on an option. Confirming a fact, giving an
    estimate, or answering a feasibility question is a CLAIM even if it starts with "yes"
    ("Yes, if the report lands Dec 1", "Contract confirmed, X was right",
    "$3M was my estimate, not a budget").
11. BLOCKER only when the span names something still unanswered or pending that the group
    needs before it can decide ("Does Relay have an EU region?", "I haven't verified that
    yet", "on-call is still unowned"). A hedged claim ("I think so but I'm not sure"), a
    stated limitation ("I don't have two engineers until Q3") or a scheduled date
    ("the report will be available Dec 1") is a claim or constraint, NOT a blocker.
12. AGREEMENT never applies to acknowledgements ("OK", "Fair", "Noted", "Right", "yes" to
    a fact). The clear case is someone restating a requirement to confirm it:
    "Confirming: $500 cap per person, all in" -> agreement.
13. A QUESTION THAT FLOATS A NEW ALTERNATIVE ("Q2?", "what about Tahoe?", "could we
    federate as a bridge?") is an OPTION (option_role=new), not a candidate_question.
14. A participant's own availability or absence is NEVER irrelevant. Absence from the very
    thing being planned ("i'm out this weekend", "can't make saturday") is a CONSTRAINT
    (plus position if it rules an option in or out). Absence that only affects something
    else ("I'm out next week, Wei covers my sign-offs") is a separate_issue.
    Never repeat medical or other private details in the gist.
15. A requirement put to the whole group ("cost split is $20 each", "everyone brings their
    own lunch") is a CONSTRAINT; individual replies confirming it are AGREEMENT.
16. Inside a side thread, a commitment is still an ACTION ("i'll bring the cake" ->
    separate_issue, action). Side-thread replies with any content are separate_issue,
    not irrelevant.

Other guidance
- "(context: ...)" notes were written by an earlier model and may be wrong. Trust the chat
  text over the note.
- Spans marked SELF_CORRECTED were retracted by their author moments later — label what they
  said, but confidence should reflect that it was withdrawn.
- Give a secondary label only if you would defend it on its own; low-confidence extras are noise.
- gist: one short normalized line per label, naming the option/requirement and the stance.

Examples (span -> labels)
- "I'd propose Nov 14-15"                        -> option, position
- "UAT should be done by then"                   -> claim
- "it's well before the holidays"                -> claim, criterion
- "Nov 14-15, Nov 21-22 and Dec 5-6 all avoid that" -> claim
- "Understood, withdrawing Dec 5-6"              -> position, option
- "UAT sign-off is scheduled for Friday Nov 6"   -> constraint
- "None, go."  (after "Anyone against Nov 14-15?") -> position
- "Nov 14-15 works for support"                  -> position
- "Nov 21-22 works too but I'd rather not stretch it" -> position
- "I'd make Nov 21-22 the backup if UAT slips"   -> option, position
- "Then it's decided: cutover Sat Nov 14 to Sun Nov 15" -> decision
- "Anyone against Nov 14-15?"                    -> process
- "Yes, runbook by Nov 9."                       -> action
- "Thanks everyone."                             -> irrelevant
More (unrelated chats):
- "Pricing: Vendor X is $50k, Vendor Y $70k" (both named earlier) -> claim
- "So it's the lake, assuming the permit comes through" -> position
- "I'll book the bus for all 30 of us"           -> action
- "Great, so everyone's happy with the lake"     -> claim
- "Then we cancel the lease and move in March"   -> option, position
- "Yes, if the permit arrives Friday" (answering "does that still work?") -> claim
- "I think they allow dogs, not 100% sure"       -> claim
- "Do they allow dogs?"                          -> blocker
- "I'm not free until June"                      -> constraint
- "Confirming: $500 cap per person, all in"      -> agreement
- "OK, fair."                                    -> irrelevant"""

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": [e.value for e in ElementType]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "gist": {"type": "string"},
                    "option_role": {
                        "type": "string", "enum": ["new", "status_change", "mention_only"],
                        "description": "required when type=option; check EARLIER CONVERSATION"},
                    "decision_kind": {
                        "type": "string",
                        "enum": ["group_closure", "personal_commitment", "consensus_claim",
                                 "proposal", "tentative_lean"],
                        "description": "required when type=decision"},
                },
                "required": ["type", "confidence", "gist"],
            },
        },
        "rationale": {"type": "string", "description": "one sentence"},
    },
    "required": ["labels", "rationale"],
}


def _line(sp: Span) -> str:
    tag = " [SELF_CORRECTED]" if sp.status == SpanStatus.self_corrected else ""
    note = f"  (context: {sp.context_note})" if sp.context_note else ""
    return f"{sp.author}: {sp.text}{tag}{note}"


def _plain(sp: Span) -> str:
    return f"{sp.author}: {sp.text}"


def _classify_one(llm: LLM, spans: list[Span], i: int) -> ClassifiedSpan:
    sp = spans[i]
    lo = max(0, i - WINDOW_BEFORE)
    earlier = spans[max(0, lo - MAX_HISTORY):lo]     # full history, compact, no notes
    before = spans[lo:i]
    after = spans[i + 1:i + 1 + WINDOW_AFTER]
    user = (
        "EARLIER CONVERSATION (for checking what was already said):\n"
        + ("\n".join(map(_plain, earlier)) or "(none)") + "\n\n"
        "CONTEXT BEFORE:\n" + ("\n".join(map(_line, before)) or "(none)") + "\n\n"
        ">>> SPAN TO LABEL:\n" + _line(sp) + "\n\n"
        "CONTEXT AFTER:\n" + ("\n".join(map(_line, after)) or "(none)")
    )
    out = llm.structured(system=CLASSIFY_SYSTEM, user=user,
                         tool_name="classify", schema=CLASSIFY_SCHEMA, max_tokens=512)
    return ClassifiedSpan(span=sp, labels=_clean(out.get("labels", [])),
                          rationale=out.get("rationale", ""))


# A decision the model itself says isn't group closure becomes what it actually is.
DECISION_FALLBACK = {
    "personal_commitment": ElementType.action,
    "consensus_claim": ElementType.claim,
    "proposal": ElementType.position,
    "tentative_lean": ElementType.position,
}


def _gate(r: dict) -> dict | None:
    """Apply the model's own justification fields. Returns the (possibly retyped) label,
    or None to drop it."""
    t = r.get("type")
    if t == "option" and r.get("option_role", "new") == "mention_only":
        return {**r, "type": "claim"}          # restating/pricing an existing option
    if t == "decision":
        kind = r.get("decision_kind", "group_closure")
        if kind != "group_closure":
            fb = DECISION_FALLBACK.get(kind)
            return {**r, "type": fb.value} if fb else None
    return r


def _clean(raw: list[dict]) -> list[Label]:
    labels: dict[ElementType, Label] = {}
    for r in raw:
        if isinstance(r, str):
            r = {"type": r, "confidence": 0.5, "gist": ""}
        elif not isinstance(r, dict):
            continue
        r = _gate(r)
        if r is None:
            continue
        try:
            lab = Label(type=ElementType(r["type"]),
                        confidence=float(r.get("confidence", 0.5)),
                        gist=r.get("gist", ""))
        except (ValueError, KeyError):
            continue
        if lab.type not in labels or lab.confidence > labels[lab.type].confidence:
            labels[lab.type] = lab   # dedupe repeated types, keep the strongest

    content = sorted((l for l in labels.values() if l.type not in NON_CONTENT),
                     key=lambda l: -l.confidence)
    content = [l for k, l in enumerate(content)
               if l.confidence >= (MIN_PRIMARY if k == 0 else MIN_SECONDARY)]
    if content:                      # content labels win over process/irrelevant
        return content
    non = sorted((l for l in labels.values() if l.type in NON_CONTENT),
                 key=lambda l: -l.confidence)
    return non[:1] or [Label(type=ElementType.irrelevant, confidence=0.5, gist="")]


def classify(spans: list[Span], llm: LLM, *, max_workers: int = 16) -> list[ClassifiedSpan]:
    """normalize() output -> one ClassifiedSpan per span, same order."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda i: _classify_one(llm, spans, i), range(len(spans))))


if __name__ == "__main__":
    import sys
    from llm import AnthropicLLM
    from normalize import normalize
    from models import Transcript
    llm = AnthropicLLM()
    t = Transcript.model_validate_json(open(sys.argv[1]).read())
    for cs in classify(normalize(t, llm), llm):
        if cs.is_irrelevant:
            continue
        tags = ", ".join(f"{l.type.value}({l.confidence:.2f})" for l in cs.labels)
        print(f"{cs.span.author:>6} | {cs.span.text!r:60} | {tags}")
