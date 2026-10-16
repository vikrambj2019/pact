# Pact decision-card and enterprise dependency fixes

Replace the matching pipeline modules in your application with the eight files
in `fixed/`. Keep their existing package locations and relative imports. No new
production dependency is introduced. The tests use a standalone pure-function
loader because API models, settings, participant mapping and storage were not
included in the supplied files.

## What changed

- Added `agreements`: one source-backed affirmation/rejection per participant and
  requirement. Budget and guest-cap affirmers are independent of venue/date votes.
- Added action `status`: `requested`, `committed`, `completed`, `cancelled`.
  Third-party assignments cannot become owner commitments merely because the
  owner was named. Repeated requests preserve an existing acceptance/completion.
- Separated an owner's promised `due` from an `external_due`, such as a venue's
  reservation deadline.
- Added `private` independently of `sensitive`. Public requirements can remain
  visible while health reasons and private contribution arrangements are omitted.
  A display guard also omits common personal health/financial reason patterns when
  a model forgets a flag, and extracts an explicitly separated requirement clause.
  This is a fallback, not a complete privacy classifier.
  Raw sources and quotes remain in the internal snapshot for audit; existing
  application access controls must protect that internal material.
- Claims carry `evidence_status`: `unverified`, `conflicting`, `superseded`.
  Participant reports are never promoted to independently verified facts.
- Added narrow, deterministic reconciliation for an explicit yes to an adjacent
  enumerated plan, normative guest-cap restatements, and a uniquely matched
  source-backed deposit deadline.
  Explicit personal contribution ceilings and named booking requests are recovered
  from source when omitted; short acknowledgments adopt a directly stated cap
  without becoming option votes. Explicit reported venue capacity is also compared with a committed guest-list
  target; an explicitly deferred collective task records a missing named owner. It never treats silence, ok, whatever or a
  qualified yes as an option vote, and it does not infer task acceptance.
- Scoped option matching and position linking by decision. Corrected mislabeled
  position decisions against unambiguous known options, preventing phantom options.
  Switching a date
  preference does not erase a venue preference. Acknowledgments do not replace
  explicit support. Dropped options require explicit reopening.
- Removed the default synthesis and summary rewriting calls. The structured
  snapshot preserves validated extraction candidates and the displayed card is
  rendered deterministically with source references. Missing/invalid citations
  and nonmatching quotes are omitted from the card.
- Retained legacy summary helpers for compatibility, but changed their entailment
  error path to remove unchecked items rather than display them.
- Removed token-overlap deduplication. Chat/transcript turns remain independent;
  emails are deduplicated only for exact same-author, same-known-timestamp copies.
- Preserved multiline messages and URLs. Fixed automatic detection of HH:MM:SS
  transcripts. Source chronology no longer interleaves document-local sequence
  numbers: dated documents are ordered by their earliest real timestamp, with
  stable document grouping. Undated transcripts still require an explicit external
  ordering policy if chronology across documents matters.
- Added a second, focused extraction pass per chunk for votes, requirement affirmations,
  commitments and separate issues. This addresses omissions observed in the first
  live benchmark (the initial one-pass result passed 21 of 26 checks).
- Passed prior claims, positions, requirements and actions to subsequent chunks
  as read-only state so corrections and later acceptances have context.
- Repaired mistyped citations only when an exact quote uniquely identifies a
  primary message. Ambiguous short replies and context-only matches are never guessed.
- Bound participant attribution to the message containing the quote, rather than
  any other cited message. Structured responses that hit the output-token limit
  raise an error instead of silently accepting truncated state.
- Checked that the extraction job belongs to the workspace being synthesized.

## Apply

1. Replace `parser.py`, `dedup.py`, `chunker.py`, `extractor.py`, `merger.py`,
   `llm.py` and `synthesizer.py` in your existing pipeline package.
2. Reparse the original document in a fresh workspace for this benchmark. Old
   duplicate flags may have hidden valid messages. Do not simply synthesize an
   old extraction job: it has no agreement/acceptance/privacy metadata.
3. Run extraction and synthesis again. `structured_snapshot.agreements` is new;
   update a custom structured UI if it displays fields individually. A UI that
   displays `executive_summary` receives the deterministic card directly.
4. Legacy actions without acceptance status default to `requested`, conservatively.
   No database migration is assumed for new JSON fields; verify this against your
   application's model/schema validation, which was not supplied here.

## Validation

Run deterministic regression tests with the standard library:

```bash
python -m unittest discover -s tests -v
```

The 51 regression tests exercise actual parsing, validation, merging and rendering functions.
They do not fabricate a model response and claim that extraction was tested.

The synthetic transcript is `tests/anniversary.txt`. `tests/run_live.py` invokes
the actual extractor via the Anthropic Messages API, carries state across chunks,
then merges and renders using the patched functions. It scores 26 answer-key
checks: exact affirmer sets, option states, ownership/acceptance, deadlines,
privacy, separate issues and citation validity.

```bash
python tests/run_live.py --credentials /absolute/path/to/local.env \
  --model claude-sonnet-5 --runs 3 --output live-results
```

The credential file is read only and never copied into output. API request
failures print redacted error summaries. No credential file is included here.
Live outputs contain only the invented benchmark discussion and its derived state.

## Limits

Extraction remains model-dependent. Deterministic rendering prevents a later
model call from rewriting state; it does not prove that extraction is correct.
Quote presence verifies provenance, not interpretation or external truth. The
scorer is a repeatable benchmark, not an exhaustive proof, and semantic-equivalent
wording can occasionally need manual scoring. Actual DB integration and UI behavior
must be checked in the complete repository. See `VALIDATION.md` for the exact
run history and the distinction between fresh live runs and replayed model results.

Photographer/secrecy/irrelevant anniversary-date topics follow the provided
benchmark's separation policy; in other products those can instead be relevant
sub-decisions. Configure that extraction policy deliberately for your use case.


## Portal-launch update

Teams headers `[10/5/26 9:02 AM] Full Name: message` are now parsed natively,
including multiline continuations. Wall-clock source timestamps have no supplied
timezone; `locator.timezone_unspecified` records that limitation.

Bare confirmations retain earlier support conditions. Generic sponsorship is not
an option vote. Approval routing never counts as approval; receipt and review are
separate gates. Duplicate gates are reconciled by explicit ticket/artifact identity,
with vendor-scoped contract identities. Requested deadlines, start dates and owner
completion deadlines remain distinct. Named bundle confirmations recover omitted
accepted milestone dates from source; an unrelated yes cannot accept them.

The focused coverage pass now audits claims, blockers and objections as well as
votes and actions. The deterministic card includes status counts and a preference
line, and retains relevant revised cost and effort estimates. Side-thread actions
are excluded from the main card.

Run the native Teams benchmark:

```bash
python tests/run_portal.py --credentials /absolute/path/to/local.env \
  --model claude-sonnet-5 --fixture tests/portal.txt --runs 1 \
  --output portal-results
```

`tests/score_portal.py` checks 25 semantic expectations from the supplied portal
card. It does not require verbatim wording. The production modules still need the
existing application's models, configuration, jobs and persistence layer, which
are not included in this standalone fixture harness.


## Independent semantic reviewer

Install `fixed/reviewer.py` alongside the existing modules. Extraction now runs
an independent forced-tool review after the initial and focused passes. Every
item in all eleven extraction sections receives a keyed pass/reject verdict,
including its fields, conditions, attribution, dates and supporting sources.
A rejected chunk gets one re-extraction attempt with review feedback. Failed or
incomplete review fails the job; it never silently removes disputed items.

The merged state is reviewed after reconciliation. Synthesis rechecks the final
structured sections, question and computed preference after reference/ID changes,
then reviews every nonempty executive-summary line and the complete card against
both the JSON and source transcript. Summary review must pass before creating an
Event or BriefRevision. Legacy extraction jobs also receive the final reviews.

Reports record keyed verdicts and complete coverage separately from the existing
quote/reference verifier. Review is an LLM semantic audit, not external fact
verification. This adds latency and model cost: up to twenty elements per review
call, with complete context supplied to each batch. Reviewer outages, response
truncation, missing/duplicate verdicts or rejections block publication. Large
transcripts must fit the model context window; no silent context truncation is
performed. The surrounding job runner should surface failed-job feedback for
correction. The public card remains deterministic; the reviewer cannot rewrite it.

Run the isolated live reviewer probes with `tests/run_reviewer.py` using the same
credentials/model arguments as the portal harness. The probes are separate from
a fresh full extraction benchmark. Never place credential files in the archive.
