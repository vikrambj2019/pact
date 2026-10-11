# Pact pipeline validation — independent reviewer update

- 58 local regression tests pass; all eight production modules compile.
- Fresh live reviewer probes accepted a sourced Dev action and rejected an
  invented Nov 6 completion deadline, invented Hal funding approval, and an
  executive-summary line claiming the invented deadline (4/4 expected outcomes).
  Evidence: `validation/reviewer/probes.json`, model `claude-sonnet-5`.
- Coverage/failure regressions exercise all eleven extraction sections, summary
  lines, missing/duplicate/unknown verdicts, unavailable review, whole-output
  contradictions, extraction retry feedback and review-before-persistence order.
- A fresh full-pipeline extraction with all new review stages and integration
  against the surrounding application's database/job runner remain outstanding.
  The standalone harness now runs the new reviews. The following older benchmark
  results predate this reviewer; replays are not evidence of reviewer acceptance.

# Earlier portal validation

## Deterministic checks

- 51 regression tests pass, including all 29 earlier tests.
- All seven production modules and standalone test scripts compile.
- Native Teams parsing produces 68 messages from seven named participants.
- Six earlier live anniversary results replayed through this version each pass
  26/26 semantic checks. These are saved-state replays, not fresh model calls.
- Four recent independently extracted portal states each replay through this
  version at 25/25, including exact gate and unassigned-action counts.

## Portal benchmark

`tests/portal.txt` is the user's unmodified transcript. `tests/score_portal.py`
checks 25 semantic expectations from the user's expected card: three options,
C dropped, B fallback, Marcus/Rosa support plus Elena's condition, Priya's proposal,
no Hal option vote, three open gates and two objections, two unassigned actions,
accepted deadlines, no Dev completion deadline, revised estimates, historical
receipt forecasts excluded, and valid source references and matching quotes.

The iterative results are retained. Model omissions and misclassifications were
observed and then repaired in extraction instructions, validation, chronological
merge rules, source reconciliation and deterministic rendering. Replay artifacts
explicitly identify their mode. Replay results must not be described as fresh
end-to-end runs of the final source.

The latest fresh native Teams run (`validation/portal-final-native/run-1.json`)
passed 23/25 before the final classification fixes. Reprocessing that actual live
state through the final code passed 25/25 (`validation/portal-replays/final-native`).
The two remaining differences were a review requirement counted as an objection
and a budget-note acknowledgment counted as an unowned task. This final result is
an explicitly labeled deterministic replay, not a fresh final-version model run.
No hand-written expected extraction is substituted for a model response.

## Scope and limits

The surrounding application models, database/job execution, participant mapping,
API validation and UI were not supplied, so integration into that application was
not tested. Source-reference checks prove quote presence, not semantic entailment
or external verification. The two fixtures do not establish general accuracy or
comprehensive privacy coverage. Teams source timestamps do not specify timezone;
the parser preserves the wall-clock value and flags that limitation.

Rosa's action is to chase the DPA before Oct 20; this does not guarantee the
counterparty's signature. Dev's completion date remains unaccepted. Operational
requirements remain public while private reasons and side-thread actions are
excluded from the main card.

No credentials are included in deliverables. Source hashes and raw scored model
outputs are included so the test results can be inspected and reproduced.
