# Portal launch test result: FAIL

Tested the supplied seven production modules without changes, using one fresh live Claude Sonnet 5 extraction run.

## Native ingestion

The original Teams export is detected as plain text: two source units, zero author labels. It should produce 68 message units and seven participants. Native end-to-end processing fails at ingestion.

## Downstream diagnostic

A test-only header conversion preserves the original message text and maps Teams headers into supported WhatsApp syntax. The converted input produces 68 messages, seven participants and three extraction chunks. Actual extraction, chronological merging, reconciliation and deterministic card rendering then ran against the unchanged modules.

Compared the actual output with the user's expected portal card. This is a manual review, not the unrelated anniversary benchmark score. One run cannot establish consistency.

| Expected behavior | Observed result |
|---|---|
| Elena retains conditional support pending security sign-off | FAIL: later generic yes becomes unconditional support |
| Jon's B funding objection stays open pending approval | FAIL: objection marked resolved; no approval occurred |
| SOC 2 receipt and unfinished review remain distinct | FAIL: receipt recorded and review action present, but review/sign-off blocker absent |
| Oct 30 checkpoint has no accepted owner | FAIL: checkpoint action missing |
| B estimates retain disputed 3 vs 5 weeks and revised $65k cost | FAIL: estimates absent from displayed card |
| Priya's A proposal remains attributed | FAIL: Priya absent from A positions |
| Hal backs the eventual selection conditionally, without approving B funding | PARTIAL: rendered as conditional A support; approval requirement remains in constraints |
| Correct current claims supersede stale reports | FAIL: old Oct 23 expected report date remains displayed after Oct 9 receipt |
| Dev's generic fine is not explicit support | PASS, but acknowledgment linked to B rather than the adjacent fallback statement being clarified |
| Exactly three options; C dropped, B fallback | PASS |
| Marcus Nov 6, Rosa DPA chase, Elena Oct 14 review, Dev token fix | PASS |
| Dev has no accepted Nov 6 deadline | PASS: fix commitment has no due date; separate Nov 6 request is not accepted |
| Budget ask to Hal by Oct 30 has no named owner | PASS |
| DPA and finding #3 remain open | PASS |
| Figma, on-call and all-hands are separate topics | PASS |

The output also calls Option B 'capped at $215k' in an agreement; $215k is its estimated total, not an approved cap. The displayed card lacks the requested concise status counts and preference line.

## Expected-card nuance

Rosa explicitly commits to chasing the DPA and says it needs to be signed before Oct 20. This is a target/dependency, not proof that she can guarantee the counterparty's signature. The actual action wording 'Chase the DPA' is more precise than treating signature itself as guaranteed.

## Files

`run-1.json` is the actual structured result; `run-1-card.txt` is the actual renderer output. Neither has been manually corrected to match the expected card. The standalone diagnostic runner and original transcript are included with the unchanged pipeline files.

No production fixes were made during this test. Credentials are excluded.
