# Pact MVP — Technical Specification

Version 0.2 · October 9, 2026 · Phase 1: single-company prototype; Phase 2: workforce identity and enterprise controls

## 1. Product contract

Pact converts meeting transcripts and email chains into an interactive decision brief. Participants challenge interpretations, add evidence, compare options, and resolve blockers. A human operator explicitly records the resulting decision as a durable, searchable enterprise artifact.

The unit of work is one decision workspace. It can contain multiple source documents, but it has one confirmed primary decision question. Tangential questions become linked separate issues. A dependency that can change the current decision stays in scope.

The MVP succeeds when a user can import a real discussion, inspect a source-grounded brief, correct or challenge it, and record a defensible decision with rationale and actions. A summary is the starting point; the decision record is the lasting output.

### Mandatory invariants

1. An assertion in a transcript is evidence that someone made a claim, not evidence that the claim is true.
2. Silence never establishes agreement, consent, or a participant's position.
3. Model recommendations and emerging preferences never authorize a decision.
4. Participant positions, options, claims, and corrections retain provenance.
5. A recorded decision is immutable. Changes create a new deliberation revision and, if explicitly confirmed by a human, a superseding decision artifact.
6. The model cannot assign authority or record a decision. Phase 1 records human-declared ownership; verified authority is a Phase 2 capability.
7. Input text is untrusted data, including instructions embedded in email bodies or attachments.

## 2. Scope

### Phase 1 — included

- Paste text; upload UTF-8 TXT, Markdown, EML, and text-based PDF.
- Import long email chains and speaker-labelled transcripts; preserve unknown authors and timestamps explicitly.
- Remove repeated email quotations without deleting newly added inline responses.
- Extract candidate decision questions; human confirms the primary question and scope.
- Produce a structured decision brief with source references.
- Interactive chat that proposes updates to the brief and asks focused questions.
- Correct attribution, positions, claims, options, priorities, and scope.
- Evidence review against uploaded material, with support and contradiction shown separately.
- Separate issues with explanations and reversible classification.
- Readiness review, explicit human recording, version history, and artifact search.
- Export JSON and Markdown.
- Evaluation using real cases and independently reviewed synthetic variants.

### Later product extensions — deferred beyond Phase 1

Live meeting bots, audio transcription, OCR for scanned PDFs, mailbox connectors, organization-wide document search, external web verification, automated task execution, automatic decisions, voting engines, enterprise knowledge graphs, and real-time multi-party co-editing. Uploads supplement the workspace without requiring these integrations.

### Phase 1 operating assumption

One company, one trusted operator, one application instance. Use a local or privately restricted pilot environment. All workspaces belong to that company. No workforce UID, employee directory, user accounts, login flow, SSO, memberships, roles, tenant IDs, or application-level access controls are required in Phase 1.

Participant names are labels extracted from sources. Decision owner and recorded-by names are manually entered metadata, not verified employee identities. Internal UUIDs remain for documents, participants, revisions, and artifacts; these are record keys, not workforce UIDs. No company ID is required on every row.

### Phase 2 — workforce identity and enterprise controls

Add authenticated users, optional workforce UID/directory linkage, participant-to-user mapping, roles, workspace access, verified owner authorization, and permission-aware search/export. Continue assuming one company; multi-company tenancy is a separate future requirement, not a Phase 2 prerequisite. Detailed migration requirements appear in section 14.

## 3. User experience

Use three main areas: source material, living decision brief, and interaction panel. Include a separate-issues drawer and decision history.

1. Create workspace; provide a title and optionally a decision question.
2. Paste or upload discussion. Display parsing status and warnings.
3. Review detected questions; confirm the primary question, scope, decision owner, and decision rule. Owner may remain unset during deliberation.
4. Inspect the initial brief. Click a source reference to see the original passage and surrounding context.
5. Engage through chat or field corrections. Show proposed material changes before applying them; show a diff after every accepted update.
6. Request a readiness review. Display unresolved evidence, trade-offs, objections, and dependencies.
7. The operator previews the final artifact, enters the decision owner and recorded-by names, and explicitly records the decision. Record accepted risks or unresolved objections rather than silently removing them.
8. Search, export, or reopen the record. Reopening preserves the original artifact.

Examples: “That was a question, not my objection”; “What is the strongest case against option A?”; “Prioritize reliability”; “This topic belongs separately”; “What prevents us from deciding?”

## 4. Proposed architecture

Use a modular monolith rather than a distributed agent system for the MVP.

| Component | Responsibility |
|---|---|
| React/TypeScript UI | Sources, brief, chat, diffs, recording, artifact search |
| Python/FastAPI API | Commands, validation, state transitions; identity/access controls in Phase 2 |
| Python worker | Parsing, staged extraction, synthesis, evidence review, exports |
| PostgreSQL | Workspace state, revisions, events, jobs, artifacts, full-text search |
| Private object storage | Original uploads and generated exports |
| LLM adapter | Provider-independent structured extraction and reasoning |

Run API and worker from the same codebase. A PostgreSQL-backed jobs table is sufficient initially; claim jobs atomically, use leases and heartbeats, and recover abandoned work. Do not require Redis or a vector database in the first build. Full-text retrieval plus exact source references support the initial corpus.

The backend owns state. The LLM reads a bounded context and returns typed proposals; it never writes database rows directly. Use Pydantic models for contracts and database migrations for schema changes. Pin actual dependency versions when implementing; this specification does not prescribe current package releases.

## 5. Domain model

All persisted entities have internal UUID record IDs, creation time, and relevant actor labels. Phase 1 has no organization ID, workforce UID, or user foreign keys. Store timestamps in UTC; retain original timezone information when available.

| Entity | Key fields |
|---|---|
| Workspace | title, primary_question, scope, lifecycle_status, current_revision_id, decision_owner_label, decision_rule |
| SourceDocument | type, original_filename, content_hash, storage_key, parser_version, parse_status, warnings |
| SourceUnit | document_id, sequence, author_label, mapped_participant_id, timestamp, original_text, normalized_text, locator, duplicate_of |
| Participant | display_label, source_aliases, resolution_status |
| BriefRevision | workspace_id, revision_number, parent_revision_id, structured_snapshot, event_id |
| ConversationMessage | author_label, role (human/assistant), text, linked_revision_id |
| ChangeProposal | base_revision_id, typed_operations, rationale, source_refs, status |
| Event | event_type, actor_type (human/AI/system), actor_label, payload, prior_revision, resulting_revision |
| DecisionArtifact | workspace_id, artifact_version, originating_revision_id, decision_owner_label, recorded_by_label, authority_verification, recorded_at, immutable_snapshot, supersedes_artifact_id |
| Job | type, status, input_revision, idempotency_key, attempt_count, lease_until, error_code |

Author labels are not authenticated identities. The operator can merge aliases or split ambiguous names within a workspace, preserving source attribution and correction history. Cross-workspace workforce identity resolution is deferred to Phase 2.

### Structured brief

The snapshot contains stable IDs for the following collections:

- Question: text, scope, objective, confirmed_by, confirmation_time.
- Criteria: name, description, human-confirmed priority; optional weight only if users explicitly choose numerical comparison.
- Constraints: description, mandatory/negotiable/unknown, basis, status.
- Options: title, description, origin (source/human/AI), arguments for and against, assumptions, dependencies.
- Arguments: proposition, argument_type, option_ids, participant_ids, source_refs, addressed_by_ids.
- Claims: proposition, kind, evidence_status, evidence_links, applicability_limits, verification_scope.
- Positions: participant_id, option_id, stance, conditions, source_refs, assertion_time, confirmation_status, supersedes_position_id.
- Objections: description, option_ids, materiality rationale, resolution_status, response_refs.
- Blockers: type, description, linked entities, resolution_status, proposed unblock action.
- Separate issues: title, question, relevance rationale, source_refs, optional owner, disposition, linked_workspace_id.
- Actions: description, owner nullable, due_date nullable, assignment_status, dependency_ids, completion_status.
- Emerging preference: optional option_id with supporting and dissenting references; never labelled consensus by inference.

Claim kinds: factual, predictive, causal, priority, constraint, or other. Evidence states: unverified, supported_within_scope, contradicted_within_scope, conflicting, unresolved. Uploaded documents can establish what a document reports; reliability and applicability remain explicit.

Position stances: support, oppose, conditional, undecided, or unknown. Phase 1 distinguishes source_reported and operator_corrected. A confirmation explicitly present in a source remains source_reported; the operator cannot mark it as authenticated participant confirmation. Participant-confirmed status is introduced in Phase 2. A later statement can supersede an earlier position, but missing timestamps or ambiguous statements require clarification.

### Source reference contract

```json
{
  "source_document_id": "uuid",
  "source_unit_id": "uuid",
  "start_char": 120,
  "end_char": 198,
  "basis": "source_reported"
}
```

Offsets are zero-based, end-exclusive Unicode code-point offsets into the immutable SourceUnit.original_text. Additional locators include transcript timestamp, email message identifier, or PDF page number. Normalize for processing, but retain a mapping back to original text. Validate bounds and existence server-side. Human contributions reference persisted chat messages or explicit correction events instead of inventing original-source citations.

## 6. Ingestion and long-input processing

### Parsing

Preserve original bytes and hash before normalization. Enforce configurable limits; proposed pilot defaults are 25 MiB per file and 250,000 extracted words per workspace. Larger input is rejected with a clear message rather than truncated silently.

For EML, decode MIME and preserve sender, recipients, date, subject, Message-ID, and textual body. Attachments become separate source documents only when explicitly selected. For pasted chains, apply conservative header and quote detection. If message boundaries cannot be reconstructed, preserve segments as unknown-author units and issue warnings.

Repeated quotations are marked by duplicate links, not removed from the canonical source. Preserve inline replies and near-duplicate text with changed wording. The extraction view may suppress exact duplicates; source inspection must show the full original.

For transcripts, create speaker turns with sequence and timestamps where present. For PDF, preserve page references; reject or flag pages without usable text. Never guess missing authors or dates.

### Staged extraction

1. Split into token-bounded chunks at message or turn boundaries. Split oversized turns with stable offsets. Configure budgets for the selected model rather than a fixed word-to-token ratio.
2. Extract candidate questions, options, claims, objections, positions, and tangents per chunk using a strict schema.
3. Merge candidates using stable references. Retain attribution and contradictions; repeated statements do not become stronger evidence through frequency alone.
4. Confirm the primary question with the user before final synthesis.
5. Construct a brief for that question; retain alternative questions as separate issues.
6. Run a coverage pass across all units for omitted material objections and decision dependencies.
7. Validate schemas and references, then persist the first revision.

Do not reduce a long chain to one free-text summary before extracting arguments. Store intermediate chunk results so retries resume without repeating completed work. Display partial processing as partial, never as a complete brief.

## 7. Interactive deliberation engine

### Request cycle

1. Validate the request and target workspace in the single-company instance. Authentication and permission checks are Phase 2 additions.
2. Load the expected revision, user message, relevant brief entities, and retrieved source context.
3. Classify the request: clarification, correction, challenge, evidence review, comparison, scope change, separate issue, readiness, or decision drafting.
4. Produce a response and optional typed change proposal.
5. Validate proposal operations and source links.
6. Apply only within the permitted mutation policy and expected revision.
7. Append an event and publish the updated revision/diff.

A question can receive an answer without changing state. Explicit, unambiguous operator instructions may apply a correction immediately and display the diff. Ambiguous or AI-inferred material changes require acceptance. Decisions, owner changes, primary-question changes, and supersession always require explicit application commands.

### LLM output contract

```json
{
  "response_text": "The capacity objection remains unresolved.",
  "source_refs": [],
  "proposal": {
    "base_revision": 7,
    "operations": [
      {
        "type": "add_blocker",
        "entity_id": "temporary_id",
        "fields": {
          "type": "unanswered_objection",
          "description": "Confirm pilot staffing capacity"
        },
        "basis_entity_ids": ["objection_uuid"]
      }
    ],
    "rationale": "The objection has no recorded response."
  },
  "follow_up_question": "Who can confirm available staffing?"
}
```

Use an allowlisted operation union; do not accept arbitrary JSON paths or SQL. Server assigns permanent IDs. Unsupported assertions are flagged or omitted. Schema failures get bounded retries; exhausted retries fail the job without committing malformed state.

### Scope classification

Classify a new point as directly_relevant, decision_dependency, separate_issue, or uncertain. Explain the relationship to the confirmed question. Uncertain classifications prompt clarification. The operator can restore a separate issue to scope or move an in-scope point out; both changes are logged.

Preserve side issues with source references. Creating another workspace from an issue is an explicit user action. Do not assign owners or deadlines as if participants accepted them.

### Evidence review

For a selected claim, retrieve relevant uploaded passages and show supporting, contradicting, and limiting evidence. Track the search corpus and source versions. A no-match result remains unverified or unresolved. Distinguish disagreement over priorities from disagreement over evidence. New evidence can invalidate readiness without modifying a previously recorded artifact.

## 8. Lifecycle, readiness, and decision authority

Lifecycle states: draft -> deliberating -> decision_recorded. A readiness assessment belongs to a specific brief revision and can become stale. Job processing state is separate from workspace lifecycle.

Readiness result: blocked, ready_with_explicit_tradeoffs, or ready_for_owner_review, with explanatory blockers and exceptions. It is advisory, not a permission grant. A model cannot prevent a human operator from recording a choice under uncertainty, but recording requires visible acknowledgment of identified unresolved material risks.

Pilot decision rule: designated owner decides after consultation. Do not implement consensus as the default. If another rule is described in source material, preserve it as context; require a human to select the supported workflow or defer that workspace. In Phase 1, the operator enters decision_owner_label and records that ownership is human-declared. The application does not verify organizational authority or infer it from titles. Phase 2 adds an authenticated owner designated through an admin-controlled workflow.

### Recording transaction

The UI presents the exact artifact preview. The operator supplies the decision owner and recorded-by labels and explicitly confirms the option, rationale, criteria, accepted risks, dissent, actions, and revisit conditions. Store authority_verification="not_verified_phase_1" in the artifact.

POST /workspaces/{id}/decisions requires expected_revision, artifact_draft_hash, and explicit_confirmation=true. The server verifies explicit human confirmation, nonempty decision owner and recorded-by labels, current revision, complete required fields, unresolved-risk acknowledgments, and idempotency key. Workforce identity and owner authorization checks are deferred to Phase 2. In one database transaction it writes an immutable artifact snapshot, a recording event, and the lifecycle update. If state changed after preview, return 409 and require a new preview.

Neither chat wording such as “sounds good” nor an LLM tool request records a decision.

### Enterprise artifact

Required fields: stable artifact ID, title, decision question, selected direction, decision_owner_label, recorded_by_label, authority_verification, recording time, rationale, criteria, alternatives and rejection reasons (or explicitly not considered), evidence references, accepted assumptions/risks, unresolved dissent, next actions, revisit conditions, originating brief revision, and lineage.

Snapshot cited excerpts and source hashes so rationale remains interpretable if a source is later removed under a retention policy. Phase 1 search and export operate across this company instance; there is no per-user visibility filtering. Phase 2 adds workspace permissions to sources, records, search, and exports.

Reopen creates a new deliberation revision referencing the current artifact. The existing recorded decision remains effective until an explicitly human-confirmed artifact supersedes it. Search shows the latest effective decision plus visible history. Invalidating a decision without a replacement requires an explicit human-confirmed withdrawal event and reason; retain the historical artifact.

Action completion updates are separate events and do not rewrite the original decision snapshot.

## 9. API surface

All endpoints are under /api/v1. Mutations accept an Idempotency-Key; state mutations include expected_revision. Reusing a key with different input returns 409. Responses include stable IDs and current revision.

| Method / route | Purpose |
|---|---|
| POST /workspaces | Create workspace |
| GET /workspaces/{id} | Workspace and current brief |
| POST /workspaces/{id}/sources | Upload or paste source; return job ID |
| POST /workspaces/{id}/question-confirmation | Confirm question and scope |
| POST /workspaces/{id}/synthesis | Start or resume initial synthesis |
| GET /jobs/{id} | Status, warnings, errors |
| GET /sources/{id}/units/{unit_id} | Original text and citation context |
| POST /workspaces/{id}/messages | Deliberation response and optional proposal |
| POST /proposals/{id}/accept | Validated revision mutation |
| POST /proposals/{id}/reject | Preserve rejection and optional rationale |
| POST /workspaces/{id}/corrections | Explicit typed human correction |
| POST /workspaces/{id}/claim-reviews | Evidence review job |
| POST /workspaces/{id}/readiness | Revision-bound readiness review |
| POST /workspaces/{id}/decision-preview | Validated immutable-artifact draft and hash |
| POST /workspaces/{id}/decisions | Explicit human-confirmed recording with owner/recorder labels |
| POST /workspaces/{id}/reopen | Start deliberation referencing existing artifact |
| POST /decisions/{id}/withdrawal | Explicit human-confirmed withdrawal with reason |
| GET /workspaces/{id}/history | Events and revisions |
| GET /decisions?q=... | Artifact search across the single-company instance |
| GET /decisions/{id} | Artifact and lineage |
| POST /decisions/{id}/exports | Generate JSON or Markdown export |

Use 409 for stale revisions/idempotency conflicts, 422 for invalid input, and explicit job error codes for parsing, model, and storage failures. Phase 2 adds 401/403 and authorization checks to every endpoint, including jobs and citations. Provide polling first; streaming text can follow without changing durable mutation semantics.

## 10. Persistence, reliability, and security

- Transactions cover revision creation, event append, and current-revision update. Use optimistic concurrency; never overwrite another accepted change silently.
- Worker jobs use bounded retries with backoff, retryable error classification, leases, and checkpointed extraction. Ignore late results whose base revision is stale; retain them as unapplied proposals if useful.
- Keep source objects private and the Phase 1 instance locally or privately restricted. Serve source and export retrieval through the application; use short-lived URLs when needed. Per-user authorization is Phase 2.
- Phase 1 has one trusted operator with the same access to all workspaces. Do not implement a role or membership system in this phase.
- Phase 1 corrections are labelled operator corrections. Verified participant self-confirmation and user mappings are Phase 2.
- Phase 1 queries must correctly scope entities to their workspaces for data integrity. Access isolation, workforce authorization, and permission filtering are Phase 2.
- Secrets are server-side environment configuration; provide .env.example with variable names only. Never store credentials in source control or browser state.
- Treat uploaded content as quoted data. Disable model execution of embedded instructions; allow only application-scoped retrieval and typed proposals.
- Exclude raw discussion text from operational logs by default. Log job IDs, durations, costs, counts, errors, model version, and prompt version. Keep audit payload access restricted.
- Deletion and retention are explicit administrative operations, including uploads, derived records, exports, and backups. Artifact immutability means ordinary edits are prohibited; it does not exempt content from authorized deletion requirements.

## 11. Evaluation dataset and acceptance criteria

Begin with 3–5 real cases the product owner understands and has permission to use. Manually annotate the primary question, material options/objections, stated positions, actual outcome where known, source references, and separate issues. Do not assume the actual historical decision is the optimal decision.

Create 20–30 synthetic variants. Preserve each real case's provenance and mark derivatives synthetic. Hold out entire case families to avoid testing on minor rewrites of development examples. An independent human reviews labels; the model generating examples is not the sole judge.

Variants cover quiet objections, repeated unsupported claims, title swaps, polished versus plain phrasing, position reversals, factual/priorities confusion, quoted email duplication, inline replies, mixed timelines, ambiguous authors, genuine tangents, blocking dependencies, missing evidence, prompt injection, and false consensus.

### Release gates

| Capability | Acceptance condition |
|---|---|
| Source grounding | Every referenced passage resolves; attribution and offsets pass validation |
| Material concern coverage | At least 90% recall on human-labelled material objections in the held-out pilot set |
| Positions | No fabricated explicit agreement, confirmation, or decision authority in the held-out set |
| Pushback | Valid corrections change the intended entities and preserve prior history |
| Scope | At least 90% correct tangent/dependency classification on labelled cases; uncertain cases may ask for clarification |
| Fair representation | No unexplained material change in captured arguments when only titles, names, or fluency change |
| Recording | Missing explicit confirmation, missing owner/recorder labels, and stale-revision attempts fail; no LLM path can record a decision |
| Versioning | Reopening and supersession preserve prior snapshots and show the current effective artifact |
| Workspace integrity | Source references and mutations cannot incorrectly bind to another workspace; no workforce-access test is required in Phase 1 |
| End-to-end | Real discussion -> brief -> challenge -> accepted revision -> human-recorded artifact -> search/export |

Targets are proposed pilot gates, not claims of achieved performance. Report numerators and denominators. Failure on fabrication or explicit-recording controls requires fixing the relevant path before release, even if aggregate scores are high.

Measure initial synthesis latency, chat latency, token/cost usage, parser warnings, correction rate, repeated objections, and whether owners can explain the final rationale from the artifact. Proposed usability target: ordinary interaction completes within 20 seconds at p95; long synthesis is asynchronous with visible progress. Establish measured baselines before making speed or quality claims.

## 12. Implementation sequence

1. Foundation: single-company instance, database migrations, storage, jobs, immutable sources, citation viewer. No auth, workforce UID, memberships, or tenancy implementation.
2. Ingestion: transcript/email parsing, quote deduplication, warnings, long-input checkpoints, candidate questions.
3. Initial brief: schema, extraction/merge, confirmed scope, coverage review, revision history.
4. Deliberation: chat, typed proposals, explicit corrections, evidence review, scope classification, diffs.
5. Closure: readiness, manually entered owner/recorder labels, decision preview, transactional recording, reopening/supersession, artifact search/export.
6. Pilot: annotated real cases, synthetic challenge families, workspace-integrity and explicit-recording tests, measured evaluation, iteration.
7. Phase 2: implement workforce identity and access controls only after validating the core deliberation workflow; see section 14.

Deliver a runnable repository with README, .env.example, migrations, fixtures, unit/integration tests, evaluation runner, and one documented end-to-end demo. Implement meaningful tests for provenance, deduplication, mutation validation, concurrency, explicit human recording, immutability, and workspace integrity.

## 13. Configuration and build assumptions

Suggested configuration: DATABASE_URL, OBJECT_STORAGE_ENDPOINT, OBJECT_STORAGE_BUCKET, OBJECT_STORAGE_ACCESS_KEY, OBJECT_STORAGE_SECRET_KEY, LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, COMPANY_NAME, MAX_UPLOAD_BYTES, MAX_WORKSPACE_WORDS, LLM_TIMEOUT_SECONDS, and MAX_JOB_ATTEMPTS. Use server-side secrets management in deployment. COMPANY_NAME is display metadata, not a tenant-routing key. AUTH_ISSUER and AUTH_AUDIENCE are Phase 2 configuration.

Model/provider choice remains configurable. Implement a single provider first behind the adapter. Do not add parallel agents for the MVP: extraction stages and proposal validation are sufficient. Keep extraction, synthesis, evidence review, and deliberation prompts versioned and independently evaluable.

This spec is ready for implementation without claiming the existing Pact repository already follows it. Before modifying that repository, inspect its architecture and map existing components to these requirements; reuse working ingestion, UI, or persistence where appropriate.


## 14. Phase 2 — identity and enterprise access

Build after Phase 1 demonstrates useful deliberation on real cases. This phase adds verified actors and governed access without redesigning the decision brief.

### Additions

- Authenticated User records with stable application IDs. Optional workforce UID and directory/SSO linkage; use immutable provider subject identifiers rather than email as the primary key.
- Membership records and viewer, participant, editor, and admin roles. Admins manage membership and designate decision owners; decision authority is an explicit designation separate from a role.
- Explicit mapping from workspace Participant records to authenticated users. Ambiguous names remain unmapped until reviewed.
- Verified participant confirmations/corrections, distinguished from source assertions and operator-entered changes.
- Owner-authorized recording, supersession, and withdrawal. Backend derives actor identity from the authenticated session, never from a submitted actor label.
- Permission checks for all workspace, source, job, citation, history, decision, search, download, and export operations.
- Permission-filtered retrieval for the LLM so a user cannot expose restricted evidence through chat.
- Access-control and authenticated-owner tests, including access revocation and stale previews.

### Migration from Phase 1

1. Preserve all existing internal UUIDs, source references, revisions, and artifact snapshots.
2. Add nullable user linkage to participant, workspace owner, chat, and event records. Retain original labels.
3. Provision initial admin and explicitly assign workspace memberships; do not grant access by matching display names automatically.
4. Store identity mapping and later verification in separate events/metadata. Never rewrite immutable Phase 1 artifacts or imply their original authority was authenticated.
5. Keep legacy artifacts labelled not_verified_phase_1. If desired, an authenticated owner can explicitly ratify a legacy record with a linked attestation or create a superseding artifact.
6. Add server-side actor and permission checks before enabling multi-user access.

### Phase 2 release gates

All unauthorized recording attempts fail; only the designated authenticated owner can record, supersede, or withdraw. Cross-workspace access tests pass for sources, search, exports, jobs, and LLM retrieval. Mapped participants can confirm their own positions; users cannot impersonate others. Legacy artifacts retain their original provenance and verification status.

The system still serves one company. Multi-company tenant schemas and organization isolation are deferred until a concrete multi-company requirement is approved.
