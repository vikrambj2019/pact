export interface Workspace {
  id: string;
  title: string;
  primary_question: string | null;
  scope: string | null;
  decision_rule: string | null;
  lifecycle_status: 'draft' | 'deliberating' | 'decision_recorded';
  decision_owner_label: string | null;
  current_revision_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface SourceDocument {
  id: string;
  workspace_id: string;
  type: string;
  original_filename: string | null;
  parse_status: 'pending' | 'done' | 'failed';
  word_count: number | null;
  warnings: string[];
  created_at: string;
}

export interface Job {
  id: string;
  type: string;
  status: 'pending' | 'running' | 'done' | 'failed';
  output_payload: Record<string, unknown> | null;
  error_code: string | null;
  error_detail: string | null;
  created_at: string;
  updated_at: string;
}

export interface EvidenceLink {
  source_unit_id: string;
  source_document_id: string;
  quote: string;
  start_char: number;
  end_char: number;
  verdict: 'supports' | 'contradicts' | 'limits';
  explanation: string;
}

export interface Claim {
  id: string;
  text: string;
  kind: string;
  author_label?: string;
  evidence_status: string;
  evidence_links?: EvidenceLink[];
  source_unit_ids: string[];
}

export interface BriefOption {
  id: string;
  title: string;
  description?: string;
  arguments_for?: string[];
  arguments_against?: string[];
  source_unit_ids: string[];
}

export interface BriefPosition {
  id: string;
  participant_label: string;
  option_id: string;
  stance: 'support' | 'oppose' | 'conditional' | 'undecided' | 'unknown';
  conditions?: string;
}

export interface EmergingPreference {
  option_id: string;
  summary: string;
}

export interface BriefSnapshot {
  question?: { text: string; scope: string | null };
  criteria?: Array<{ id: string; name: string; description?: string }>;
  constraints?: Array<{ id: string; text: string; kind: string; basis?: string }>;
  options?: BriefOption[];
  claims?: Claim[];
  positions?: BriefPosition[];
  objections?: Array<{ id: string; text: string; resolution_status: string; author_label?: string }>;
  blockers?: Array<{ id: string; text: string; kind: string; resolution_status: string }>;
  actions?: Array<{ id: string; text: string; owner_label?: string }>;
  separate_issues?: Array<{ id: string; title: string; rationale: string }>;
  emerging_preference?: EmergingPreference | null;
}

export interface BriefData {
  workspace_id: string;
  revision_id: string;
  revision_number: number;
  created_at: string;
  brief: BriefSnapshot;
}

export interface ConversationMessage {
  id: string;
  role: 'human' | 'assistant';
  author_label: string;
  text: string;
  linked_revision_id: string | null;
  created_at: string;
}

export interface ProposedOperation {
  type: string;
  [key: string]: unknown;
}

export interface ChangeProposal {
  id: string;
  workspace_id: string;
  base_revision_id: string;
  status: 'pending' | 'accepted' | 'rejected';
  rationale: string;
  typed_operations: ProposedOperation[];
  rejection_rationale: string | null;
  created_at: string;
  resolved_at: string | null;
}

export interface MessageResponse {
  message_id: string;
  response_text: string;
  request_type: string;
  follow_up_question?: string | null;
  proposal?: { id: string; status: string; rationale: string; operations: ProposedOperation[] } | null;
}

export interface DecisionArtifact {
  artifact_id: string;
  artifact_version: number;
  workspace_id: string;
  lifecycle_status?: string;
  decision_owner_label: string;
  recorded_by_label: string;
  authority_verification?: string;
  recorded_at: string;
  originating_revision_id?: string;
  supersedes_artifact_id: string | null;
  withdrawn_at: string | null;
  immutable_snapshot?: Record<string, unknown>;
}
