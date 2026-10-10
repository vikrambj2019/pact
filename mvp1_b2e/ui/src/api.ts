import type {
  BriefData, ChangeProposal, ConversationMessage, DecisionArtifact,
  Job, MessageResponse, SourceDocument, Workspace,
} from './types';

const BASE = '/api/v1';

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const msg = typeof err.detail === 'string' ? err.detail : JSON.stringify(err);
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return req<T>(path, {
    method: 'POST',
    body: body != null ? JSON.stringify(body) : undefined,
  });
}

export const api = {
  workspaces: {
    list: () => req<Workspace[]>('/workspaces'),
    get: (id: string) => req<Workspace>(`/workspaces/${id}`),
    create: (body: { title: string; decision_question?: string; decision_rule?: string }) =>
      post<Workspace>('/workspaces', body),
    confirmQuestion: (
      id: string,
      body: {
        primary_question: string;
        scope?: string;
        decision_owner_label?: string;
        decision_rule?: string;
        expected_revision: number;
      },
    ) => post<Workspace>(`/workspaces/${id}/question-confirmation`, body),
  },

  sources: {
    list: (wsId: string) => req<SourceDocument[]>(`/workspaces/${wsId}/sources`),
    upload: async (wsId: string, file: File): Promise<{ source_document_id: string; job_id: string }> => {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`${BASE}/workspaces/${wsId}/sources`, { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(typeof err.detail === 'string' ? err.detail : JSON.stringify(err));
      }
      return res.json();
    },
    uploadPasted: async (wsId: string, text: string): Promise<{ source_document_id: string; job_id: string }> => {
      const form = new FormData();
      form.append('pasted_text', text);
      const res = await fetch(`${BASE}/workspaces/${wsId}/sources`, { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(typeof err.detail === 'string' ? err.detail : JSON.stringify(err));
      }
      return res.json();
    },
  },

  jobs: {
    get: (id: string) => req<Job>(`/jobs/${id}`),
  },

  brief: {
    extract: (wsId: string) =>
      post<{ job_id: string; status: string }>(`/workspaces/${wsId}/brief/extract`),
    synthesize: (wsId: string) =>
      post<{ job_id: string; status: string }>(`/workspaces/${wsId}/brief/synthesize`),
    get: (wsId: string) => req<BriefData>(`/workspaces/${wsId}/brief`),
  },

  deliberation: {
    messages: (wsId: string) => req<ConversationMessage[]>(`/workspaces/${wsId}/messages`),
    send: (wsId: string, body: { author_label: string; text: string; expected_revision?: number }) =>
      post<MessageResponse>(`/workspaces/${wsId}/messages`, body),
    getProposal: (wsId: string, proposalId: string) =>
      req<ChangeProposal>(`/workspaces/${wsId}/proposals/${proposalId}`),
    acceptProposal: (wsId: string, proposalId: string, expectedRevision: number) =>
      post<{ new_revision_number: number; new_revision_id: string }>(
        `/workspaces/${wsId}/proposals/${proposalId}/accept`,
        { expected_revision: expectedRevision },
      ),
    rejectProposal: (wsId: string, proposalId: string, rationale: string) =>
      post<{ status: string }>(`/workspaces/${wsId}/proposals/${proposalId}/reject`, { rationale }),
  },

  claims: {
    review: (wsId: string, claimId: string) =>
      post<{ job_id: string; claim_id: string; status: string }>(
        `/workspaces/${wsId}/claims/${claimId}/review`,
      ),
  },

  decisions: {
    preview: (wsId: string) =>
      req<{ preview: Record<string, unknown>; revision_number: number; lifecycle_status: string }>(
        `/workspaces/${wsId}/decisions/preview`,
      ),
    record: (
      wsId: string,
      body: {
        decision_owner_label: string;
        recorded_by_label: string;
        expected_revision: number;
        rationale: string;
        selected_option_id?: string;
        notes?: string;
      },
    ) => post<DecisionArtifact>(`/workspaces/${wsId}/decisions`, body),
    list: (wsId: string) =>
      req<{ decisions: DecisionArtifact[]; lifecycle_status: string }>(
        `/workspaces/${wsId}/decisions`,
      ),
    get: (wsId: string, artifactId: string) =>
      req<DecisionArtifact>(`/workspaces/${wsId}/decisions/${artifactId}`),
  },
};
