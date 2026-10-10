import { useCallback, useEffect, useState } from 'react';
import { api } from '../api';
import type { BriefData, BriefOption, Claim, Job, Workspace } from '../types';
import JobPoller from './JobPoller';

interface Props {
  workspace: Workspace;
  onWorkspaceUpdate: () => void;
}

const EVIDENCE_COLOR: Record<string, string> = {
  unverified: 'bg-gray-100 text-gray-500',
  unresolved: 'bg-gray-100 text-gray-500',
  supported_within_scope: 'bg-green-100 text-green-700',
  contradicted_within_scope: 'bg-red-100 text-red-700',
  conflicting: 'bg-amber-100 text-amber-700',
};

function Section({ title, children, count }: { title: string; children: React.ReactNode; count?: number }) {
  const [open, setOpen] = useState(true);
  if (!count && count !== undefined) return null;
  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
      >
        <span className="text-sm font-semibold text-gray-700">{title}</span>
        <div className="flex items-center gap-2">
          {count != null && <span className="text-xs text-gray-400">{count}</span>}
          <span className="text-gray-400 text-xs">{open ? '▲' : '▼'}</span>
        </div>
      </button>
      {open && <div className="p-4 space-y-3 bg-white">{children}</div>}
    </div>
  );
}

function ClaimCard({ claim, workspaceId, onReviewed }: {
  claim: Claim;
  workspaceId: string;
  onReviewed: () => void;
}) {
  const [reviewJobId, setReviewJobId] = useState<string | null>(null);
  const [error, setError] = useState('');

  async function startReview() {
    setError('');
    try {
      const res = await api.claims.review(workspaceId, claim.id);
      setReviewJobId(res.job_id);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="border border-gray-100 rounded-md p-3">
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm text-gray-800">{claim.text}</p>
          {claim.author_label && (
            <p className="text-xs text-gray-400 mt-0.5">{claim.author_label}</p>
          )}
          {claim.evidence_links && claim.evidence_links.length > 0 && (
            <div className="mt-2 space-y-1">
              {claim.evidence_links.map((link, i) => (
                <div key={i} className={`text-xs px-2 py-1 rounded flex gap-2 ${link.verdict === 'supports' ? 'bg-green-50 text-green-700' : link.verdict === 'contradicts' ? 'bg-red-50 text-red-700' : 'bg-gray-50 text-gray-600'}`}>
                  <span className="font-medium capitalize">{link.verdict}</span>
                  <span className="italic truncate">"{link.quote}"</span>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="shrink-0 flex flex-col items-end gap-1.5">
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${EVIDENCE_COLOR[claim.evidence_status] ?? 'bg-gray-100 text-gray-500'}`}>
            {claim.evidence_status.replace(/_/g, ' ')}
          </span>
          {reviewJobId ? (
            <JobPoller
              jobId={reviewJobId}
              onDone={() => { setReviewJobId(null); onReviewed(); }}
              onFailed={() => setReviewJobId(null)}
            />
          ) : (
            <button
              onClick={startReview}
              className="text-xs text-indigo-600 hover:underline"
            >
              Run review
            </button>
          )}
        </div>
      </div>
      {error && <p className="text-xs text-red-600 mt-1">{error}</p>}
    </div>
  );
}

export default function BriefTab({ workspace, onWorkspaceUpdate }: Props) {
  const [brief, setBrief] = useState<BriefData | null>(null);
  const [briefError, setBriefError] = useState('');
  const [extractJobId, setExtractJobId] = useState<string | null>(null);
  const [synthJobId, setSynthJobId] = useState<string | null>(null);
  const [actionError, setActionError] = useState('');

  const loadBrief = useCallback(() => {
    api.brief.get(workspace.id)
      .then(setBrief)
      .catch((e) => setBriefError(e.message));
  }, [workspace.id]);

  useEffect(() => { loadBrief(); }, [loadBrief]);

  async function handleExtract() {
    setActionError('');
    try {
      const res = await api.brief.extract(workspace.id);
      setExtractJobId(res.job_id);
    } catch (e: unknown) {
      setActionError(e instanceof Error ? e.message : String(e));
    }
  }

  async function handleSynthesize() {
    setActionError('');
    try {
      const res = await api.brief.synthesize(workspace.id);
      setSynthJobId(res.job_id);
    } catch (e: unknown) {
      setActionError(e instanceof Error ? e.message : String(e));
    }
  }

  function onJobDone(job: Job) {
    setExtractJobId(null);
    setSynthJobId(null);
    if (job.status === 'failed') {
      setActionError(job.error_detail ?? `${job.type} job failed`);
    } else {
      onWorkspaceUpdate();
      loadBrief();
    }
  }

  const b = brief?.brief;
  const optionById = Object.fromEntries((b?.options ?? []).map((o: BriefOption) => [o.id, o]));
  const isDraft = workspace.lifecycle_status === 'draft';

  return (
    <div className="space-y-6">
      {/* Actions */}
      <div className="flex flex-wrap items-center gap-3">
        {isDraft ? (
          <p className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-md px-3 py-2">
            Confirm the decision question in the Sources tab before extracting.
          </p>
        ) : (
          <>
            {extractJobId ? (
              <JobPoller jobId={extractJobId} onDone={onJobDone} onFailed={onJobDone} />
            ) : (
              <button
                onClick={handleExtract}
                className="bg-white border border-gray-300 text-sm px-4 py-2 rounded-md hover:bg-gray-50 transition-colors"
              >
                Extract
              </button>
            )}
            {synthJobId ? (
              <JobPoller jobId={synthJobId} onDone={onJobDone} onFailed={onJobDone} />
            ) : (
              <button
                onClick={handleSynthesize}
                className="bg-indigo-600 text-white text-sm px-4 py-2 rounded-md hover:bg-indigo-700 transition-colors"
              >
                Synthesize brief
              </button>
            )}
          </>
        )}
        {actionError && <p className="text-sm text-red-600">{actionError}</p>}
      </div>

      {briefError && !brief && (
        <p className="text-sm text-gray-400">{briefError === 'Not Found' || briefError.includes('404') ? 'No brief yet — run extract then synthesize.' : briefError}</p>
      )}

      {b && (
        <div className="space-y-4">
          <div className="text-xs text-gray-400">
            Revision {brief!.revision_number} · {new Date(brief!.created_at).toLocaleString()}
          </div>

          {b.question && (
            <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-4">
              <p className="text-sm font-semibold text-indigo-900">{b.question.text}</p>
              {b.question.scope && <p className="text-xs text-indigo-600 mt-1">Scope: {b.question.scope}</p>}
            </div>
          )}

          {b.emerging_preference && (
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <p className="text-xs font-semibold text-amber-700 uppercase tracking-wide mb-1">Emerging preference</p>
              <p className="text-sm text-amber-900">
                {optionById[b.emerging_preference.option_id]?.title ?? b.emerging_preference.option_id}
              </p>
              <p className="text-sm text-amber-700 mt-1">{b.emerging_preference.summary}</p>
            </div>
          )}

          {(b.options ?? []).length > 0 && (
            <Section title="Options" count={b.options!.length}>
              {b.options!.map((opt) => (
                <div key={opt.id} className="border border-gray-100 rounded-md p-3">
                  <p className="text-sm font-semibold text-gray-800">{opt.title}</p>
                  {opt.description && <p className="text-sm text-gray-600 mt-1">{opt.description}</p>}
                  <div className="grid grid-cols-2 gap-3 mt-2">
                    {(opt.arguments_for ?? []).length > 0 && (
                      <div>
                        <p className="text-xs font-medium text-green-600 mb-1">For</p>
                        <ul className="space-y-0.5">{opt.arguments_for!.map((a, i) => <li key={i} className="text-xs text-gray-600">+ {a}</li>)}</ul>
                      </div>
                    )}
                    {(opt.arguments_against ?? []).length > 0 && (
                      <div>
                        <p className="text-xs font-medium text-red-600 mb-1">Against</p>
                        <ul className="space-y-0.5">{opt.arguments_against!.map((a, i) => <li key={i} className="text-xs text-gray-600">− {a}</li>)}</ul>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </Section>
          )}

          {(b.claims ?? []).length > 0 && (
            <Section title="Claims" count={b.claims!.length}>
              {b.claims!.map((claim) => (
                <ClaimCard
                  key={claim.id}
                  claim={claim}
                  workspaceId={workspace.id}
                  onReviewed={loadBrief}
                />
              ))}
            </Section>
          )}

          {(b.positions ?? []).length > 0 && (
            <Section title="Positions" count={b.positions!.length}>
              {b.positions!.map((pos) => (
                <div key={pos.id} className="flex items-start gap-3 text-sm">
                  <span className="font-medium text-gray-700 min-w-28">{pos.participant_label}</span>
                  <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${pos.stance === 'support' ? 'bg-green-100 text-green-700' : pos.stance === 'oppose' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'}`}>
                    {pos.stance}
                  </span>
                  <span className="text-gray-500">{optionById[pos.option_id]?.title ?? pos.option_id}</span>
                  {pos.conditions && <span className="text-gray-400 italic text-xs">({pos.conditions})</span>}
                </div>
              ))}
            </Section>
          )}

          {(b.criteria ?? []).length > 0 && (
            <Section title="Criteria" count={b.criteria!.length}>
              {b.criteria!.map((c) => (
                <div key={c.id} className="text-sm">
                  <span className="font-medium text-gray-700">{c.name}</span>
                  {c.description && <span className="text-gray-500"> — {c.description}</span>}
                </div>
              ))}
            </Section>
          )}

          {(b.constraints ?? []).length > 0 && (
            <Section title="Constraints" count={b.constraints!.length}>
              {b.constraints!.map((c) => (
                <div key={c.id} className="text-sm flex gap-2">
                  <span className={`shrink-0 text-xs px-1.5 py-0.5 rounded font-medium ${c.kind === 'mandatory' ? 'bg-red-100 text-red-600' : 'bg-gray-100 text-gray-500'}`}>{c.kind}</span>
                  <span className="text-gray-700">{c.text}</span>
                </div>
              ))}
            </Section>
          )}

          {(b.objections ?? []).length > 0 && (
            <Section title="Objections" count={b.objections!.length}>
              {b.objections!.map((o) => (
                <div key={o.id} className="text-sm flex gap-2">
                  <span className={`shrink-0 text-xs px-1.5 py-0.5 rounded font-medium ${o.resolution_status === 'resolved' ? 'bg-green-100 text-green-600' : 'bg-amber-100 text-amber-600'}`}>{o.resolution_status}</span>
                  <span className="text-gray-700">{o.text}</span>
                </div>
              ))}
            </Section>
          )}

          {(b.blockers ?? []).length > 0 && (
            <Section title="Blockers" count={b.blockers!.length}>
              {b.blockers!.map((bl) => (
                <div key={bl.id} className="text-sm flex gap-2">
                  <span className={`shrink-0 text-xs px-1.5 py-0.5 rounded font-medium ${bl.resolution_status === 'resolved' ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'}`}>{bl.resolution_status}</span>
                  <span className="text-gray-700">{bl.text}</span>
                </div>
              ))}
            </Section>
          )}

          {(b.actions ?? []).length > 0 && (
            <Section title="Actions" count={b.actions!.length}>
              {b.actions!.map((a) => (
                <div key={a.id} className="text-sm flex gap-2">
                  <span className="text-gray-700">{a.text}</span>
                  {a.owner_label && <span className="text-gray-400 text-xs">[{a.owner_label}]</span>}
                </div>
              ))}
            </Section>
          )}

          {(b.separate_issues ?? []).length > 0 && (
            <Section title="Separate issues" count={b.separate_issues!.length}>
              {b.separate_issues!.map((s) => (
                <div key={s.id} className="text-sm">
                  <p className="font-medium text-gray-700">{s.title}</p>
                  <p className="text-gray-500 text-xs mt-0.5">{s.rationale}</p>
                </div>
              ))}
            </Section>
          )}
        </div>
      )}
    </div>
  );
}
