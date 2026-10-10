import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { Job, SourceDocument, Workspace } from '../types';
import JobPoller from './JobPoller';

interface Props {
  workspace: Workspace;
  onWorkspaceUpdate: (ws: Workspace) => void;
  onConfirmed: () => void;
}

const PARSE_STATUS_COLOR: Record<string, string> = {
  pending: 'text-amber-600',
  done: 'text-green-600',
  failed: 'text-red-600',
};

export default function SourcesTab({ workspace, onWorkspaceUpdate, onConfirmed }: Props) {
  const [sources, setSources] = useState<SourceDocument[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [activeJobs, setActiveJobs] = useState<Map<string, string>>(new Map()); // docId -> jobId
  const [jobErrors, setJobErrors] = useState<Map<string, string>>(new Map()); // docId -> error
  const fileRef = useRef<HTMLInputElement>(null);

  // Pasted text
  const [showPaste, setShowPaste] = useState(false);
  const [pasteText, setPasteText] = useState('');
  const [pasting, setPasting] = useState(false);

  // Question confirmation form
  const [question, setQuestion] = useState(workspace.primary_question ?? '');
  const [scope, setScope] = useState(workspace.scope ?? '');
  const [owner, setOwner] = useState(workspace.decision_owner_label ?? '');
  const [rule, setRule] = useState(workspace.decision_rule ?? '');
  const [confirming, setConfirming] = useState(false);
  const [confirmError, setConfirmError] = useState('');

  const loadSources = () =>
    api.sources.list(workspace.id).then(setSources).catch(() => {});

  useEffect(() => { loadSources(); }, [workspace.id]);

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError('');
    try {
      const res = await api.sources.upload(workspace.id, file);
      setActiveJobs((m) => new Map(m).set(res.source_document_id, res.job_id));
      await loadSources();
    } catch (err: unknown) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  }

  async function handlePaste(e: React.FormEvent) {
    e.preventDefault();
    if (!pasteText.trim()) return;
    setPasting(true);
    setUploadError('');
    try {
      const res = await api.sources.uploadPasted(workspace.id, pasteText.trim());
      setActiveJobs((m) => new Map(m).set(res.source_document_id, res.job_id));
      setPasteText('');
      setShowPaste(false);
      await loadSources();
    } catch (err: unknown) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setPasting(false);
    }
  }

  function handleJobDone(docId: string, job: Job) {
    setActiveJobs((m) => { const n = new Map(m); n.delete(docId); return n; });
    if (job.status === 'failed') {
      setJobErrors((m) => new Map(m).set(docId, job.error_detail ?? 'Parse failed'));
    }
    loadSources();
  }

  async function handleConfirm(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setConfirming(true);
    setConfirmError('');
    try {
      const updated = await api.workspaces.confirmQuestion(workspace.id, {
        primary_question: question.trim(),
        scope: scope.trim() || undefined,
        decision_owner_label: owner.trim() || undefined,
        decision_rule: rule.trim() || undefined,
        expected_revision: 0,
      });
      onWorkspaceUpdate(updated);
      onConfirmed();
    } catch (err: unknown) {
      setConfirmError(err instanceof Error ? err.message : String(err));
    } finally {
      setConfirming(false);
    }
  }

  const isDraft = workspace.lifecycle_status === 'draft';

  return (
    <div className="space-y-8">
      {/* Upload */}
      <section>
        <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">Source documents</h2>
        <div className="flex gap-2 mb-3">
          <label className={`cursor-pointer bg-white border border-gray-300 text-sm px-4 py-2 rounded-md hover:bg-gray-50 transition-colors ${uploading ? 'opacity-50 pointer-events-none' : ''}`}>
            {uploading ? 'Uploading…' : 'Upload file (.txt, .md, .eml)'}
            <input ref={fileRef} type="file" accept=".txt,.md,.eml" className="hidden" onChange={handleFileUpload} />
          </label>
          <button
            onClick={() => setShowPaste(!showPaste)}
            className="bg-white border border-gray-300 text-sm px-4 py-2 rounded-md hover:bg-gray-50 transition-colors"
          >
            Paste text
          </button>
        </div>

        {showPaste && (
          <form onSubmit={handlePaste} className="mb-3 space-y-2">
            <textarea
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 h-32 resize-y"
              placeholder="Paste email, transcript, or notes…"
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
            />
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={pasting || !pasteText.trim()}
                className="bg-indigo-600 text-white text-sm px-3 py-1.5 rounded-md hover:bg-indigo-700 disabled:opacity-50 transition-colors"
              >
                {pasting ? 'Saving…' : 'Save'}
              </button>
              <button type="button" onClick={() => { setShowPaste(false); setPasteText(''); }} className="text-sm text-gray-500 hover:text-gray-700">Cancel</button>
            </div>
          </form>
        )}

        {uploadError && <p className="text-sm text-red-600 mb-2">{uploadError}</p>}

        {sources.length === 0 ? (
          <p className="text-sm text-gray-400">No sources uploaded yet.</p>
        ) : (
          <ul className="space-y-2">
            {sources.map((doc) => {
              const jobId = activeJobs.get(doc.id);
              return (
                <li key={doc.id} className="bg-white border border-gray-200 rounded-md px-4 py-3 flex items-center justify-between gap-4">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-gray-800 truncate">
                      {doc.original_filename ?? `Pasted text (${doc.type})`}
                    </p>
                    <p className="text-xs text-gray-400 mt-0.5">
                      {doc.word_count != null ? `~${doc.word_count.toLocaleString()} words · ` : ''}
                      {doc.type}
                    </p>
                  </div>
                  <div className="shrink-0">
                    {jobId ? (
                      <JobPoller
                        jobId={jobId}
                        onDone={(job) => handleJobDone(doc.id, job)}
                        onFailed={(job) => handleJobDone(doc.id, job)}
                      />
                    ) : (
                      <div className="text-right">
                        <span className={`text-xs font-medium ${PARSE_STATUS_COLOR[doc.parse_status] ?? 'text-gray-500'}`}>
                          {doc.parse_status}
                        </span>
                        {jobErrors.get(doc.id) && (
                          <p className="text-xs text-red-500 mt-0.5 max-w-48 text-right">{jobErrors.get(doc.id)}</p>
                        )}
                      </div>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {/* Question confirmation (only in draft) */}
      {isDraft && (
        <section>
          <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">Confirm decision question</h2>
          <p className="text-sm text-gray-500 mb-4">
            Confirm the question and scope before running extraction. This advances the workspace to <em>Deliberating</em>.
          </p>
          <form onSubmit={handleConfirm} className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Primary question *</label>
              <input
                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                placeholder="e.g. Which CRM should we adopt?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Scope</label>
              <input
                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                placeholder="e.g. North America sales team only"
                value={scope}
                onChange={(e) => setScope(e.target.value)}
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Decision owner</label>
                <input
                  className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="e.g. CTO"
                  value={owner}
                  onChange={(e) => setOwner(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Decision rule</label>
                <input
                  className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="e.g. Steering committee consensus"
                  value={rule}
                  onChange={(e) => setRule(e.target.value)}
                />
              </div>
            </div>
            {confirmError && <p className="text-sm text-red-600">{confirmError}</p>}
            <button
              type="submit"
              disabled={confirming || !question.trim()}
              className="bg-indigo-600 text-white text-sm px-4 py-2 rounded-md hover:bg-indigo-700 disabled:opacity-50 transition-colors"
            >
              {confirming ? 'Confirming…' : 'Confirm question'}
            </button>
          </form>
        </section>
      )}
    </div>
  );
}
