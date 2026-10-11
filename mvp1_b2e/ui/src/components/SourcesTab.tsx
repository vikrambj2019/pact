import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { BriefData, BriefOption, Job, SourceDocument, Workspace } from '../types';
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

const EVIDENCE_COLOR: Record<string, string> = {
  supported_within_scope: 'bg-green-100 text-green-700',
  contradicted_within_scope: 'bg-red-100 text-red-700',
  conflicting: 'bg-amber-100 text-amber-700',
  unverified: 'bg-gray-100 text-gray-500',
  unresolved: 'bg-gray-100 text-gray-500',
};

type BuildPhase = 'idle' | 'extracting' | 'synthesizing' | 'done';

function BriefSummary({ data }: { data: BriefData }) {
  const b = data.brief;
  const optionById = Object.fromEntries(
    (b.options ?? []).map((o: BriefOption) => [o.id, o])
  );

  return (
    <div className="space-y-4">
      <p className="text-xs text-gray-400">Revision {data.revision_number} · {new Date(data.created_at).toLocaleString()}</p>

      {b.question && (
        <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-4">
          <p className="text-sm font-semibold text-indigo-900">{b.question.text}</p>
          {b.question.scope && <p className="text-xs text-indigo-600 mt-1">Scope: {b.question.scope}</p>}
        </div>
      )}

      {b.emerging_preference && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <p className="text-xs font-semibold text-amber-700 uppercase tracking-wide mb-1">Emerging preference</p>
          <p className="text-sm font-medium text-amber-900">
            {optionById[b.emerging_preference.option_id]?.title ?? b.emerging_preference.option_id}
          </p>
          <p className="text-sm text-amber-700 mt-1">{b.emerging_preference.summary}</p>
        </div>
      )}

      {(b.options ?? []).length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            Options ({b.options!.length})
          </p>
          <div className="grid gap-2">
            {b.options!.map((opt) => (
              <div key={opt.id} className="bg-white border border-gray-200 rounded-md px-3 py-2">
                <p className="text-sm font-medium text-gray-800">{opt.title}</p>
                {opt.description && <p className="text-xs text-gray-500 mt-0.5">{opt.description}</p>}
                <div className="grid grid-cols-2 gap-2 mt-1.5">
                  {(opt.arguments_for ?? []).length > 0 && (
                    <ul className="space-y-0.5">
                      {opt.arguments_for!.map((a, i) => (
                        <li key={i} className="text-xs text-green-700">+ {a}</li>
                      ))}
                    </ul>
                  )}
                  {(opt.arguments_against ?? []).length > 0 && (
                    <ul className="space-y-0.5">
                      {opt.arguments_against!.map((a, i) => (
                        <li key={i} className="text-xs text-red-600">− {a}</li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {(b.claims ?? []).length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            Claims ({b.claims!.length})
          </p>
          <div className="space-y-1.5">
            {b.claims!.map((claim) => (
              <div key={claim.id} className="flex items-start gap-2">
                <span className={`shrink-0 text-xs px-1.5 py-0.5 rounded font-medium ${EVIDENCE_COLOR[claim.evidence_status] ?? 'bg-gray-100 text-gray-500'}`}>
                  {claim.evidence_status.replace(/_/g, ' ')}
                </span>
                <span className="text-sm text-gray-700">{claim.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {(b.blockers ?? []).filter(bl => bl.resolution_status === 'open').length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-md px-3 py-2">
          <p className="text-xs font-semibold text-red-700 mb-1">Open blockers</p>
          {b.blockers!.filter(bl => bl.resolution_status === 'open').map(bl => (
            <p key={bl.id} className="text-xs text-red-600">{bl.text}</p>
          ))}
        </div>
      )}
    </div>
  );
}

export default function SourcesTab({ workspace, onWorkspaceUpdate, onConfirmed }: Props) {
  const [sources, setSources] = useState<SourceDocument[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [activeJobs, setActiveJobs] = useState<Map<string, string>>(new Map());
  const [jobErrors, setJobErrors] = useState<Map<string, string>>(new Map());
  const fileRef = useRef<HTMLInputElement>(null);

  const [showPaste, setShowPaste] = useState(false);
  const [pasteText, setPasteText] = useState('');
  const [pasting, setPasting] = useState(false);

  // Build brief
  const [phase, setPhase] = useState<BuildPhase>('idle');
  const [extractJobId, setExtractJobId] = useState<string | null>(null);
  const [synthJobId, setSynthJobId] = useState<string | null>(null);
  const [buildError, setBuildError] = useState('');
  const [brief, setBrief] = useState<BriefData | null>(null);

  const loadSources = useCallback(() =>
    api.sources.list(workspace.id).then(setSources).catch(() => {}),
    [workspace.id],
  );

  // Load existing brief on mount
  useEffect(() => {
    loadSources();
    api.brief.get(workspace.id)
      .then((b) => { setBrief(b); setPhase('done'); })
      .catch(() => {});
  }, [workspace.id, loadSources]);

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

  async function handleBuild() {
    setBuildError('');
    setPhase('extracting');
    try {
      const res = await api.brief.extract(workspace.id);
      setExtractJobId(res.job_id);
    } catch (err: unknown) {
      setBuildError(err instanceof Error ? err.message : String(err));
      setPhase('idle');
    }
  }

  async function onExtractDone(job: Job) {
    setExtractJobId(null);
    if (job.status === 'failed') {
      setBuildError(job.error_detail ?? 'Extraction failed');
      setPhase('idle');
      return;
    }
    setPhase('synthesizing');
    try {
      const res = await api.brief.synthesize(workspace.id);
      setSynthJobId(res.job_id);
    } catch (err: unknown) {
      setBuildError(err instanceof Error ? err.message : String(err));
      setPhase('idle');
    }
  }

  async function onSynthDone(job: Job) {
    setSynthJobId(null);
    if (job.status === 'failed') {
      setBuildError(job.error_detail ?? 'Synthesis failed');
      setPhase('idle');
      return;
    }
    const b = await api.brief.get(workspace.id).catch(() => null);
    setBrief(b);
    setPhase('done');
    onConfirmed(); // reload workspace
  }

  const hasParsedSource = sources.some((s) => s.parse_status === 'done');
  const canBuild = hasParsedSource;

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
                      {doc.original_filename ?? `Pasted text`}
                    </p>
                    <p className="text-xs text-gray-400 mt-0.5">
                      {doc.word_count != null ? `~${doc.word_count.toLocaleString()} words · ` : ''}{doc.type}
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


      {/* Build brief */}
      {canBuild && (
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
              Decision brief
            </h2>
            {phase === 'done' && (
              <button
                onClick={handleBuild}
                className="bg-white border border-gray-300 text-sm px-3 py-1.5 rounded-md hover:bg-gray-50 transition-colors"
              >
                Rebuild brief
              </button>
            )}
          </div>

          {phase === 'idle' && (
            <button
              onClick={handleBuild}
              className="bg-indigo-600 text-white text-sm px-5 py-2 rounded-md hover:bg-indigo-700 transition-colors"
            >
              Build brief
            </button>
          )}

          {(phase === 'extracting' || phase === 'synthesizing') && (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${phase === 'extracting' ? 'bg-indigo-500 animate-pulse' : 'bg-green-500'}`} />
                <span className={`text-sm ${phase === 'extracting' ? 'text-indigo-700 font-medium' : 'text-gray-400'}`}>Extract</span>
              </div>
              <span className="text-gray-300">→</span>
              <div className="flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${phase === 'synthesizing' ? 'bg-indigo-500 animate-pulse' : 'bg-gray-200'}`} />
                <span className={`text-sm ${phase === 'synthesizing' ? 'text-indigo-700 font-medium' : 'text-gray-400'}`}>Synthesize</span>
              </div>
              <span className="text-xs text-gray-400 ml-2">
                {phase === 'extracting' ? 'Reading sources…' : 'Writing brief…'}
              </span>
            </div>
          )}

          {extractJobId && (
            <div className="hidden">
              <JobPoller jobId={extractJobId} onDone={onExtractDone} onFailed={onExtractDone} />
            </div>
          )}
          {synthJobId && (
            <div className="hidden">
              <JobPoller jobId={synthJobId} onDone={onSynthDone} onFailed={onSynthDone} />
            </div>
          )}

          {buildError && <p className="text-sm text-red-600 mt-2">{buildError}</p>}

          {phase === 'done' && brief && (
            <div className="mt-4">
              <BriefSummary data={brief} />
            </div>
          )}
        </section>
      )}
    </div>
  );
}
