import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { BriefOption, DecisionArtifact, Workspace } from '../types';

interface Props {
  workspace: Workspace;
  onWorkspaceUpdate: () => void;
}

export default function DecisionTab({ workspace, onWorkspaceUpdate }: Props) {
  const [artifacts, setArtifacts] = useState<DecisionArtifact[]>([]);
  const [options, setOptions] = useState<BriefOption[]>([]);
  const [revisionNumber, setRevisionNumber] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const autoFilledRef = useRef(false);

  // Form state
  const [owner, setOwner] = useState(workspace.decision_owner_label ?? '');
  const [recorder, setRecorder] = useState('');
  const [rationale, setRationale] = useState('');
  const [selectedOptionId, setSelectedOptionId] = useState('');
  const [notes, setNotes] = useState('');
  const [recording, setRecording] = useState(false);
  const [recordError, setRecordError] = useState('');
  const [showPreview, setShowPreview] = useState(false);

  // Selected artifact detail
  const [expandedArtifact, setExpandedArtifact] = useState<string | null>(null);
  const [detailCache, setDetailCache] = useState<Map<string, DecisionArtifact>>(new Map());

  const load = useCallback(() => {
    const p1 = api.decisions.list(workspace.id)
      .then((res) => setArtifacts(res.decisions))
      .catch(() => {});
    const p2 = api.brief.get(workspace.id)
      .then((b) => {
        setOptions(b.brief.options ?? []);
        setRevisionNumber(b.revision_number);
        // Auto-fill from emerging_preference on first load
        if (!autoFilledRef.current && b.brief.emerging_preference) {
          autoFilledRef.current = true;
          const ep = b.brief.emerging_preference;
          if (ep.option_id) setSelectedOptionId(ep.option_id);
          if (ep.summary) setRationale(ep.summary);
        }
      })
      .catch(() => {});
    Promise.all([p1, p2]).finally(() => setLoading(false));
  }, [workspace.id]);

  useEffect(() => { load(); }, [load]);

  async function handleRecord(e: React.FormEvent) {
    e.preventDefault();
    if (!owner.trim() || !recorder.trim() || !rationale.trim() || revisionNumber == null) return;
    setRecording(true);
    setRecordError('');
    try {
      await api.decisions.record(workspace.id, {
        decision_owner_label: owner.trim(),
        recorded_by_label: recorder.trim(),
        expected_revision: revisionNumber,
        rationale: rationale.trim(),
        selected_option_id: selectedOptionId || undefined,
        notes: notes.trim() || undefined,
      });
      await load();
      onWorkspaceUpdate();
      setRationale('');
      setNotes('');
      setSelectedOptionId('');
      setShowPreview(false);
      autoFilledRef.current = false;
    } catch (e: unknown) {
      setRecordError(e instanceof Error ? e.message : String(e));
    } finally {
      setRecording(false);
    }
  }

  if (loading) return <p className="text-sm text-gray-400">Loading…</p>;

  const canRecord = workspace.lifecycle_status !== 'draft' && workspace.current_revision_id;
  const selectedOption = options.find((o) => o.id === selectedOptionId);

  return (
    <div className="space-y-8">
      {/* Record form */}
      {canRecord && (
        <section>
          <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-4">Record decision</h2>
          <form onSubmit={handleRecord} className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Decision owner *</label>
                <input
                  className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="e.g. CTO"
                  value={owner}
                  onChange={(e) => setOwner(e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Recorded by *</label>
                <input
                  className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="Your name"
                  value={recorder}
                  onChange={(e) => setRecorder(e.target.value)}
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Selected option <span className="text-gray-400 font-normal">(optional — leave blank for defer)</span>
              </label>
              <select
                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white"
                value={selectedOptionId}
                onChange={(e) => setSelectedOptionId(e.target.value)}
              >
                <option value="">— None / Defer —</option>
                {options.map((opt) => (
                  <option key={opt.id} value={opt.id}>{opt.title}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Rationale *</label>
              <textarea
                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 h-24 resize-y"
                placeholder="Why was this decision made? What evidence or discussion led to this conclusion?"
                value={rationale}
                onChange={(e) => setRationale(e.target.value)}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Notes <span className="text-gray-400 font-normal">(optional)</span>
              </label>
              <textarea
                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 h-16 resize-y"
                placeholder="Meeting notes, conditions, follow-ups…"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
              />
            </div>

            {recordError && <p className="text-sm text-red-600">{recordError}</p>}

            <div className="flex items-center gap-3">
              <button
                type="submit"
                disabled={recording || !owner.trim() || !recorder.trim() || !rationale.trim()}
                className="bg-green-600 text-white text-sm px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                {recording ? 'Recording…' : 'Record decision'}
              </button>
              <button
                type="button"
                onClick={() => setShowPreview((v) => !v)}
                className="text-sm text-indigo-600 hover:text-indigo-800 underline"
              >
                {showPreview ? 'Hide preview' : 'Preview artifact'}
              </button>
            </div>
          </form>

          {/* Live artifact preview — renders from current form state, no API call */}
          {showPreview && (
            <div className="mt-4 bg-indigo-50 border border-indigo-200 rounded-lg p-4">
              <p className="text-xs font-semibold text-indigo-600 uppercase tracking-wide mb-3">Artifact preview (live)</p>
              <div className="space-y-2">
                <Row label="Question" value={workspace.primary_question ?? undefined} />
                <Row label="Scope" value={workspace.scope ?? undefined} />
                <Row label="Decision rule" value={workspace.decision_rule ?? undefined} />
                <Row label="Revision" value={revisionNumber != null ? String(revisionNumber) : undefined} />
                <Row label="Selected option" value={selectedOption?.title ?? '— None / Defer —'} />
                <Row label="Decision owner" value={owner.trim() || undefined} />
                <Row label="Recorded by" value={recorder.trim() || undefined} />
                <Row label="Rationale" value={rationale.trim() || undefined} />
                {notes.trim() && <Row label="Notes" value={notes.trim()} />}
              </div>
            </div>
          )}
        </section>
      )}

      {!canRecord && (
        <div className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-md px-4 py-3">
          Synthesize a brief before recording a decision.
        </div>
      )}

      {/* History */}
      {artifacts.length > 0 && (
        <section>
          <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-4">Decision history</h2>
          <ul className="space-y-3">
            {artifacts.map((a) => (
              <li key={a.artifact_id} className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                <button
                  onClick={() => {
                    const next = expandedArtifact === a.artifact_id ? null : a.artifact_id;
                    setExpandedArtifact(next);
                    if (next && !detailCache.has(next)) {
                      api.decisions.get(workspace.id, next)
                        .then((d) => setDetailCache((m) => new Map(m).set(next, d)))
                        .catch(() => {});
                    }
                  }}
                  className="w-full flex items-start justify-between px-4 py-3 text-left hover:bg-gray-50 transition-colors"
                >
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-800">v{a.artifact_version}</span>
                      {a.withdrawn_at && (
                        <span className="text-xs bg-red-100 text-red-600 px-1.5 py-0.5 rounded">withdrawn</span>
                      )}
                    </div>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {a.decision_owner_label} · recorded by {a.recorded_by_label} · {new Date(a.recorded_at).toLocaleString()}
                    </p>
                    {a.supersedes_artifact_id && (
                      <p className="text-xs text-gray-400 mt-0.5">Supersedes a previous decision</p>
                    )}
                  </div>
                  <span className="text-xs text-gray-400 shrink-0 mt-0.5">{expandedArtifact === a.artifact_id ? '▲' : '▼'}</span>
                </button>

                {expandedArtifact === a.artifact_id && (() => {
                  const detail = detailCache.get(a.artifact_id);
                  if (!detail) {
                    return (
                      <div className="border-t border-gray-100 px-4 py-3 bg-gray-50">
                        <span className="text-xs text-gray-400">Loading…</span>
                      </div>
                    );
                  }
                  const snap = detail.immutable_snapshot ?? {};
                  return (
                    <div className="border-t border-gray-100 px-4 py-3 space-y-2 bg-gray-50">
                      <Row label="Selected option" value={(snap.selected_option_title as string | null) ?? '— none —'} />
                      <Row label="Rationale" value={snap.rationale as string} />
                      <Row label="Notes" value={snap.notes as string | undefined} />
                      <Row label="Question" value={snap.primary_question as string} />
                      <Row label="Revision" value={String(snap.revision_number)} />
                    </div>
                  );
                })()}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value?: string | null }) {
  if (!value) return null;
  return (
    <div className="grid grid-cols-3 gap-2 text-sm">
      <span className="text-gray-500 font-medium">{label}</span>
      <span className="col-span-2 text-gray-800">{String(value)}</span>
    </div>
  );
}
