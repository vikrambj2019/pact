import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { Workspace } from '../types';

const LIFECYCLE_LABEL: Record<string, string> = {
  draft: 'Draft',
  deliberating: 'Deliberating',
  decision_recorded: 'Decision Recorded',
};

const LIFECYCLE_COLOR: Record<string, string> = {
  draft: 'bg-gray-100 text-gray-600',
  deliberating: 'bg-blue-100 text-blue-700',
  decision_recorded: 'bg-green-100 text-green-700',
};

export default function Home() {
  const navigate = useNavigate();
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Create form
  const [showCreate, setShowCreate] = useState(false);
  const [title, setTitle] = useState('');
  const [question, setQuestion] = useState('');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState('');

  useEffect(() => {
    api.workspaces.list()
      .then(setWorkspaces)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!title.trim()) return;
    setCreating(true);
    setCreateError('');
    try {
      const ws = await api.workspaces.create({ title: title.trim(), decision_question: question.trim() || undefined });
      navigate(`/w/${ws.id}`);
    } catch (e: unknown) {
      setCreateError(e instanceof Error ? e.message : String(e));
      setCreating(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold text-gray-900">Pact</h1>
        <button
          onClick={() => setShowCreate(true)}
          className="bg-indigo-600 text-white text-sm px-4 py-2 rounded-md hover:bg-indigo-700 transition-colors"
        >
          New workspace
        </button>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-10">
        {showCreate && (
          <div className="mb-8 bg-white border border-gray-200 rounded-lg p-6">
            <h2 className="text-base font-semibold text-gray-900 mb-4">New workspace</h2>
            <form onSubmit={handleCreate} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Title</label>
                <input
                  className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="e.g. Cloud vendor selection"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Decision question <span className="text-gray-400 font-normal">(optional)</span>
                </label>
                <input
                  className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder="e.g. Which CRM should we adopt?"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                />
              </div>
              {createError && <p className="text-sm text-red-600">{createError}</p>}
              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={creating || !title.trim()}
                  className="bg-indigo-600 text-white text-sm px-4 py-2 rounded-md hover:bg-indigo-700 disabled:opacity-50 transition-colors"
                >
                  {creating ? 'Creating…' : 'Create'}
                </button>
                <button
                  type="button"
                  onClick={() => { setShowCreate(false); setTitle(''); setQuestion(''); setCreateError(''); }}
                  className="text-sm text-gray-600 px-4 py-2 rounded-md hover:bg-gray-100 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        )}

        {loading ? (
          <p className="text-sm text-gray-500">Loading…</p>
        ) : error ? (
          <p className="text-sm text-red-600">{error}</p>
        ) : workspaces.length === 0 ? (
          <div className="text-center py-20 text-gray-400">
            <p className="text-lg font-medium">No workspaces yet</p>
            <p className="text-sm mt-1">Create one to get started</p>
          </div>
        ) : (
          <ul className="space-y-3">
            {workspaces.map((ws) => (
              <li key={ws.id}>
                <button
                  onClick={() => navigate(`/w/${ws.id}`)}
                  className="w-full text-left bg-white border border-gray-200 rounded-lg px-5 py-4 hover:border-indigo-300 hover:shadow-sm transition-all"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <p className="font-medium text-gray-900 truncate">{ws.title}</p>
                      {ws.primary_question && (
                        <p className="text-sm text-gray-500 mt-0.5 truncate">{ws.primary_question}</p>
                      )}
                    </div>
                    <span className={`shrink-0 text-xs px-2 py-0.5 rounded-full font-medium ${LIFECYCLE_COLOR[ws.lifecycle_status] ?? 'bg-gray-100 text-gray-600'}`}>
                      {LIFECYCLE_LABEL[ws.lifecycle_status] ?? ws.lifecycle_status}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 mt-2">
                    {new Date(ws.created_at).toLocaleDateString()}
                  </p>
                </button>
              </li>
            ))}
          </ul>
        )}
      </main>
    </div>
  );
}
