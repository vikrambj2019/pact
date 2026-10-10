import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { api } from '../api';
import type { Workspace } from '../types';
import SourcesTab from '../components/SourcesTab';
import BriefTab from '../components/BriefTab';
import ChatTab from '../components/ChatTab';
import DecisionTab from '../components/DecisionTab';

type Tab = 'sources' | 'brief' | 'chat' | 'decision';

const LIFECYCLE_COLOR: Record<string, string> = {
  draft: 'bg-gray-100 text-gray-600',
  deliberating: 'bg-blue-100 text-blue-700',
  decision_recorded: 'bg-green-100 text-green-700',
};

const LIFECYCLE_LABEL: Record<string, string> = {
  draft: 'Draft',
  deliberating: 'Deliberating',
  decision_recorded: 'Decision recorded',
};

export default function WorkspacePage() {
  const { workspaceId } = useParams<{ workspaceId: string }>();
  const navigate = useNavigate();
  const [ws, setWs] = useState<Workspace | null>(null);
  const [error, setError] = useState('');
  const [tab, setTab] = useState<Tab>('sources');

  const reload = () => {
    if (!workspaceId) return;
    api.workspaces.get(workspaceId)
      .then(setWs)
      .catch((e) => setError(e.message));
  };

  useEffect(reload, [workspaceId]);

  if (error) return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <p className="text-red-600 mb-4">{error}</p>
        <button onClick={() => navigate('/')} className="text-sm text-indigo-600 hover:underline">
          Back to workspaces
        </button>
      </div>
    </div>
  );

  if (!ws) return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <p className="text-gray-400 text-sm">Loading…</p>
    </div>
  );

  const tabs: { id: Tab; label: string }[] = [
    { id: 'sources', label: 'Sources' },
    { id: 'brief', label: 'Brief' },
    { id: 'chat', label: 'Chat' },
    { id: 'decision', label: 'Decision' },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2 mb-0.5">
              <button onClick={() => navigate('/')} className="text-sm text-gray-400 hover:text-gray-600">
                Workspaces
              </button>
              <span className="text-gray-300">/</span>
              <span className="text-sm text-gray-700 font-medium truncate">{ws.title}</span>
            </div>
            {ws.primary_question && (
              <p className="text-xs text-gray-500 truncate">{ws.primary_question}</p>
            )}
          </div>
          <span className={`shrink-0 text-xs px-2 py-0.5 rounded-full font-medium ${LIFECYCLE_COLOR[ws.lifecycle_status] ?? 'bg-gray-100 text-gray-600'}`}>
            {LIFECYCLE_LABEL[ws.lifecycle_status] ?? ws.lifecycle_status}
          </span>
        </div>

        {/* Tabs */}
        <nav className="flex gap-0 mt-4 -mb-px">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                tab === t.id
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8">
        {tab === 'sources' && (
          <SourcesTab workspace={ws} onWorkspaceUpdate={setWs} onConfirmed={reload} />
        )}
        {tab === 'brief' && (
          <BriefTab workspace={ws} onWorkspaceUpdate={reload} />
        )}
        {tab === 'chat' && (
          <ChatTab workspace={ws} onWorkspaceUpdate={reload} />
        )}
        {tab === 'decision' && (
          <DecisionTab workspace={ws} onWorkspaceUpdate={reload} />
        )}
      </main>
    </div>
  );
}
