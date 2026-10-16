import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { ChangeProposal, ConversationMessage, Workspace } from '../types';

interface Props {
  workspace: Workspace;
  onWorkspaceUpdate: () => void;
}

interface PendingProposal {
  id: string;
  rationale: string;
  operationCount: number;
}

export default function ChatTab({ workspace, onWorkspaceUpdate }: Props) {
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [text, setText] = useState('');
  const [authorLabel, setAuthorLabel] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const [pendingProposal, setPendingProposal] = useState<PendingProposal | null>(null);
  const [proposalDetail, setProposalDetail] = useState<ChangeProposal | null>(null);
  const [proposalAction, setProposalAction] = useState<'accepting' | 'rejecting' | null>(null);
  const [rejectReason, setRejectReason] = useState('');
  const [revisionNumber, setRevisionNumber] = useState<number | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const loadMessages = useCallback(() =>
    api.deliberation.messages(workspace.id)
      .then(setMessages)
      .catch(() => {}),
    [workspace.id],
  );

  const loadRevision = useCallback(() =>
    api.brief.get(workspace.id)
      .then((b) => setRevisionNumber(b.revision_number))
      .catch(() => {}),
    [workspace.id],
  );

  useEffect(() => {
    loadMessages();
    loadRevision();
  }, [loadMessages, loadRevision]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim() || !authorLabel.trim()) return;
    setSending(true);
    setError('');
    setPendingProposal(null);

    try {
      const res = await api.deliberation.send(workspace.id, {
        author_label: authorLabel.trim(),
        text: text.trim(),
        expected_revision: revisionNumber ?? undefined,
      });
      setText('');
      await loadMessages();

      if (res.proposal) {
        setPendingProposal({
          id: res.proposal.id,
          rationale: res.proposal.rationale,
          operationCount: res.proposal.operations?.length ?? 0,
        });
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
    }
  }

  async function viewProposalDetail() {
    if (!pendingProposal) return;
    const detail = await api.deliberation.getProposal(workspace.id, pendingProposal.id);
    setProposalDetail(detail);
  }

  async function handleAccept() {
    if (!pendingProposal || revisionNumber == null) return;
    setProposalAction('accepting');
    try {
      await api.deliberation.acceptProposal(workspace.id, pendingProposal.id, revisionNumber);
      setPendingProposal(null);
      setProposalDetail(null);
      await loadRevision();
      onWorkspaceUpdate();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setProposalAction(null);
    }
  }

  async function handleReject(e: React.FormEvent) {
    e.preventDefault();
    if (!pendingProposal) return;
    setProposalAction('rejecting');
    try {
      await api.deliberation.rejectProposal(workspace.id, pendingProposal.id, rejectReason || 'Rejected by user');
      setPendingProposal(null);
      setProposalDetail(null);
      setRejectReason('');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setProposalAction(null);
    }
  }

  const canChat = workspace.lifecycle_status !== 'draft' && workspace.current_revision_id;

  if (!canChat) {
    return (
      <div className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-md px-4 py-3">
        Synthesize a brief first before starting deliberation.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4" style={{ height: 'calc(100vh - 200px)' }}>
      {/* Message history */}
      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {messages.length === 0 && (
          <p className="text-sm text-gray-400">No messages yet. Start the deliberation.</p>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`flex ${m.role === 'human' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-xl rounded-lg px-4 py-3 text-sm ${
              m.role === 'human'
                ? 'bg-indigo-600 text-white'
                : 'bg-white border border-gray-200 text-gray-800'
            }`}>
              {m.role === 'human' && (
                <p className="text-xs font-medium opacity-75 mb-1">{m.author_label}</p>
              )}
              <p className="whitespace-pre-wrap">{m.text}</p>
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Pending proposal */}
      {pendingProposal && (
        <div className="border border-indigo-200 bg-indigo-50 rounded-lg p-4 space-y-3">
          <div className="flex items-start justify-between gap-2">
            <div>
              <p className="text-sm font-semibold text-indigo-800">Change proposal</p>
              <p className="text-xs text-indigo-600 mt-0.5">{pendingProposal.operationCount} operation(s) · {pendingProposal.rationale.slice(0, 100)}</p>
            </div>
            <button onClick={viewProposalDetail} className="text-xs text-indigo-600 hover:underline shrink-0">
              View details
            </button>
          </div>

          {proposalDetail && (
            <div className="bg-white border border-indigo-100 rounded-md p-3 max-h-40 overflow-y-auto">
              <pre className="text-xs text-gray-600 whitespace-pre-wrap font-mono">
                {JSON.stringify(proposalDetail.typed_operations, null, 2)}
              </pre>
            </div>
          )}

          <div className="flex gap-2 items-center">
            <button
              onClick={handleAccept}
              disabled={proposalAction !== null}
              className="bg-green-600 text-white text-sm px-4 py-1.5 rounded-md hover:bg-green-700 disabled:opacity-50 transition-colors"
            >
              {proposalAction === 'accepting' ? 'Accepting…' : 'Accept'}
            </button>
            <button
              onClick={() => setProposalDetail(null)}
              disabled={proposalAction !== null}
              className="bg-white border border-gray-300 text-sm px-4 py-1.5 rounded-md hover:bg-gray-50 disabled:opacity-50 transition-colors"
              title="Reject"
            >
              Reject
            </button>
          </div>

          <form onSubmit={handleReject} className="flex gap-2">
            <input
              className="flex-1 border border-gray-300 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder="Rejection reason (optional)"
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
            />
            <button
              type="submit"
              disabled={proposalAction !== null}
              className="bg-red-600 text-white text-sm px-3 py-1.5 rounded-md hover:bg-red-700 disabled:opacity-50 transition-colors"
            >
              {proposalAction === 'rejecting' ? 'Rejecting…' : 'Confirm reject'}
            </button>
          </form>
        </div>
      )}

      {/* Input */}
      <form onSubmit={handleSend} className="space-y-2">
        {error && <p className="text-sm text-red-600">{error}</p>}
        <div className="flex gap-2">
          <input
            className="w-32 border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            placeholder="Your name"
            value={authorLabel}
            onChange={(e) => setAuthorLabel(e.target.value)}
            required
          />
          <input
            className="flex-1 border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            placeholder="Message…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            required
          />
          <button
            type="submit"
            disabled={sending || !text.trim() || !authorLabel.trim()}
            className="bg-indigo-600 text-white text-sm px-4 py-2 rounded-md hover:bg-indigo-700 disabled:opacity-50 transition-colors"
          >
            {sending ? '…' : 'Send'}
          </button>
        </div>
      </form>
    </div>
  );
}
