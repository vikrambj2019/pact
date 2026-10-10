import { useEffect, useState } from 'react';
import { api } from '../api';
import type { Job } from '../types';

interface Props {
  jobId: string;
  onDone: (job: Job) => void;
  onFailed: (job: Job) => void;
  intervalMs?: number;
}

export default function JobPoller({ jobId, onDone, onFailed, intervalMs = 2500 }: Props) {
  const [status, setStatus] = useState<string>('pending');

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const job = await api.jobs.get(jobId);
        if (cancelled) return;
        setStatus(job.status);
        if (job.status === 'done') { onDone(job); return; }
        if (job.status === 'failed') { onFailed(job); return; }
        setTimeout(poll, intervalMs);
      } catch {
        if (!cancelled) setTimeout(poll, intervalMs * 2);
      }
    };

    poll();
    return () => { cancelled = true; };
  }, [jobId, onDone, onFailed, intervalMs]);

  return (
    <span className="inline-flex items-center gap-1.5 text-sm text-amber-700">
      <span className="animate-spin h-3 w-3 border-2 border-amber-700 border-t-transparent rounded-full" />
      {status === 'pending' ? 'Queued…' : 'Running…'}
    </span>
  );
}
