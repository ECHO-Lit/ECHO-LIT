import { useCallback, useEffect, useRef, useState } from 'react';
import { CreateJobInput, JobStatus, cancelJob, runJob } from '@/lib/jobs';

const TERMINAL = new Set(['success', 'failure', 'cancelled']);

interface Run {
  controller: AbortController;
  jobId?: string;
  /** The server reported a terminal state; DELETE would then remove the result, not cancel. */
  settled: boolean;
  /** Abandoned by a newer run; cancel the job as soon as its id is known. */
  superseded: boolean;
}

function cancelQuietly(jobId: string) {
  // Best effort: the job may have finished or been reaped in the meantime.
  void cancelJob(jobId).catch(() => {});
}

export function useJob<T = unknown>() {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [result, setResult] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const run = useRef<Run | null>(null);

  // Nobody will read an abandoned run's result. Stopping only its polling left
  // the job on the server, holding the single analysis worker while the next
  // job queued behind it.
  const abandon = useCallback(() => {
    const previous = run.current;
    if (!previous) return;
    run.current = null;
    previous.superseded = true;
    previous.controller.abort();
    if (previous.jobId && !previous.settled) cancelQuietly(previous.jobId);
    setStatus(null);
  }, []);

  const start = useCallback(async (input: CreateJobInput) => {
    abandon();
    const current: Run = { controller: new AbortController(), settled: false, superseded: false };
    run.current = current;
    setError(null);
    setResult(null);
    try {
      const value = await runJob<T>(input, {
        signal: current.controller.signal,
        onCreated: (jobId) => {
          current.jobId = jobId;
          if (current.superseded) cancelQuietly(jobId);
        },
        onProgress: (next) => {
          current.settled = TERMINAL.has(next.status);
          if (run.current === current) setStatus(next);
        },
      });
      if (run.current === current) setResult(value);
      return value;
    } catch (caught) {
      if (run.current === current && (caught as Error).name !== 'AbortError') setError((caught as Error).message);
      throw caught;
    }
  }, [abandon]);

  // The current run's id, not `status.job_id`: status lags a new submit by one
  // poll, and until then would name the previous job.
  const cancel = useCallback(async () => {
    const current = run.current;
    if (current?.jobId && !current.settled) await cancelJob(current.jobId);
  }, []);

  const stopPolling = useCallback(() => run.current?.controller.abort(), []);
  // Unmount only stops polling: a long job (a lens fit, say) is left to finish
  // so its outcome is there when the user comes back.
  useEffect(() => () => run.current?.controller.abort(), []);
  return { start, cancel, abandon, stopPolling, status, result, error, isRunning: !!status && !TERMINAL.has(status.status) };
}
