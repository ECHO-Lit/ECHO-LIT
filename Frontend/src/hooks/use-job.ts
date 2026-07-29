import { useCallback, useEffect, useRef, useState } from 'react';
import { CreateJobInput, JobStatus, cancelJob, runJob } from '@/lib/jobs';

export function useJob<T = unknown>() {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [result, setResult] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);

  const start = useCallback(async (input: CreateJobInput) => {
    controller.current?.abort();
    controller.current = new AbortController();
    setError(null);
    setResult(null);
    try {
      const value = await runJob<T>(input, {
        signal: controller.current.signal,
        onProgress: setStatus,
      });
      setResult(value);
      return value;
    } catch (caught) {
      if ((caught as Error).name !== 'AbortError') setError((caught as Error).message);
      throw caught;
    }
  }, []);

  const cancel = useCallback(async () => {
    if (status?.job_id) await cancelJob(status.job_id);
  }, [status?.job_id]);

  const stopPolling = useCallback(() => controller.current?.abort(), []);
  useEffect(() => () => controller.current?.abort(), []);
  return { start, cancel, stopPolling, status, result, error, isRunning: !!status && !['success', 'failure', 'cancelled'].includes(status.status) };
}
