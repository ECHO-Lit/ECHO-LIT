import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { cancelJob, getJob, getJobResult, JobStatus } from '@/lib/jobs';

const TERMINAL = new Set(['success', 'failure', 'cancelled']);
const isTerminal = (status?: JobStatus) => !!status && TERMINAL.has(status.status);

/**
 * Non-blocking job lifecycle on top of TanStack Query: a mutation to submit,
 * a self-stopping poll for status, and a dependent query for the result that
 * only fires once the job succeeds. Complements the older `useJob` hook
 * (hooks/use-job.ts), which polls via a hand-rolled promise loop that does
 * not survive an unmount/remount and cannot share state between components.
 */
export function useAnalysisJob<TResult, TSubmitArgs>(
  submit: (args: TSubmitArgs) => Promise<{ job_id: string }>,
  queryKeyPrefix: string,
) {
  const queryClient = useQueryClient();
  const activeKey = [queryKeyPrefix, 'active'] as const;
  // `activeKey` is written via setQueryData/read via getQueryData below, never
  // through useQuery -- so it has zero subscribers and is eligible for the
  // default 5-minute gcTime the instant it's set. FR10 jobs with explanations
  // on commonly run 10-30+ minutes, so partway through, this entry would get
  // silently evicted, jobId would go undefined, the status query would
  // disable itself, and the panel would revert to the idle "Run Analysis"
  // button while the backend kept processing unaware. Pin it so it never gets
  // GC'd out from under a still-running job.
  queryClient.setQueryDefaults(activeKey, { gcTime: Infinity });

  const start = useMutation({
    mutationFn: submit,
    onSuccess: ({ job_id }) => {
      queryClient.setQueryData(activeKey, job_id);
      queryClient.removeQueries({ queryKey: [queryKeyPrefix, 'job'] });
    },
  });

  const jobId = queryClient.getQueryData<string>(activeKey);

  const status = useQuery({
    queryKey: [queryKeyPrefix, 'job', jobId],
    queryFn: ({ signal }) => getJob(jobId as string, signal),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data as JobStatus | undefined;
      if (isTerminal(data)) return false;
      return data?.status === 'queued' ? 1000 : 2000;
    },
    // FR10 jobs with explanations on run 10-30+ minutes. `false` here (the
    // library default) pauses polling the instant the browser tab loses
    // visibility, so a user who alt-tabs away comes back to a UI that never
    // updated. Poll through backgrounding for jobs this long.
    refetchIntervalInBackground: true,
    // Defense in depth alongside PredictionPanel's forceMount on the
    // Fairness TabsContent: even with forceMount, any future unmount of this
    // hook's owner (a different route, a conditional render, etc.) would
    // otherwise drop this query's only observer, making it eligible for the
    // default 5-minute garbage collection -- silently resetting
    // isRunning/progress to nothing mid-job even though the backend keeps
    // going. Pin it so a remount always finds the last-known status.
    gcTime: Infinity,
    staleTime: 0,
  });

  const result = useQuery({
    queryKey: [queryKeyPrefix, 'job', jobId, 'result'],
    queryFn: ({ signal }) => getJobResult<TResult>(jobId as string, signal),
    enabled: !!jobId && status.data?.status === 'success',
    staleTime: Infinity,
    gcTime: 30 * 60 * 1000,
  });

  const reset = () => {
    queryClient.removeQueries({ queryKey: activeKey });
    queryClient.removeQueries({ queryKey: [queryKeyPrefix, 'job'] });
  };

  return {
    start: start.mutateAsync,
    cancel: () => jobId && cancelJob(jobId),
    reset,
    jobId,
    status: status.data,
    progress: status.data?.progress,
    result: result.data,
    isSubmitting: start.isPending,
    isRunning: !!status.data && !isTerminal(status.data),
    error:
      (start.error as Error | undefined)?.message ??
      status.data?.error?.message ??
      (result.error as Error | undefined)?.message ??
      null,
  };
}
