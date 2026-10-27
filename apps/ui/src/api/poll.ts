// Job polling (design §11.2): pollJob(id, {intervalMs: 1000, sinceRevision, signal}) polls
// GET /jobs/{id} until the job is done, failed or cancelled, and backs off to 5 s on errors.
// With since_revision the server answers `result: null, unchanged: true` while the network
// revision has not moved; the last full result is kept, so every JobOut handed out carries it.
import { useEffect, useRef, useState } from "react";
import { ApiError, isAbort, toApiError } from "./http";
import { getJob } from "./predict";
import type { JobOut, NetworkOut } from "./types.gen";

export const TERMINAL_STATUSES: ReadonlySet<JobOut["status"]> = new Set(["done", "failed", "cancelled"]);

export function isTerminal(job: JobOut | null | undefined): boolean {
  return !!job && TERMINAL_STATUSES.has(job.status);
}

/** The network revision a job result carries (network results only). */
export function jobRevision(job: JobOut | null | undefined): number | undefined {
  if (!job || job.result_kind !== "network" || !job.result) return undefined;
  return (job.result as NetworkOut).revision;
}

export interface PollOptions {
  intervalMs?: number;
  /** Error backoff ceiling (default 5000 ms). */
  maxIntervalMs?: number;
  sinceRevision?: number;
  signal?: AbortSignal;
  /** Called with every new state of the job (merged with the last full result). */
  onUpdate?: (job: JobOut) => void;
  /** Called on a failed poll that will be retried (network error, 5xx). */
  onError?: (err: ApiError, attempt: number) => void;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("aborted", "AbortError"));
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        reject(new DOMException("aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

function merge(job: JobOut, last: JobOut | null): JobOut {
  if (!job.unchanged || !last) return job;
  return { ...job, result: last.result, result_kind: job.result_kind ?? last.result_kind };
}

function changed(a: JobOut, b: JobOut | null): boolean {
  return (
    !b ||
    !a.unchanged ||
    a.status !== b.status ||
    a.progress.updated_at !== b.progress.updated_at ||
    a.finished_at !== b.finished_at
  );
}

/** Poll until the job ends; resolves with its final state. 4xx answers (404 JOB_NOT_FOUND)
 *  reject; network errors and 5xx back off and retry. Rejects with AbortError on abort. */
export async function pollJob(jobId: string, opts: PollOptions = {}): Promise<JobOut> {
  const base = opts.intervalMs ?? 1000;
  const ceiling = opts.maxIntervalMs ?? 5000;
  let interval = base;
  let revision = opts.sinceRevision;
  let last: JobOut | null = null;
  let failures = 0;
  for (;;) {
    try {
      const job = merge(await getJob(jobId, { sinceRevision: revision }, { signal: opts.signal }), last);
      if (changed(job, last)) opts.onUpdate?.(job);
      last = job;
      revision = jobRevision(job) ?? revision;
      interval = base;
      failures = 0;
      if (isTerminal(job)) return job;
    } catch (err) {
      if (isAbort(err)) throw err;
      const e = toApiError(err);
      if (e.status >= 400 && e.status < 500) throw e;
      failures += 1;
      opts.onError?.(e, failures);
      interval = Math.min(ceiling, interval * 2);
    }
    await sleep(interval, opts.signal);
  }
}

export interface JobPollState {
  job: JobOut | null;
  error: ApiError | null;
  done: boolean;
}

/** React hook: poll `jobId` (null: idle) and re-render on every update. */
export function useJobPoll(jobId: string | null, opts: { intervalMs?: number } = {}): JobPollState {
  const [state, setState] = useState<JobPollState>({ job: null, error: null, done: false });
  const interval = useRef(opts.intervalMs);
  interval.current = opts.intervalMs;
  useEffect(() => {
    setState({ job: null, error: null, done: false });
    if (!jobId) return;
    const ctl = new AbortController();
    pollJob(jobId, {
      intervalMs: interval.current,
      signal: ctl.signal,
      onUpdate: (job) => setState({ job, error: null, done: isTerminal(job) }),
      onError: (error) => setState((s) => ({ ...s, error })),
    }).then(
      (job) => setState({ job, error: null, done: true }),
      (err) => {
        if (!isAbort(err)) setState((s) => ({ ...s, error: toApiError(err), done: true }));
      },
    );
    return () => ctl.abort();
  }, [jobId]);
  return state;
}
