// The PMechDB index status (GET /browse/status; design §10, §11.8). The index is built only on
// request: browse-build-index posts /browse/index (an index job already queued or running is
// returned instead of a new one) and the job is polled until it ends; meanwhile curated-only
// search keeps working. A missing steps.parquet is state "unavailable" (409 on POST), shown here.
// DOM: div[data-testid=browse-index-status][data-state] with button[data-testid=browse-build-index]
// and, while a job is followed, span[data-testid=browse-index-job][data-status].
import { useEffect, useRef, useState } from "react";
import { ApiError, buildBrowseIndex, toApiError } from "../../api/client";
import { useJobPoll } from "../../api/poll";
import type { IndexStatusOut, JobOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import type { LoadState } from "./useLoad";

const STATE_TEXT: Record<string, string> = {
  ready: "ready",
  building: "being built",
  missing: "not built yet",
  stale: "out of date (PMechDB steps changed)",
  unavailable: "unavailable",
};

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "never";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function jobProgress(job: JobOut): string {
  const p = job.progress;
  const count = p.total ? ` ${p.done}/${p.total}` : "";
  return `${p.phase}${count}${p.message ? `: ${p.message}` : ""}`;
}

export interface IndexStatusProps {
  status: LoadState<IndexStatusOut>;
  /** The index job to follow (from the hash, or the one the status reports as building). */
  jobId: string | null;
  onStarted: (jobId: string) => void;
  onFinished: (job: JobOut) => void;
}

function useFinished(job: JobOut | null, done: boolean, onFinished: (job: JobOut) => void) {
  const seen = useRef<string | null>(null);
  const cb = useRef(onFinished);
  cb.current = onFinished;
  useEffect(() => {
    if (job && done && seen.current !== job.job_id) {
      seen.current = job.job_id;
      cb.current(job);
    }
  }, [job, done]);
}

function useBuild(onStarted: (jobId: string) => void) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const build = async () => {
    setBusy(true);
    setError(null);
    try {
      const created = await buildBrowseIndex();
      onStarted(created.job_id);
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(false);
    }
  };
  return { busy, error, build };
}

function JobInfo({ jobId, job, done }: { jobId: string; job: JobOut | null; done: boolean }) {
  return (
    <span className="br-job" data-testid="browse-index-job" data-status={job?.status ?? "loading"}>
      job <code>{jobId}</code> {job ? (done ? job.status : jobProgress(job)) : "…"}
    </span>
  );
}

/** Poll the status every 3 s while an index is building without a job we can follow. */
function usePollWhileBuilding(state: string, jobId: string | null, reload: () => void) {
  useEffect(() => {
    if (state !== "building" || jobId) return;
    const t = window.setInterval(reload, 3000);
    return () => window.clearInterval(t);
  }, [state, jobId, reload]);
}

export function IndexStatus({ status, jobId, onStarted, onFinished }: IndexStatusProps) {
  const poll = useJobPoll(jobId);
  useFinished(poll.job, poll.done, onFinished);
  const { busy, error, build } = useBuild(onStarted);
  const pm = status.data?.pmechdb;
  const state = pm?.state ?? (status.error ? "error" : "loading");
  usePollWhileBuilding(state, jobId, status.reload);
  const canBuild = state === "missing" || state === "stale";
  const running = !!poll.job && !poll.done;
  return (
    <div className="br-index" data-testid="browse-index-status" data-state={state}>
      <span>
        PMechDB index: <span className={state === "ready" ? "badge badge-ok" : "badge badge-warn"}>{STATE_TEXT[state] ?? state}</span>
      </span>
      {pm && (
        <span className="muted">
          {pm.n_steps} steps, built {fmtDate(pm.built_at)}
          {pm.reason ? `; ${pm.reason}` : ""}
        </span>
      )}
      {status.data && <span className="muted">curated steps: {status.data.curated_steps}</span>}
      {(canBuild || state === "building" || state === "unavailable") && (
        <button
          type="button"
          className="btn btn-small"
          data-testid="browse-build-index"
          disabled={!canBuild || busy || running}
          title={state === "unavailable" ? (pm?.reason ?? "data/processed/pmechdb/steps.parquet is missing") : undefined}
          onClick={build}
        >
          {busy ? "starting…" : running || state === "building" ? "building…" : "Build PMechDB index"}
        </button>
      )}
      {jobId && <JobInfo jobId={jobId} job={poll.job} done={poll.done} />}
      {!canBuild && state !== "ready" && state !== "loading" && <span className="muted">Searches of curated steps work meanwhile.</span>}
      <ErrorBanner error={status.error} title="Cannot read the index status" />
      <ErrorBanner error={error} title="Index build not started" testId="browse-index-error" />
      {poll.error && <ErrorBanner error={poll.error} title={`Cannot follow job ${jobId}`} />}
      {poll.job?.error && <ErrorBanner error={ApiError.fromBody(poll.job.error, 0)} title="Index build failed" />}
    </div>
  );
}
