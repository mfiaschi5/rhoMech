// JobProgress: the status of a background job (predict or override): phase, counts, the worker
// task it waits on, a cancel button and the job error (design §5.3, §11.8).
//
// Props:
//   job        JobOut | null              the latest poll (null before the first answer)
//   pollError  ApiError | null            a failed poll (network down, 404)
//   title      string                     heading (default "Job")
//
// DOM: predict-job-status[data-status], predict-job-progress, predict-job-cancel (while not
// terminal), predict-job-error.
import { useState } from "react";
import { ApiError, cancelJob, toApiError } from "../../api/client";
import { isTerminal } from "../../api/poll";
import type { ErrorBody, JobOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import { PendingNote } from "../../components/PendingNote";
import { fmtSeconds, shortId } from "../../components/format";

export interface JobProgressProps {
  job: JobOut | null;
  pollError?: ApiError | null;
  title?: string;
}

function ProgressBar({ job }: { job: JobOut }) {
  const p = job.progress;
  const frac = p.total ? Math.min(1, p.done / p.total) : isTerminal(job) ? 1 : null;
  return (
    <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={p.total ?? undefined} aria-valuenow={p.done}>
      <div className={`progress-fill${frac === null ? " indeterminate" : ""}`} style={{ width: `${Math.round((frac ?? 0.3) * 100)}%` }} />
    </div>
  );
}

function Counts({ job }: { job: JobOut }) {
  const p = job.progress;
  return (
    <div className="row muted job-counts">
      <span>
        phase <strong>{p.phase || "–"}</strong>
      </span>
      {p.total !== null && (
        <span>
          {p.done} / {p.total}
        </span>
      )}
      {p.depth !== null && (
        <span>
          depth {p.depth}
          {p.max_steps !== null ? ` / ${p.max_steps}` : ""}
        </span>
      )}
      <span>
        {p.n_nodes} nodes, {p.n_edges} edges, frontier {p.frontier}
      </span>
      {p.eta_s !== null && <span>eta {fmtSeconds(p.eta_s)}</span>}
      {job.partial && <span className="badge badge-warn">partial network</span>}
    </div>
  );
}

function CancelButtons({ jobId }: { jobId: string }) {
  const [cancelError, setCancelError] = useState<ApiError | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const cancel = async (force: boolean) => {
    setCancelling(true);
    try {
      await cancelJob(jobId, { force });
      setCancelError(null);
    } catch (err) {
      setCancelError(toApiError(err));
    } finally {
      setCancelling(false);
    }
  };
  return (
    <>
      <button type="button" className="btn btn-small" data-testid="predict-job-cancel" disabled={cancelling} onClick={() => void cancel(false)}>
        cancel
      </button>
      <button type="button" className="btn btn-small" disabled={cancelling} title="also kill the worker task" onClick={() => void cancel(true)}>
        force cancel
      </button>
      <ErrorBanner error={cancelError} title="Cannot cancel" testId="predict-job-cancel-error" />
    </>
  );
}

export function JobProgress({ job, pollError, title = "Job" }: JobProgressProps) {
  const status = job?.status ?? "loading";
  const task = job?.progress.current_task ?? null;
  return (
    <section className="panel job-progress-panel" data-testid="predict-job-progress" data-job-id={job?.job_id ?? ""} data-kind={job?.kind ?? ""}>
      <div className="row job-head">
        <strong>{title}</strong>
        {job && <code className="code">{shortId(job.job_id, 17)}</code>}
        <span className={`badge job-status job-status-${status}`} data-testid="predict-job-status" data-status={status}>
          {status}
        </span>
        {job && !isTerminal(job) && <CancelButtons jobId={job.job_id} />}
      </div>
      {job && <ProgressBar job={job} />}
      {job && <Counts job={job} />}
      {job?.progress.message && <div className="muted">{job.progress.message}</div>}
      {task && !isTerminal(job) && (
        <div className="row muted">
          <PendingNote label={task.label} worker={null} />
          <span>
            {task.name}, {fmtSeconds(task.elapsed_s)} ({task.priority}, {task.queue_depth} queued)
          </span>
        </div>
      )}
      {job?.error && (
        <ErrorBanner
          error={bodyError(job.error)}
          title={`${job.kind} job ${job.status}`}
          testId="predict-job-error"
        />
      )}
      <ErrorBanner error={pollError} title="Cannot poll the job" testId="predict-job-poll-error" />
    </section>
  );
}

/** An ErrorBody (job.error, a node failure) as the ApiError the banner shows with its code;
 *  status 0: the error happened inside the job, not on this request. */
export function bodyError(body: ErrorBody | null | undefined): ApiError | null {
  return body ? ApiError.fromBody(body, 0) : null;
}
