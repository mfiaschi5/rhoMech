// JobLine: one batch job of this page (export or re-check), polled with api/poll.ts until it ends.
// The job id lives in the hash (export=<id>, recheck=<id>, recheckall=<id>), so a reload rejoins
// the job and shows its report again. `onFinished` runs once per job when it reaches a terminal
// status, so the caller can reload what the job changed.
import { useEffect, useRef, type ReactNode } from "react";
import { ApiError } from "../../api/client";
import { useJobPoll } from "../../api/poll";
import type { JobOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";

export interface JobLineProps {
  jobId: string | null;
  testId: string;
  title: string;
  onFinished?: (job: JobOut) => void;
  children?: (job: JobOut) => ReactNode;
}

function progressText(job: JobOut): string {
  const p = job.progress;
  const count = p.total ? ` ${p.done}/${p.total}` : p.done ? ` ${p.done}` : "";
  const task = p.current_task ? ` (worker: ${p.current_task.label})` : "";
  return `${p.phase}${count}${p.message ? `: ${p.message}` : ""}${task}`;
}

export function JobLine({ jobId, testId, title, onFinished, children }: JobLineProps) {
  const { job, error, done } = useJobPoll(jobId);
  const finished = useRef<string | null>(null);
  const callback = useRef(onFinished);
  callback.current = onFinished;
  useEffect(() => {
    if (job && done && finished.current !== job.job_id) {
      finished.current = job.job_id;
      callback.current?.(job);
    }
  }, [job, done]);
  if (!jobId) return null;
  const status = job?.status ?? (error ? "error" : "loading");
  return (
    <div className="cu-job" data-testid={testId} data-status={status} data-job-id={jobId}>
      <div className="cu-job-head">
        <strong>{title}</strong> <code className="code">{jobId}</code>{" "}
        <span className={status === "done" ? "badge badge-ok" : status === "failed" ? "badge badge-error" : "badge"}>
          {status}
        </span>
        {job && !done && (
          <span className="pending-note">
            <span className="spinner" aria-hidden="true" /> {progressText(job)}
          </span>
        )}
        {job?.partial && <span className="badge badge-warn">partial</span>}
      </div>
      {error && <ErrorBanner error={error} title={`Cannot follow job ${jobId}`} />}
      {job?.error && <ErrorBanner error={ApiError.fromBody(job.error, 0)} title={`${title} failed`} />}
      {job && done && job.status === "done" && children?.(job)}
    </div>
  );
}
