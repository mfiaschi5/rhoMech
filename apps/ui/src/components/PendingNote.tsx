// PendingNote: what a pending computation waits for (design §6.1 pending protocol).
//
// Props:
//   label   string | null | undefined           PendingOut.label (the onPending callback's first
//                                                argument); nothing renders for null
//   worker  WorkerStatusOut | null | undefined  PendingOut.worker
//
// Text: "queued behind: <label of the task the worker is running>" when the worker is busy with
// another task, else "computing: <label>". DOM: <span data-testid="pending-note">.
import type { WorkerStatusOut } from "../api/types.gen";

export interface PendingNoteProps {
  label: string | null | undefined;
  worker?: WorkerStatusOut | null;
}

export function PendingNote({ label, worker }: PendingNoteProps) {
  if (!label) return null;
  const ahead = worker?.busy && worker.current_label && worker.current_label !== label ? worker.current_label : null;
  const queue = worker && worker.queue_depth > 0 ? ` (${worker.queue_depth} queued)` : "";
  const elapsed = worker?.elapsed_s != null ? `, running ${Math.round(worker.elapsed_s)} s` : "";
  return (
    <span className="pending-note" data-testid="pending-note" role="status" title={`waiting for: ${label}`}>
      <span className="spinner" aria-hidden="true" />
      {ahead ? `queued behind: ${ahead}${elapsed}${queue}` : `computing: ${label}${queue}`}
      {worker?.unstable && <span className="badge badge-error">worker unstable</span>}
    </span>
  );
}
