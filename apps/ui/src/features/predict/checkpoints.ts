// The network predictor's checkpoint on the Predict page (pure; node --test runs it): the
// default choice (the server's rule, serve.models.default_checkpoint: the usable non-smoke run
// with the best validation top-1, best.pt before last.pt; a smoke run only when nothing else is
// usable), the picker's labels, the {source, run, file} a request names, the job header's line,
// and the body of an expansion that keeps the network's predictor and checkpoint.

export interface ListedCheckpoint {
  source: string;
  run: string;
  file: string;
  usable: boolean;
  smoke: boolean;
  encoder: string | null;
  epoch: number | null;
  metrics: Record<string, number | null>;
  reason: string | null;
}

export interface CheckpointRef {
  source: string;
  run: string;
  file: string;
}

export interface RecordedCheckpoint extends CheckpointRef {
  sha256: string;
  epoch: number | null;
  val_top1: number | null;
  encoder: string | null;
  smoke: boolean;
}

/** "source/run/file": the checkpoint's id in the picker. */
export const checkpointId = (c: CheckpointRef) => `${c.source}/${c.run}/${c.file}`;

/** {source, run, file} of a listed or recorded checkpoint (what a request names). */
export const refOf = (c: CheckpointRef): CheckpointRef => ({ source: c.source, run: c.run, file: c.file });

const top1 = (c: ListedCheckpoint) => (typeof c.metrics.val_top1 === "number" ? c.metrics.val_top1 : -1);

function rank(a: ListedCheckpoint, b: ListedCheckpoint): number {
  return top1(b) - top1(a) || Number(b.file === "best") - Number(a.file === "best") || checkpointId(a).localeCompare(checkpointId(b));
}

/** The default checkpoint of the network predictor (the server's rule), null when none is usable. */
export function defaultNetworkCheckpoint<C extends ListedCheckpoint>(list: readonly C[]): C | null {
  const usable = list.filter((c) => c.usable);
  const real = usable.filter((c) => !c.smoke);
  return [...(real.length ? real : usable)].sort(rank)[0] ?? null;
}

const pct = (x: number | null | undefined) => (typeof x === "number" ? `${(100 * x).toFixed(1)}%` : null);

/** One option of the picker: "flex_first/best.pt · flex · epoch 19 · val top-1 60.9%", then
 *  " · smoke" or " (refused)". */
export function pickerLabel(c: ListedCheckpoint): string {
  const parts = [`${c.run}/${c.file}.pt`];
  if (c.encoder) parts.push(c.encoder);
  if (c.epoch !== null) parts.push(`epoch ${c.epoch}`);
  const t = pct(c.metrics.val_top1);
  if (t) parts.push(`val top-1 ${t}`);
  if (c.smoke) parts.push("smoke");
  const label = parts.join(" · ");
  return c.usable ? label : `${label} (refused)`;
}

/** The job header's line of a recorded checkpoint: "flex_first/best.pt · epoch 19 · val top-1
 *  60.9% · sha 1a2b3c4d". */
export function recordedLabel(c: RecordedCheckpoint): string {
  const parts = [`${c.run}/${c.file}.pt`];
  if (c.encoder) parts.push(c.encoder);
  if (c.epoch !== null) parts.push(`epoch ${c.epoch}`);
  const t = pct(c.val_top1);
  if (t) parts.push(`val top-1 ${t}`);
  parts.push(`sha ${c.sha256.slice(0, 8)}`);
  return parts.join(" · ");
}

export interface ExpandBody {
  node_id: string;
  steps: number;
  predictor?: string;
  checkpoint?: CheckpointRef | null;
}

/** The expansion of one state with the network's own predictor and checkpoint (a user network
 *  names none: the server expands it with the floor). */
export function expandBody(provenance: { predictor: string; checkpoint?: RecordedCheckpoint | null } | null | undefined, nodeId: string, steps: number): ExpandBody {
  const body: ExpandBody = { node_id: nodeId, steps };
  if (!provenance || provenance.predictor === "user") return body;
  body.predictor = provenance.predictor;
  if (provenance.checkpoint) body.checkpoint = refOf(provenance.checkpoint);
  return body;
}
