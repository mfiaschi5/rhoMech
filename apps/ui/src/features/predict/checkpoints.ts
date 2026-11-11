// The network predictor's checkpoint on the Predict page (pure; node --test runs it): the
// choice (the user's, else the server's default, GET /models default_checkpoint: the rule lives
// only in serve.ml.checkpoints.default_checkpoint), which checkpoints the predictor runs (usable
// and forward: a backward run predicts the previous step), the picker's labels, the {source,
// run, file} a request names, the job header's line, and the body of an expansion that keeps
// the network's predictor and checkpoint.

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
  direction?: string; // "forward" | "backward" (GET /models); absent: forward
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

/** A checkpoint the network predictor runs: usable and forward. */
export const predictable = (c: ListedCheckpoint) => c.usable && (c.direction ?? "forward") === "forward";

/** The picker's checkpoint: the chosen id when the predictor can run it, else the server's
 *  default (`defaultId`, GET /models default_checkpoint), null when neither is listed. */
export function chooseCheckpoint<C extends ListedCheckpoint>(list: readonly C[], value: string | null, defaultId: string | null | undefined): C | null {
  const byId = (id: string) => list.find((c) => checkpointId(c) === id && predictable(c)) ?? null;
  return (value ? byId(value) : null) ?? (defaultId ? byId(defaultId) : null);
}

const pct = (x: number | null | undefined) => (typeof x === "number" ? `${(100 * x).toFixed(1)}%` : null);

/** One option of the picker: "flex_first/best.pt · flex · epoch 19 · val top-1 60.9%", then
 *  " · smoke", " · backward (previous step)" or " (refused)". */
export function pickerLabel(c: ListedCheckpoint): string {
  const parts = [`${c.run}/${c.file}.pt`];
  if (c.encoder) parts.push(c.encoder);
  if (c.epoch !== null) parts.push(`epoch ${c.epoch}`);
  const t = pct(c.metrics.val_top1);
  if (t) parts.push(`val top-1 ${t}`);
  if (c.smoke) parts.push("smoke");
  if (c.direction === "backward") parts.push("backward (previous step)");
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
