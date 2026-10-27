// Pure logic of the embedding trace view (EmbeddingView.tsx): the checkpoint choice, the trace
// page's URL with its theme and view hash, and what is remembered per browser. No React, no
// fetch: unit-tested with `node --experimental-strip-types --test` (embedding.test.ts).
//
// The trace page (ksmech-eval embedding, served at /api/embedding/{id}/page) keeps its view in
// its own URL hash: #sel=<token>&mode=<colouring>&layer=<0..L>&head=<-1..H-1>. The view reads
// that hash from the iframe and stores it per trace id, so a trace reopens where it was left.

export interface CheckpointLike {
  source: string;
  run: string;
  file: string;
  usable: boolean;
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

export type PageTheme = "light" | "dark";

export const CHECKPOINT_KEY = "ksmech.embedding.checkpoint";
export const VIEW_KEY_PREFIX = "ksmech.embedding.view.";
export const VIEW_PARAMS = ["sel", "mode", "layer", "head"] as const;
const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$/;
const MODE_RE = /^[a-z][a-z0-9_]{0,31}$/;

/** "source/run/file", the checkpoint's id in the picker and in storage. */
export const checkpointKey = (c: CheckpointRef) => `${c.source}/${c.run}/${c.file}`;

/** The parts of a checkpoint key, null when it is not one. */
export function parseCheckpointKey(key: string | null | undefined): CheckpointRef | null {
  const parts = (key ?? "").split("/");
  if (parts.length !== 3 || !parts.every((p) => NAME_RE.test(p))) return null;
  return { source: parts[0], run: parts[1], file: parts[2] };
}

const top1 = (c: CheckpointLike) => c.metrics.val_top1 ?? -1;

/** The checkpoint to preselect: the remembered one while it is usable, else the usable one
 *  with the best validation top-1 (best.pt before last.pt on a tie), else null. */
export function defaultCheckpoint<C extends CheckpointLike>(list: readonly C[], remembered: string | null): C | null {
  const usable = list.filter((c) => c.usable);
  const kept = usable.find((c) => checkpointKey(c) === remembered);
  if (kept) return kept;
  const order = [...usable].sort((a, b) => top1(b) - top1(a) || Number(b.file === "best") - Number(a.file === "best") || checkpointKey(a).localeCompare(checkpointKey(b)));
  return order[0] ?? null;
}

/** One line of the picker: "flex_first/best.pt · flex · epoch 29 · val top-1 58.7%". */
export function checkpointLabel(c: CheckpointLike): string {
  const parts = [`${c.run}/${c.file}.pt`];
  if (c.encoder) parts.push(c.encoder);
  if (c.epoch !== null) parts.push(`epoch ${c.epoch}`);
  const t = c.metrics.val_top1;
  if (typeof t === "number") parts.push(`val top-1 ${(100 * t).toFixed(1)}%`);
  const label = parts.join(" · ");
  return c.usable ? label : `${label} (refused)`;
}

const intIn = (v: string | null, lo: number, hi: number) => v !== null && /^-?\d{1,4}$/.test(v) && +v >= lo && +v <= hi;

/** The page's view hash with only its own parameters, each well formed ("" when none):
 *  "#sel=5&mode=cos&layer=3&head=-1". */
export function cleanViewHash(hash: string | null | undefined): string {
  const q = new URLSearchParams((hash ?? "").replace(/^#/, ""));
  const out = new URLSearchParams();
  for (const k of VIEW_PARAMS) {
    const v = q.get(k);
    const ok = k === "mode" ? v !== null && MODE_RE.test(v) : k === "head" ? intIn(v, -1, 9999) : intIn(v, 0, 9999);
    if (ok && v !== null) out.set(k, v);
  }
  const s = out.toString();
  return s ? `#${s}` : "";
}

/** The page URL: `pageUrl` (the server's /api/embedding/{id}/page), the theme of the app when
 *  it overrides the browser's, and the view hash. */
export function tracePageUrl(pageUrl: string, theme: PageTheme | null, hash: string | null | undefined): string {
  const q = theme ? `?theme=${theme}` : "";
  return `${pageUrl}${q}${cleanViewHash(hash)}`;
}

/** The app's theme override (`<html data-theme>`), null when it follows the browser. */
export function themeOf(value: string | null | undefined): PageTheme | null {
  return value === "light" || value === "dark" ? value : null;
}

export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/** A read that never throws (blocked storage reads as empty). */
export function readStored(store: KeyValueStore | null, key: string): string | null {
  try {
    return store ? store.getItem(key) : null;
  } catch {
    return null;
  }
}

/** A write that never throws (blocked storage: the value lasts for this page only). */
export function writeStored(store: KeyValueStore | null, key: string, value: string): void {
  try {
    store?.setItem(key, value);
  } catch {
    // storage blocked
  }
}

export const viewKey = (embeddingId: string) => `${VIEW_KEY_PREFIX}${embeddingId}`;
