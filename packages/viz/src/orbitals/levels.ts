// Layout of the energy-level diagram: canonical MOs grouped by degenerate level, members side by
// side; the energy window keeps the frontier region readable; the zoom and pan of the visible
// energy range; the frontier labels kept apart. Pure functions (node --experimental-strip-types).
//
// Degeneracy: the server groups the canonical MOs at 0.05 eV (`level`, orbital_sets.
// degenerate_levels: ascending energy, a new level when an MO lies more than the tolerance above
// the first MO of the current level). groupLevels(rows, tol) regroups them client-side the same
// way at any threshold the user picks, and frontierBlocks(rows, tol) re-derives the HOMO / LUMO
// blocks (orbitals.frontier.frontier_blocks: occupation >= 0.5 is occupied; HOMO-1 / LUMO+1 the
// next distinct level), so at 0.05 eV both agree with the server.
import type { OrbitalRowOut } from "../apiTypes";

export interface LevelGroup {
  level: number;
  energy: number;
  rows: OrbitalRowOut[];
}

/** The degeneracy threshold (eV): default, bounds and input step. */
export const DEGENERACY = { dflt: 0.05, min: 0.001, max: 0.5, step: 0.001 } as const;

/** A valid threshold from anything (storage, the input). */
export function sanitizeDegeneracy(x: unknown): number {
  const v = typeof x === "number" ? x : typeof x === "string" && x.trim() !== "" ? Number(x) : NaN;
  if (!Number.isFinite(v)) return DEGENERACY.dflt;
  return Math.round(Math.min(DEGENERACY.max, Math.max(DEGENERACY.min, v)) * 1000) / 1000;
}

/** Level id per MO index at threshold `tol` (the server's rule, see the header). */
export function degenerateLevels(rows: readonly OrbitalRowOut[], tol: number): Map<number, number> {
  const out = new Map<number, number>();
  let level = -1;
  let first: number | null = null;
  const sorted = [...rows].filter((r) => Number.isFinite(r.energy_ev)).sort((a, b) => a.energy_ev - b.energy_ev || a.index - b.index);
  for (const r of sorted) {
    if (first === null || r.energy_ev - first > tol + 1e-12) {
      level += 1;
      first = r.energy_ev;
    }
    out.set(r.index, level);
  }
  return out;
}

/** The degenerate groups, ascending energy: regrouped at `tol` when given, else the server's
 *  `level` (one group per MO without one). */
export function groupLevels(rows: readonly OrbitalRowOut[], tol?: number): LevelGroup[] {
  const ids = tol === undefined ? null : degenerateLevels(rows, tol);
  const by = new Map<number, OrbitalRowOut[]>();
  for (const r of rows) {
    const k = ids ? (ids.get(r.index) ?? -1 - r.index) : (r.level ?? r.index);
    const list = by.get(k) ?? [];
    list.push(r);
    by.set(k, list);
  }
  return [...by.entries()]
    .map(([level, rs]) => ({ level, rows: rs.sort((a, b) => a.index - b.index), energy: rs.reduce((s, r) => s + r.energy_ev, 0) / rs.length }))
    .sort((a, b) => a.energy - b.energy);
}

const OCCUPIED = 0.5;

/** The MOs of `members` within `tol` of `anchor`'s energy. */
const block = (members: readonly OrbitalRowOut[], anchor: OrbitalRowOut, tol: number) => members.filter((r) => Math.abs(r.energy_ev - anchor.energy_ev) <= tol + 1e-12);

/** The frontier label of each MO in a HOMO-1 / HOMO / LUMO / LUMO+1 block at `tol` (the server's
 *  frontier_blocks; MOs in no block are absent). */
export function frontierBlocks(rows: readonly OrbitalRowOut[], tol: number): Map<number, string> {
  const out = new Map<number, string>();
  const occ = rows.filter((r) => r.occupation >= OCCUPIED && Number.isFinite(r.energy_ev));
  const virt = rows.filter((r) => r.occupation < OCCUPIED && Number.isFinite(r.energy_ev));
  const put = (rs: OrbitalRowOut[], f: string) => rs.forEach((r) => out.set(r.index, f));
  if (occ.length) {
    const homo = occ.reduce((a, b) => (b.index > a.index ? b : a));
    put(block(occ, homo, tol), "homo");
    const below = occ.filter((r) => r.energy_ev < homo.energy_ev - tol);
    if (below.length) put(block(below, below.reduce((a, b) => (b.energy_ev > a.energy_ev ? b : a)), tol), "homo_minus_1");
  }
  if (virt.length) {
    const lumo = virt.reduce((a, b) => (b.energy_ev < a.energy_ev ? b : a));
    put(block(virt, lumo, tol), "lumo");
    const above = virt.filter((r) => r.energy_ev > lumo.energy_ev + tol);
    if (above.length) put(block(above, above.reduce((a, b) => (b.energy_ev < a.energy_ev ? b : a)), tol), "lumo_plus_1");
  }
  return out;
}

export type Window = "frontier" | "all";
export type Range = readonly [number, number];

/** The energy range drawn: all levels, or HOMO − 12 eV to LUMO + 8 eV. */
export function energyWindow(rows: readonly OrbitalRowOut[], w: Window): [number, number] {
  const es = rows.map((r) => r.energy_ev).filter(Number.isFinite);
  if (!es.length) return [-1, 1];
  let lo = Math.min(...es);
  let hi = Math.max(...es);
  if (w === "frontier") {
    const homo = rows.find((r) => r.frontier === "homo")?.energy_ev;
    const lumo = rows.find((r) => r.frontier === "lumo")?.energy_ev;
    if (homo !== undefined) lo = Math.max(lo, homo - 12);
    if (lumo !== undefined) hi = Math.min(hi, lumo + 8);
  }
  const pad = Math.max(0.5, (hi - lo) * 0.04);
  return [lo - pad, hi + pad];
}

/** The smallest visible span (eV): a 1000x zoom of a 50 eV spectrum, enough to split 0.05 eV. */
export const MIN_SPAN = 0.05;

/** `r` scaled by `factor` (< 1 zooms in) about energy `at`, kept within [MIN_SPAN, `limit`'s
 *  span x 4] and inside `limit` widened by its span. */
export function zoomRange(r: Range, factor: number, at: number, limit: Range): [number, number] {
  const span = r[1] - r[0];
  const full = limit[1] - limit[0];
  const next = Math.min(full * 4, Math.max(MIN_SPAN, span * factor));
  const k = next / span;
  return clampRange([at - (at - r[0]) * k, at + (r[1] - at) * k], limit);
}

/** `r` moved by `dE` eV (positive: towards higher energies), kept near `limit`. */
export const panRange = (r: Range, dE: number, limit: Range): [number, number] => clampRange([r[0] + dE, r[1] + dE], limit);

/** Keep a range overlapping `limit`: its centre no further out than `limit`'s ends. */
export function clampRange(r: Range, limit: Range): [number, number] {
  const c = (r[0] + r[1]) / 2;
  const shift = c < limit[0] ? limit[0] - c : c > limit[1] ? limit[1] - c : 0;
  return [r[0] + shift, r[1] + shift];
}

/** x positions (left edges) of `n` side-by-side lines centred in [x0, x1]. */
export function sideBySide(n: number, x0: number, x1: number, maxW: number, gap: number): { xs: number[]; w: number } {
  const w = Math.min(maxW, (x1 - x0 - gap * (n - 1)) / Math.max(1, n));
  const total = n * w + (n - 1) * gap;
  const start = x0 + (x1 - x0 - total) / 2;
  return { xs: Array.from({ length: n }, (_, i) => start + i * (w + gap)), w };
}

/** Label positions (y, px) for wanted positions `ys`, at least `gap` apart, in [lo, hi] when
 *  they fit: a sweep down then up keeps each as close to its level as the others allow. */
export function spreadLabels(ys: readonly number[], gap: number, lo: number, hi: number): number[] {
  const order = ys.map((y, i) => [y, i] as const).sort((a, b) => a[0] - b[0]);
  const out = order.map(([y]) => y);
  for (let i = 0; i < out.length; i++) out[i] = Math.max(out[i], i === 0 ? lo : out[i - 1] + gap);
  for (let i = out.length - 1; i >= 0; i--) out[i] = Math.min(out[i], i === out.length - 1 ? hi : out[i + 1] - gap);
  for (let i = 0; i < out.length; i++) out[i] = Math.max(out[i], i === 0 ? lo : out[i - 1] + gap);
  const res = new Array<number>(ys.length);
  order.forEach(([, idx], k) => (res[idx] = out[k]));
  return res;
}

/** "homo_minus_1" -> "HOMO−1"; null for non-frontier orbitals. */
export function frontierLabel(f: string | null | undefined): string | null {
  if (!f) return null;
  return f
    .toUpperCase()
    .replace("_MINUS_", "−")
    .replace("_PLUS_", "+");
}

/** The weight of each canonical MO in an LMO's spectral composition (k -> w). */
export function spectralWeights(lmo: OrbitalRowOut | undefined): Map<number, number> {
  const out = new Map<number, number>();
  for (const [k, , w] of lmo?.spectral ?? []) out.set(k, w);
  return out;
}
