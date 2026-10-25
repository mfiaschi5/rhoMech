// Layout of the energy-level diagram: canonical MOs grouped by degenerate level (the server's
// `level`, MOs within the degeneracy tolerance), members side by side; the energy window keeps
// the frontier region readable. Pure functions.
import type { OrbitalRowOut } from "../../../api/types.gen";

export interface LevelGroup {
  level: number;
  energy: number;
  rows: OrbitalRowOut[];
}

export function groupLevels(rows: readonly OrbitalRowOut[]): LevelGroup[] {
  const by = new Map<number, OrbitalRowOut[]>();
  for (const r of rows) {
    const k = r.level ?? r.index;
    const list = by.get(k) ?? [];
    list.push(r);
    by.set(k, list);
  }
  return [...by.entries()]
    .map(([level, rs]) => ({ level, rows: rs, energy: rs.reduce((s, r) => s + r.energy_ev, 0) / rs.length }))
    .sort((a, b) => a.energy - b.energy);
}

export type Window = "frontier" | "all";

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

/** x positions (left edges) of `n` side-by-side lines centred in [x0, x1]. */
export function sideBySide(n: number, x0: number, x1: number, maxW: number, gap: number): { xs: number[]; w: number } {
  const w = Math.min(maxW, (x1 - x0 - gap * (n - 1)) / Math.max(1, n));
  const total = n * w + (n - 1) * gap;
  const start = x0 + (x1 - x0 - total) / 2;
  return { xs: Array.from({ length: n }, (_, i) => start + i * (w + gap)), w };
}

/** "homo_minus_1" -> "HOMO−1"; null for non-frontier orbitals. */
export function frontierLabel(f: string | null): string | null {
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
