// Screen geometry of a pool drawing (pure): atom positions under the sketch view, hit testing of
// atoms, bonds and forming-bond overlays, and the arrow context of arrowPath.ts.
import type { View } from "../workbench/sketch/geometry";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { fitView, segmentDistance, toScreen } from "../workbench/sketch/geometry.ts";
// @ts-ignore TS5097: see above
import { BOND_LENGTH } from "../workbench/sketch/model.ts";
import type { ArrowGeomContext, Vec } from "./arrowPath";
import type { PoolStructure } from "./poolStructure";
import type { MatrixIO } from "./v9types";

export const ATOM_HIT_PX = 13;
export const BOND_HIT_PX = 8;
export const MAX_SCALE = 40;

export interface PoolView {
  view: View;
  /** Pool atom index -> screen position (px in the canvas). */
  xy: Map<number, Vec>;
  bondPx: number;
  centroid: Vec;
}

export function poolView(s: PoolStructure, width: number, height: number): PoolView {
  const view = fitView(s.drawn.map((a) => [a.x, a.y] as Vec), width, height, MAX_SCALE, 34);
  const xy = new Map<number, Vec>(s.drawn.map((a) => [a.index, toScreen(view, [a.x, a.y])] as const));
  const pts = [...xy.values()];
  const centroid: Vec = pts.length ? [pts.reduce((t, p) => t + p[0], 0) / pts.length, pts.reduce((t, p) => t + p[1], 0) / pts.length] : [width / 2, height / 2];
  return { view, xy, bondPx: view.s * BOND_LENGTH, centroid };
}

export function arrowContext(s: PoolStructure, pv: PoolView): ArrowGeomContext {
  const nbrs = new Map<number, number[]>();
  for (const b of s.bonds) {
    nbrs.set(b.i, [...(nbrs.get(b.i) ?? []), b.j]);
    nbrs.set(b.j, [...(nbrs.get(b.j) ?? []), b.i]);
  }
  return {
    pos: (i) => pv.xy.get(i),
    neighbours: (i) => (nbrs.get(i) ?? []).map((k) => pv.xy.get(k)).filter((p): p is Vec => !!p),
    centroid: pv.centroid,
    bondPx: pv.bondPx,
  };
}

export type PoolHit = { kind: "atom"; i: number } | { kind: "bond"; i: number; j: number; t: number } | null;

/** The pairs a click can hit as bonds: the reactant's bonds and R's non-bonded pairs. */
export function hitPairs(s: PoolStructure, m: MatrixIO | null): [number, number][] {
  const pairs = new Map<string, [number, number]>();
  for (const b of s.bonds) pairs.set(b.i < b.j ? `${b.i}-${b.j}` : `${b.j}-${b.i}`, [Math.min(b.i, b.j), Math.max(b.i, b.j)]);
  for (const [i, j] of m?.bonds ?? []) pairs.set(`${i}-${j}`, [i, j]);
  return [...pairs.values()];
}

/** The atom within ATOM_HIT_PX of p (nearest), else the bond within BOND_HIT_PX (nearest; t is
 *  the position along it from i), else null. */
export function hitTest(pv: PoolView, pairs: readonly [number, number][], p: Vec): PoolHit {
  let best: PoolHit = null;
  let bestD = ATOM_HIT_PX;
  for (const [i, q] of pv.xy) {
    const d = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (d < bestD) {
      bestD = d;
      best = { kind: "atom", i };
    }
  }
  if (best) return best;
  bestD = BOND_HIT_PX;
  for (const [i, j] of pairs) {
    const a = pv.xy.get(i);
    const b = pv.xy.get(j);
    if (!a || !b) continue;
    const d = segmentDistance(p, a, b);
    if (d < bestD) {
      const L2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2;
      const t = L2 > 0 ? ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / L2 : 0.5;
      bestD = d;
      best = { kind: "bond", i, j, t: Math.max(0, Math.min(1, t)) };
    }
  }
  return best;
}

/** The end of a bond a gesture means: the one the press was clearly near (|t - 0.5| > 0.12),
 *  else the one nearer to where it went (`toward`). */
export function nearEnd(pv: PoolView, hit: Extract<PoolHit, { kind: "bond" }>, toward: Vec | null): number {
  if (Math.abs(hit.t - 0.5) > 0.12 || !toward) return hit.t < 0.5 ? hit.i : hit.j;
  const a = pv.xy.get(hit.i);
  const b = pv.xy.get(hit.j);
  if (!a || !b) return hit.i;
  return Math.hypot(a[0] - toward[0], a[1] - toward[1]) <= Math.hypot(b[0] - toward[0], b[1] - toward[1]) ? hit.i : hit.j;
}
