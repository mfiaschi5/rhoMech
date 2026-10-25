// The layout of a projection: wedges from the depth, then the layout fix, trying a few slightly
// tilted views when the exact one leaves defects. A 3D projection often lines atoms up along
// the view axis (an atom on a bond, two atoms on one spot) in a way no local relaxation can
// untangle; turning the molecule by 8-15 degrees about the screen axes (the depth is known)
// separates them while the drawing still looks like the view. Each candidate is scored by its
// remaining defects plus TILT_COST per degree; the exact view wins whenever it has none.
import type { SketchModel } from "./model";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { centroid, setPositions } from "./model.ts";
// @ts-ignore TS5097: see above
import { assignStereo, depthParities } from "./stereo.ts";
// @ts-ignore TS5097: see above
import { ANCHOR_PROJECTION, fixLayout, layoutMetrics, type FixOptions, type FixResult, type LayoutMetrics } from "./layoutFix.ts";

/** Tilts (degrees about the screen x and y axes), tried in this order. */
export const TILTS: readonly (readonly [number, number])[] = [
  [0, 0],
  [8, 0],
  [-8, 0],
  [0, 8],
  [0, -8],
  [8, 8],
  [8, -8],
  [-8, 8],
  [-8, -8],
  [15, 0],
  [-15, 0],
  [0, 15],
  [0, -15],
];
const TILT_COST = 0.15;

/** Defects as one number (lower is better). */
export function layoutPenalty(x: LayoutMetrics): number {
  return 10 * x.labelOverlaps + 5 * x.closeContacts + 5 * x.atomOnBond + 2 * x.smallAngles + 0.5 * x.crossings + 20 * x.bondLengthRms;
}

const clean = (x: LayoutMetrics): boolean => x.labelOverlaps + x.closeContacts + x.atomOnBond + x.smallAngles === 0;

/** The drawing turned by (ax, ay) degrees about the screen x and y axes through its centroid
 *  (x, y and the depth z; atoms without depth are left alone). */
export function tilt(m: SketchModel, ax: number, ay: number): SketchModel {
  if (ax === 0 && ay === 0) return m;
  const [cx, cy] = centroid(m);
  const [sa, ca] = [Math.sin((ax * Math.PI) / 180), Math.cos((ax * Math.PI) / 180)];
  const [sb, cb] = [Math.sin((ay * Math.PI) / 180), Math.cos((ay * Math.PI) / 180)];
  const pos = new Map<number, [number, number, number]>();
  for (const a of m.atoms) {
    if (a.z === undefined) continue;
    const x = a.x - cx;
    // about x: y, z turn; then about y: x, z turn
    const y1 = (a.y - cy) * ca - a.z * sa;
    const z1 = (a.y - cy) * sa + a.z * ca;
    const x2 = x * cb + z1 * sb;
    const z2 = -x * sb + z1 * cb;
    pos.set(a.id, [cx + x2, cy + y1, z2]);
  }
  return setPositions(m, pos);
}

export interface ProjectedLayout extends FixResult {
  /** The tilt used, degrees about screen x and y. */
  tilt: readonly [number, number];
}

/** Wedges for `centres` from the depth, then the fix; the best of the tilts (see above). */
export function projectedLayout(m: SketchModel, centres: Iterable<number>, opts: FixOptions = {}): ProjectedLayout {
  const ids = [...centres];
  const fixOpts = { anchor: ANCHOR_PROJECTION, ...opts };
  const before = layoutMetrics(m, fixOpts);
  let best: ProjectedLayout | null = null;
  let bestScore = Infinity;
  for (const t of TILTS) {
    const turned = tilt(m, t[0], t[1]);
    const marked = assignStereo(turned, depthParities(turned, ids), { useDepth: true, clearOthers: true });
    const r = fixLayout(marked, fixOpts);
    const score = layoutPenalty(r.after) + TILT_COST * (Math.abs(t[0]) + Math.abs(t[1])) + (r.stereoKept ? 0 : 50);
    if (score < bestScore - 1e-9) {
      best = { ...r, before, tilt: t };
      bestScore = score;
    }
    if (t[0] === 0 && t[1] === 0 && clean(r.after) && r.stereoKept) break;
  }
  return best as ProjectedLayout;
}
