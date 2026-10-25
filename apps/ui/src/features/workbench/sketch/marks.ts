// Small marks around an atom or a bond, in screen pixels (y down): the radical dots and the CIP
// labels (R / S / r / s / ? on atoms, E / Z on double bonds). Pure: drawing.buildScene places
// them and SketchCanvas only draws them.
//
// Placement: a mark goes into the largest free angle around its atom, where "taken" are the
// directions of its bonds, of the H count of its label (left or right) and of its charge
// superscript (up-right). Its distance clears the label box (or the bare vertex) by a margin.
// When an atom has both, the radical dots take the largest gap and the CIP label the largest
// gap left with the dots' direction taken too. A bond label sits beside the bond's midpoint, on
// the side away from the second line of a double bond.
import type { Vec } from "./geometry";

/** Box of a label about its atom (screen px, y down), as drawing.labelBox returns it. */
export interface Box {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/** Gap (px) between a bare vertex and the nearest dot; between a label box and a mark. */
export const VERTEX_GAP_PX = 6;
export const LABEL_GAP_PX = 3;
/** Distance (px) between the two dots of a diradical, centre to centre. */
export const DOT_PAIR_PX = 4.6;
export const DOT_R_PX = 1.7;

const TAU = 2 * Math.PI;
const norm = (a: number) => ((a % TAU) + TAU) % TAU;

/** The bisector (radians, atan2(dy, dx) with y down) of the largest free angular gap among the
 *  `taken` directions; `fallback` (straight up) when nothing is taken. */
export function freeAngle(taken: readonly number[], fallback = -Math.PI / 2): number {
  if (taken.length === 0) return fallback;
  const sorted = taken.map(norm).sort((a, b) => a - b);
  if (sorted.length === 1) return norm(sorted[0] + Math.PI);
  let best = -1;
  let at = fallback;
  for (let i = 0; i < sorted.length; i++) {
    const a = sorted[i];
    const b = i + 1 < sorted.length ? sorted[i + 1] : sorted[0] + TAU;
    // ties go to the gap nearest straight up (a chemist's habit)
    const gap = b - a;
    const mid = norm(a + gap / 2);
    const upness = -Math.sin(mid) * 1e-6;
    if (gap + upness > best + 1e-9) {
      best = gap + upness;
      at = mid;
    }
  }
  return at;
}

/** Distance from the atom centre along `angle` at which the ray leaves `box` (0 without one). */
export function boxExit(box: Box | null, angle: number): number {
  if (!box) return 0;
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);
  const tx = dx > 1e-9 ? box.x1 / dx : dx < -1e-9 ? box.x0 / dx : Infinity;
  const ty = dy > 1e-9 ? box.y1 / dy : dy < -1e-9 ? box.y0 / dy : Infinity;
  return Math.max(0, Math.min(tx, ty));
}

/** The point at `dist` px from `at` along `angle`. */
export const along = (at: Vec, angle: number, dist: number): Vec => [at[0] + Math.cos(angle) * dist, at[1] + Math.sin(angle) * dist];

/** How far from the atom centre a mark of half-size `half` px sits along `angle`. */
export function markDistance(box: Box | null, angle: number, half: number): number {
  return box ? boxExit(box, angle) + LABEL_GAP_PX + half : VERTEX_GAP_PX + half;
}

/** The centres of `n` radical dots (1 or 2) at `angle` from the atom: two dots sit side by side,
 *  across the direction. */
export function radicalDots(at: Vec, box: Box | null, angle: number, n: number): Vec[] {
  if (n <= 0) return [];
  const c = along(at, angle, markDistance(box, angle, DOT_R_PX));
  if (n === 1) return [c];
  const px = -Math.sin(angle) * (DOT_PAIR_PX / 2);
  const py = Math.cos(angle) * (DOT_PAIR_PX / 2);
  return [
    [c[0] + px, c[1] + py],
    [c[0] - px, c[1] - py],
  ];
}

/** The directions taken around an atom: its bonds (screen positions of the neighbours), the H
 *  side of its label and its charge superscript. */
export function takenAngles(at: Vec, neighbours: readonly Vec[], hSide: "left" | "right" | null, charged: boolean): number[] {
  const out = neighbours.map((q) => Math.atan2(q[1] - at[1], q[0] - at[0]));
  if (hSide) out.push(hSide === "right" ? 0 : Math.PI);
  if (charged) out.push(-Math.PI / 4);
  return out;
}

/** The anchor of a bond's E/Z label: beside the midpoint of p-q, `dist` px out on the side
 *  away from the second line (`side`, as drawing.doubleBondSide; 0: the + normal side). */
export function bondMarkAt(p: Vec, q: Vec, side: -1 | 0 | 1, dist = 10): Vec {
  const dx = q[0] - p[0];
  const dy = q[1] - p[1];
  const l = Math.hypot(dx, dy) || 1;
  const k = side === 0 ? 1 : -side;
  const extra = side === 0 ? 2.25 : 0;
  return [(p[0] + q[0]) / 2 + (-dy / l) * (dist + extra) * k, (p[1] + q[1]) / 2 + (dx / l) * (dist + extra) * k];
}
