// "Fix layout": a short, deterministic position-based relaxation of a 2D drawing (model units)
// that repairs what a 3D projection or a hand drawing gets wrong, without touching the
// topology. Each sweep (Gauss-Seidel, fixed order) applies, as position corrections:
//
//   bond length   each bond back to BOND_LENGTH (stiffness 1; a bond shorter than 0.05 L is
//                 first given a deterministic direction, growthAngle from the other neighbours)
//   angles        two bonds at one atom closer than ANGLE_PUSH open up (weight W_ANGLE)
//   contacts      non-bonded atoms closer than CONTACT_PUSH L move apart (W_SOFT)
//   atom on bond  an atom closer than ATOM_BOND_PUSH L to a bond it is not on moves off (W_SOFT)
//   labels        a label box (drawing.labelBox: symbol, H count, charge; px / pxPerUnit) moves
//                 off other labels, atoms and bonds it is not on (W_SOFT)
//   anchor        every atom back towards its start by `anchor` (keeps the 3D view's look)
//
// It stops when no atom moves more than 1e-4 L in a sweep, or after maxIter sweeps. The
// metrics count with the same geometry but the stricter thresholds (CONTACT_MIN, ...), so a
// converged layout reports zero. Wedge meaning is kept: the configurations drawn before (stereo
// drawnParities) are drawn again on the moved layout (assignStereo, with the depth as a guide
// when the drawing has one).
import type { SketchModel } from "./model";
import type { Vec } from "./geometry";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { BOND_LENGTH, bondOrderSum, hasDepth, setPositions } from "./model.ts";
// @ts-ignore TS5097: see above
import { DEG, fromAngle, growthAngle, segmentDistance } from "./geometry.ts";
// @ts-ignore TS5097: see above
import { atomLabel, labelBox } from "./drawing.ts";
// @ts-ignore TS5097: see above
import { fallbackHCount } from "./elements.ts";
// @ts-ignore TS5097: see above
import { findCrossings } from "./crossings.ts";
// @ts-ignore TS5097: see above
import { assignStereo, drawnParities } from "./stereo.ts";

const L = BOND_LENGTH;
export const CONTACT_MIN = 0.6;
const CONTACT_PUSH = 0.68;
export const ATOM_BOND_MIN = 0.3;
const ATOM_BOND_PUSH = 0.36;
export const MIN_ANGLE = 35 * DEG;
const ANGLE_PUSH = 40 * DEG;
const LABEL_PAD_PX = 3;
const W_SOFT = 0.5;
const W_ANGLE = 0.3;
/** Anchor weights: after a projection, and for "Fix layout" on a drawing. */
export const ANCHOR_PROJECTION = 0.02;
export const ANCHOR_HAND = 0.005;

export interface LayoutMetrics {
  labelOverlaps: number;
  closeContacts: number;
  atomOnBond: number;
  /** sqrt(mean((d / BOND_LENGTH - 1)^2)) over the bonds. */
  bondLengthRms: number;
  crossings: number;
  smallAngles: number;
}

export interface FixOptions {
  anchor?: number;
  maxIter?: number;
  /** Implicit H by atom id (RDKit's); default valences otherwise. */
  hCounts?: ReadonlyMap<number, number> | null;
  /** Screen scale the label boxes are measured at (px per model unit). */
  pxPerUnit?: number;
}

export interface FixResult {
  model: SketchModel;
  before: LayoutMetrics;
  after: LayoutMetrics;
  iterations: number;
  /** False when a wedge configuration could not be drawn again after the move. */
  stereoKept: boolean;
}

/** Positions as a flat mutable array, with the graph by index. */
interface Frame {
  m: SketchModel;
  x: Float64Array;
  y: Float64Array;
  bonds: [number, number][];
  adj: Set<number>[];
  idx: Map<number, number>;
}

function frameOf(m: SketchModel): Frame {
  const idx = new Map(m.atoms.map((a, i) => [a.id, i]));
  const x = Float64Array.from(m.atoms.map((a) => a.x));
  const y = Float64Array.from(m.atoms.map((a) => a.y));
  const bonds: [number, number][] = [];
  const adj = m.atoms.map(() => new Set<number>());
  for (const b of m.bonds) {
    const i = idx.get(b.a);
    const j = idx.get(b.b);
    if (i === undefined || j === undefined) continue;
    bonds.push([i, j]);
    adj[i].add(j);
    adj[j].add(i);
  }
  return { m, x, y, bonds, adj, idx };
}

const modelAt = (f: Frame): SketchModel => ({ ...f.m, atoms: f.m.atoms.map((a, i) => ({ ...a, x: f.x[i], y: f.y[i] })) });

interface Box {
  i: number;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/** The label boxes (model units, absolute) of the labelled atoms at the frame's positions. */
function labelBoxes(f: Frame, opts: FixOptions): Box[] {
  const s = opts.pxPerUnit ?? 25;
  const m = modelAt(f);
  const out: Box[] = [];
  m.atoms.forEach((a, i) => {
    const h = opts.hCounts?.get(a.id) ?? fallbackHCount(a.el, a.charge, bondOrderSum(m, a.id));
    const label = atomLabel(m, a, h, { s, ox: 0, oy: 0 });
    if (!label) return;
    const b = labelBox(label);
    out.push({ i, x0: a.x + b.x0 / s, x1: a.x + b.x1 / s, y0: a.y - b.y1 / s, y1: a.y - b.y0 / s });
  });
  return out;
}

/** Distance from box b (inflated by pad) to segment p-q: <= 0 means they touch. */
function boxSegmentGap(b: Box, p: Vec, q: Vec, pad: number): number {
  const cx = (b.x0 + b.x1) / 2;
  const cy = (b.y0 + b.y1) / 2;
  const dx = q[0] - p[0];
  const dy = q[1] - p[1];
  const l2 = dx * dx + dy * dy;
  const t = l2 > 1e-12 ? Math.max(0, Math.min(1, ((cx - p[0]) * dx + (cy - p[1]) * dy) / l2)) : 0;
  // the closest point of the segment to the box: sample the clamp of the box centre, then
  // refine with the box edges (exact enough for short label boxes)
  let best = Infinity;
  for (const tt of [0, t, 1, ...edgeParams(b, p, q)]) {
    const px = p[0] + dx * tt;
    const py = p[1] + dy * tt;
    const ex = Math.max(b.x0 - pad - px, 0, px - b.x1 - pad);
    const ey = Math.max(b.y0 - pad - py, 0, py - b.y1 - pad);
    best = Math.min(best, Math.hypot(ex, ey));
  }
  return best;
}

/** Parameters where segment p-q meets the lines of the box edges (inside [0, 1]). */
function edgeParams(b: Box, p: Vec, q: Vec): number[] {
  const out: number[] = [];
  const dx = q[0] - p[0];
  const dy = q[1] - p[1];
  for (const xe of [b.x0, b.x1]) if (Math.abs(dx) > 1e-12) out.push((xe - p[0]) / dx);
  for (const ye of [b.y0, b.y1]) if (Math.abs(dy) > 1e-12) out.push((ye - p[1]) / dy);
  return out.filter((t) => t > 0 && t < 1);
}

const boxOverlap = (a: Box, b: Box, pad: number): [number, number] => [
  Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0) + pad,
  Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0) + pad,
];

const inBox = (b: Box, px: number, py: number, pad: number): boolean =>
  px > b.x0 - pad && px < b.x1 + pad && py > b.y0 - pad && py < b.y1 + pad;

/** The angle gaps (radians) between consecutive bonds around atom i, smallest first. */
function angleGaps(f: Frame, i: number): { a: number; b: number; gap: number }[] {
  const ns = [...f.adj[i]].map((j) => ({ j, t: Math.atan2(f.y[j] - f.y[i], f.x[j] - f.x[i]) })).sort((p, q) => p.t - q.t || p.j - q.j);
  if (ns.length < 2) return [];
  return ns.map((n, k) => {
    const next = ns[(k + 1) % ns.length];
    const gap = k + 1 < ns.length ? next.t - n.t : next.t + 2 * Math.PI - n.t;
    return { a: n.j, b: next.j, gap };
  });
}

/** The layout's defects (see the module comment for the thresholds). */
export function layoutMetrics(m: SketchModel, opts: FixOptions = {}): LayoutMetrics {
  const f = frameOf(m);
  const n = m.atoms.length;
  let closeContacts = 0;
  let atomOnBond = 0;
  let smallAngles = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (!f.adj[i].has(j) && Math.hypot(f.x[i] - f.x[j], f.y[i] - f.y[j]) < CONTACT_MIN * L) closeContacts++;
    }
    for (const [a, b] of f.bonds) {
      if (a !== i && b !== i && segmentDistance([f.x[i], f.y[i]], [f.x[a], f.y[a]], [f.x[b], f.y[b]]) < ATOM_BOND_MIN * L) atomOnBond++;
    }
    smallAngles += angleGaps(f, i).filter((g) => g.gap < MIN_ANGLE).length;
  }
  const sq = f.bonds.map(([a, b]) => (Math.hypot(f.x[a] - f.x[b], f.y[a] - f.y[b]) / L - 1) ** 2);
  const bondLengthRms = sq.length ? Math.sqrt(sq.reduce((s, v) => s + v, 0) / sq.length) : 0;
  return { labelOverlaps: countLabelOverlaps(f, opts), closeContacts, atomOnBond, bondLengthRms, crossings: findCrossings(m).length, smallAngles };
}

function countLabelOverlaps(f: Frame, opts: FixOptions): number {
  const boxes = labelBoxes(f, opts);
  let count = 0;
  for (let k = 0; k < boxes.length; k++) {
    const b = boxes[k];
    for (let l = k + 1; l < boxes.length; l++) {
      const [ox, oy] = boxOverlap(b, boxes[l], 0);
      if (ox > 0 && oy > 0) count++;
    }
    for (let j = 0; j < f.x.length; j++) if (j !== b.i && inBox(b, f.x[j], f.y[j], 0)) count++;
    for (const [p, q] of f.bonds) {
      if (p === b.i || q === b.i) continue;
      if (boxSegmentGap(b, [f.x[p], f.y[p]], [f.x[q], f.y[q]], 0) <= 0) count++;
    }
  }
  return count;
}

function move(f: Frame, i: number, dx: number, dy: number): void {
  f.x[i] += dx;
  f.y[i] += dy;
}

/** A deterministic unit direction for a coincident pair. */
const spread = (i: number, j: number): Vec => fromAngle(((i * 0.618034 + j * 0.381966) % 1) * 2 * Math.PI);

/** Bonds shorter than 0.05 L get their far atom placed along growthAngle (once, up front). */
function openCollapsedBonds(f: Frame): void {
  for (const [i, j] of f.bonds) {
    if (Math.hypot(f.x[j] - f.x[i], f.y[j] - f.y[i]) >= 0.05 * L) continue;
    const others = [...f.adj[i]].filter((k) => k !== j && Math.hypot(f.x[k] - f.x[i], f.y[k] - f.y[i]) >= 0.05 * L);
    const t = growthAngle([f.x[i], f.y[i]], others.map((k): Vec => [f.x[k], f.y[k]]), [], L);
    const d = fromAngle(t, 0.5 * L);
    f.x[j] = f.x[i] + d[0];
    f.y[j] = f.y[i] + d[1];
  }
}

function relaxBonds(f: Frame): void {
  for (const [i, j] of f.bonds) {
    const dx = f.x[j] - f.x[i];
    const dy = f.y[j] - f.y[i];
    const d = Math.hypot(dx, dy);
    const [ux, uy] = d > 1e-9 ? [dx / d, dy / d] : spread(i, j);
    const c = (d - L) / 2;
    move(f, i, ux * c, uy * c);
    move(f, j, -ux * c, -uy * c);
  }
}

function relaxAngles(f: Frame): void {
  for (let i = 0; i < f.x.length; i++) {
    for (const g of angleGaps(f, i)) {
      if (g.gap >= ANGLE_PUSH) continue;
      const turn = ((ANGLE_PUSH - g.gap) / 2) * W_ANGLE;
      for (const [j, sgn] of [[g.a, -1], [g.b, 1]] as const) {
        const rx = f.x[j] - f.x[i];
        const ry = f.y[j] - f.y[i];
        const c = Math.cos(sgn * turn);
        const s = Math.sin(sgn * turn);
        move(f, j, rx * c - ry * s - rx, rx * s + ry * c - ry);
      }
    }
  }
}

function relaxContacts(f: Frame): void {
  const n = f.x.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (f.adj[i].has(j)) continue;
      const dx = f.x[j] - f.x[i];
      const dy = f.y[j] - f.y[i];
      const d = Math.hypot(dx, dy);
      if (d >= CONTACT_PUSH * L) continue;
      const [ux, uy] = d > 1e-9 ? [dx / d, dy / d] : spread(i, j);
      const c = ((CONTACT_PUSH * L - d) / 2) * W_SOFT;
      move(f, i, -ux * c, -uy * c);
      move(f, j, ux * c, uy * c);
    }
  }
}

/** Push point i off segment a-b to `want`, the segment ends the other way (by t). */
function pushOffSegment(f: Frame, i: number, a: number, b: number, want: number): void {
  const ex = f.x[b] - f.x[a];
  const ey = f.y[b] - f.y[a];
  const l2 = ex * ex + ey * ey;
  const t = l2 > 1e-12 ? Math.max(0, Math.min(1, ((f.x[i] - f.x[a]) * ex + (f.y[i] - f.y[a]) * ey) / l2)) : 0;
  const cx = f.x[a] + ex * t;
  const cy = f.y[a] + ey * t;
  const d = Math.hypot(f.x[i] - cx, f.y[i] - cy);
  if (d >= want) return;
  const l = Math.sqrt(l2) || 1;
  // off the line: away from the closest point; on it: the left normal of a->b
  const [ux, uy] = d > 1e-9 ? [(f.x[i] - cx) / d, (f.y[i] - cy) / d] : [-ey / l, ex / l];
  const c = (want - d) * W_SOFT;
  move(f, i, (ux * c) / 2, (uy * c) / 2);
  move(f, a, (-ux * c * (1 - t)) / 2, (-uy * c * (1 - t)) / 2);
  move(f, b, (-ux * c * t) / 2, (-uy * c * t) / 2);
}

function relaxAtomOnBond(f: Frame): void {
  for (let i = 0; i < f.x.length; i++) {
    for (const [a, b] of f.bonds) if (a !== i && b !== i) pushOffSegment(f, i, a, b, ATOM_BOND_PUSH * L);
  }
}

/** Separate two boxes (atoms i, j) along the axis of least overlap. */
function separateBoxes(f: Frame, a: Box, b: Box, pad: number): void {
  const [ox, oy] = boxOverlap(a, b, pad);
  if (ox <= 0 || oy <= 0) return;
  const alongX = ox <= oy;
  const dc = alongX ? b.x0 + b.x1 - a.x0 - a.x1 : b.y0 + b.y1 - a.y0 - a.y1;
  const sgn = dc > 1e-12 ? 1 : dc < -1e-12 ? -1 : a.i < b.i ? 1 : -1;
  const c = ((alongX ? ox : oy) / 2) * W_SOFT * sgn;
  move(f, a.i, alongX ? -c : 0, alongX ? 0 : -c);
  move(f, b.i, alongX ? c : 0, alongX ? 0 : c);
}

function relaxLabels(f: Frame, opts: FixOptions): void {
  const pad = LABEL_PAD_PX / (opts.pxPerUnit ?? 25);
  const boxes = labelBoxes(f, opts);
  for (let k = 0; k < boxes.length; k++) {
    for (let l = k + 1; l < boxes.length; l++) separateBoxes(f, boxes[k], boxes[l], pad);
  }
  for (const b of boxes) {
    const pt: Box = { i: 0, x0: 0, x1: 0, y0: 0, y1: 0 };
    for (let j = 0; j < f.x.length; j++) {
      if (j === b.i || !inBox(b, f.x[j], f.y[j], pad)) continue;
      separateBoxes(f, b, { ...pt, i: j, x0: f.x[j], x1: f.x[j], y0: f.y[j], y1: f.y[j] }, pad);
    }
    const hw = (b.x1 - b.x0) / 2 + pad;
    const hh = (b.y1 - b.y0) / 2 + pad;
    for (const [p, q] of f.bonds) {
      if (p === b.i || q === b.i || boxSegmentGap(b, [f.x[p], f.y[p]], [f.x[q], f.y[q]], pad) > 0) continue;
      pushOffSegment(f, b.i, p, q, Math.hypot(hw, hh));
    }
  }
}

function relaxAnchor(f: Frame, x0: Float64Array, y0: Float64Array, w: number): void {
  if (w <= 0) return;
  for (let i = 0; i < f.x.length; i++) move(f, i, (x0[i] - f.x[i]) * w, (y0[i] - f.y[i]) * w);
}

/** The relaxed positions by atom id and the number of sweeps. */
export function relaxLayout(m: SketchModel, opts: FixOptions = {}): { pos: Map<number, [number, number]>; iterations: number } {
  const f = frameOf(m);
  openCollapsedBonds(f);
  const x0 = Float64Array.from(f.x);
  const y0 = Float64Array.from(f.y);
  const maxIter = opts.maxIter ?? 300;
  const anchor = opts.anchor ?? ANCHOR_HAND;
  let it = 0;
  while (it < maxIter) {
    it++;
    const px = Float64Array.from(f.x);
    const py = Float64Array.from(f.y);
    relaxBonds(f);
    relaxAngles(f);
    relaxContacts(f);
    relaxAtomOnBond(f);
    relaxLabels(f, opts);
    relaxAnchor(f, x0, y0, anchor);
    // converged when the sweep's net motion is negligible (the terms may still pull against
    // each other, which per-step moves would count)
    let net = 0;
    for (let i = 0; i < f.x.length; i++) net = Math.max(net, Math.hypot(f.x[i] - px[i], f.y[i] - py[i]));
    if (net < 1e-4 * L) break;
  }
  return { pos: new Map(m.atoms.map((a, i) => [a.id, [f.x[i], f.y[i]]])), iterations: it };
}

/** Relax the drawing and draw its wedge configurations again (see the module comment). */
export function fixLayout(m: SketchModel, opts: FixOptions = {}): FixResult {
  const before = layoutMetrics(m, opts);
  if (m.atoms.length < 2) return { model: m, before, after: before, iterations: 0, stereoKept: true };
  const targets = drawnParities(m);
  const { pos, iterations } = relaxLayout(m, opts);
  let out = setPositions(m, pos);
  if (targets.size) out = assignStereo(out, targets, { useDepth: hasDepth(out) });
  const now = drawnParities(out);
  const stereoKept = [...targets].every(([c, s]) => now.get(c) === s);
  return { model: out, before, after: layoutMetrics(out, opts), iterations, stereoKept };
}
