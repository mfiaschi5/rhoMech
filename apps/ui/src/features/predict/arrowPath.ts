// Curved-arrow geometry of a reading of R, in screen pixels (pure; node --test runs it).
//
// Ends (the textbook convention):
//   a lone pair on a     the point 0.38 bond lengths from a in its widest free direction (where
//                        its dots are drawn), leaving along that direction;
//   a bond (a, c)        the bond's midpoint;
//   a forming bond (a, b) the midpoint when a and b are within 1.6 bond lengths, else the point
//                        0.45 bond lengths short of b on the line from a (the arrow points at b);
//   a new lone pair on k the lone-pair point of k.
// The curve is a cubic Bézier whose two control points sit on one side of the chord, away from
// the drawing's centroid (arrows bow outwards, off the skeleton); several arrows between the
// same two ends alternate sides. A lone-pair tail bends the first control point along the lone
// pair's direction. The tail is trimmed by `gapTail` px, the head by `gapHead` px; the stroke stops
// at the base of a filled triangular head whose axis is the curve's tangent at the tip.
import type { ArrowReadingOut } from "./v9types";

export type Vec = readonly [number, number];

export interface ArrowGeomContext {
  /** Screen position of pool atom i. */
  pos(i: number): Vec | undefined;
  /** Screen positions of the atoms bonded to i. */
  neighbours(i: number): Vec[];
  /** Centroid of the drawing (arrows bow away from it). */
  centroid: Vec;
  /** A standard bond's length in px. */
  bondPx: number;
}

export interface ArrowPath {
  tail: Vec;
  /** The arrowhead's tip. */
  tip: Vec;
  c1: Vec;
  c2: Vec;
  /** The visible stroke (tail trimmed, ending at the head's base) as SVG path data. */
  d: string;
  /** Tip, left and right corners of the head. */
  head: [Vec, Vec, Vec];
  /** The full curve at t = 0.5 (a handle). */
  mid: Vec;
}

export const LONE_PAIR_OFFSET = 0.38;
export const NEAR_BOND = 1.6;
export const BOW = 0.32;

const add = (p: Vec, q: Vec): Vec => [p[0] + q[0], p[1] + q[1]];
const sub = (p: Vec, q: Vec): Vec => [p[0] - q[0], p[1] - q[1]];
const mul = (p: Vec, k: number): Vec => [p[0] * k, p[1] * k];
const len = (p: Vec): number => Math.hypot(p[0], p[1]);
const unit = (p: Vec): Vec => {
  const l = len(p);
  return l > 1e-9 ? [p[0] / l, p[1] / l] : [0, -1];
};
export const midpoint = (a: Vec, b: Vec): Vec => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

export function cubicPoint(p0: Vec, c1: Vec, c2: Vec, p3: Vec, t: number): Vec {
  const u = 1 - t;
  const a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
  return [a * p0[0] + b * c1[0] + c * c2[0] + d * p3[0], a * p0[1] + b * c1[1] + c * c2[1] + d * p3[1]];
}

export function cubicTangent(p0: Vec, c1: Vec, c2: Vec, p3: Vec, t: number): Vec {
  const u = 1 - t;
  const a = 3 * u * u, b = 6 * u * t, c = 3 * t * t;
  return [
    a * (c1[0] - p0[0]) + b * (c2[0] - c1[0]) + c * (p3[0] - c2[0]),
    a * (c1[1] - p0[1]) + b * (c2[1] - c1[1]) + c * (p3[1] - c2[1]),
  ];
}

/** The direction (unit) of the widest gap between the bonds at `at`; away from `away` when
 *  there is no bond. */
export function freeDirection(at: Vec, nbrs: readonly Vec[], away: Vec): Vec {
  if (nbrs.length === 0) {
    const d = sub(at, away);
    return len(d) > 1e-6 ? unit(d) : [0, -1];
  }
  const angles = nbrs.map((q) => Math.atan2(q[1] - at[1], q[0] - at[0])).sort((x, y) => x - y);
  let best = angles[0] + Math.PI;
  let gap = -1;
  for (let k = 0; k < angles.length; k++) {
    const a = angles[k];
    const b = k + 1 < angles.length ? angles[k + 1] : angles[0] + 2 * Math.PI;
    if (b - a > gap) {
      gap = b - a;
      best = (a + b) / 2;
    }
  }
  return [Math.cos(best), Math.sin(best)];
}

/** Where the dots of a lone pair on atom i are drawn, and the direction they face. */
export function lonePairPoint(ctx: ArrowGeomContext, i: number): { at: Vec; dir: Vec } | null {
  const p = ctx.pos(i);
  if (!p) return null;
  const dir = freeDirection(p, ctx.neighbours(i), ctx.centroid);
  return { at: add(p, mul(dir, LONE_PAIR_OFFSET * ctx.bondPx)), dir };
}

/** The point an arrow points at for the forming bond (a, b). */
export function formingBondPoint(a: Vec, b: Vec, bondPx: number): Vec {
  const d = sub(b, a);
  if (len(d) <= NEAR_BOND * bondPx) return midpoint(a, b);
  return sub(b, mul(unit(d), 0.45 * bondPx));
}

export interface ArrowEnds {
  tail: Vec;
  head: Vec;
  /** The lone pair's direction at a lone-pair tail. */
  tailDir: Vec | null;
}

/** The tail and head points of one arrow of a reading; null when an atom has no position. */
export function arrowEnds(a: ArrowReadingOut, ctx: ArrowGeomContext): ArrowEnds | null {
  const P = (i: number | undefined) => (i === undefined ? undefined : ctx.pos(i));
  let tail: Vec, tailDir: Vec | null = null;
  if (a.source.length === 1) {
    const lp = lonePairPoint(ctx, a.source[0]);
    if (!lp) return null;
    tail = lp.at;
    tailDir = lp.dir;
  } else {
    const [p, q] = [P(a.source[0]), P(a.source[1])];
    if (!p || !q) return null;
    tail = midpoint(p, q);
  }
  let head: Vec;
  if (a.target.length === 1) {
    const lp = lonePairPoint(ctx, a.target[0]);
    if (!lp) return null;
    head = lp.at;
  } else {
    const [p, q] = [P(a.target[0]), P(a.target[1])];
    if (!p || !q) return null;
    head = formingBondPoint(p, q, ctx.bondPx);
  }
  return { tail, head, tailDir };
}

export interface BezierOptions {
  centroid: Vec;
  /** +1 bows away from the centroid, -1 towards it. */
  side?: 1 | -1;
  bow?: number;
  tailDir?: Vec | null;
  /** Bow length floor (px), so very short arrows still arc. */
  minSpan?: number;
  gapTail?: number;
  gapHead?: number;
  headLength?: number;
  headWidth?: number;
}

/** The control points of the full (untrimmed) curve. */
export function controlPoints(tail: Vec, head: Vec, o: BezierOptions): [Vec, Vec] {
  const v = sub(head, tail);
  const L = len(v);
  const span = Math.max(L, o.minSpan ?? 24);
  let n: Vec = [-unit(v)[1], unit(v)[0]];
  const m = midpoint(tail, head);
  if ((m[0] - o.centroid[0]) * n[0] + (m[1] - o.centroid[1]) * n[1] < 0) n = mul(n, -1);
  const bow = (o.bow ?? BOW) * span * (o.side ?? 1);
  let c1 = add(add(tail, mul(v, 1 / 3)), mul(n, bow));
  const c2 = add(add(tail, mul(v, 2 / 3)), mul(n, bow));
  if (o.tailDir) c1 = add(tail, mul(unit(add(unit(o.tailDir), mul(unit(sub(c1, tail)), 1))), 0.5 * span));
  return [c1, c2];
}

/** The parameter t at arc length `dist` from the start (fromEnd: from the end), by sampling. */
export function tAtArcLength(p0: Vec, c1: Vec, c2: Vec, p3: Vec, dist: number, fromEnd = false): number {
  const N = 64;
  let acc = 0;
  let prev = cubicPoint(p0, c1, c2, p3, fromEnd ? 1 : 0);
  for (let k = 1; k <= N; k++) {
    const t = fromEnd ? 1 - k / N : k / N;
    const q = cubicPoint(p0, c1, c2, p3, t);
    const step = len(sub(q, prev));
    if (acc + step >= dist) {
      const f = step > 0 ? (dist - acc) / step : 0;
      const tPrev = fromEnd ? 1 - (k - 1) / N : (k - 1) / N;
      return tPrev + (t - tPrev) * f;
    }
    acc += step;
    prev = q;
  }
  return fromEnd ? 0 : 1;
}

/** The sub-curve [t0, t1] of a cubic as its four points (de Casteljau, twice). */
export function subCubic(p0: Vec, c1: Vec, c2: Vec, p3: Vec, t0: number, t1: number): [Vec, Vec, Vec, Vec] {
  const lerp = (a: Vec, b: Vec, t: number): Vec => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  const split = (a: Vec, b: Vec, c: Vec, d: Vec, t: number): [Vec, Vec, Vec, Vec] => {
    const ab = lerp(a, b, t), bc = lerp(b, c, t), cd = lerp(c, d, t);
    const abc = lerp(ab, bc, t), bcd = lerp(bc, cd, t);
    return [lerp(abc, bcd, t), bcd, cd, d];
  };
  const [q0, q1, q2, q3] = split(p0, c1, c2, p3, t0);
  const u = t0 < 1 ? (t1 - t0) / (1 - t0) : 1;
  const ab = lerp(q0, q1, u), bc = lerp(q1, q2, u), cd = lerp(q2, q3, u);
  const abc = lerp(ab, bc, u), bcd = lerp(bc, cd, u);
  return [q0, ab, abc, lerp(abc, bcd, u)];
}

const f2 = (v: number) => (Math.round(v * 100) / 100).toString();

export function bezierArrow(tail: Vec, head: Vec, o: BezierOptions): ArrowPath {
  const [c1, c2] = controlPoints(tail, head, o);
  const L = len(sub(head, tail));
  const k = Math.min(1, (0.5 * Math.max(L, 1)) / Math.max((o.gapTail ?? 4) + (o.gapHead ?? 4), 1));
  const t0 = tAtArcLength(tail, c1, c2, head, (o.gapTail ?? 4) * k);
  const t1 = tAtArcLength(tail, c1, c2, head, (o.gapHead ?? 4) * k, true);
  const tip = cubicPoint(tail, c1, c2, head, t1);
  const dir = unit(cubicTangent(tail, c1, c2, head, t1));
  const hl = o.headLength ?? 9;
  const hw = (o.headWidth ?? 7) / 2;
  const base = sub(tip, mul(dir, hl));
  const perp: Vec = [-dir[1], dir[0]];
  const headPts: [Vec, Vec, Vec] = [tip, add(base, mul(perp, hw)), sub(base, mul(perp, hw))];
  const tb = Math.max(t0, tAtArcLength(tail, c1, c2, head, (o.gapHead ?? 4) * k + hl * 0.8, true));
  const [a, b, c, d] = subCubic(tail, c1, c2, head, t0, tb);
  const path = `M ${f2(a[0])} ${f2(a[1])} C ${f2(b[0])} ${f2(b[1])} ${f2(c[0])} ${f2(c[1])} ${f2(d[0])} ${f2(d[1])}`;
  return { tail, tip, c1, c2, d: path, head: headPts, mid: cubicPoint(tail, c1, c2, head, 0.5) };
}

/** The paths of a reading's arrows; arrows with an unplaced atom are null. Arrows sharing their
 *  two ends alternate sides. */
export function layoutReading(arrows: readonly ArrowReadingOut[], ctx: ArrowGeomContext): (ArrowPath | null)[] {
  const seen = new Map<string, number>();
  return arrows.map((a) => {
    const ends = arrowEnds(a, ctx);
    if (!ends) return null;
    const key = [a.source.join(","), a.target.join(",")].sort().join("|");
    const k = seen.get(key) ?? 0;
    seen.set(key, k + 1);
    const s = ctx.bondPx / 40;
    return bezierArrow(ends.tail, ends.head, {
      centroid: ctx.centroid,
      side: k % 2 === 0 ? 1 : -1,
      bow: BOW * (1 + 0.5 * Math.floor(k / 2)),
      tailDir: ends.tailDir,
      minSpan: 0.9 * ctx.bondPx,
      gapTail: 3 * s,
      gapHead: 4 * s,
      headLength: Math.max(7, 9 * s),
      headWidth: Math.max(5.5, 7 * s),
    });
  });
}
