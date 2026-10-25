// Plane geometry of the sketcher: vectors, growth angles, regular polygons, zigzag chains and
// hit tests. Pure functions without imports, so they run under node --experimental-strip-types.
// Angles are radians, counter-clockwise, in the model frame (y up).

export type Vec = readonly [number, number];

const TAU = 2 * Math.PI;
export const DEG = Math.PI / 180;

export const add = (p: Vec, q: Vec): Vec => [p[0] + q[0], p[1] + q[1]];
export const sub = (p: Vec, q: Vec): Vec => [p[0] - q[0], p[1] - q[1]];
export const scale = (p: Vec, k: number): Vec => [p[0] * k, p[1] * k];
export const len = (p: Vec): number => Math.hypot(p[0], p[1]);
export const dist = (p: Vec, q: Vec): number => Math.hypot(p[0] - q[0], p[1] - q[1]);
export const cross = (p: Vec, q: Vec): number => p[0] * q[1] - p[1] * q[0];
export const dot = (p: Vec, q: Vec): number => p[0] * q[0] + p[1] * q[1];
export const fromAngle = (t: number, r = 1): Vec => [r * Math.cos(t), r * Math.sin(t)];
export const angleOf = (p: Vec): number => Math.atan2(p[1], p[0]);

export function rotate(p: Vec, t: number, about: Vec = [0, 0]): Vec {
  const c = Math.cos(t);
  const s = Math.sin(t);
  const x = p[0] - about[0];
  const y = p[1] - about[1];
  return [about[0] + c * x - s * y, about[1] + s * x + c * y];
}

/** An angle in [0, 2 pi). */
export function normAngle(t: number): number {
  const r = t % TAU;
  return r < 0 ? r + TAU : r;
}

/** Snap an angle to the nearest multiple of `step` (30 degrees by default). */
export function snapAngle(t: number, step = 30 * DEG): number {
  return Math.round(t / step) * step;
}

/** The first bond of an isolated atom points up-right at 30 degrees (zigzag-ready). */
export const FIRST_BOND_ANGLE = 30 * DEG;

/** Bisector of the largest angular gap between the neighbour directions. One neighbour: the
 *  opposite direction. None: `fallback`. */
export function largestGapDirection(centre: Vec, nbrs: readonly Vec[], fallback = -90 * DEG): number {
  if (nbrs.length === 0) return fallback;
  const angles = nbrs.map((q) => normAngle(angleOf(sub(q, centre)))).sort((a, b) => a - b);
  if (angles.length === 1) return normAngle(angles[0] + Math.PI);
  let best = 0;
  let bestGap = -1;
  for (let i = 0; i < angles.length; i++) {
    const next = i + 1 < angles.length ? angles[i + 1] : angles[0] + TAU;
    const gap = next - angles[i];
    if (gap > bestGap + 1e-9) {
      bestGap = gap;
      best = angles[i] + gap / 2;
    }
  }
  return normAngle(best);
}

/** Direction of a new bond grown from `centre`. No neighbour: 30 degrees. One neighbour: 120
 *  degrees from it, on the side whose end is farthest from the `others` atoms (this gives the
 *  zigzag of a chain). Two or more: the bisector of the largest free gap. */
export function growthAngle(centre: Vec, nbrs: readonly Vec[], others: readonly Vec[], length: number): number {
  if (nbrs.length === 0) return FIRST_BOND_ANGLE;
  if (nbrs.length > 1) return largestGapDirection(centre, nbrs);
  const t = angleOf(sub(nbrs[0], centre));
  const cands = [t - 120 * DEG, t + 120 * DEG].map(normAngle);
  const room = (a: number) => {
    const end = add(centre, fromAngle(a, length));
    return others.reduce((m, q) => Math.min(m, dist(end, q)), Infinity);
  };
  const [r0, r1] = cands.map(room);
  if (Math.abs(r0 - r1) > 1e-6) return r0 > r1 ? cands[0] : cands[1];
  // Tie (nothing else around): prefer the more horizontal end (a horizontal zigzag), then the
  // rightward one, then the upward one; deterministic.
  const pref = (a: number) => Math.abs(Math.cos(a)) + 0.1 * Math.cos(a) + 0.01 * Math.sin(a);
  return pref(cands[0]) >= pref(cands[1]) ? cands[0] : cands[1];
}

/** Circumradius of a regular n-gon with side `side`. */
export const circumradius = (n: number, side: number): number => side / (2 * Math.sin(Math.PI / n));

/** Regular n-gon around `centre`, pointy top (vertex 0 straight up), counter-clockwise. */
export function polygonAt(centre: Vec, n: number, side: number): Vec[] {
  const r = circumradius(n, side);
  return Array.from({ length: n }, (_, k) => add(centre, fromAngle(90 * DEG + (k * TAU) / n, r)));
}

/** Regular n-gon with vertex 0 at `p` and its centre in direction `dir` from p. */
export function polygonFromVertex(p: Vec, dir: number, n: number, side: number): Vec[] {
  const c = add(p, fromAngle(dir, circumradius(n, side)));
  return Array.from({ length: n }, (_, k) => rotate(p, (k * TAU) / n, c));
}

/** Regular n-gon on the edge p->q (vertex 0 = p, vertex 1 = q), built on side `side` of the
 *  edge (+1 left of p->q, -1 right). The side length is |pq|. */
export function polygonOnEdge(p: Vec, q: Vec, n: number, side: 1 | -1): Vec[] {
  const e = sub(q, p);
  const l = len(e);
  const normal: Vec = scale([-e[1], e[0]], side / l);
  const apothem = l / (2 * Math.tan(Math.PI / n));
  const c = add(scale(add(p, q), 0.5), scale(normal, apothem));
  const step = cross(sub(p, c), sub(q, c)) > 0 ? TAU / n : -TAU / n;
  return Array.from({ length: n }, (_, k) => (k === 0 ? p : k === 1 ? q : rotate(p, k * step, c)));
}

/** Which side of the line p->q a point lies on: +1 left, -1 right, 0 on it. */
export function sideOf(p: Vec, q: Vec, x: Vec): 1 | -1 | 0 {
  const c = cross(sub(q, p), sub(x, p));
  return Math.abs(c) < 1e-9 ? 0 : c > 0 ? 1 : -1;
}

/** Zigzag chain points from `start` towards `target` (the drag), the drag direction snapped to
 *  30 degrees; bonds of length `bond` at +-30 degrees about it. Excludes `start`. `flip` mirrors
 *  the zigzag. */
export function chainPoints(start: Vec, target: Vec, bond: number, flip = false): Vec[] {
  const d = sub(target, start);
  const along = bond * Math.cos(30 * DEG);
  const n = Math.max(1, Math.round(len(d) / along));
  const phi = snapAngle(angleOf(d));
  const u = fromAngle(phi);
  const v: Vec = [-u[1], u[0]];
  const h = (flip ? -1 : 1) * bond * Math.sin(30 * DEG);
  return Array.from({ length: n }, (_, i) => add(add(start, scale(u, (i + 1) * along)), scale(v, (i + 1) % 2 ? h : 0)));
}

/** Distance from x to the segment p-q. */
export function segmentDistance(x: Vec, p: Vec, q: Vec): number {
  const e = sub(q, p);
  const l2 = dot(e, e);
  if (l2 < 1e-12) return dist(x, p);
  const t = Math.max(0, Math.min(1, dot(sub(x, p), e) / l2));
  return dist(x, add(p, scale(e, t)));
}

export interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export function rectFrom(p: Vec, q: Vec): Rect {
  return { x0: Math.min(p[0], q[0]), y0: Math.min(p[1], q[1]), x1: Math.max(p[0], q[0]), y1: Math.max(p[1], q[1]) };
}

export const inRect = (r: Rect, p: Vec): boolean => p[0] >= r.x0 && p[0] <= r.x1 && p[1] >= r.y0 && p[1] <= r.y1;

/** Screen <-> model transform: X = s x + ox, Y = -s y + oy (model y up, screen y down). */
export interface View {
  s: number;
  ox: number;
  oy: number;
}

export const toScreen = (v: View, p: Vec): Vec => [v.s * p[0] + v.ox, -v.s * p[1] + v.oy];
export const toModel = (v: View, p: Vec): Vec => [(p[0] - v.ox) / v.s, -(p[1] - v.oy) / v.s];

/** The view that fits the points in a w x h box with `pad` px margins, at most `maxScale`. */
export function fitView(points: readonly Vec[], w: number, h: number, maxScale: number, pad = 28): View {
  if (points.length === 0) return { s: maxScale, ox: w / 2, oy: h / 2 };
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const sx = (w - 2 * pad) / Math.max(x1 - x0, 1e-6);
  const sy = (h - 2 * pad) / Math.max(y1 - y0, 1e-6);
  const s = Math.max(4, Math.min(maxScale, sx, sy));
  return { s, ox: w / 2 - (s * (x0 + x1)) / 2, oy: h / 2 + (s * (y0 + y1)) / 2 };
}
