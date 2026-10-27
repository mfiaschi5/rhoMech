// Textbook geometry of a step drawing, in screen px (pure; node --test runs it): where the
// lone-pair dots of an atom go, and curved arrows routed clear of each other and of the drawing.
//
// Lone pairs (lonePairSlots): a pair sits in one of eight directions around its atom (the four
// sides first, then the diagonals), clear of the bonds, the H of the label and the charge; the
// slots are chosen greedily, most open first, and a `prefer` direction (where the
// arrow that takes the pair is heading) is served first (bonds keep 40° clear, the label's H
// and charge 35°; a side beats a diagonal unless it is much more crowded). pairDots puts the two dots just outside
// the label's box in that direction.
//
// Arrows (routeArrows): each arrow is a cubic from its tail (a lone pair, leaving along the
// pair's direction, or a bond's midpoint) to its head. Candidates bow to either side of the
// chord by 0.28, 0.42 or 0.6 of its length. A candidate costs: crossing a bond (3 each, the
// first and last sixth of the curve exempt, where it leaves or meets a bond), passing within an
// atom's radius (up to 2 per sample, end atoms exempt), bowing towards the drawing's centre
// (0.3) and its bow (small). Two arrows that cross cost 5. Up to 5 arrows every combination is
// tried (6^n); beyond, one arrow at a time, in order.
import type { Vec } from "../pool/arrowPath";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { cubicPoint } from "../pool/arrowPath.ts";

const DEG = Math.PI / 180;
export const SLOT_ANGLES = [-90, 0, 90, 180, -45, 45, 135, -135].map((d) => d * DEG);
export const SLOT_CLEARANCE = 40 * DEG;
export const BLOCK_CLEARANCE = 35 * DEG;
/** A side slot beats a diagonal one with up to this much less room. */
export const SIDE_BONUS = 20 * DEG;

export const angDist = (a: number, b: number): number => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
const minDist = (a: number, list: readonly number[]) => list.reduce((m, t) => Math.min(m, angDist(a, t)), Math.PI);

/** The directions (screen radians, y down) of `n` lone pairs around an atom whose bonds leave
 *  at `bondDirs` (40° clearance), whose label's H or charge takes `blocked` (35°) and whose
 *  `hard` slots would collide with something else (those exact slots only). */
export function lonePairSlots(bondDirs: readonly number[], blocked: readonly number[], n: number, prefer?: number | null, hard: readonly number[] = []): number[] {
  const out: number[] = [];
  const room = (a: number) => Math.min(minDist(a, bondDirs) - SLOT_CLEARANCE, minDist(a, blocked) - BLOCK_CLEARANCE, minDist(a, hard) - 10 * DEG);
  const free = (a: number) => room(a) >= 0 && minDist(a, out) >= 60 * DEG;
  if (n > 0 && prefer != null) {
    const ok = SLOT_ANGLES.filter(free).sort((a, b) => angDist(a, prefer) - angDist(b, prefer));
    if (ok.length) out.push(ok[0]);
  }
  while (out.length < n) {
    const cands = SLOT_ANGLES.filter((a) => !out.some((o) => angDist(o, a) < 1e-6));
    if (!cands.length) break;
    // most open first; the sides win ties (SLOT_ANGLES order)
    let best = cands[0];
    let score = -Infinity;
    for (const a of cands) {
      const side = SLOT_ANGLES.indexOf(a) < 4 ? SIDE_BONUS : 0;
      const raw = Math.min(room(a), minDist(a, out) - 60 * DEG);
      const s = raw + (raw >= 0 ? side : 0);
      if (s > score + 1e-6) [best, score] = [a, s];
    }
    out.push(best);
  }
  return out;
}

export interface LabelBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** Distance from the box's origin (the atom) to its edge along `angle`. */
export function boxReach(box: LabelBox, angle: number): number {
  const [c, s] = [Math.cos(angle), Math.sin(angle)];
  const tx = c > 1e-9 ? box.x1 / c : c < -1e-9 ? box.x0 / c : Infinity;
  const ty = s > 1e-9 ? box.y1 / s : s < -1e-9 ? box.y0 / s : Infinity;
  return Math.max(0, Math.min(tx, ty));
}

/** The two dots of a pair in direction `angle` from `at`, `reach + gap` out, `sep` apart. */
export function pairDots(at: Vec, angle: number, dist: number, sep: number): [Vec, Vec] {
  const [c, s] = [Math.cos(angle), Math.sin(angle)];
  const m: Vec = [at[0] + c * dist, at[1] + s * dist];
  return [
    [m[0] - s * (sep / 2), m[1] + c * (sep / 2)],
    [m[0] + s * (sep / 2), m[1] - c * (sep / 2)],
  ];
}

export interface ArrowSpec {
  tail: Vec;
  head: Vec;
  /** The lone pair's direction at a lone-pair tail. */
  tailDir?: Vec | null;
  /** The direction the head's lone pair faces (a bond_to_lone_pair head arrives from it). */
  headDir?: Vec | null;
  /** Atoms at the arrow's ends (never obstacles to it). */
  ends: readonly number[];
  /** The ends to use when the arrow bows to side -1 (a lone-pair head sits on the bow side). */
  alt?: ArrowSpec | null;
}

export interface RouteObstacles {
  atoms: readonly { i: number; at: Vec; r: number }[];
  bonds: readonly (readonly [Vec, Vec])[];
}

export interface RoutedArrow {
  tail: Vec;
  head: Vec;
  c1: Vec;
  c2: Vec;
  side: 1 | -1;
  bow: number;
}

export const BOWS = [0.28, 0.42, 0.6];
const SAMPLES = 24;

const sub = (p: Vec, q: Vec): Vec => [p[0] - q[0], p[1] - q[1]];
const add = (p: Vec, q: Vec): Vec => [p[0] + q[0], p[1] + q[1]];
const mul = (p: Vec, k: number): Vec => [p[0] * k, p[1] * k];
const len = (p: Vec) => Math.hypot(p[0], p[1]);
const unit = (p: Vec): Vec => (len(p) > 1e-9 ? mul(p, 1 / len(p)) : [0, -1]);

/** The control points of one candidate. */
export function candidate(a: ArrowSpec, side: 1 | -1, bow: number, minSpan: number): RoutedArrow {
  const v = sub(a.head, a.tail);
  const span = Math.max(len(v), minSpan);
  const n: Vec = mul([-unit(v)[1], unit(v)[0]], side * bow * span);
  let c1 = add(add(a.tail, mul(v, 1 / 3)), n);
  let c2 = add(add(a.tail, mul(v, 2 / 3)), n);
  if (a.tailDir) c1 = add(a.tail, mul(unit(add(unit(a.tailDir), unit(sub(c1, a.tail)))), 0.5 * span));
  if (a.headDir) c2 = add(a.head, mul(unit(add(unit(a.headDir), unit(sub(c2, a.head)))), 0.45 * span));
  return { tail: a.tail, head: a.head, c1, c2, side, bow };
}

export function polyline(r: RoutedArrow, n = SAMPLES): Vec[] {
  return Array.from({ length: n + 1 }, (_, k) => cubicPoint(r.tail, r.c1, r.c2, r.head, k / n));
}

/** Do segments p1-p2 and q1-q2 properly cross? */
export function segmentsCross(p1: Vec, p2: Vec, q1: Vec, q2: Vec): boolean {
  const d = (a: Vec, b: Vec, c: Vec) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const [d1, d2, d3, d4] = [d(q1, q2, p1), d(q1, q2, p2), d(p1, p2, q1), d(p1, p2, q2)];
  return d1 * d2 < 0 && d3 * d4 < 0;
}

/** Crossings between two polylines. */
export function crossings(a: readonly Vec[], b: readonly Vec[]): number {
  let n = 0;
  for (let i = 0; i + 1 < a.length; i++) for (let j = 0; j + 1 < b.length; j++) if (segmentsCross(a[i], a[i + 1], b[j], b[j + 1])) n++;
  return n;
}

function soloCost(a: ArrowSpec, r: RoutedArrow, pts: readonly Vec[], obs: RouteObstacles, centroid: Vec): number {
  let cost = 0.4 * (r.bow - BOWS[0]);
  const lo = Math.ceil(pts.length / 6);
  const inner = pts.slice(lo, pts.length - lo);
  for (const [p, q] of obs.bonds) cost += 3 * crossings(inner, [p, q]);
  for (const o of obs.atoms) {
    if (a.ends.includes(o.i)) continue;
    for (const p of inner) {
      const d = len(sub(p, o.at));
      if (d < o.r) cost += (2 * (o.r - d)) / o.r;
    }
  }
  const mid = cubicPoint(r.tail, r.c1, r.c2, r.head, 0.5);
  const chordMid = mul(add(r.tail, r.head), 0.5);
  const out = sub(mid, chordMid);
  if (out[0] * (chordMid[0] - centroid[0]) + out[1] * (chordMid[1] - centroid[1]) < 0) cost += 0.3;
  return cost;
}

interface Option {
  r: RoutedArrow;
  pts: Vec[];
  cost: number;
}

function optionsFor(a: ArrowSpec, obs: RouteObstacles, centroid: Vec, minSpan: number): Option[] {
  const out: Option[] = [];
  for (const side of [1, -1] as const)
    for (const bow of BOWS) {
      const spec = side === -1 && a.alt ? a.alt : a;
      const r = candidate(spec, side, bow, minSpan);
      const pts = polyline(r);
      out.push({ r, pts, cost: soloCost(spec, r, pts, obs, centroid) });
    }
  return out;
}

/** The arrows routed (see the header). */
export function routeArrows(specs: readonly ArrowSpec[], obs: RouteObstacles, centroid: Vec, minSpan = 24): RoutedArrow[] {
  const opts = specs.map((a) => optionsFor(a, obs, centroid, minSpan));
  if (!specs.length) return [];
  const pair = (x: Option, y: Option) => 5 * crossings(x.pts, y.pts);
  if (specs.length > 5) {
    const chosen: Option[] = [];
    for (const o of opts) chosen.push(o.reduce((b, c) => (c.cost + chosen.reduce((s, x) => s + pair(c, x), 0) < b.cost + chosen.reduce((s, x) => s + pair(b, x), 0) ? c : b)));
    return chosen.map((o) => o.r);
  }
  let best: { pick: number[]; cost: number } = { pick: [], cost: Infinity };
  const pick: number[] = [];
  const walk = (k: number, acc: number) => {
    if (acc >= best.cost) return;
    if (k === opts.length) return void (best = { pick: [...pick], cost: acc });
    for (let c = 0; c < opts[k].length; c++) {
      const o = opts[k][c];
      const extra = pick.reduce((s, p, m) => s + pair(o, opts[m][p]), 0);
      pick.push(c);
      walk(k + 1, acc + o.cost + extra);
      pick.pop();
    }
  };
  walk(0, 0);
  return best.pick.map((c, k) => opts[k][c].r);
}

/** Crossings between routed arrows (0 for a clean scheme). */
export function arrowCrossings(rs: readonly RoutedArrow[]): number {
  let n = 0;
  const ps = rs.map((r) => polyline(r));
  for (let a = 0; a < ps.length; a++) for (let b = a + 1; b < ps.length; b++) n += crossings(ps[a], ps[b]);
  return n;
}
