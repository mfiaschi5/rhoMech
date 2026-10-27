// "Tidy fragments": separate fragments of a sketch (salts, counter-ions, solvent) spread apart so
// that none touches another. Each fragment moves rigidly (a translation of all its atoms: bond
// lengths, angles, wedges and depths unchanged, so the topology and the stereo stay exactly the
// same); the largest fragment stays where it is.
//
// The distance between two fragments is measured on what the drawing style draws: the ink boxes
// of the labels (labelGlyphs.ts, real glyph metrics at the style's size), the bond strokes and
// the bare vertices. A fragment closer than the clearance (half a bond length, at least a font
// size) to an already placed one is pushed away along the line between the nearest points (the
// centroid direction when that is degenerate), repeatedly, until it is clear. A Screen-style
// sketch is measured at the 100 % zoom, where its fixed-pixel labels have their usual size.
//
// The Sketcher applies it after Clean as its own undoable step, and on the "Tidy fragments"
// button. Pure: runs under node --experimental-strip-types.
import type { SketchModel } from "@ksmech/viz/sketch/model";
import type { Vec, View } from "@ksmech/viz/sketch/geometry";
import type { DrawingStyle } from "@ksmech/viz/sketch/style";
import type { LabelStyle } from "@ksmech/viz/sketch/drawing";
import type { Box } from "@ksmech/viz/sketch/labelGlyphs";
import { buildScene } from "@ksmech/viz/sketch/drawing";
import { atomInk } from "@ksmech/viz/sketch/labelLayout";
import { BOND_LENGTH, moveAtoms, neighbors } from "@ksmech/viz/sketch/model";
import { drawMetrics, pxPerPt, SCREEN_STYLE } from "@ksmech/viz/sketch/style";

/** Pixels per model unit of the Screen style at 100 % (Sketcher.DEFAULT_SCALE). */
const SCREEN_SCALE = 25;

/** The connected components (atom ids), largest first, ties in atom order. */
export function fragments(m: SketchModel): number[][] {
  const seen = new Set<number>();
  const out: number[][] = [];
  for (const a of m.atoms) {
    if (seen.has(a.id)) continue;
    const comp: number[] = [];
    const stack = [a.id];
    seen.add(a.id);
    while (stack.length) {
      const u = stack.pop() as number;
      comp.push(u);
      for (const v of neighbors(m, u)) if (!seen.has(v)) (seen.add(v), stack.push(v));
    }
    out.push(comp);
  }
  return out.sort((x, y) => y.length - x.length);
}

/** What a fragment draws, in output units: label ink boxes and bond / vertex segments. */
interface Ink {
  boxes: Box[];
  segs: [Vec, Vec][];
}

export interface TidyOptions {
  style?: DrawingStyle;
  /** Further styles whose clearance must hold too (the Sketcher passes the publication
   *  presets, so a salt tidied on screen still does not touch in an ACS export). */
  alsoFor?: readonly DrawingStyle[];
  labelStyle?: LabelStyle;
}

function measureView(style: DrawingStyle): { view: View; M: ReturnType<typeof drawMetrics> } {
  const s = SCREEN_SCALE;
  return { view: { s, ox: 0, oy: 0 }, M: drawMetrics(style, pxPerPt(style, s)) };
}

function fragmentInk(m: SketchModel, hCounts: ReadonlyMap<number, number> | null, comps: number[][], o: TidyOptions): { inks: Ink[]; clearance: number; s: number } {
  const style = o.style ?? SCREEN_STYLE;
  const { view, M } = measureView(style);
  const scene = buildScene(m, view, hCounts, { metrics: M, labelStyle: o.labelStyle, placeLabels: false });
  const which = new Map<number, number>();
  comps.forEach((c, i) => c.forEach((id) => which.set(id, i)));
  const inks: Ink[] = comps.map(() => ({ boxes: [], segs: [] }));
  for (const a of scene.atoms) {
    const ink = inks[which.get(a.atom.id) ?? 0];
    ink.boxes.push(...atomInk(a, M));
    ink.segs.push([a.at, a.at]);
  }
  for (const b of scene.bonds) inks[which.get(b.bond.a) ?? 0].segs.push([b.axis.p, b.axis.q]);
  const bondPx = view.s * BOND_LENGTH;
  return { inks, clearance: Math.max(0.5 * bondPx, M.font), s: view.s };
}

const clampT = (t: number) => Math.max(0, Math.min(1, t));

/** The nearest points of two segments (sampled ends and projections; exact enough here). */
function nearestSegs(a: [Vec, Vec], b: [Vec, Vec]): [Vec, Vec] {
  const proj = (p: Vec, s: [Vec, Vec]): Vec => {
    const d: Vec = [s[1][0] - s[0][0], s[1][1] - s[0][1]];
    const l2 = d[0] * d[0] + d[1] * d[1];
    const t = l2 < 1e-12 ? 0 : clampT(((p[0] - s[0][0]) * d[0] + (p[1] - s[0][1]) * d[1]) / l2);
    return [s[0][0] + d[0] * t, s[0][1] + d[1] * t];
  };
  const cands: [Vec, Vec][] = [
    [a[0], proj(a[0], b)],
    [a[1], proj(a[1], b)],
    [proj(b[0], a), b[0]],
    [proj(b[1], a), b[1]],
  ];
  return cands.reduce((best, c) => (Math.hypot(c[0][0] - c[1][0], c[0][1] - c[1][1]) < Math.hypot(best[0][0] - best[1][0], best[0][1] - best[1][1]) ? c : best));
}

/** Nearest points of a box and a point (the point clamped into the box). */
const clampToBox = (p: Vec, b: Box): Vec => [Math.max(b.x0, Math.min(b.x1, p[0])), Math.max(b.y0, Math.min(b.y1, p[1]))];

function nearestBoxBox(a: Box, b: Box): [Vec, Vec] {
  const ax = Math.max(a.x0, Math.min(a.x1, (b.x0 + b.x1) / 2));
  const ay = Math.max(a.y0, Math.min(a.y1, (b.y0 + b.y1) / 2));
  const pb = clampToBox([ax, ay], b);
  return [clampToBox(pb, a), pb];
}

const boxSegs = (b: Box): [Vec, Vec][] => [
  [[b.x0, b.y0], [b.x1, b.y0]],
  [[b.x1, b.y0], [b.x1, b.y1]],
  [[b.x1, b.y1], [b.x0, b.y1]],
  [[b.x0, b.y1], [b.x0, b.y0]],
];

const inside = (p: Vec, b: Box) => p[0] >= b.x0 && p[0] <= b.x1 && p[1] >= b.y0 && p[1] <= b.y1;

/** The nearest pair of points between two inks and their distance (0 when they overlap). */
function nearest(a: Ink, b: Ink): { d: number; pa: Vec; pb: Vec } {
  let best = { d: Infinity, pa: [0, 0] as Vec, pb: [0, 0] as Vec };
  const consider = (pa: Vec, pb: Vec, overlap = false) => {
    const d = overlap ? 0 : Math.hypot(pa[0] - pb[0], pa[1] - pb[1]);
    if (d < best.d) best = { d, pa, pb };
  };
  const aSegs = [...a.segs, ...a.boxes.flatMap(boxSegs)];
  const bSegs = [...b.segs, ...b.boxes.flatMap(boxSegs)];
  for (const sa of aSegs) for (const sb of bSegs) consider(...nearestSegs(sa, sb));
  for (const ba of a.boxes) for (const bb of b.boxes) if (ba.x0 < bb.x1 && bb.x0 < ba.x1 && ba.y0 < bb.y1 && bb.y0 < ba.y1) consider(...nearestBoxBox(ba, bb), true);
  for (const ba of a.boxes) for (const s of b.segs) if (inside(s[0], ba)) consider(s[0], s[0], true);
  for (const bb of b.boxes) for (const s of a.segs) if (inside(s[0], bb)) consider(s[0], s[0], true);
  return best;
}

const centroidOf = (ink: Ink): Vec => {
  const pts = ink.segs.flatMap((s) => s);
  return [pts.reduce((t, p) => t + p[0], 0) / pts.length, pts.reduce((t, p) => t + p[1], 0) / pts.length];
};

/** The screen-space push (px) that moves fragment `f` clear of `placed`, or null when clear. */
function pushOf(f: Ink, placed: Ink[], clearance: number): Vec | null {
  let worst: { d: number; pa: Vec; pb: Vec; from: Ink } | null = null;
  for (const p of placed) {
    const n = nearest(p, f);
    if (n.d < clearance - 1e-6 && (!worst || n.d < worst.d)) worst = { ...n, from: p };
  }
  if (!worst) return null;
  let dir: Vec = [worst.pb[0] - worst.pa[0], worst.pb[1] - worst.pa[1]];
  if (worst.d < 1e-6) {
    const [c0, c1] = [centroidOf(worst.from), centroidOf(f)];
    dir = [c1[0] - c0[0], c1[1] - c0[1]];
  }
  const l = Math.hypot(dir[0], dir[1]);
  const u: Vec = l < 1e-9 ? [1, 0] : [dir[0] / l, dir[1] / l];
  const step = clearance - worst.d + clearance * 0.05;
  return [u[0] * step, u[1] * step];
}

const shiftInk = (ink: Ink, v: Vec): Ink => ({
  boxes: ink.boxes.map((b) => ({ x0: b.x0 + v[0], x1: b.x1 + v[0], y0: b.y0 + v[1], y1: b.y1 + v[1] })),
  segs: ink.segs.map(([p, q]) => [[p[0] + v[0], p[1] + v[1]], [q[0] + v[0], q[1] + v[1]]] as [Vec, Vec]),
});

/** The sketch with its fragments spread apart (see the header); `m` itself when nothing
 *  touches or there is a single fragment. */
export function tidyFragments(m: SketchModel, hCounts: ReadonlyMap<number, number> | null, o: TidyOptions = {}): SketchModel {
  const comps = fragments(m);
  if (comps.length < 2) return m;
  const styles = [o.style ?? SCREEN_STYLE, ...(o.alsoFor ?? []).filter((x) => x !== o.style)];
  // pushes only ever move fragments apart, so after one pass per style every clearance holds
  // (a second round catches a push in a later style that undid nothing but is cheap to check)
  let out = m;
  for (let round = 0; round < 2; round++) for (const style of styles) out = tidyIn(out, hCounts, comps, { ...o, style });
  return out;
}

function tidyIn(m: SketchModel, hCounts: ReadonlyMap<number, number> | null, comps: number[][], o: TidyOptions): SketchModel {
  const { inks, clearance, s } = fragmentInk(m, hCounts, comps, o);
  const placed: Ink[] = [inks[0]];
  let out = m;
  for (let i = 1; i < comps.length; i++) {
    let ink = inks[i];
    const total: [number, number] = [0, 0];
    for (let iter = 0; iter < 40; iter++) {
      const v = pushOf(ink, placed, clearance);
      if (!v) break;
      ink = shiftInk(ink, v);
      total[0] += v[0];
      total[1] += v[1];
    }
    placed.push(ink);
    if (total[0] !== 0 || total[1] !== 0) out = moveAtoms(out, comps[i], total[0] / s, -total[1] / s);
  }
  return out;
}

/** True when some fragment is closer than the clearance to another (the button's state). */
export function fragmentsTouch(m: SketchModel, hCounts: ReadonlyMap<number, number> | null, o: TidyOptions = {}): boolean {
  return tidyFragments(m, hCounts, o) !== m;
}
