// From sides (stepGeometry.ts) to what the step drawings render, in screen px (pure; node --test
// runs it): a side as a workbench sketch model (sideModel; atom id = source pool atom + an
// offset per layer), the view that fits a scheme in its box (fitScheme), the lone-pair dots of
// the atoms a step touches (lonePairLayout, stepRoute.ts slots) and the textbook arrows of a
// reading routed clear of the drawing (routeReading).
//
// Arrow ends (textbook):
//   lone_pair_to_bond  from the donor's lone pair (the slot facing the target) to the attacked
//                      atom's label edge, or the forming bond's midpoint when the atoms are
//                      within NEAR_BOND bond lengths;
//   bond_to_bond       from the breaking bond's midpoint to the same kind of head;
//   bond_to_lone_pair  from the bond's midpoint to the label edge of the atom that takes the
//                      pair, 50° off the bond on the side the arrow bows to, arriving from outside.
import type { SketchModel } from "../workbench/sketch/model";
import type { Scene } from "../workbench/sketch/drawing";
import type { DrawMetrics } from "../workbench/sketch/style";
import type { View } from "../workbench/sketch/geometry";
import type { ArrowReadingOut } from "./v9types";
import type { Vec } from "./arrowPath";
import type { Side, XY } from "./stepGeometry";
import type { LabelBox, RouteObstacles } from "./stepRoute";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { labelBox, labelGlyphs } from "../workbench/sketch/drawing.ts";
// @ts-ignore TS5097: see above
import { BOND_LENGTH } from "../workbench/sketch/model.ts";
// @ts-ignore TS5097: see above
import { NEAR_BOND } from "./arrowPath.ts";
// @ts-ignore TS5097: see above
import { SLOT_ANGLES, boxReach, lonePairSlots, pairDots, routeArrows, type ArrowSpec, type RoutedArrow } from "./stepRoute.ts";

export interface SideModel {
  model: SketchModel;
  hCounts: Map<number, number>;
}

/** A side as a sketch (atom ids: pool atom + idOffset; bond ids after them), at `xy` (model
 *  units) or the side's own positions; `bonds` overrides the bond orders (0: not drawn). The
 *  atoms R touches are labelled; with `labelCarbons` false a touched carbon stays a vertex (the
 *  textbook look). Terminal carbons with H are always written out (H2C=, H3C-). */
export function sideModel(side: Side, idOffset = 0, xy?: XY | null, bonds?: ReadonlyMap<string, number> | null, labelCarbons = true): SideModel {
  const degree = new Map<number, number>();
  for (const b of side.bonds.values()) for (const i of [b.i, b.j]) degree.set(i, (degree.get(i) ?? 0) + 1);
  const atoms = [...side.atoms.values()].map((a) => {
    const p = xy?.get(a.index) ?? [a.x, a.y];
    // a terminal carbon reads as its group (H2C=, H3C-), never as a bare line end
    const terminal = a.el === "C" && (degree.get(a.index) ?? 0) <= 1 && a.hCount > 0;
    const flag = terminal || (a.labelled && (labelCarbons || a.el !== "C"));
    return { id: a.index + idOffset, el: a.el, x: p[0], y: p[1], charge: a.charge, ...(flag ? { explicitH: true } : {}) };
  });
  let next = Math.max(0, ...atoms.map((a) => a.id)) + 1;
  const list = bonds ? [...bonds].map(([key, order]) => ({ key, order })) : [...side.bonds].map(([key, b]) => ({ key, order: b.order }));
  const ids = new Set(atoms.map((a) => a.id));
  const out = [];
  for (const { key, order } of list) {
    const [i, j] = key.split("-").map(Number);
    if (order < 1 || !ids.has(i + idOffset) || !ids.has(j + idOffset)) continue;
    out.push({ id: next++, a: i + idOffset, b: j + idOffset, order: Math.min(3, order) as 1 | 2 | 3, stereo: "none" as const });
  }
  const hCounts = new Map([...side.atoms.values()].map((a) => [a.index + idOffset, a.hCount] as const));
  return { model: { atoms, bonds: out, nextId: next }, hCounts };
}

/** The bond key ("i-j", pool atoms) of each bond id of a side model. */
export function bondKeys(m: SketchModel, idOffset = 0): Map<number, string> {
  return new Map(m.bonds.map((b) => [b.id, `${Math.min(b.a, b.b) - idOffset}-${Math.max(b.a, b.b) - idOffset}`] as const));
}

export interface FitOptions {
  /** Margin around the atoms, in bond lengths (labels, dots and arrows live there). */
  padBonds: number;
  /** Largest and smallest standard bond, px. */
  maxBondPx: number;
  minBondPx: number;
  /** Height limits (px). */
  maxHeight: number;
  minHeight: number;
}

export const DEFAULT_FIT: FitOptions = { padBonds: 0.9, maxBondPx: 46, minBondPx: 18, maxHeight: 420, minHeight: 150 };

/** The view and height that fit model box `b` in `width` px: as large as the limits allow,
 *  centred, the height following the content (no tall empty canvas). */
export function fitScheme(b: { x0: number; y0: number; x1: number; y1: number }, width: number, o: FitOptions = DEFAULT_FIT): { view: View; height: number; bondPx: number } {
  const padM = o.padBonds * BOND_LENGTH;
  const w = b.x1 - b.x0 + 2 * padM;
  const h = b.y1 - b.y0 + 2 * padM;
  const sMax = o.maxBondPx / BOND_LENGTH;
  const sMin = o.minBondPx / BOND_LENGTH;
  const s = Math.max(sMin, Math.min(sMax, width / Math.max(w, 1e-6), o.maxHeight / Math.max(h, 1e-6)));
  const height = Math.max(o.minHeight, Math.min(o.maxHeight, h * s));
  return { view: { s, ox: width / 2 - (s * (b.x0 + b.x1)) / 2, oy: height / 2 + (s * (b.y0 + b.y1)) / 2 }, height, bondPx: s * BOND_LENGTH };
}

export interface AtomGeom {
  at: Vec;
  /** The label's box about the atom (a bare vertex: a small square). */
  box: LabelBox;
  /** The element symbol's ink box about the atom (where lone pairs and arrow heads sit). */
  core: LabelBox;
  /** Ink boxes of the label's other runs (H, H count, charge) about the atom. */
  extras: LabelBox[];
  bondDirs: number[];
  /** Directions the label's H and charge take. */
  blocked: number[];
}

/** Per atom id: position, label box, bond directions and label-blocked directions. */
export function atomGeometry(scene: Scene, M: DrawMetrics): Map<number, AtomGeom> {
  const out = new Map<number, AtomGeom>();
  const dirs = new Map<number, number[]>();
  for (const b of scene.bonds) {
    const [p, q] = [b.axis.p, b.axis.q];
    dirs.set(b.bond.a, [...(dirs.get(b.bond.a) ?? []), Math.atan2(q[1] - p[1], q[0] - p[0])]);
    dirs.set(b.bond.b, [...(dirs.get(b.bond.b) ?? []), Math.atan2(p[1] - q[1], p[0] - q[0])]);
  }
  for (const a of scene.atoms) {
    const l = a.label;
    // the directions of the label's H, H count and charge as drawn (placement may move them)
    const glyphs = l ? labelGlyphs(l, M) : [];
    const blocked = glyphs.filter((g) => g.kind !== "symbol").map((g) => Math.atan2(g.y - (g.kind === "charge" ? g.size * 0.36 : g.size * 0.36), g.x + g.w / 2));
    const r = M.font * 0.2;
    const sym = glyphs[0] ?? null;
    const hh = sym ? sym.size * 0.38 : r;
    const core = sym ? { x0: sym.x, x1: sym.x + sym.w, y0: -hh, y1: hh } : { x0: -r, y0: -r, x1: r, y1: r };
    const extras = glyphs.slice(1).map((g) => (g.kind === "charge" ? { x0: g.x, x1: g.x + g.w, y0: g.y - g.size * 0.36 - g.w / 2, y1: g.y - g.size * 0.36 + g.w / 2 } : { x0: g.x, x1: g.x + g.w, y0: g.y - g.size * 0.72, y1: g.y }));
    out.set(a.atom.id, { at: a.at, box: l ? labelBox(l, M) : core, core, extras, bondDirs: dirs.get(a.atom.id) ?? [], blocked });
  }
  return out;
}

export interface LonePairMark {
  /** Atom id. */
  id: number;
  angles: number[];
  dots: [Vec, Vec][];
  /** Centre of each pair. */
  centres: Vec[];
}

export const dotRadius = (M: DrawMetrics): number => Math.max(1.3, M.font * 0.085);

/** Lone-pair dots of the atoms in `want` (id -> {n, prefer}). A slot whose pair would land on
 *  another atom's label, a bond or a pair already placed counts as blocked. */
export function lonePairLayout(geom: ReadonlyMap<number, AtomGeom>, want: ReadonlyMap<number, { n: number; prefer?: number | null }>, M: DrawMetrics, bonds: readonly (readonly [Vec, Vec])[] = []): Map<number, LonePairMark> {
  const out = new Map<number, LonePairMark>();
  const placed: Vec[] = [];
  for (const [id, w] of want) {
    const g = geom.get(id);
    if (!g || w.n <= 0) continue;
    const dist = (a: number) => boxReach(g.core, a) + M.font * 0.2;
    const centre = (a: number): Vec => [g.at[0] + Math.cos(a) * dist(a), g.at[1] + Math.sin(a) * dist(a)];
    const clash = (a: number) => {
      const c = centre(a);
      for (const [j, o] of geom) if (j !== id && insideBox(c, o, M.font * 0.25)) return true;
      const pad = dotRadius(M) + M.font * 0.06;
      const dots = pairDots(g.at, a, dist(a), M.font * 0.34);
      if (g.extras.some((b) => dots.some((d) => d[0] - g.at[0] > b.x0 - pad && d[0] - g.at[0] < b.x1 + pad && d[1] - g.at[1] > b.y0 - pad && d[1] - g.at[1] < b.y1 + pad))) return true;
      if (placed.some((p) => Math.hypot(p[0] - c[0], p[1] - c[1]) < M.font * 0.95)) return true;
      return bonds.some(([p, q]) => segDist(c, p, q) < M.font * 0.22);
    };
    const angles = lonePairSlots(g.bondDirs, g.blocked, w.n, w.prefer, SLOT_ANGLES.filter(clash));
    const dots = angles.map((a) => pairDots(g.at, a, dist(a), M.font * 0.34));
    const centres = angles.map(centre);
    placed.push(...centres);
    out.set(id, { id, angles, dots, centres });
  }
  return out;
}

function insideBox(c: Vec, g: AtomGeom, pad: number): boolean {
  const [x, y] = [c[0] - g.at[0], c[1] - g.at[1]];
  return x > g.box.x0 - pad && x < g.box.x1 + pad && y > g.box.y0 - pad && y < g.box.y1 + pad;
}

function segDist(c: Vec, p: Vec, q: Vec): number {
  const [dx, dy] = [q[0] - p[0], q[1] - p[1]];
  const L2 = dx * dx + dy * dy;
  const t = L2 > 0 ? Math.max(0, Math.min(1, ((c[0] - p[0]) * dx + (c[1] - p[1]) * dy) / L2)) : 0;
  return Math.hypot(c[0] - p[0] - t * dx, c[1] - p[1] - t * dy);
}

const sub = (p: Vec, q: Vec): Vec => [p[0] - q[0], p[1] - q[1]];
const len = (p: Vec) => Math.hypot(p[0], p[1]);
const unit = (p: Vec): Vec => (len(p) > 1e-9 ? [p[0] / len(p), p[1] / len(p)] : [0, -1]);
const mid = (p: Vec, q: Vec): Vec => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];

/** The direction each donor's arrow heads in (for its lone-pair slot): atom id -> angle. */
export function donorPreferences(arrows: readonly ArrowReadingOut[], geom: ReadonlyMap<number, AtomGeom>, idOf: (i: number) => number): Map<number, number> {
  const out = new Map<number, number>();
  for (const a of arrows) {
    if (a.source.length !== 1) continue;
    const s = geom.get(idOf(a.source[0]));
    const t = a.target.map((i) => geom.get(idOf(i))?.at).filter((p): p is Vec => !!p);
    if (!s || !t.length) continue;
    const goal = t.length === 2 ? (a.target[0] === a.source[0] ? t[1] : a.target[1] === a.source[0] ? t[0] : mid(t[0], t[1])) : t[0];
    out.set(idOf(a.source[0]), Math.atan2(goal[1] - s.at[1], goal[0] - s.at[0]));
  }
  return out;
}

/** The head of an arrow that points at atom `b` coming from `a` (a forming bond). */
function bondHead(a: AtomGeom, b: AtomGeom, bondPx: number, gap: number): Vec {
  const d = sub(b.at, a.at);
  if (len(d) <= NEAR_BOND * bondPx) return mid(a.at, b.at);
  const u = unit(d);
  const back = boxReach(b.box, Math.atan2(-u[1], -u[0])) + gap;
  return [b.at[0] - u[0] * back, b.at[1] - u[1] * back];
}

/** One arrow's spec; null when an atom is not drawn. `side` matters for a lone-pair head only. */
function specOf(a: ArrowReadingOut, geom: ReadonlyMap<number, AtomGeom>, pairs: ReadonlyMap<number, LonePairMark>, idOf: (i: number) => number, bondPx: number, gap: number, side: 1 | -1): ArrowSpec | null {
  const G = (i: number) => geom.get(idOf(i));
  let tail: Vec;
  let tailDir: Vec | null = null;
  if (a.source.length === 1) {
    const lp = pairs.get(idOf(a.source[0]));
    const g = G(a.source[0]);
    if (!g) return null;
    const ang = lp?.angles[0] ?? 0;
    tailDir = [Math.cos(ang), Math.sin(ang)];
    tail = lp ? [lp.centres[0][0] + tailDir[0] * gap * 0.6, lp.centres[0][1] + tailDir[1] * gap * 0.6] : g.at;
  } else {
    const [p, q] = [G(a.source[0]), G(a.source[1])];
    if (!p || !q) return null;
    tail = mid(p.at, q.at);
  }
  if (a.target.length === 2) {
    const [x, y] = [G(a.target[0]), G(a.target[1])];
    if (!x || !y) return null;
    const fromFirst = a.source.includes(a.target[0]) && !a.source.includes(a.target[1]);
    const head = fromFirst ? bondHead(x, y, bondPx, gap) : a.source.includes(a.target[1]) ? bondHead(y, x, bondPx, gap) : mid(x.at, y.at);
    return { tail, head, tailDir, ends: [...a.source, ...a.target].map(idOf) };
  }
  const t = G(a.target[0]);
  if (!t) return null;
  const partner = a.source.find((i) => i !== a.target[0]);
  const away = partner !== undefined && G(partner) ? unit(sub(t.at, G(partner)!.at)) : unit(sub(t.at, tail));
  const base = Math.atan2(away[1], away[0]);
  const chord = unit(sub(t.at, tail));
  // the bow side's normal decides which way off the bond the head goes; among 35-80° off it,
  // the direction farthest from the atom's drawn lone pairs and bonds
  const nx = -chord[1] * side;
  const ny = chord[0] * side;
  const turn = Math.sign(Math.cos(base) * ny - Math.sin(base) * nx) || 1;
  const taken = [...(pairs.get(idOf(a.target[0]))?.angles ?? []), ...t.bondDirs];
  const room = (x: number) => taken.reduce((m, y) => Math.min(m, Math.abs(Math.atan2(Math.sin(x - y), Math.cos(x - y)))), Math.PI);
  const hang = [35, 50, 65, 80].map((d) => base + (turn * d * Math.PI) / 180).reduce((b, x) => (room(x) > room(b) + 1e-6 ? x : b));
  const hd: Vec = [Math.cos(hang), Math.sin(hang)];
  const reach = boxReach(t.core, hang) + gap * 1.6;
  return { tail, head: [t.at[0] + hd[0] * reach, t.at[1] + hd[1] * reach], tailDir, headDir: hd, ends: [...a.source, ...a.target].map(idOf) };
}

/** Obstacles for the router: every atom (its label's reach) and every bond. */
export function obstaclesOf(scene: Scene, geom: ReadonlyMap<number, AtomGeom>, M: DrawMetrics): RouteObstacles {
  const atoms = [...geom].map(([i, g]) => ({ i, at: g.at, r: Math.max(M.font * 0.55, (g.box.x1 - g.box.x0) / 2 + M.font * 0.15) }));
  return { atoms, bonds: scene.bonds.map((b) => [b.axis.p, b.axis.q] as const) };
}

/** The reading's arrows routed (stepRoute.ts) on a drawn scene; idOf maps pool atoms to ids. */
export function routeReading(arrows: readonly ArrowReadingOut[], scene: Scene, geom: ReadonlyMap<number, AtomGeom>, pairs: ReadonlyMap<number, LonePairMark>, M: DrawMetrics, idOf: (i: number) => number = (i) => i): (RoutedArrow | null)[] {
  const bondPx = M.bond;
  const gap = M.font * 0.22;
  const pts = [...geom.values()].map((g) => g.at);
  const centroid: Vec = pts.length ? [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length] : [0, 0];
  const obs = obstaclesOf(scene, geom, M);
  const specs = arrows.map((a) => {
    const plus = specOf(a, geom, pairs, idOf, bondPx, gap, 1);
    return plus ? { ...plus, alt: specOf(a, geom, pairs, idOf, bondPx, gap, -1) } : null;
  });
  const live = specs.map((s, k) => (s ? k : -1)).filter((k) => k >= 0);
  const routed = routeArrows(live.map((k) => specs[k]!), obs, centroid, 0.9 * bondPx);
  const out: (RoutedArrow | null)[] = arrows.map(() => null);
  live.forEach((k, n) => (out[k] = routed[n]));
  return out;
}

export { BOND_LENGTH };
