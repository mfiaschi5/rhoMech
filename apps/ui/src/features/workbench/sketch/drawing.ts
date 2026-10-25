// What the canvas draws, in screen pixels: atom labels (ChemDraw-style: carbon is a bare vertex
// unless charged, isolated or flagged explicitH; heteroatoms carry their implicit H count),
// bond strokes (double bonds offset into rings, centred at terminal ends, triple bonds, wedge
// and hash bonds), all as plain data. Pure: SketchCanvas only turns it into SVG elements.
//
// Options (SceneOptions, the page's sketch preferences): the label font size, the label style
// (skeletal = ChemDraw's carbon vertices; all = every carbon labelled; no-h = no implicit H
// drawn) and the CIP labels to place (cip.ts, the R/S toggle). Radical dots and CIP labels go
// into the free space around their atom (marks.ts).
//
// Crossing breaks (model.breakCrossings): where two bonds cross (crossings.ts) the bond behind
// is drawn with a gap of BREAK_GAP_PX on each side of the front bond's outline, on every one
// of its strokes (all lines of a multiple bond; the ticks of a hash; a wedge in two pieces).
import type { SketchAtom, SketchBond, SketchModel } from "./model";
import type { Vec, View } from "./geometry";
import type { StereoLabels } from "./cip";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { findCrossings } from "./crossings.ts";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { atomById, bondOrderSum, neighbors, smallestRingThrough } from "./model.ts";
// @ts-ignore TS5097: see above
import { add, len, scale, sideOf, sub, toScreen } from "./geometry.ts";
// @ts-ignore TS5097: see above
import { chargeLabel, fallbackHCount } from "./elements.ts";
// @ts-ignore TS5097: see above
import { along, bondMarkAt, freeAngle, markDistance, radicalDots, takenAngles } from "./marks.ts";

export const FONT_PX = 13;
const DOUBLE_GAP_PX = 4.5;
const INNER_SHORTEN = 0.14;
const WEDGE_HALF_PX = 3.6;
/** Clearance (px) between the front bond's outline and the ends of the broken bond behind. */
export const BREAK_GAP_PX = 3.2;
const CHARGE_FONT_PX = 10;
/** Half the size (px) of a CIP label such as "(R)", for its clearance. */
const CIP_HALF_PX = 7;

export type LabelStyle = "skeletal" | "all" | "no-h";
export const LABEL_STYLES: readonly LabelStyle[] = ["skeletal", "all", "no-h"];

export interface SceneOptions {
  /** Label font size (px); FONT_PX by default. */
  fontPx?: number;
  labelStyle?: LabelStyle;
  /** CIP labels to draw (null: none). */
  cip?: StereoLabels | null;
}

/** A small text mark (a CIP label) at a screen point. */
export interface Mark {
  text: string;
  at: Vec;
}

export interface AtomLabel {
  el: string;
  hCount: number;
  hSide: "left" | "right";
  charge: string;
}

export interface Seg {
  p: Vec;
  q: Vec;
}

export interface BondShape {
  lines: Seg[];
  /** Filled wedge polygons (one; two or more when a crossing breaks it). */
  wedges?: Vec[][];
}

export interface AtomScene {
  atom: SketchAtom;
  at: Vec;
  label: AtomLabel | null;
  /** Centres of the radical dots (screen px); empty without unpaired electrons. */
  dots: Vec[];
  cip?: Mark;
}

export interface BondScene {
  bond: SketchBond;
  shape: BondShape;
  /** Full-length centre line (hit target, hover and selection halo). */
  axis: Seg;
  /** Crossings at which this bond is behind and drawn broken (0 without breaks). */
  broken: number;
  cip?: Mark;
}

export interface Scene {
  atoms: AtomScene[];
  bonds: BondScene[];
  fontPx: number;
}

const H_LEFT_WHEN_ALONE = new Set(["O", "S", "Se", "F", "Cl", "Br", "I"]);

/** The label of an atom, or null for a bare carbon vertex. */
export function atomLabel(m: SketchModel, atom: SketchAtom, hCount: number, view: View, style: LabelStyle = "skeletal"): AtomLabel | null {
  const nbrs = neighbors(m, atom.id);
  const shown = atom.el !== "C" || atom.charge !== 0 || atom.explicitH === true || nbrs.length === 0 || style === "all";
  if (!shown) return null;
  if (style === "no-h" && atom.explicitH !== true && nbrs.length > 0) hCount = 0;
  let hSide: "left" | "right" = "right";
  if (nbrs.length === 0) hSide = H_LEFT_WHEN_ALONE.has(atom.el) ? "left" : "right";
  else {
    const me = toScreen(view, [atom.x, atom.y]);
    const dx = nbrs.reduce((s, id) => {
      const n = atomById(m, id);
      return n ? s + Math.sign(toScreen(view, [n.x, n.y])[0] - me[0] + 1e-9) : s;
    }, 0);
    hSide = dx > 0 ? "left" : "right";
  }
  return { el: atom.el, hCount, hSide, charge: chargeLabel(atom.charge) };
}

/** Half the estimated glyph width of an element symbol (capital ~0.62 em, lowercase ~0.5 em). */
export const labelHalfWidth = (el: string, font = FONT_PX): number => ((0.62 + 0.5 * (el.length - 1)) * font) / 2;

/** Width (px) of the H part of a label ("H", "H2"...); 0 without H. */
export const hLabelWidth = (hCount: number, font = FONT_PX): number => (hCount > 0 ? font * 0.68 + (hCount > 1 ? font * 0.45 : 0) : 0);

/** The box (screen px about the atom, y down) of everything a label draws: the symbol, the H
 *  count on its side and the charge superscript. SketchCanvas places the glyphs by the same
 *  numbers. */
export function labelBox(label: AtomLabel, font = FONT_PX): { x0: number; x1: number; y0: number; y1: number } {
  const half = labelHalfWidth(label.el, font);
  const hw = hLabelWidth(label.hCount, font);
  const cf = (CHARGE_FONT_PX * font) / FONT_PX;
  const right = label.hSide === "right";
  let x0 = -half - (right ? 0 : hw);
  let x1 = half + (right ? hw : 0);
  let y1 = font * 0.5 + (label.hCount > 1 ? 3 : 0);
  let y0 = -font * 0.5;
  if (label.charge) {
    const cx = half + (right ? hw : 0) + 1;
    x1 = Math.max(x1, cx + label.charge.length * cf * 0.62);
    y0 = Math.min(y0, -font * 0.55 - cf * 0.5);
  }
  x0 = Math.min(x0, -half);
  y1 = Math.max(y1, font * 0.5);
  return { x0, x1, y0, y1 };
}

/** Pixels a bond stops short of a labelled atom. */
export function labelTrim(label: AtomLabel | null, font = FONT_PX): number {
  if (!label) return 0;
  return label.el.length > 1 ? font * 0.72 : font * 0.55;
}

/** -1 / +1: the side (of p->q on screen) the second line of a double bond goes; 0: centred. */
export function doubleBondSide(m: SketchModel, b: SketchBond, at: (id: number) => Vec): -1 | 0 | 1 {
  const p = at(b.a);
  const q = at(b.b);
  const ring = smallestRingThrough(m, b.a, b.b);
  if (ring) {
    const c = scale(ring.map(at).reduce((s, v) => add(s, v), [0, 0] as Vec), 1 / ring.length);
    return sideOf(p, q, c) || 1;
  }
  const na = neighbors(m, b.a).filter((x) => x !== b.b);
  const nb = neighbors(m, b.b).filter((x) => x !== b.a);
  if (na.length === 0 && nb.length === 0) return 0;
  if ((na.length === 0 && nb.length >= 2) || (nb.length === 0 && na.length >= 2)) return 0;
  const sum = [...na, ...nb].reduce((s, x) => s + sideOf(p, q, at(x)), 0);
  return sum > 0 ? 1 : sum < 0 ? -1 : 1;
}

function trimmed(p: Vec, q: Vec, tp: number, tq: number): Seg {
  const d = sub(q, p);
  const l = len(d) || 1;
  const u = scale(d, 1 / l);
  return { p: add(p, scale(u, Math.min(tp, l / 2))), q: sub(q, scale(u, Math.min(tq, l / 2))) };
}

function offset(s: Seg, k: number): Seg {
  const d = sub(s.q, s.p);
  const l = len(d) || 1;
  const n: Vec = [-d[1] / l, d[0] / l];
  return { p: add(s.p, scale(n, k)), q: add(s.q, scale(n, k)) };
}

function shorten(s: Seg, f: number, atP: boolean, atQ: boolean): Seg {
  const d = sub(s.q, s.p);
  return { p: atP ? add(s.p, scale(d, f)) : s.p, q: atQ ? sub(s.q, scale(d, f)) : s.q };
}

/** The strokes of one bond between screen points p and q. `tp`/`tq` trim the ends at labels. */
export function bondShape(
  p: Vec,
  q: Vec,
  b: Pick<SketchBond, "order" | "stereo">,
  side: -1 | 0 | 1,
  tp: number,
  tq: number,
): BondShape {
  const main = trimmed(p, q, tp, tq);
  if (b.order === 1 && b.stereo === "wedge") return { lines: [], wedges: [wedgePolygon(main)] };
  if (b.order === 1 && b.stereo === "hash") return { lines: hashLines(main) };
  if (b.order === 1) return { lines: [main] };
  if (b.order === 2 && side === 0) {
    return { lines: [offset(main, DOUBLE_GAP_PX / 2), offset(main, -DOUBLE_GAP_PX / 2)] };
  }
  if (b.order === 2) {
    const inner = shorten(offset(main, side * DOUBLE_GAP_PX), INNER_SHORTEN, tp === 0, tq === 0);
    return { lines: [main, inner] };
  }
  return {
    lines: [main, ...[1, -1].map((k) => shorten(offset(main, k * DOUBLE_GAP_PX), 0.08, tp === 0, tq === 0))],
  };
}

function wedgePolygon(s: Seg): Vec[] {
  const d = sub(s.q, s.p);
  const l = len(d) || 1;
  const n: Vec = [-d[1] / l, d[0] / l];
  return [add(s.p, scale(n, 0.4)), add(s.q, scale(n, WEDGE_HALF_PX)), sub(s.q, scale(n, WEDGE_HALF_PX)), sub(s.p, scale(n, 0.4))];
}

function hashLines(s: Seg): Seg[] {
  const d = sub(s.q, s.p);
  const l = len(d) || 1;
  const n: Vec = [-d[1] / l, d[0] / l];
  const count = Math.max(4, Math.round(l / 4.5));
  return Array.from({ length: count }, (_, i) => {
    const t = (i + 0.5) / count;
    const c = add(s.p, scale(d, t));
    const w = 0.6 + (WEDGE_HALF_PX - 0.6) * t;
    return { p: add(c, scale(n, w)), q: sub(c, scale(n, w)) };
  });
}

/** A crossing as the drawing sees it: the point (px), the front bond's direction and the
 *  half-width of its outline (px). */
interface Cut {
  at: Vec;
  dir: Vec;
  half: number;
}

/** Half-width (px) of what a bond draws at parameter t (for the gap of the bond behind). */
function outlineHalf(b: Pick<SketchBond, "order" | "stereo">, t: number): number {
  if (b.order === 2) return DOUBLE_GAP_PX;
  if (b.order === 3) return DOUBLE_GAP_PX + 0.8;
  return b.stereo === "wedge" || b.stereo === "hash" ? 0.6 + (WEDGE_HALF_PX - 0.6) * t : 0.8;
}

/** The arclength interval [s0, s1] of segment s that the cut removes, or null when s runs
 *  parallel to the front bond. */
function cutInterval(s: Seg, c: Cut): [number, number] | null {
  const d = sub(s.q, s.p);
  const l = len(d);
  const den = d[0] * c.dir[1] - d[1] * c.dir[0];
  if (l < 1e-9 || Math.abs(den) < 1e-9 * l) return null;
  const w = sub(c.at, s.p);
  const along = ((w[0] * c.dir[1] - w[1] * c.dir[0]) / den) * l;
  const sin = Math.abs(den) / l;
  const g = (BREAK_GAP_PX + c.half) / Math.max(sin, 0.35);
  return [along - g, along + g];
}

function cutSeg(s: Seg, c: Cut): Seg[] {
  const iv = cutInterval(s, c);
  const l = len(sub(s.q, s.p));
  if (!iv || iv[1] <= 0 || iv[0] >= l) return [s];
  const u = scale(sub(s.q, s.p), 1 / l);
  const out: Seg[] = [];
  if (iv[0] > 0.6) out.push({ p: s.p, q: add(s.p, scale(u, iv[0])) });
  if (l - iv[1] > 0.6) out.push({ p: add(s.p, scale(u, iv[1])), q: s.q });
  return out;
}

const lerp = (p: Vec, q: Vec, t: number): Vec => [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];

/** A wedge polygon [narrow left, wide left, wide right, narrow right] without the cut. */
function cutWedge(w: Vec[], axis: Seg, c: Cut): Vec[][] {
  const iv = cutInterval(axis, c);
  const l = len(sub(axis.q, axis.p));
  if (!iv || iv[1] <= 0 || iv[0] >= l) return [w];
  const piece = (t0: number, t1: number): Vec[] => [lerp(w[0], w[1], t0), lerp(w[0], w[1], t1), lerp(w[3], w[2], t1), lerp(w[3], w[2], t0)];
  const out: Vec[][] = [];
  if (iv[0] > 0.6) out.push(piece(0, iv[0] / l));
  if (l - iv[1] > 0.6) out.push(piece(iv[1] / l, 1));
  return out;
}

/** The shape of bond `b` (centre line `main`) with the crossing cuts applied. */
export function breakShape(shape: BondShape, b: Pick<SketchBond, "order" | "stereo">, main: Seg, cuts: readonly Cut[]): BondShape {
  if (cuts.length === 0) return shape;
  let lines = shape.lines;
  let wedges = shape.wedges;
  for (const c of cuts) {
    if (b.order === 1 && b.stereo === "hash") {
      const iv = cutInterval(main, c);
      lines = lines.filter((tick) => {
        const mid = scale(add(tick.p, tick.q), 0.5);
        const along = len(sub(mid, main.p));
        return !iv || along < iv[0] || along > iv[1];
      });
    } else lines = lines.flatMap((s) => cutSeg(s, c));
    if (wedges) wedges = wedges.flatMap((w) => cutWedge(w, main, c));
  }
  return wedges ? { lines, wedges } : { lines };
}

/** The cuts of each bond behind a crossing (screen px), by bond id. */
function crossingCuts(m: SketchModel, view: View, at: (id: number) => Vec): Map<number, Cut[]> {
  const out = new Map<number, Cut[]>();
  const byId = new Map(m.bonds.map((b) => [b.id, b]));
  for (const c of findCrossings(m)) {
    const front = byId.get(c.front);
    if (!front) continue;
    const d = sub(at(front.b), at(front.a));
    const l = len(d) || 1;
    const cut: Cut = { at: toScreen(view, c.at), dir: scale(d, 1 / l), half: outlineHalf(front, c.tFront) };
    out.set(c.back, [...(out.get(c.back) ?? []), cut]);
  }
  return out;
}

/** The radical dots and the CIP label of one atom, placed in the free space around it. */
function atomMarks(m: SketchModel, s: AtomScene, pos: (id: number) => Vec, font: number, cip: StereoLabels | null): Pick<AtomScene, "dots" | "cip"> {
  const n = s.atom.radical ?? 0;
  const code = cip?.atoms.get(s.atom.id);
  if (!n && !code) return { dots: [] };
  const box = s.label ? labelBox(s.label, font) : null;
  const hSide = s.label && s.label.hCount > 0 ? s.label.hSide : null;
  const taken = takenAngles(s.at, neighbors(m, s.atom.id).map(pos), hSide, !!s.label?.charge);
  const dotAngle = freeAngle(taken);
  const dots = radicalDots(s.at, box, dotAngle, n);
  if (!code) return { dots };
  const a = freeAngle(n ? [...taken, dotAngle] : taken);
  return { dots, cip: { text: `(${code})`, at: along(s.at, a, markDistance(box, a, CIP_HALF_PX)) } };
}

/** Everything to draw, in screen pixels. `hCounts` (RDKit's, by atom id) wins over the default
 *  valence fallback. Crossing breaks follow model.breakCrossings. */
export function buildScene(m: SketchModel, view: View, hCounts: ReadonlyMap<number, number> | null, opts: SceneOptions = {}): Scene {
  const font = opts.fontPx ?? FONT_PX;
  const cip = opts.cip ?? null;
  const at = new Map<number, Vec>(m.atoms.map((a) => [a.id, toScreen(view, [a.x, a.y])]));
  const pos = (id: number) => at.get(id) ?? ([0, 0] as Vec);
  const atoms: AtomScene[] = m.atoms.map((atom) => {
    const h = hCounts?.get(atom.id) ?? fallbackHCount(atom.el, atom.charge, bondOrderSum(m, atom.id), atom.radical ?? 0);
    const s: AtomScene = { atom, at: pos(atom.id), label: atomLabel(m, atom, h, view, opts.labelStyle), dots: [] };
    return { ...s, ...atomMarks(m, s, pos, font, cip) };
  });
  const labels = new Map(atoms.map((s) => [s.atom.id, s.label]));
  const cuts = m.breakCrossings ? crossingCuts(m, view, pos) : new Map<number, Cut[]>();
  const bonds: BondScene[] = m.bonds.map((bond) => {
    const p = pos(bond.a);
    const q = pos(bond.b);
    const side = bond.order === 2 ? doubleBondSide(m, bond, pos) : 0;
    const tp = labelTrim(labels.get(bond.a) ?? null, font);
    const tq = labelTrim(labels.get(bond.b) ?? null, font);
    const mine = cuts.get(bond.id) ?? [];
    const shape = breakShape(bondShape(p, q, bond, side, tp, tq), bond, trimmed(p, q, tp, tq), mine);
    const code = cip?.bonds.get(bond.id);
    const mark = code ? { cip: { text: `(${code})`, at: bondMarkAt(p, q, side) } } : {};
    return { bond, shape, axis: { p, q }, broken: mine.length, ...mark };
  });
  return { atoms, bonds, fontPx: font };
}
