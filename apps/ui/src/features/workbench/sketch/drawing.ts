// What the canvas draws, in output units (screen pixels on the canvas, points in an export):
// atom labels (ChemDraw-style: carbon is a bare vertex unless charged, isolated or flagged
// explicitH, or the style labels carbons or terminal methyls; heteroatoms carry their implicit H
// count), bond strokes (double bonds offset into rings, centred at terminal ends unless the
// style offsets them, triple bonds, wedge and hash bonds), all as plain data. Pure: SketchCanvas
// turns it into SVG elements, exportImage.ts into an SVG file or a PDF.
//
// Every dimension comes from the drawing style (style.ts) as DrawMetrics in output units: line,
// bold (wedge) and hash widths, hash spacing, double-bond spacing (% of the fixed bond length),
// the margin around labels (bonds stop at the label's ink box plus the margin), font sizes, the
// crossing gap. Without metrics the Screen style at FONT_PX is used.
//
// Options (SceneOptions): the metrics, the label style (skeletal = ChemDraw's carbon vertices;
// all = every carbon labelled; no-h = no implicit H drawn) and the CIP labels to place (cip.ts,
// the R/S toggle). Radical dots and CIP labels go into the free space around their atom
// (marks.ts).
//
// Crossing breaks (model.breakCrossings): where two bonds cross (crossings.ts) the bond behind
// is drawn with a gap of metrics.breakGap on each side of the front bond's outline, on every one
// of its strokes (all lines of a multiple bond; the ticks of a hash; a wedge in pieces).
import type { SketchAtom, SketchBond, SketchModel } from "./model";
import type { Vec, View } from "./geometry";
import type { StereoLabels } from "./cip";
import type { DrawMetrics } from "./style";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { findCrossings } from "./crossings.ts";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { atomById, bondOrderSum, neighbors, smallestRingThrough } from "./model.ts";
// @ts-ignore TS5097: see above
import { add, len, scale, sideOf, sub, toScreen } from "./geometry.ts";
// @ts-ignore TS5097: see above
import { chargeLabel, fallbackHCount } from "./elements.ts";
// @ts-ignore TS5097: see above
import { along, bondMarkAt, boxExit, freeAngle, markDistance, radicalDots, takenAngles, DOT_R_PX } from "./marks.ts";
// @ts-ignore TS5097: see above
import { screenMetrics } from "./style.ts";
// @ts-ignore TS5097: see above
import { CAP_HEIGHT_EM, textWidthEm } from "./fontMetrics.ts";

export const FONT_PX = 13;
/** Clearance (px) of the Screen style between the front bond's outline and the broken bond. */
export const BREAK_GAP_PX = 3.2;
/** The bond length (px) the mark sizes of marks.ts are drawn for (Screen at 100 %). */
const MARK_BOND_PX = 37.5;

export type LabelStyle = "skeletal" | "all" | "no-h";
export const LABEL_STYLES: readonly LabelStyle[] = ["skeletal", "all", "no-h"];

export interface SceneOptions {
  /** Label font size (px) of the default Screen metrics; ignored with `metrics`. */
  fontPx?: number;
  /** The style's dimensions in output units (style.drawMetrics). */
  metrics?: DrawMetrics;
  labelStyle?: LabelStyle;
  /** CIP labels to draw (null: none). */
  cip?: StereoLabels | null;
}

/** A small text mark (a CIP label) at a point. */
export interface Mark {
  text: string;
  at: Vec;
}

export type ChargeSlot = "ur" | "ul" | "lr" | "ll";
export type HSide = "left" | "right" | "above" | "below";

export interface AtomLabel {
  el: string;
  hCount: number;
  /** Where the H count goes: right or left of the symbol, or above / below it when bonds
   *  leave on both sides near the horizontal (an NH in a chain). */
  hSide: HSide;
  charge: string;
  /** Where the charge sign sits: upper right unless a bond or the H count is there. */
  chargeSlot?: ChargeSlot;
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
  /** Centres of the radical dots; empty without unpaired electrons. */
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
  metrics: DrawMetrics;
  /** Radius of a radical dot. */
  dotR: number;
}

const H_LEFT_WHEN_ALONE = new Set(["O", "S", "Se", "F", "Cl", "Br", "I"]);
const DEG = Math.PI / 180;
const SLOT_ANGLE: Record<ChargeSlot, number> = { ur: -45 * DEG, ul: -135 * DEG, lr: 45 * DEG, ll: 135 * DEG };
const SLOTS: ChargeSlot[] = ["ur", "ul", "lr", "ll"];
const H_ANGLE: Record<HSide, number> = { right: 0, left: Math.PI, above: -90 * DEG, below: 90 * DEG };

/** Smallest angular distance from `a` to the `taken` directions (pi when none). */
const angularGap = (a: number, taken: readonly number[]) => taken.reduce((m, t) => Math.min(m, Math.abs(Math.atan2(Math.sin(a - t), Math.cos(a - t)))), Math.PI);

/** The first charge slot at least 40 degrees from every taken direction (screen angles), else
 *  the one farthest from them. */
export function chargeSlot(taken: readonly number[]): ChargeSlot {
  const free = SLOTS.find((s) => angularGap(SLOT_ANGLE[s], taken) > 40 * DEG);
  return free ?? SLOTS.reduce((b, s) => (angularGap(SLOT_ANGLE[s], taken) > angularGap(SLOT_ANGLE[b], taken) ? s : b));
}

/** The screen direction of a charge slot and of an H side (marks.ts keeps them free). */
export const chargeSlotAngle = (s: ChargeSlot | undefined): number => SLOT_ANGLE[s ?? "ur"];
export const hSideAngle = (s: HSide): number => H_ANGLE[s];

/** The H side for bonds leaving at screen angles `bonds`: right or left (away from the bonds,
 *  as ChemDraw), or above / below when the horizontal sides are both within 45 degrees of a
 *  bond. */
export function hSideFor(bonds: readonly number[], el: string): HSide {
  if (bonds.length === 0) return H_LEFT_WHEN_ALONE.has(el) ? "left" : "right";
  const dx = bonds.reduce((s, t) => s + Math.sign(Math.cos(t) + 1e-9), 0);
  const first: HSide = dx > 0 ? "left" : "right";
  const order: HSide[] = [first, first === "left" ? "right" : "left", "above", "below"];
  return order.find((h) => angularGap(H_ANGLE[h], bonds) >= 45 * DEG) ?? first;
}

/** Which carbons a style labels besides the charged / isolated / explicit-H ones. */
export interface LabelRules {
  style?: LabelStyle;
  carbons?: boolean;
  methyls?: boolean;
}

function shownCarbon(atom: SketchAtom, nbrs: number, hCount: number, r: LabelRules): boolean {
  if (atom.charge !== 0 || atom.explicitH === true || nbrs === 0 || r.style === "all" || r.carbons) return true;
  return !!r.methyls && nbrs === 1 && hCount === 3;
}

/** The label of an atom, or null for a bare carbon vertex. */
export function atomLabel(m: SketchModel, atom: SketchAtom, hCount: number, view: View, rules: LabelStyle | LabelRules = "skeletal"): AtomLabel | null {
  const r: LabelRules = typeof rules === "string" ? { style: rules } : rules;
  const nbrs = neighbors(m, atom.id);
  if (atom.el === "C" && !shownCarbon(atom, nbrs.length, hCount, r)) return null;
  if (r.style === "no-h" && atom.explicitH !== true && nbrs.length > 0) hCount = 0;
  const me = toScreen(view, [atom.x, atom.y]);
  const pts = nbrs.map((id) => atomById(m, id)).filter((n): n is SketchAtom => !!n).map((n) => toScreen(view, [n.x, n.y]));
  const bonds = pts.map((q) => Math.atan2(q[1] - me[1], q[0] - me[0]));
  const hSide = hSideFor(bonds, atom.el);
  const charge = chargeLabel(atom.charge);
  if (!charge) return { el: atom.el, hCount, hSide, charge };
  const taken = hCount > 0 ? [...bonds, H_ANGLE[hSide]] : bonds;
  return { el: atom.el, hCount, hSide, charge, chargeSlot: chargeSlot(taken) };
}

type FontArg = number | DrawMetrics;
const metricsOf = (f: FontArg | undefined): DrawMetrics => (typeof f === "object" ? f : screenMetrics(f ?? FONT_PX));

/** Half the width of an element symbol: Helvetica metrics for publication fonts, else the
 *  UI-font estimate (capital ~0.62 em, lowercase ~0.5 em). */
export function labelHalfWidth(el: string, f: FontArg = FONT_PX): number {
  const M = metricsOf(f);
  return M.metricText ? (textWidthEm(el) * M.font) / 2 : ((0.62 + 0.5 * (el.length - 1)) * M.font) / 2;
}

const hWidth = (M: DrawMetrics) => (M.metricText ? textWidthEm("H") * M.font : M.font * 0.68);
const digitsWidth = (n: number, M: DrawMetrics) => (n <= 1 ? 0 : M.metricText ? textWidthEm(String(n)) * M.sub : M.font * 0.45);

/** Width of the H part of a label ("H", "H2"...); 0 without H. */
export function hLabelWidth(hCount: number, f: FontArg = FONT_PX): number {
  const M = metricsOf(f);
  return hCount > 0 ? hWidth(M) + digitsWidth(hCount, M) : 0;
}

/** One glyph run of a label: start x and baseline y about the atom (y down), and its size. */
export interface Glyph {
  text: string;
  x: number;
  y: number;
  size: number;
  kind: "symbol" | "h" | "count" | "charge";
}

/** Baseline offset that puts the middle of a capital on the atom point. */
const baseline = (size: number) => (CAP_HEIGHT_EM / 2) * size;

/** Where the charge sign starts (x, about the atom) and its vertical centre (y, down). */
export function chargeAnchor(label: AtomLabel, f: FontArg = FONT_PX): Vec {
  const M = metricsOf(f);
  const half = labelHalfWidth(label.el, M);
  const hw = hLabelWidth(label.hCount, M);
  const slot = label.chargeSlot ?? "ur";
  const y = slot[0] === "u" ? -M.font * 0.55 : M.font * 0.55;
  if (slot[1] === "r") return [half + (label.hSide === "right" ? hw : 0) + chargeGap(M), y];
  return [-half - (label.hSide === "left" ? hw : 0) - chargeGap(M) - chargeWidth(label, M), y];
}

/** Gap between the symbol (or its H) and the charge sign. */
const chargeGap = (M: DrawMetrics) => (M.metricText ? M.font * 0.04 : 1);

/** Width of the charge mark (text, or the circled sign). */
export function chargeWidth(label: AtomLabel, f: FontArg = FONT_PX): number {
  const M = metricsOf(f);
  if (M.chargePlacement === "circled" && label.charge.length === 1) return M.charge * 1.1;
  return M.metricText ? textWidthEm(label.charge) * M.charge : label.charge.length * M.charge * 0.62;
}

/** The glyph runs of a label (symbol, H, H count, charge) about the atom. SketchCanvas and
 *  the export both draw these, so the screen and the files agree. */
export function labelGlyphs(label: AtomLabel, f: FontArg = FONT_PX): Glyph[] {
  const M = metricsOf(f);
  const half = labelHalfWidth(label.el, M);
  const base = baseline(M.font);
  const out: Glyph[] = [{ text: label.el, x: -half, y: base, size: M.font, kind: "symbol" }];
  if (label.hCount > 0) {
    const hw = hLabelWidth(label.hCount, M);
    const pos: Record<HSide, Vec> = {
      right: [half, base],
      left: [-half - hw, base],
      above: [-hw / 2, base - M.font * 0.95],
      below: [-hw / 2, base + M.font * 0.95],
    };
    const [x, y] = pos[label.hSide];
    out.push({ text: "H", x, y, size: M.font, kind: "h" });
    if (label.hCount > 1) out.push({ text: String(label.hCount), x: x + hWidth(M), y: y + M.font * 0.3, size: M.sub, kind: "count" });
  }
  if (label.charge) {
    const [cx, cy] = chargeAnchor(label, M);
    out.push({ text: label.charge, x: cx, y: cy + baseline(M.charge), size: M.charge, kind: "charge" });
  }
  return out;
}

/** The box (about the atom, y down) of everything a label draws. */
export function labelBox(label: AtomLabel, f: FontArg = FONT_PX): { x0: number; x1: number; y0: number; y1: number } {
  const M = metricsOf(f);
  const box = { x0: Infinity, x1: -Infinity, y0: -M.font * 0.5, y1: M.font * 0.5 };
  for (const g of labelGlyphs(label, M)) {
    const w = g.kind === "charge" ? chargeWidth(label, M) : g.kind === "count" ? digitsWidth(label.hCount, M) : g.kind === "h" ? hWidth(M) : labelHalfWidth(label.el, M) * 2;
    box.x0 = Math.min(box.x0, g.x);
    box.x1 = Math.max(box.x1, g.x + w);
    box.y0 = Math.min(box.y0, g.y - g.size * 0.86);
    box.y1 = Math.max(box.y1, g.y + g.size * (g.kind === "count" ? 0.2 : 0.14));
  }
  return box;
}

/** How far a bond leaving a labelled atom at screen angle `angle` (radians, y down) stops short
 *  of the atom: the ray's exit from the label's ink box (the symbol at cap height, a horizontal
 *  H part on its side) plus the style's margin. 0 for a bare vertex. */
export function labelTrim(label: AtomLabel | null, f: FontArg = FONT_PX, angle = 0): number {
  if (!label) return 0;
  const M = metricsOf(f);
  const half = labelHalfWidth(label.el, M);
  const hw = hLabelWidth(label.hCount, M);
  const capHalf = (CAP_HEIGHT_EM * M.font) / 2;
  const box = { x0: -half - (label.hSide === "left" ? hw : 0), x1: half + (label.hSide === "right" ? hw : 0), y0: -capHalf, y1: capHalf };
  return boxExit(box, angle) + M.margin;
}

/** -1 / +1: the side (of p->q on screen) the second line of a double bond goes; 0: centred.
 *  In a ring: toward the ring centre. A chain end next to a branch point (C=O on a ring):
 *  centred, or with terminalDouble "offset" toward the substituents. */
export function doubleBondSide(m: SketchModel, b: SketchBond, at: (id: number) => Vec, terminal: "centred" | "offset" = "centred"): -1 | 0 | 1 {
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
  const branchEnd = (na.length === 0 && nb.length >= 2) || (nb.length === 0 && na.length >= 2);
  if (branchEnd && terminal === "centred") return 0;
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

/** The strokes of one bond between points p and q. `tp`/`tq` trim the ends at labels. */
export function bondShape(
  p: Vec,
  q: Vec,
  b: Pick<SketchBond, "order" | "stereo">,
  side: -1 | 0 | 1,
  tp: number,
  tq: number,
  M: DrawMetrics = screenMetrics(),
): BondShape {
  const main = trimmed(p, q, tp, tq);
  const gap = M.doubleGap;
  if (b.order === 1 && b.stereo === "wedge") return { lines: [], wedges: [wedgePiece(main, 0, 1, M)] };
  if (b.order === 1 && b.stereo === "hash") return { lines: hashLines(main, M) };
  if (b.order === 1) return { lines: [main] };
  if (b.order === 2 && side === 0) return { lines: [offset(main, gap / 2), offset(main, -gap / 2)] };
  if (b.order === 2) {
    const inner = shorten(offset(main, side * gap), M.innerShorten, tp === 0, tq === 0);
    return { lines: [main, inner] };
  }
  return {
    lines: [main, ...[1, -1].map((k) => shorten(offset(main, k * gap), M.innerShorten * 0.6, tp === 0, tq === 0))],
  };
}

/** Half-width of a wedge at parameter t (0 at the stereocentre, 1 at the wide end): a classic
 *  wedge grows linearly from a point; a tapered one starts at the line width, reaches the bold
 *  width at 70 % and stays bold to its end. */
export function wedgeHalf(t: number, M: DrawMetrics): number {
  const bold = M.bold / 2;
  if (M.wedgeShape === "tapered") {
    const n = M.line / 2;
    return t >= 0.7 ? bold : n + ((bold - n) * t) / 0.7;
  }
  const n = M.line * 0.27;
  return n + (bold - n) * t;
}

/** The wedge outline between parameters t0 < t1 of the axis s (a closed polygon). */
function wedgePiece(s: Seg, t0: number, t1: number, M: DrawMetrics): Vec[] {
  const d = sub(s.q, s.p);
  const l = len(d) || 1;
  const n: Vec = [-d[1] / l, d[0] / l];
  const ts = [t0, ...(M.wedgeShape === "tapered" && t0 < 0.7 && t1 > 0.7 ? [0.7] : []), t1];
  const at = (t: number) => add(s.p, scale(d, t));
  const left = ts.map((t) => add(at(t), scale(n, wedgeHalf(t, M))));
  const right = ts.map((t) => sub(at(t), scale(n, wedgeHalf(t, M)))).reverse();
  return [...left, ...right];
}

function hashLines(s: Seg, M: DrawMetrics): Seg[] {
  const d = sub(s.q, s.p);
  const l = len(d) || 1;
  const n: Vec = [-d[1] / l, d[0] / l];
  const count = Math.max(3, Math.round(l / M.hashSpacing));
  const narrow = M.hashWidth * 0.4;
  const bold = M.bold / 2;
  return Array.from({ length: count }, (_, i) => {
    const t = (i + 0.5) / count;
    const c = add(s.p, scale(d, t));
    const w = narrow + (bold - narrow) * t;
    return { p: add(c, scale(n, w)), q: sub(c, scale(n, w)) };
  });
}

/** A crossing as the drawing sees it: the point, the front bond's direction and the
 *  half-width of its outline. */
interface Cut {
  at: Vec;
  dir: Vec;
  half: number;
}

/** Half-width of what a bond draws at parameter t (for the gap of the bond behind). */
function outlineHalf(b: Pick<SketchBond, "order" | "stereo">, t: number, M: DrawMetrics): number {
  if (b.order === 2) return M.doubleGap;
  if (b.order === 3) return M.doubleGap + M.line / 2;
  return b.stereo === "wedge" || b.stereo === "hash" ? wedgeHalf(t, M) : M.line / 2;
}

/** The arclength interval [s0, s1] of segment s that the cut removes, or null when s runs
 *  parallel to the front bond. */
function cutInterval(s: Seg, c: Cut, M: DrawMetrics): [number, number] | null {
  const d = sub(s.q, s.p);
  const l = len(d);
  const den = d[0] * c.dir[1] - d[1] * c.dir[0];
  if (l < 1e-9 || Math.abs(den) < 1e-9 * l) return null;
  const w = sub(c.at, s.p);
  const along = ((w[0] * c.dir[1] - w[1] * c.dir[0]) / den) * l;
  const sin = Math.abs(den) / l;
  const g = (M.breakGap + c.half) / Math.max(sin, 0.35);
  return [along - g, along + g];
}

/** Pieces shorter than this are dropped (output units). */
const minPiece = (M: DrawMetrics) => M.line * 0.4;

function cutSeg(s: Seg, c: Cut, M: DrawMetrics): Seg[] {
  const iv = cutInterval(s, c, M);
  const l = len(sub(s.q, s.p));
  if (!iv || iv[1] <= 0 || iv[0] >= l) return [s];
  const u = scale(sub(s.q, s.p), 1 / l);
  const out: Seg[] = [];
  if (iv[0] > minPiece(M)) out.push({ p: s.p, q: add(s.p, scale(u, iv[0])) });
  if (l - iv[1] > minPiece(M)) out.push({ p: add(s.p, scale(u, iv[1])), q: s.q });
  return out;
}

/** The parameter intervals of the wedge left after the cuts, as polygons. */
function cutWedges(axis: Seg, cuts: readonly Cut[], M: DrawMetrics): Vec[][] {
  const l = len(sub(axis.q, axis.p)) || 1;
  let pieces: [number, number][] = [[0, 1]];
  for (const c of cuts) {
    const iv = cutInterval(axis, c, M);
    if (!iv) continue;
    const [a, b] = [iv[0] / l, iv[1] / l];
    pieces = pieces.flatMap(([t0, t1]): [number, number][] => {
      if (b <= t0 || a >= t1) return [[t0, t1]];
      const out: [number, number][] = [];
      if ((a - t0) * l > minPiece(M)) out.push([t0, a]);
      if ((t1 - b) * l > minPiece(M)) out.push([b, t1]);
      return out;
    });
  }
  return pieces.map(([t0, t1]) => wedgePiece(axis, t0, t1, M));
}

/** The shape of bond `b` (centre line `main`) with the crossing cuts applied. */
export function breakShape(shape: BondShape, b: Pick<SketchBond, "order" | "stereo">, main: Seg, cuts: readonly Cut[], M: DrawMetrics = screenMetrics()): BondShape {
  if (cuts.length === 0) return shape;
  let lines = shape.lines;
  for (const c of cuts) {
    if (b.order === 1 && b.stereo === "hash") {
      const iv = cutInterval(main, c, M);
      lines = lines.filter((tick) => {
        const mid = scale(add(tick.p, tick.q), 0.5);
        const along = len(sub(mid, main.p));
        return !iv || along < iv[0] || along > iv[1];
      });
    } else lines = lines.flatMap((s) => cutSeg(s, c, M));
  }
  return shape.wedges ? { lines, wedges: cutWedges(main, cuts, M) } : { lines };
}

/** The cuts of each bond behind a crossing, by bond id. */
function crossingCuts(m: SketchModel, view: View, at: (id: number) => Vec, M: DrawMetrics): Map<number, Cut[]> {
  const out = new Map<number, Cut[]>();
  const byId = new Map(m.bonds.map((b) => [b.id, b]));
  for (const c of findCrossings(m)) {
    const front = byId.get(c.front);
    if (!front) continue;
    const d = sub(at(front.b), at(front.a));
    const l = len(d) || 1;
    const cut: Cut = { at: toScreen(view, c.at), dir: scale(d, 1 / l), half: outlineHalf(front, c.tFront, M) };
    out.set(c.back, [...(out.get(c.back) ?? []), cut]);
  }
  return out;
}

/** The radical dots and the CIP label of one atom, placed in the free space around it. */
function atomMarks(m: SketchModel, s: AtomScene, pos: (id: number) => Vec, M: DrawMetrics, cip: StereoLabels | null): Pick<AtomScene, "dots" | "cip"> {
  const n = s.atom.radical ?? 0;
  const code = cip?.atoms.get(s.atom.id);
  if (!n && !code) return { dots: [] };
  const k = M.bond / MARK_BOND_PX;
  const box = s.label ? labelBox(s.label, M) : null;
  const taken = takenAngles(s.at, neighbors(m, s.atom.id).map(pos), null, false);
  if (s.label && s.label.hCount > 0) taken.push(hSideAngle(s.label.hSide));
  if (s.label?.charge) taken.push(chargeSlotAngle(s.label.chargeSlot));
  const dotAngle = freeAngle(taken);
  const dots = radicalDots(s.at, box, dotAngle, n, k);
  if (!code) return { dots };
  const a = freeAngle(n ? [...taken, dotAngle] : taken);
  return { dots, cip: { text: `(${code})`, at: along(s.at, a, markDistance(box, a, M.font * 0.54, k)) } };
}

/** Everything to draw, in output units. `hCounts` (RDKit's, by atom id) wins over the default
 *  valence fallback. Crossing breaks follow model.breakCrossings. */
export function buildScene(m: SketchModel, view: View, hCounts: ReadonlyMap<number, number> | null, opts: SceneOptions = {}): Scene {
  const M = opts.metrics ?? screenMetrics(opts.fontPx ?? FONT_PX);
  const cip = opts.cip ?? null;
  const rules: LabelRules = { style: opts.labelStyle, carbons: M.labelCarbons, methyls: M.labelMethyls };
  const at = new Map<number, Vec>(m.atoms.map((a) => [a.id, toScreen(view, [a.x, a.y])]));
  const pos = (id: number) => at.get(id) ?? ([0, 0] as Vec);
  const atoms: AtomScene[] = m.atoms.map((atom) => {
    const h = hCounts?.get(atom.id) ?? fallbackHCount(atom.el, atom.charge, bondOrderSum(m, atom.id), atom.radical ?? 0);
    const s: AtomScene = { atom, at: pos(atom.id), label: atomLabel(m, atom, h, view, rules), dots: [] };
    return { ...s, ...atomMarks(m, s, pos, M, cip) };
  });
  const labels = new Map(atoms.map((s) => [s.atom.id, s.label]));
  const cuts = m.breakCrossings ? crossingCuts(m, view, pos, M) : new Map<number, Cut[]>();
  const bonds: BondScene[] = m.bonds.map((bond) => {
    const p = pos(bond.a);
    const q = pos(bond.b);
    const side = bond.order === 2 ? doubleBondSide(m, bond, pos, M.terminalDouble) : 0;
    const ang = Math.atan2(q[1] - p[1], q[0] - p[0]);
    const tp = labelTrim(labels.get(bond.a) ?? null, M, ang);
    const tq = labelTrim(labels.get(bond.b) ?? null, M, ang + Math.PI);
    const mine = cuts.get(bond.id) ?? [];
    const shape = breakShape(bondShape(p, q, bond, side, tp, tq, M), bond, trimmed(p, q, tp, tq), mine, M);
    const code = cip?.bonds.get(bond.id);
    const mark = code ? { cip: { text: `(${code})`, at: bondMarkAt(p, q, side, 10 * (M.bond / MARK_BOND_PX)) } } : {};
    return { bond, shape, axis: { p, q }, broken: mine.length, ...mark };
  });
  return { atoms, bonds, fontPx: M.font, metrics: M, dotR: DOT_R_PX * (M.bond / MARK_BOND_PX) };
}
