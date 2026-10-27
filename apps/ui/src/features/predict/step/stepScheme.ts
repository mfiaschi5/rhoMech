// The textbook scheme of a step, reactants -> product in one row (pure; node --test runs it):
// the model layout (schemeGeometry: the reactant pool as drawn, the product fitted onto it and
// moved SCHEME_GAP bond lengths to its right, level with it, and the reaction arrow between)
// and its drawing in a view (drawScheme: the two workbench scenes, the lone pairs of the atoms
// the step touches, the routed curved arrows). StepTextbook and StepOrbital draw it on screen,
// stepExport.ts writes it in points.
import type { Scene } from "../../workbench/sketch/drawing";
import type { View } from "../../workbench/sketch/geometry";
import type { DrawMetrics, DrawingStyle } from "../../workbench/sketch/style";
import type { Vec } from "../arrowPath";
import type { Side, SideAtom, XY } from "./stepGeometry";
import type { AtomGeom, LonePairMark } from "./stepScene";
import type { RoutedArrow } from "./stepRoute";
import type { ArrowReadingOut, MatrixIO } from "../v9types";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { buildScene } from "../../workbench/sketch/drawing.ts";
// @ts-ignore TS5097: see above
import { toScreen } from "../../workbench/sketch/geometry.ts";
// @ts-ignore TS5097: see above
import { BOND_LENGTH } from "../../workbench/sketch/model.ts";
// @ts-ignore TS5097: see above
import { drawMetrics, pxPerPt } from "../../workbench/sketch/style.ts";
// @ts-ignore TS5097: see above
import { touchedAtoms } from "../poolStructure.ts";
// @ts-ignore TS5097: see above
import { textWidthEm } from "../../workbench/sketch/fontMetrics.ts";
// @ts-ignore TS5097: see above
import { bounds, shifted, sideFragments, spreadFragments, xyOf } from "./stepGeometry.ts";
// @ts-ignore TS5097: see above
import { atomGeometry, donorPreferences, lonePairLayout, routeReading, sideModel } from "./stepScene.ts";

/** Product atom ids are pool atom + PRODUCT_ID (the two scenes never share an id). */
export const PRODUCT_ID = 100000;
/** Space between the reactants and the product, in bond lengths (the reaction arrow's room). */
export const SCHEME_GAP = 4.2;

/** The space between reactants and product (bond lengths) that fits the reaction arrow under
 *  and over its texts: `size` is the text size in bond lengths. */
export function schemeGapFor(texts: readonly (string | undefined)[], size: number): number {
  const w = Math.max(0, ...texts.filter((t): t is string => !!t).map((t) => textWidthEm(t) * size));
  return Math.max(SCHEME_GAP, w + 2.4);
}

export interface SchemeInput {
  source: Side;
  product: Side | null;
  /** Product positions fitted onto the reactant (stepGeometry.alignProduct). */
  aligned: XY | null;
  matrix: MatrixIO;
  arrows: readonly ArrowReadingOut[];
}

export interface SchemeGeometry {
  react: XY;
  prod: XY | null;
  box: { x0: number; y0: number; x1: number; y1: number };
  /** The reaction arrow (model units), null without a product. */
  rxn: { from: Vec; to: Vec } | null;
}

/** Half the width (model units) of an atom's label as drawn, from its text: `charW` is a
 *  character's width in bond lengths (ACS 1996 ~0.42, the screen styles ~0.22). */
export function labelHalfWidth(a: SideAtom, charW: number, labelCarbons: boolean): number {
  const methyl = a.el === "C" && a.hCount === 3;
  const shown = a.el !== "C" || a.charge !== 0 || methyl || (labelCarbons && a.labelled);
  if (!shown) return 0.12 * BOND_LENGTH;
  const chars = a.el.length + (a.hCount > 0 ? (a.hCount > 1 ? 2 : 1) : 0) + (a.charge ? 1.2 : 0);
  return ((chars * charW) / 2 + (a.lone ? 0.3 : 0)) * BOND_LENGTH;
}

export interface SpreadOptions {
  charW: number;
  labelCarbons: boolean;
  /** Clear room between fragments' labels, in bond lengths. */
  gapBonds: number;
}

export const TEXTBOOK_SPREAD: SpreadOptions = { charW: 0.42, labelCarbons: false, gapBonds: 1.5 };

/** A side's positions with its fragments spread so their labels keep `gapBonds` apart. */
export function spreadSide(side: Side, xy: XY, o: SpreadOptions): XY {
  const half = (i: number) => {
    const a = side.atoms.get(i);
    return a ? labelHalfWidth(a, o.charW, o.labelCarbons) : 0;
  };
  return spreadFragments(xy, sideFragments(side), half, o.gapBonds * BOND_LENGTH);
}

export function schemeGeometry(inp: SchemeInput, gapBonds = SCHEME_GAP, spread: SpreadOptions = TEXTBOOK_SPREAD): SchemeGeometry {
  const react = spreadSide(inp.source, xyOf(inp.source), spread);
  const rb = bounds([react]) ?? { x0: 0, x1: 0, y0: 0, y1: 0 };
  if (!inp.product || !inp.aligned || !inp.aligned.size) return { react, prod: null, box: rb, rxn: null };
  const aligned = spreadSide(inp.product, inp.aligned, spread);
  const pb0 = bounds([aligned])!;
  // the arrow sits between the drawn extents: the reactants' rightmost label edge and the
  // product's leftmost one, with half a bond of air on each side
  const half = (side: Side) => (i: number) => {
    const a = side.atoms.get(i);
    return a ? labelHalfWidth(a, spread.charW, spread.labelCarbons) : 0;
  };
  const right = Math.max(...[...react].map(([i, p]) => p[0] + half(inp.source)(i)));
  const left = Math.min(...[...aligned].map(([i, p]) => p[0] - half(inp.product!)(i)));
  const m = 0.5 * BOND_LENGTH;
  const arrowLen = Math.max(2.4, gapBonds - 1.8) * BOND_LENGTH;
  // (gapBonds - 1.8: the half bond of air on each side and the arrowhead's overhang)
  const prod = shifted(aligned, right + 2 * m + arrowLen - left, (rb.y0 + rb.y1) / 2 - (pb0.y0 + pb0.y1) / 2);
  const box = bounds([react, prod])!;
  const y = (rb.y0 + rb.y1) / 2;
  return { react, prod, box, rxn: { from: [right + m, y], to: [right + m + arrowLen, y] } };
}

export interface SchemeDrawing {
  view: View;
  M: DrawMetrics;
  react: Scene;
  prod: Scene | null;
  geomR: Map<number, AtomGeom>;
  geomP: Map<number, AtomGeom> | null;
  pairsR: Map<number, LonePairMark>;
  pairsP: Map<number, LonePairMark>;
  arrows: (RoutedArrow | null)[];
  /** Screen positions of the reactant atoms by pool atom (the R overlay's). */
  xyR: Map<number, Vec>;
  rxn: { from: Vec; to: Vec } | null;
  /** Atom ids of the new lone pair on each atom that gains one (product scene; pair 0). */
  newPairs: Set<number>;
}

const sceneBonds = (s: Scene): [Vec, Vec][] => s.bonds.map((b) => [b.axis.p, b.axis.q]);

/** The lone pairs to draw: touched atoms that have some, on each side. */
function wantPairs(side: Side, touched: ReadonlySet<number>, offset: number, prefer: ReadonlyMap<number, number>): Map<number, { n: number; prefer?: number | null }> {
  const out = new Map<number, { n: number; prefer?: number | null }>();
  for (const i of touched) {
    const a = side.atoms.get(i);
    if (a?.lone) out.set(i + offset, { n: a.lone, prefer: prefer.get(i + offset) ?? null });
  }
  return out;
}

/** Atoms that gain a lone pair, each facing away from the partner it lost (product screen px). */
function gainPrefs(m: MatrixIO, xy: (i: number) => Vec | undefined): Map<number, number> {
  const out = new Map<number, number>();
  for (const [i, d] of m.lone) {
    if (d <= 0) continue;
    const broken = m.bonds.find(([a, b, x]) => x < 0 && (a === i || b === i));
    const partner = broken ? (broken[0] === i ? broken[1] : broken[0]) : undefined;
    const [p, q] = [xy(i), partner !== undefined ? xy(partner) : undefined];
    if (p && q) out.set(i, Math.atan2(p[1] - q[1], p[0] - q[0]));
  }
  return out;
}

/** The scheme drawn in `view` with `style` (metrics follow the zoom for publication styles). */
export function drawScheme(inp: SchemeInput, g: SchemeGeometry, view: View, style: DrawingStyle, metricText = false, labelCarbons = true): SchemeDrawing {
  const M = { ...drawMetrics(style, pxPerPt(style, view.s)), ...(metricText ? { metricText: true } : {}) };
  const rm = sideModel(inp.source, 0, g.react, null, labelCarbons);
  const react = buildScene(rm.model, view, rm.hCounts, { metrics: M });
  const geomR = atomGeometry(react, M);
  const touched = touchedAtoms(inp.matrix, inp.arrows);
  const pairsR = lonePairLayout(geomR, wantPairs(inp.source, touched, 0, donorPreferences(inp.arrows, geomR, (i) => i)), M, sceneBonds(react));
  const arrows = routeReading(inp.arrows, react, geomR, pairsR, M);
  let prod: Scene | null = null;
  let geomP: Map<number, AtomGeom> | null = null;
  let pairsP = new Map<number, LonePairMark>();
  const newPairs = new Set<number>();
  if (inp.product && g.prod) {
    const pm = sideModel(inp.product, PRODUCT_ID, g.prod, null, labelCarbons);
    prod = buildScene(pm.model, view, pm.hCounts, { metrics: M });
    geomP = atomGeometry(prod, M);
    const gp = geomP;
    const prefs = gainPrefs(inp.matrix, (i) => gp.get(i + PRODUCT_ID)?.at);
    const keyed = new Map([...prefs].map(([i, a]) => [i + PRODUCT_ID, a] as const));
    pairsP = lonePairLayout(geomP, wantPairs(inp.product, touched, PRODUCT_ID, keyed), M, sceneBonds(prod));
    for (const i of prefs.keys()) newPairs.add(i + PRODUCT_ID);
  }
  const xyR = new Map([...g.react].map(([i, p]) => [i, toScreen(view, p)] as const));
  const rxn = g.rxn ? { from: toScreen(view, g.rxn.from), to: toScreen(view, g.rxn.to) } : null;
  return { view, M, react, prod, geomR, geomP, pairsR, pairsP, arrows, xyR, rxn, newPairs };
}
