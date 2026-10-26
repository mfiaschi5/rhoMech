// The textbook scheme as a publication file (pure; node --test runs it): the same layout as the
// screen (stepScheme.ts) drawn in points at the style's true size (ACS 1996: 14.4 pt bonds),
// black on white or transparent. Molecules, lone pairs, the reaction arrow and its text are the
// workbench export's primitives (exportImage.ts, written by its toSvg); the curved arrows are
// added as cubic <path>s with filled heads, so they stay true curves in the file.
import type { DrawingStyle } from "../workbench/sketch/style";
import type { Prim } from "../workbench/sketch/exportImage";
import type { DrawMetrics } from "../workbench/sketch/style";
import type { Vec } from "./arrowPath";
import type { SchemeInput } from "./stepScheme";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { primsBox, scenePrimitives, toSvg } from "../workbench/sketch/exportImage.ts";
// @ts-ignore TS5097: see above
import { cssFontFamily } from "../workbench/sketch/style.ts";
// @ts-ignore TS5097: see above
import { textWidthEm } from "../workbench/sketch/fontMetrics.ts";
// @ts-ignore TS5097: see above
import { BOND_LENGTH } from "../workbench/sketch/model.ts";
// @ts-ignore TS5097: see above
import { arrowFromControls, cubicPoint, type ArrowStyle } from "./arrowPath.ts";
// @ts-ignore TS5097: see above
import { drawScheme, schemeGapFor, schemeGeometry } from "./stepScheme.ts";
// @ts-ignore TS5097: see above
import { dotRadius } from "./stepScene.ts";

/** Curved-arrow sizes for a drawing's metrics (screen and file alike). */
export function arrowStyleFor(M: DrawMetrics): ArrowStyle & { stroke: number } {
  return { gapTail: M.font * 0.06, gapHead: M.font * 0.1, headLength: M.bond * 0.27, headWidth: M.bond * 0.2, stroke: Math.max(M.line * 1.3, M.bond * 0.03) };
}

export interface SchemeFile {
  svg: string;
  widthPt: number;
  heightPt: number;
}

export interface SchemeFileOptions {
  above?: string;
  below?: string;
  background?: string | null;
  padPt?: number;
  title?: string;
  labelCarbons?: boolean;
}

const f = (x: number) => String(Math.round(x * 1000) / 1000);
const BLACK = "#000000";

function textPrim(text: string, cx: number, baseline: number, size: number): Prim {
  return { k: "text", x: cx - (textWidthEm(text) * size) / 2, y: baseline, text, size, weight: 400, italic: false, colour: BLACK };
}

/** Text over and under the reaction arrow, as a fraction of the label font. */
export const RXN_TEXT = { above: 0.72, below: 0.62 };

/** The scheme file (see the header). */
export function schemeSvg(inp: SchemeInput, style: DrawingStyle, o: SchemeFileOptions = {}): SchemeFile {
  const g = schemeGeometry(inp, schemeGapFor([o.above, o.below], (style.fontSizePt * RXN_TEXT.above) / style.bondLengthPt));
  const view = { s: style.bondLengthPt / BOND_LENGTH, ox: 0, oy: 0 };
  const d = drawScheme(inp, g, view, style, true, o.labelCarbons ?? true);
  const M = d.M;
  const prims: Prim[] = [...scenePrimitives(d.react, false), ...(d.prod ? scenePrimitives(d.prod, false) : [])];
  const r = dotRadius(M);
  for (const m of [...d.pairsR.values(), ...d.pairsP.values()]) for (const pair of m.dots) for (const c of pair) prims.push({ k: "circle", c, r, colour: BLACK, fill: true, w: 0 });
  if (d.rxn) {
    const { from, to } = d.rxn;
    const hl = M.font * 0.9;
    prims.push({ k: "line", p: from, q: [to[0] - hl * 0.8, to[1]], w: M.line, colour: BLACK });
    prims.push({ k: "poly", pts: [to, [to[0] - hl, to[1] - hl * 0.42], [to[0] - hl * 0.78, to[1]], [to[0] - hl, to[1] + hl * 0.42]], colour: BLACK });
    const cx = (from[0] + to[0]) / 2;
    if (o.above) prims.push(textPrim(o.above, cx, from[1] - M.font * 0.45, M.font * RXN_TEXT.above));
    if (o.below) prims.push(textPrim(o.below, cx, from[1] + M.font * 1.0, M.font * RXN_TEXT.below));
  }
  const st = arrowStyleFor(M);
  const curves = d.arrows.filter((a): a is NonNullable<typeof a> => !!a);
  const curvePts: Vec[] = curves.flatMap((a) => Array.from({ length: 17 }, (_, k) => cubicPoint(a.tail, a.c1, a.c2, a.head, k / 16)));
  const box = primsBox([...prims, ...curvePts.map((c): Prim => ({ k: "circle", c, r: st.headWidth ?? 2, colour: BLACK, fill: true, w: 0 }))]) ?? { x0: 0, y0: 0, x1: 0, y1: 0 };
  const pad = o.padPt ?? 4;
  const [dx, dy] = [pad - box.x0, pad - box.y0];
  const mv = (v: Vec): Vec => [v[0] + dx, v[1] + dy];
  const moved = prims.map((p): Prim => (p.k === "line" ? { ...p, p: mv(p.p), q: mv(p.q) } : p.k === "poly" ? { ...p, pts: p.pts.map(mv) } : p.k === "circle" ? { ...p, c: mv(p.c) } : { ...p, x: p.x + dx, y: p.y + dy }));
  const layout = { prims: moved, widthPt: box.x1 - box.x0 + 2 * pad, heightPt: box.y1 - box.y0 + 2 * pad, fontFamily: cssFontFamily(style.fontFamily, false), fontWeight: style.fontWeight, bondLengthPt: style.bondLengthPt, lineWidthPt: style.lineWidthPt };
  const base = toSvg(layout, { background: o.background ?? null, title: o.title ?? "ksmech step" });
  const arrows = curves
    .map((a) => {
      const p = arrowFromControls(mv(a.tail), mv(a.c1), mv(a.c2), mv(a.head), st);
      return `<path d="${p.d}" fill="none" stroke="${BLACK}" stroke-width="${f(st.stroke)}" stroke-linecap="round"/><polygon points="${p.head.map((v: Vec) => `${f(v[0])},${f(v[1])}`).join(" ")}" fill="${BLACK}"/>`;
    })
    .join("");
  return { svg: base.replace(/<\/svg>$/, `<g class="arrows">${arrows}</g></svg>`), widthPt: layout.widthPt, heightPt: layout.heightPt };
}
