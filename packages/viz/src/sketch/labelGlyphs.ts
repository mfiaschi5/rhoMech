// The glyph geometry of an atom label, in output units about the atom point (y down): where the
// element symbol, the H, the H-count subscript and the charge sign go, how wide they are (real
// Helvetica / Arial advance widths from fontMetrics.ts for the publication fonts, the UI-font
// estimate for Screen) and the ink box of each run (cap height for capitals and digits, the
// sign's own height for a charge). drawing.ts, labelLayout.ts (collision-aware placement),
// SketchCanvas and the export all read these, so the screen, the placement and the files agree.
//
// A label may carry a placement from labelLayout.ts: `scale` shrinks the subscript and the
// charge (never below the style editor's 40 % of the font), `offset` displaces the whole label
// (render only, with a leader to the atom point).
//
// Pure: runs under node --experimental-strip-types.
import type { Vec } from "./geometry";
import type { AtomLabel, HSide } from "./drawing";
import type { DrawMetrics } from "./style";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { screenMetrics } from "./style.ts";
// @ts-ignore TS5097: see above
import { CAP_HEIGHT_EM, textWidthEm } from "./fontMetrics.ts";

export const FONT_PX = 13;

/** An axis-aligned box (output units, y down). */
export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export type FontArg = number | DrawMetrics;
export const metricsOf = (f: FontArg | undefined): DrawMetrics => (typeof f === "object" ? f : screenMetrics(f ?? FONT_PX));

/** Half the width of an element symbol: Helvetica metrics for publication fonts, else the
 *  UI-font estimate (capital ~0.62 em, lowercase ~0.5 em). */
export function labelHalfWidth(el: string, f: FontArg = FONT_PX): number {
  const M = metricsOf(f);
  return M.metricText ? (textWidthEm(el) * M.font) / 2 : ((0.62 + 0.5 * (el.length - 1)) * M.font) / 2;
}

const hWidth = (M: DrawMetrics) => (M.metricText ? textWidthEm("H") * M.font : M.font * 0.68);
const scaleOf = (label: AtomLabel) => label.scale ?? 1;
const subSize = (label: AtomLabel, M: DrawMetrics) => M.sub * scaleOf(label);
const chargeSize = (label: AtomLabel, M: DrawMetrics) => M.charge * scaleOf(label);

function digitsWidth(n: number, M: DrawMetrics, size = M.sub): number {
  if (n <= 1) return 0;
  return M.metricText ? textWidthEm(String(n)) * size : (M.font * 0.45 * size) / M.sub;
}

/** Width of the H part of a label ("H", "H2"...); 0 without H. */
export function hLabelWidth(hCount: number, f: FontArg = FONT_PX, scale = 1): number {
  const M = metricsOf(f);
  return hCount > 0 ? hWidth(M) + digitsWidth(hCount, M, M.sub * scale) : 0;
}

/** One glyph run of a label: start x and baseline y about the atom (y down), size and width. */
export interface Glyph {
  text: string;
  x: number;
  y: number;
  size: number;
  w: number;
  kind: "symbol" | "h" | "count" | "charge";
  /** A charge drawn as a circled sign (its ink is the circle). */
  circled?: boolean;
  /** The label is displaced from its atom (labelLayout.ts). */
  displaced?: boolean;
}

/** Baseline offset that puts the middle of a capital on the atom point. */
const baseline = (size: number) => (CAP_HEIGHT_EM / 2) * size;

/** Gap between the symbol (or its H) and the charge sign. */
const chargeGap = (M: DrawMetrics) => (M.metricText ? M.font * 0.04 : 1);

/** Width of the charge mark (text, or the circled sign). */
export function chargeWidth(label: AtomLabel, f: FontArg = FONT_PX): number {
  const M = metricsOf(f);
  const size = chargeSize(label, M);
  if (M.chargePlacement === "circled" && label.charge.length === 1) return size * 1.1;
  return M.metricText ? textWidthEm(label.charge) * size : label.charge.length * size * 0.62;
}

const offsetOf = (label: AtomLabel): Vec => label.offset ?? [0, 0];

/** Where the charge sign starts (x, about the atom) and its vertical centre (y, down). */
export function chargeAnchor(label: AtomLabel, f: FontArg = FONT_PX): Vec {
  const M = metricsOf(f);
  const [ox, oy] = offsetOf(label);
  const half = labelHalfWidth(label.el, M);
  const hw = hLabelWidth(label.hCount, M, scaleOf(label));
  const slot = label.chargeSlot ?? "ur";
  const cw = chargeWidth(label, M);
  const right = ox + half + (label.hSide === "right" ? hw : 0) + chargeGap(M);
  const left = ox - half - (label.hSide === "left" ? hw : 0) - chargeGap(M) - cw;
  if (slot === "u" || slot === "d") return [ox - cw / 2, oy + (slot === "u" ? -1 : 1) * M.font * 0.82];
  if (slot === "r" || slot === "l") return [slot === "r" ? right + chargeGap(M) : left - chargeGap(M), oy];
  const y = oy + (slot[0] === "u" ? -M.font * 0.55 : M.font * 0.55);
  return [slot[1] === "r" ? right : left, y];
}

/** The H run's start (x) and baseline (y) about the label origin. */
function hStart(side: HSide, half: number, hw: number, M: DrawMetrics): Vec {
  const base = baseline(M.font);
  if (side === "right") return [half, base];
  if (side === "left") return [-half - hw, base];
  return [-hw / 2, base + (side === "above" ? -M.font * 0.95 : M.font * 0.95)];
}

/** The glyph runs of a label (symbol, H, H count, charge) about the atom, its offset applied. */
export function labelGlyphs(label: AtomLabel, f: FontArg = FONT_PX): Glyph[] {
  const M = metricsOf(f);
  const [ox, oy] = offsetOf(label);
  const half = labelHalfWidth(label.el, M);
  const out: Glyph[] = [{ text: label.el, x: ox - half, y: oy + baseline(M.font), size: M.font, w: 2 * half, kind: "symbol", ...(label.offset ? { displaced: true } : {}) }];
  if (label.hCount > 0) {
    const sub = subSize(label, M);
    const [x, y] = hStart(label.hSide, half, hLabelWidth(label.hCount, M, scaleOf(label)), M);
    out.push({ text: "H", x: ox + x, y: oy + y, size: M.font, w: hWidth(M), kind: "h" });
    if (label.hCount > 1) {
      out.push({ text: String(label.hCount), x: ox + x + hWidth(M), y: oy + y + M.font * 0.3, size: sub, w: digitsWidth(label.hCount, M, sub), kind: "count" });
    }
  }
  if (label.charge) {
    const [cx, cy] = chargeAnchor(label, M);
    const size = chargeSize(label, M);
    const circled = M.chargePlacement === "circled" && label.charge.length === 1;
    out.push({ text: label.charge, x: cx, y: cy + baseline(size), size, w: chargeWidth(label, M), kind: "charge", ...(circled ? { circled } : {}) });
  }
  return out;
}

const DESCENDERS = /[gjpqy]/;

/** The ink box of one glyph run (about the atom): capitals and digits at cap height, a
 *  descender where the symbol has one, a charge sign about its centre line. */
export function glyphInk(g: Glyph): Box {
  if (g.kind === "charge") {
    const cy = g.y - baseline(g.size);
    const h = g.circled ? g.w / 2 : (/\d/.test(g.text) ? 0.37 : 0.3) * g.size;
    return { x0: g.x, x1: g.x + g.w, y0: cy - h, y1: cy + h };
  }
  const top = g.y - CAP_HEIGHT_EM * g.size;
  const bottom = g.y + (DESCENDERS.test(g.text) ? 0.21 * g.size : 0);
  return { x0: g.x, x1: g.x + g.w, y0: top, y1: bottom };
}

/** The ink boxes of a label's glyph runs, in glyph order (about the atom). */
export const labelInk = (label: AtomLabel, f: FontArg = FONT_PX): Box[] => labelGlyphs(label, metricsOf(f)).map(glyphInk);

/** The box (about the atom, y down) of everything a label draws. */
export function labelBox(label: AtomLabel, f: FontArg = FONT_PX): Box {
  const M = metricsOf(f);
  const oy = offsetOf(label)[1];
  const box = { x0: Infinity, x1: -Infinity, y0: oy - M.font * 0.5, y1: oy + M.font * 0.5 };
  for (const g of labelGlyphs(label, M)) {
    box.x0 = Math.min(box.x0, g.x);
    box.x1 = Math.max(box.x1, g.x + g.w);
    box.y0 = Math.min(box.y0, g.y - g.size * 0.86);
    box.y1 = Math.max(box.y1, g.y + g.size * (g.kind === "count" ? 0.2 : 0.14));
  }
  return box;
}
