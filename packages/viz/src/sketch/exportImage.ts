// Publication export of a sketch: the drawing in a drawing style (style.ts) at its true
// physical size, as a list of primitives (lines, filled polygons, circles, text with explicit
// baselines) that the SVG writer here and the PDF writer (pdf.ts) both render, so the two
// files are the same picture. Units are points (1/72 in): a standard model bond is exactly the
// style's fixed bond length (ACS 1996: 14.4 pt = 0.508 cm). Only the molecule is drawn: no
// selection, hover or highlight, optional CIP labels, optional white background.
//
// exportLayout(model, hCounts, style, opts) -> { prims, widthPt, heightPt }
// toSvg(layout, opts) -> an SVG document (width/height in pt, viewBox in pt)
// pixelSize(layout, dpi) -> the PNG size at a resolution; pngWithDpi(bytes, dpi) writes the
// resolution into a PNG (a pHYs chunk) so Word and LaTeX place it at its physical size.
//
// Pure (no DOM): runs under node --experimental-strip-types.
import type { SketchModel } from "./model";
import type { StereoLabels } from "./cip";
import type { AtomScene, LabelStyle, Scene } from "./drawing";
import type { DrawingStyle, DrawMetrics } from "./style";
import type { Vec } from "./geometry";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { buildScene, chargeAnchor, labelGlyphs } from "./drawing.ts";
// @ts-ignore TS5097: see above
import { cssFontFamily, drawMetrics, MODEL_BOND } from "./style.ts";
// @ts-ignore TS5097: see above
import { CENTRAL_BASELINE_EM, textWidthEm } from "./fontMetrics.ts";

export type Prim =
  | { k: "line"; p: Vec; q: Vec; w: number; colour: string }
  | { k: "poly"; pts: Vec[]; colour: string }
  | { k: "circle"; c: Vec; r: number; colour: string; fill: boolean; w: number }
  | { k: "text"; x: number; y: number; text: string; size: number; weight: 400 | 700; italic: boolean; colour: string };

export interface ExportLayout {
  prims: Prim[];
  widthPt: number;
  heightPt: number;
  /** CSS font-family of the text. */
  fontFamily: string;
  fontWeight: 400 | 700;
  /** The style's fixed bond length (pt) and line width (pt), for checks and the LaTeX snippet. */
  bondLengthPt: number;
  lineWidthPt: number;
}

export interface ExportOptions {
  cip?: StereoLabels | null;
  labelStyle?: LabelStyle;
  /** Margin (pt) around the drawing; default 2 pt. */
  padPt?: number;
}

/** Label colours of the colour styles (the sketcher's light-theme element tokens). */
const ELEMENT_COLOURS: Record<string, string> = {
  N: "#2850d8", O: "#d0261c", S: "#a08000", P: "#d06a00", F: "#1f8a3c", Cl: "#1f8a3c", Br: "#9a2a2a", I: "#7a1f9a", B: "#c06a6a", Si: "#8a6a30",
};
const BLACK = "#000000";

export function labelColour(el: string, colour: boolean): string {
  if (!colour || el === "C" || el === "H") return BLACK;
  return ELEMENT_COLOURS[el] ?? "#6a4fa0";
}

/** The text pieces of one atom label (drawing.labelGlyphs), with the circled charge drawn as a
 *  circle around its sign. */
function labelPrims(s: AtomScene, M: DrawMetrics): Prim[] {
  const label = s.label;
  if (!label) return [];
  const [x, y] = s.at;
  const colour = labelColour(label.el, M.colour);
  const leader: Prim[] = s.leader ? [{ k: "line", p: s.leader.p, q: s.leader.q, w: M.line * 0.6, colour: BLACK }] : [];
  return leader.concat(
    labelGlyphs(label, M).flatMap((g): Prim[] => {
      if (g.kind === "charge") return chargePrims(g.text, x + g.x, y + chargeAnchor(label, M)[1], M, colour, g.size);
      return [{ k: "text", x: x + g.x, y: y + g.y, text: g.text, size: g.size, weight: M.fontWeight, italic: false, colour }];
    }),
  );
}

/** The charge sign at its glyph size (`size`: the placement may shrink it). */
function chargePrims(charge: string, x: number, cy: number, M: DrawMetrics, colour: string, chargeSize = M.charge): Prim[] {
  if (M.chargePlacement === "circled" && charge.length === 1) {
    const r = chargeSize * 0.55;
    const size = chargeSize * 0.95;
    const w = textWidthEm(charge) * size;
    return [
      { k: "circle", c: [x + r, cy], r, colour, fill: false, w: Math.max(M.line * 0.7, 0.3 * M.perPt) },
      { k: "text", x: x + r - w / 2, y: cy + size * 0.3, text: charge, size, weight: 400, italic: false, colour },
    ];
  }
  return [{ k: "text", x, y: cy + CENTRAL_BASELINE_EM * chargeSize, text: charge, size: chargeSize, weight: M.fontWeight, italic: false, colour }];
}

/** Everything a scene draws, as primitives (no UI decoration). */
export function scenePrimitives(scene: Scene, withCip: boolean): Prim[] {
  const M = scene.metrics;
  const out: Prim[] = [];
  for (const b of scene.bonds) {
    const w = b.bond.order === 1 && b.bond.stereo === "hash" ? M.hashWidth : M.line;
    for (const l of b.shape.lines) out.push({ k: "line", p: l.p, q: l.q, w, colour: BLACK });
    for (const pts of b.shape.wedges ?? []) out.push({ k: "poly", pts, colour: BLACK });
  }
  for (const a of scene.atoms) {
    out.push(...labelPrims(a, M));
    for (const d of a.dots) out.push({ k: "circle", c: d, r: scene.dotR, colour: BLACK, fill: true, w: 0 });
  }
  if (withCip) {
    const size = M.font * 0.75;
    const marks = [...scene.atoms.map((a) => a.cip), ...scene.bonds.map((b) => b.cip)].filter((m) => !!m);
    for (const m of marks) {
      const w = textWidthEm(m.text) * size;
      out.push({ k: "text", x: m.at[0] - w / 2, y: m.at[1] + CENTRAL_BASELINE_EM * size, text: m.text, size, weight: 400, italic: true, colour: BLACK });
    }
  }
  return out;
}

interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

function primBox(p: Prim): Box {
  if (p.k === "line") {
    const h = p.w / 2;
    return { x0: Math.min(p.p[0], p.q[0]) - h, x1: Math.max(p.p[0], p.q[0]) + h, y0: Math.min(p.p[1], p.q[1]) - h, y1: Math.max(p.p[1], p.q[1]) + h };
  }
  if (p.k === "poly") {
    const xs = p.pts.map((v) => v[0]);
    const ys = p.pts.map((v) => v[1]);
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  }
  if (p.k === "circle") {
    const r = p.r + p.w / 2;
    return { x0: p.c[0] - r, x1: p.c[0] + r, y0: p.c[1] - r, y1: p.c[1] + r };
  }
  const w = textWidthEm(p.text) * p.size;
  return { x0: p.x, x1: p.x + w, y0: p.y - 0.75 * p.size, y1: p.y + 0.22 * p.size };
}

/** The bounding box of primitives (null when there are none). */
export function primsBox(prims: readonly Prim[]): Box | null {
  if (prims.length === 0) return null;
  return prims.map(primBox).reduce((a, b) => ({ x0: Math.min(a.x0, b.x0), y0: Math.min(a.y0, b.y0), x1: Math.max(a.x1, b.x1), y1: Math.max(a.y1, b.y1) }));
}

const shiftV = (v: Vec, dx: number, dy: number): Vec => [v[0] + dx, v[1] + dy];

function shift(p: Prim, dx: number, dy: number): Prim {
  if (p.k === "line") return { ...p, p: shiftV(p.p, dx, dy), q: shiftV(p.q, dx, dy) };
  if (p.k === "poly") return { ...p, pts: p.pts.map((v) => shiftV(v, dx, dy)) };
  if (p.k === "circle") return { ...p, c: shiftV(p.c, dx, dy) };
  return { ...p, x: p.x + dx, y: p.y + dy };
}

/** The sketch in `style` at true size (pt), its box `padPt` beyond the ink. */
export function exportLayout(m: SketchModel, hCounts: ReadonlyMap<number, number> | null, style: DrawingStyle, opts: ExportOptions = {}): ExportLayout {
  const metrics = { ...drawMetrics(style, 1), metricText: true };
  const view = { s: style.bondLengthPt / MODEL_BOND, ox: 0, oy: 0 };
  const scene = buildScene(m, view, hCounts, { metrics, labelStyle: opts.labelStyle, cip: opts.cip ?? null });
  const raw = scenePrimitives(scene, !!opts.cip);
  const pad = opts.padPt ?? 2;
  const box = primsBox(raw) ?? { x0: 0, y0: 0, x1: 0, y1: 0 };
  const prims = raw.map((p) => shift(p, pad - box.x0, pad - box.y0));
  return {
    prims,
    widthPt: box.x1 - box.x0 + 2 * pad,
    heightPt: box.y1 - box.y0 + 2 * pad,
    fontFamily: cssFontFamily(style.fontFamily, false),
    fontWeight: style.fontWeight,
    bondLengthPt: style.bondLengthPt,
    lineWidthPt: style.lineWidthPt,
  };
}

const f = (x: number) => String(Math.round(x * 1000) / 1000);
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function svgPrim(p: Prim): string {
  if (p.k === "line") return `<line x1="${f(p.p[0])}" y1="${f(p.p[1])}" x2="${f(p.q[0])}" y2="${f(p.q[1])}" stroke="${p.colour}" stroke-width="${f(p.w)}"/>`;
  if (p.k === "poly") return `<polygon points="${p.pts.map((v) => `${f(v[0])},${f(v[1])}`).join(" ")}" fill="${p.colour}"/>`;
  if (p.k === "circle") {
    const paint = p.fill ? `fill="${p.colour}"` : `fill="none" stroke="${p.colour}" stroke-width="${f(p.w)}"`;
    return `<circle cx="${f(p.c[0])}" cy="${f(p.c[1])}" r="${f(p.r)}" ${paint}/>`;
  }
  const extra = `${p.italic ? ' font-style="italic"' : ""}${p.weight === 700 ? ' font-weight="bold"' : ""}`;
  return `<text x="${f(p.x)}" y="${f(p.y)}" font-size="${f(p.size)}" fill="${p.colour}"${extra}>${esc(p.text)}</text>`;
}

/** An SVG document of the layout: width and height in pt, the viewBox in pt (1 user unit =
 *  1 pt), every text with an explicit baseline and start position (no dominant-baseline, which
 *  Word and Inkscape read differently). */
export function toSvg(l: ExportLayout, opts: { background?: string | null; title?: string } = {}): string {
  const W = f(l.widthPt);
  const H = f(l.heightPt);
  const bg = opts.background ? `<rect width="${W}" height="${H}" fill="${opts.background}"/>` : "";
  const title = opts.title ? `<title>${esc(opts.title)}</title>` : "";
  const shapes = l.prims.filter((p) => p.k !== "text").map(svgPrim).join("");
  const texts = l.prims.filter((p) => p.k === "text").map(svgPrim).join("");
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="${W}pt" height="${H}pt" viewBox="0 0 ${W} ${H}">` +
    title +
    bg +
    `<g stroke-linecap="round" stroke-linejoin="round">${shapes}</g>` +
    `<g font-family="${esc(l.fontFamily)}"${l.fontWeight === 700 ? ' font-weight="bold"' : ""}>${texts}</g>` +
    `</svg>`
  );
}

/** The PNG size (px) of the layout at `dpi`. */
export function pixelSize(l: ExportLayout, dpi: number): { w: number; h: number } {
  return { w: Math.max(1, Math.round((l.widthPt / 72) * dpi)), h: Math.max(1, Math.round((l.heightPt / 72) * dpi)) };
}

/** Physical size of the layout in cm. */
export const sizeCm = (l: ExportLayout): { w: number; h: number } => ({ w: (l.widthPt / 72) * 2.54, h: (l.heightPt / 72) * 2.54 });

let crcTable: Uint32Array | null = null;

export function crc32(bytes: Uint8Array): number {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c >>> 0;
    }
  }
  let c = 0xffffffff;
  for (const b of bytes) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** The PNG with its resolution set to `dpi` (a pHYs chunk after IHDR, replacing any). */
export function pngWithDpi(png: Uint8Array, dpi: number): Uint8Array {
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
  const chunks: Uint8Array[] = [png.subarray(0, 8)];
  let at = 8;
  while (at + 8 <= png.length) {
    const n = view.getUint32(at);
    const type = String.fromCharCode(...png.subarray(at + 4, at + 8));
    const end = at + 12 + n;
    if (type !== "pHYs") chunks.push(png.subarray(at, end));
    if (type === "IHDR") chunks.push(physChunk(dpi));
    at = end;
  }
  const out = new Uint8Array(chunks.reduce((s, c) => s + c.length, 0));
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

function physChunk(dpi: number): Uint8Array {
  const ppm = Math.round(dpi / 0.0254);
  const c = new Uint8Array(21);
  const v = new DataView(c.buffer);
  v.setUint32(0, 9);
  c.set([0x70, 0x48, 0x59, 0x73], 4); // "pHYs"
  v.setUint32(8, ppm);
  v.setUint32(12, ppm);
  c[16] = 1; // unit: metre
  v.setUint32(17, crc32(c.subarray(4, 17)));
  return c;
}
