// LaTeX output of a sketch, for Overleaf and other LaTeX documents:
//
// latexFigure(layout, file)  a figure environment that includes the exported PDF at its
//                            physical width (\includegraphics[width=<cm>cm]{scheme.pdf}).
// toChemfig(model, hCounts, style)  chemfig code (\usepackage{chemfig}) that redraws the
//                            molecule natively in LaTeX: each bond with its absolute angle from
//                            the sketch's 2D coordinates (and a length factor when it differs
//                            from the standard bond by more than 5 %), branches in parentheses,
//                            ring closures as chemfig hooks (?[a] ... ?[a,<bond>]), double bonds
//                            offset to the ring side (=^ / =_), wedges and hashes as Cram bonds
//                            (< > <: >:), heteroatom labels with their H count and charge
//                            (OH, H_2N, N^{+}) on the side the drawing's label placement
//                            chose (labelLayout.ts; a charge off the upper right as
//                            \charge{<angle>=...}{N}), atom sep and widths from the style.
//                            Not supported: radicals (dropped, noted in a comment), explicit
//                            crossing breaks, circled charges and per-element colours.
// Pure: runs under node --experimental-strip-types.
import type { SketchAtom, SketchBond, SketchModel } from "./model";
import type { Vec } from "./geometry";
import type { DrawingStyle } from "./style";
import type { ExportLayout } from "./exportImage";
import type { AtomLabel, ChargeSlot } from "./drawing";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { buildScene, doubleBondSide } from "./drawing.ts";
// @ts-ignore TS5097: see above
import { fallbackHCount } from "./elements.ts";
// @ts-ignore TS5097: see above
import { bondOrderSum, neighbors } from "./model.ts";
// @ts-ignore TS5097: see above
import { drawMetrics, MODEL_BOND } from "./style.ts";

const cm = (pt: number) => Math.round(((pt / 72) * 2.54) * 100) / 100;

/** A figure environment including `file` at the layout's physical width. */
export function latexFigure(l: ExportLayout, file = "scheme.pdf", caption = "Caption."): string {
  return [
    "% needs \\usepackage{graphicx}; upload " + file + " to the project",
    "\\begin{figure}[htbp]",
    "  \\centering",
    `  \\includegraphics[width=${cm(l.widthPt)}cm]{${file}}`,
    `  \\caption{${caption}}`,
    "  \\label{fig:scheme}",
    "\\end{figure}",
  ].join("\n");
}

interface Graph {
  atoms: Map<number, SketchAtom>;
  bondOf: Map<string, SketchBond>;
  nbrs: (id: number) => number[];
}

const key = (a: number, b: number) => (a < b ? `${a}-${b}` : `${b}-${a}`);

function graph(m: SketchModel): Graph {
  const atoms = new Map(m.atoms.map((a) => [a.id, a]));
  const bondOf = new Map(m.bonds.map((b) => [key(b.a, b.b), b]));
  return { atoms, bondOf, nbrs: (id) => neighbors(m, id) };
}

interface Tree {
  order: number[];
  children: Map<number, number[]>;
  /** Ring-closure bonds as [earlier atom, later atom]. */
  closures: [number, number][];
}

/** A depth-first spanning tree from `root` (neighbours in id order) and its ring closures. */
function spanningTree(g: Graph, root: number, seen: Set<number>): Tree {
  const order: number[] = [];
  const children = new Map<number, number[]>();
  const closures: [number, number][] = [];
  const done = new Set<string>();
  const visit = (u: number, parent: number | null) => {
    seen.add(u);
    order.push(u);
    children.set(u, []);
    for (const v of [...g.nbrs(u)].sort((a, b) => a - b)) {
      if (v === parent || done.has(key(u, v))) continue;
      done.add(key(u, v));
      if (seen.has(v)) closures.push([v, u]);
      else {
        children.get(u)?.push(v);
        visit(v, u);
      }
    }
  };
  visit(root, null);
  return { order, children, closures };
}

/** chemfig angle of each charge slot the label placement may choose (labelLayout.ts). */
const SLOT_DEG: Record<ChargeSlot, number> = { ur: 45, ul: 135, lr: -45, ll: -135, u: 90, d: -90, r: 0, l: 180 };

/** The chemfig text of an atom: "" for a bare carbon vertex. The H side and the charge slot
 *  follow the drawing's label placement when given (an H above or below has no chemfig form:
 *  it goes left or right by the bonds); a charge off the upper right is a \charge. */
function atomText(g: Graph, a: SketchAtom, hCount: number, placed?: AtomLabel | null): string {
  const nb = g.nbrs(a.id);
  if (a.el === "C" && a.charge === 0 && nb.length > 0 && a.explicitH !== true) return "";
  const sign = a.charge === 0 ? "" : `${Math.abs(a.charge) > 1 ? Math.abs(a.charge) : ""}${a.charge > 0 ? "+" : "-"}`;
  const h = hCount > 0 ? `H${hCount > 1 ? `_${hCount}` : ""}` : "";
  const byBonds = nb.reduce((s, id) => s + Math.sign((g.atoms.get(id)?.x ?? a.x) - a.x + 1e-9), 0) <= 0;
  const side = placed?.hSide;
  const right = side === "right" ? true : side === "left" ? false : byBonds;
  const slot = placed?.chargeSlot ?? "ur";
  // \charge sits on the symbol's node: only where the H run is not (else the superscript)
  const slotSide = slot === "u" || slot === "d" ? null : slot.endsWith("r") ? "right" : "left";
  const clearOfH = !h || slotSide === null || (slotSide === "right") !== right;
  if (sign && slot !== "ur" && slot !== "r" && clearOfH) {
    const el = `\\charge{${SLOT_DEG[slot]}=$\\scriptstyle ${sign}$}{${a.el}}`;
    return h && !right ? `${h}${el}` : `${el}${h}`;
  }
  const charge = sign ? `^{${sign}}` : "";
  const body = h && !right ? `${h}${a.el}` : `${a.el}${h}`;
  return body + charge;
}

/** The bond code from `from` to `to` (order, stereo, offset side) without the angle. */
function bondCode(m: SketchModel, g: Graph, from: number, to: number): string {
  const b = g.bondOf.get(key(from, to));
  if (!b) return "-";
  if (b.order === 3) return "~";
  if (b.order === 2) {
    const pos = (id: number): Vec => {
      const a = g.atoms.get(id);
      return a ? [a.x, a.y] : [0, 0];
    };
    const side = doubleBondSide(m, b, pos);
    const left = b.a === from ? side : -side;
    return left > 0 ? "=^" : left < 0 ? "=_" : "=";
  }
  if (b.stereo === "wedge") return b.a === from ? "<" : ">";
  if (b.stereo === "hash") return b.a === from ? "<:" : ">:";
  return "-";
}

/** The [:angle(,length)] argument of a bond from `from` to `to`. */
function bondArg(g: Graph, from: number, to: number): string {
  const p = g.atoms.get(from);
  const q = g.atoms.get(to);
  if (!p || !q) return "";
  const dx = q.x - p.x;
  const dy = q.y - p.y;
  let deg = Math.round((Math.atan2(dy, dx) * 180) / Math.PI);
  if (deg <= -180) deg += 360;
  const f = Math.hypot(dx, dy) / MODEL_BOND;
  return Math.abs(f - 1) > 0.05 ? `[:${deg},${Math.round(f * 100) / 100}]` : `[:${deg}]`;
}

const hookName = (i: number) => (i < 26 ? String.fromCharCode(97 + i) : `h${i}`);

/** chemfig code of one connected fragment from `root`. */
function fragment(m: SketchModel, g: Graph, root: number, hOf: (a: SketchAtom) => number, seen: Set<number>, labels: ReadonlyMap<number, AtomLabel | null>): string {
  const t = spanningTree(g, root, seen);
  const hooks = new Map<number, string[]>();
  t.closures.forEach(([early, late], i) => {
    const name = hookName(i);
    const code = bondCode(m, g, early, late);
    const arg = code === "-" ? `?[${name}]` : `?[${name},{${code}}]`;
    hooks.set(early, [...(hooks.get(early) ?? []), `?[${name}]`]);
    hooks.set(late, [...(hooks.get(late) ?? []), arg]);
  });
  const emit = (u: number): string => {
    const a = g.atoms.get(u);
    const text = a ? atomText(g, a, hOf(a), labels.get(u)) : "";
    const kids = t.children.get(u) ?? [];
    const branch = (v: number) => `${bondCode(m, g, u, v)}${bondArg(g, u, v)}${emit(v)}`;
    const side = kids.slice(0, -1).map((v) => `(${branch(v)})`).join("");
    const main = kids.length ? branch(kids[kids.length - 1]) : "";
    const atom = text.length > 1 || text.includes("^") || text.includes("\\") ? `{${text}}` : text;
    return `${atom}${(hooks.get(u) ?? []).join("")}${side}${main}`;
  };
  return `\\chemfig{${emit(root)}}`;
}

/** chemfig options from a drawing style (bp = PostScript point, TeX's 1/72 in). */
export function chemfigOptions(s: DrawingStyle): string {
  const bp = (x: number) => `${Math.round(x * 100) / 100}bp`;
  const gap = (s.bondSpacingPct / 100) * s.bondLengthPt;
  return [
    `atom sep=${bp(s.bondLengthPt)}`,
    `bond offset=${bp(s.marginPt)}`,
    `double bond sep=${bp(gap)}`,
    `cram width=${bp(s.boldWidthPt)}`,
    `cram dash width=${bp(s.hashWidthPt)}`,
    `cram dash sep=${bp(Math.max(0.3, s.hashSpacingPt - s.hashWidthPt))}`,
    `bond style={line width=${bp(s.lineWidthPt)}}`,
  ].join(",");
}

/** The labels as the drawing places them in `style` (H side, charge slot), by atom id. */
function placedLabels(m: SketchModel, hCounts: ReadonlyMap<number, number> | null, style: DrawingStyle): Map<number, AtomLabel | null> {
  const scene = buildScene(m, { s: style.bondLengthPt / MODEL_BOND, ox: 0, oy: 0 }, hCounts, { metrics: { ...drawMetrics(style, 1), metricText: true } });
  return new Map(scene.atoms.map((s) => [s.atom.id, s.label]));
}

/** chemfig code of the whole sketch in `style` (one \chemfig per fragment). */
export function toChemfig(m: SketchModel, hCounts: ReadonlyMap<number, number> | null, style: DrawingStyle): string {
  const g = graph(m);
  const hOf = (a: SketchAtom) => hCounts?.get(a.id) ?? fallbackHCount(a.el, a.charge, bondOrderSum(m, a.id), a.radical ?? 0);
  const seen = new Set<number>();
  const frags: string[] = [];
  const labels = placedLabels(m, hCounts, style);
  for (const a of [...m.atoms].sort((x, y) => x.id - y.id)) if (!seen.has(a.id)) frags.push(fragment(m, g, a.id, hOf, seen, labels));
  const sans = style.fontFamily !== "times";
  const radicals = m.atoms.some((a) => (a.radical ?? 0) > 0);
  return [
    "% needs \\usepackage{chemfig}; generated by the ksmech workbench",
    ...(radicals ? ["% radical dots are not reproduced"] : []),
    "\\begingroup",
    `\\setchemfig{${chemfigOptions(style)}}`,
    ...(sans ? ["\\renewcommand*\\printatom[1]{\\ensuremath{\\mathsf{#1}}}"] : []),
    `\\fontsize{${style.fontSizePt}bp}{${Math.round(style.fontSizePt * 1.2 * 10) / 10}bp}\\selectfont`,
    frags.join("\\quad\n"),
    "\\endgroup",
  ].join("\n");
}
