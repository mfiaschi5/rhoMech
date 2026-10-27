// Collision-aware placement of atom labels (drawing.buildScene runs it on every scene, so the
// canvas, the SVG / PDF / PNG export and the overlap report agree), in output units, y down.
//
// Each labelled atom starts from ChemDraw's choice (drawing.atomLabel: the H count away from
// the bonds, the charge in the first free slot). When that placement collides with another
// label, with a bond, or with itself (an H-count subscript under a charge), the alternatives
// are scored by their overlap (ink boxes of labelGlyphs.ts, real glyph metrics at the style's
// size, inflated by half the style's margin) plus a small preference cost:
//   1. the H side (left, right, above, below) and the charge slot (ur, ul, lr, ll);
//   2. then the subscript and the charge shrunk to 85 % and 72 % (never below the style
//      editor's 40 % of the font);
//   3. as a last resort, for a terminal or isolated atom only (bonds must still meet at the
//      symbol), a render-only displacement of the whole label (up to 0.9 font) with a short
//      leader line from the atom point.
// Whatever still overlaps is reported (Scene.issues; the canvas shows the count as
// data-label-issues). A few greedy passes settle neighbouring labels against each other.
//
// Bond ends are trimmed against the final ink boxes (each line of a double / triple bond and
// the edges of a wedge separately, plus the margin), so a bond never runs into a label.
//
// sceneOverlaps(scene) is the box metric of the tests: label-label (two atoms' ink boxes),
// label-self (non-adjacent runs of one label) and label-bond (a drawn stroke or wedge in an ink
// box) overlaps.
//
// Pure: runs under node --experimental-strip-types.
import type { Vec } from "./geometry";
import type { AtomLabel, AtomScene, ChargeSlot, HSide, Scene, Seg } from "./drawing";
import type { DrawMetrics } from "./style";
import type { Box, Glyph } from "./labelGlyphs";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { glyphInk, labelGlyphs } from "./labelGlyphs.ts";

/** A bond as the placement sees it: its atoms, its axis (ends trimmed at the symbols) and the
 *  half-width of what it draws. */
export interface PlacementBond {
  a: number;
  b: number;
  p: Vec;
  q: Vec;
  half: number;
}

export interface LabelIssue {
  atom: number;
  kind: "label-label" | "label-bond" | "label-self";
  /** Overlap in font units (area / font^2 or length / font). */
  amount: number;
}

export interface Placement {
  labels: Map<number, AtomLabel>;
  leaders: Map<number, Seg>;
  issues: LabelIssue[];
}

const H_SIDES: HSide[] = ["right", "left", "above", "below"];
const SLOTS: ChargeSlot[] = ["ur", "ul", "lr", "ll", "u", "d", "r", "l"];
const SCALES = [1, 0.85, 0.72];
/** An overlap worth this many preference points per font^2 (any overlap beats any preference). */
const OVERLAP_WEIGHT = 1e4;
const EPS = 1e-4;

export const inflate = (b: Box, d: number): Box => ({ x0: b.x0 - d, y0: b.y0 - d, x1: b.x1 + d, y1: b.y1 + d });
export const shiftBox = (b: Box, v: Vec): Box => ({ x0: b.x0 + v[0], y0: b.y0 + v[1], x1: b.x1 + v[0], y1: b.y1 + v[1] });

export function overlapArea(a: Box, b: Box): number {
  const w = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0);
  const h = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
  return w > 0 && h > 0 ? w * h : 0;
}

/** The parameter interval (arclength from p) of segment p-q inside `box` (Liang-Barsky), or
 *  null when it misses. */
export function segInBox(p: Vec, q: Vec, box: Box): [number, number] | null {
  const d: Vec = [q[0] - p[0], q[1] - p[1]];
  const l = Math.hypot(d[0], d[1]);
  let t0 = 0;
  let t1 = 1;
  const edges: [number, number][] = [
    [-d[0], p[0] - box.x0],
    [d[0], box.x1 - p[0]],
    [-d[1], p[1] - box.y0],
    [d[1], box.y1 - p[1]],
  ];
  for (const [pp, qq] of edges) {
    if (Math.abs(pp) < 1e-12) {
      if (qq < 0) return null;
      continue;
    }
    const r = qq / pp;
    if (pp < 0) t0 = Math.max(t0, r);
    else t1 = Math.min(t1, r);
    if (t0 > t1) return null;
  }
  return [t0 * l, t1 * l];
}

export const segBoxLength = (p: Vec, q: Vec, box: Box): number => {
  const iv = segInBox(p, q, box);
  return iv ? Math.max(0, iv[1] - iv[0]) : 0;
};

/** Glyph pairs of one label that may touch: the symbol and its H, the H and its count. */
const adjacent = (a: Glyph, b: Glyph) => {
  const k = [a.kind, b.kind].sort().join();
  return k === "h,symbol" || k === "count,h";
};

/** The candidate placements of a label with their preference costs (the default first). */
export function candidates(label: AtomLabel, M: DrawMetrics): { label: AtomLabel; pref: number }[] {
  const sides = label.hCount > 0 ? [label.hSide, ...H_SIDES.filter((s) => s !== label.hSide)] : [label.hSide];
  const slots = label.charge ? [label.chargeSlot ?? "ur", ...SLOTS.filter((s) => s !== (label.chargeSlot ?? "ur"))] : [undefined];
  const shrinkable = label.hCount > 1 || !!label.charge;
  const minScale = Math.max(0.4 * M.font / Math.max(M.sub, 1e-9), label.charge ? (0.4 * M.font) / Math.max(M.charge, 1e-9) : 0);
  const scales = shrinkable ? SCALES.filter((k) => k === 1 || k >= minScale) : [1];
  const out: { label: AtomLabel; pref: number }[] = [];
  scales.forEach((k, ki) =>
    sides.forEach((hSide, si) =>
      slots.forEach((chargeSlot, ci) => {
        const slotPref = SLOTS.indexOf(chargeSlot ?? "ur") < 4 ? ci * 0.4 : 1.6 + ci * 0.1;
        const pref = si * (si > 1 ? 1.5 : 1) + (chargeSlot ? slotPref : 0) + ki * 2.5;
        out.push({ label: { ...label, hSide, ...(chargeSlot ? { chargeSlot } : {}), ...(k !== 1 ? { scale: k } : {}) }, pref });
      }),
    ),
  );
  return out;
}

interface Item {
  id: number;
  at: Vec;
  label: AtomLabel;
  /** Absolute ink boxes of the current placement, with their glyphs. */
  glyphs: Glyph[];
  boxes: Box[];
  near: number[];
  bonds: PlacementBond[];
  /** Bonds at the atom: only a terminal (or isolated) atom's label may be displaced. */
  degree: number;
}

function inkOf(label: AtomLabel, at: Vec, M: DrawMetrics): { glyphs: Glyph[]; boxes: Box[] } {
  const glyphs = labelGlyphs(label, M);
  return { glyphs, boxes: glyphs.map((g) => shiftBox(glyphInk(g), at)) };
}

/** The clearance a label keeps from another atom's label: the style's margin, at least a fifth
 *  of the font; a charge sign keeps 0.3 font (a sign next to another symbol reads as its). */
export const labelGap = (M: DrawMetrics, charge = false): number => Math.max(M.margin, (charge ? 0.3 : 0.2) * M.font);
/** The clearance a label keeps from a bond's stroke. */
export const bondGap = (M: DrawMetrics): number => M.margin * 0.75;

interface Overlap {
  self: number;
  label: number;
  bond: number;
}

const total = (o: Overlap) => o.self + o.label + o.bond;

/** The overlaps of `item` placed as (glyphs, boxes) against the other items' current boxes and
 *  the bonds near it, in font units. */
function overlapOf(item: Item, ink: { glyphs: Glyph[]; boxes: Box[] }, items: Item[], M: DrawMetrics): Overlap {
  const f2 = M.font * M.font;
  const o: Overlap = { self: 0, label: 0, bond: 0 };
  const { glyphs, boxes } = ink;
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) if (!adjacent(glyphs[i], glyphs[j])) o.self += overlapArea(boxes[i], boxes[j]) / f2;
    const gap = labelGap(M, glyphs[i].kind === "charge");
    for (const k of item.near) for (const b of items[k].boxes) o.label += overlapArea(inflate(boxes[i], gap), b) / f2;
    for (const bd of item.bonds) {
      const incident = bd.a === item.id || bd.b === item.id;
      if (incident && glyphs[i].kind === "symbol" && !glyphs[i].displaced) continue;
      o.bond += segBoxLength(bd.p, bd.q, inflate(boxes[i], bondGap(M) + bd.half)) / M.font;
    }
  }
  return o;
}

const near = (a: Box, b: Box, r: number) => a.x0 - r < b.x1 && b.x0 - r < a.x1 && a.y0 - r < b.y1 && b.y0 - r < a.y1;

const unionBox = (bs: Box[]): Box => bs.reduce((u, b) => ({ x0: Math.min(u.x0, b.x0), y0: Math.min(u.y0, b.y0), x1: Math.max(u.x1, b.x1), y1: Math.max(u.y1, b.y1) }));

/** The items with their neighbourhoods (labels and bonds within reach of any placement). */
function makeItems(atoms: readonly AtomScene[], bonds: readonly PlacementBond[], M: DrawMetrics): Item[] {
  const items: Item[] = atoms.filter((s) => s.label).map((s) => ({ id: s.atom.id, at: s.at, label: s.label as AtomLabel, ...inkOf(s.label as AtomLabel, s.at, M), near: [], bonds: [], degree: bonds.filter((b) => b.a === s.atom.id || b.b === s.atom.id).length }));
  const reach = M.font * 2.6;
  const extent = items.map((it) => unionBox(it.boxes));
  items.forEach((it, i) => {
    it.near = items.map((_, j) => j).filter((j) => j !== i && near(extent[i], extent[j], 2 * reach));
    it.bonds = bonds.filter((b) => near(extent[i], unionBox([{ x0: b.p[0], y0: b.p[1], x1: b.p[0], y1: b.p[1] }, { x0: b.q[0], y0: b.q[1], x1: b.q[0], y1: b.q[1] }]), reach + b.half));
  });
  return items;
}

/** The best (lowest overlap, then preference) candidate of one item. */
function bestCandidate(item: Item, items: Item[], M: DrawMetrics, base: AtomLabel): { label: AtomLabel; ov: Overlap; score: number } {
  let best = { label: item.label, ov: overlapOf(item, item, items, M), score: Infinity };
  for (const c of candidates(base, M)) {
    const ov = overlapOf(item, inkOf(c.label, item.at, M), items, M);
    const score = total(ov) * OVERLAP_WEIGHT + c.pref;
    if (score < best.score - 1e-9) best = { label: c.label, ov, score };
  }
  return best;
}

/** A displaced copy of the best placement, when one clears every overlap. */
function displaced(item: Item, items: Item[], M: DrawMetrics, from: AtomLabel): { label: AtomLabel; ov: Overlap } | null {
  let best: { label: AtomLabel; ov: Overlap; score: number } | null = null;
  for (const mag of [0.3, 0.6, 0.9]) {
    for (let k = 0; k < 16; k++) {
      const a = (k / 16) * 2 * Math.PI;
      const offset: Vec = [Math.cos(a) * mag * M.font, Math.sin(a) * mag * M.font];
      const label = { ...from, offset };
      const ov = overlapOf(item, inkOf(label, item.at, M), items, M);
      const score = total(ov) * OVERLAP_WEIGHT + mag;
      if (total(ov) < EPS && (!best || score < best.score)) best = { label, ov, score };
    }
    if (best) return best;
  }
  return null;
}

function setItem(item: Item, label: AtomLabel, M: DrawMetrics): void {
  item.label = label;
  Object.assign(item, inkOf(label, item.at, M));
}

/** The leader from the atom point to the displaced symbol's box (null when short). */
function leaderOf(item: Item, M: DrawMetrics): Seg | null {
  const off = item.label.offset;
  if (!off) return null;
  const sym = inflate(item.boxes[0], M.margin);
  const iv = segInBox(item.at, [item.at[0] + off[0], item.at[1] + off[1]], sym);
  const l = Math.hypot(off[0], off[1]);
  const stop = iv ? iv[0] : l;
  if (stop < M.line * 2) return null;
  return { p: item.at, q: [item.at[0] + (off[0] / l) * stop, item.at[1] + (off[1] / l) * stop] };
}

/** Place every label of `atoms` (see the header); `bonds` are the scene's bonds. */
export function placeLabels(atoms: readonly AtomScene[], bonds: readonly PlacementBond[], M: DrawMetrics): Placement {
  const items = makeItems(atoms, bonds, M);
  const base = new Map(items.map((it) => [it.id, it.label]));
  for (let pass = 0; pass < 3; pass++) {
    let changed = false;
    for (const it of items) {
      if (total(overlapOf(it, it, items, M)) < EPS && it.label === base.get(it.id)) continue;
      const best = bestCandidate(it, items, M, base.get(it.id) as AtomLabel);
      if (best.label !== it.label) {
        setItem(it, best.label, M);
        changed = true;
      }
    }
    if (!changed) break;
  }
  return finish(items, M);
}

/** Displace what still overlaps, then the leaders and the issues. */
function finish(items: Item[], M: DrawMetrics): Placement {
  const issues: LabelIssue[] = [];
  const leaders = new Map<number, Seg>();
  for (const it of items) {
    let ov = overlapOf(it, it, items, M);
    if (it.degree <= 1 && (ov.label > EPS || ov.bond > EPS)) {
      const d = displaced(it, items, M, it.label);
      if (d) {
        setItem(it, d.label, M);
        ov = d.ov;
      }
    }
    const lead = leaderOf(it, M);
    if (lead) leaders.set(it.id, lead);
    if (ov.label >= EPS) issues.push({ atom: it.id, kind: "label-label", amount: ov.label });
    if (ov.bond >= EPS) issues.push({ atom: it.id, kind: "label-bond", amount: ov.bond });
    if (ov.self >= EPS) issues.push({ atom: it.id, kind: "label-self", amount: ov.self });
  }
  return { labels: new Map(items.map((it) => [it.id, it.label])), leaders, issues };
}

/** The interval of p-q (arclength) within `r` of `box` (the box grown by a margin with round
 *  corners, as a margin around the glyph outline is), or null. */
export function segInRoundedBox(p: Vec, q: Vec, box: Box, r: number): [number, number] | null {
  const parts: ([number, number] | null)[] = [
    segInBox(p, q, { x0: box.x0 - r, x1: box.x1 + r, y0: box.y0, y1: box.y1 }),
    segInBox(p, q, { x0: box.x0, x1: box.x1, y0: box.y0 - r, y1: box.y1 + r }),
  ];
  for (const c of [[box.x0, box.y0], [box.x1, box.y0], [box.x0, box.y1], [box.x1, box.y1]] as Vec[]) parts.push(segInCircle(p, q, c, r));
  const hit = parts.filter((x): x is [number, number] => !!x);
  return hit.length ? [Math.min(...hit.map((x) => x[0])), Math.max(...hit.map((x) => x[1]))] : null;
}

function segInCircle(p: Vec, q: Vec, c: Vec, r: number): [number, number] | null {
  const d: Vec = [q[0] - p[0], q[1] - p[1]];
  const l = Math.hypot(d[0], d[1]);
  if (l < 1e-12 || r <= 0) return null;
  const u: Vec = [d[0] / l, d[1] / l];
  const w: Vec = [c[0] - p[0], c[1] - p[1]];
  const t = w[0] * u[0] + w[1] * u[1];
  const h2 = r * r - (w[0] * w[0] + w[1] * w[1] - t * t);
  if (h2 <= 0) return null;
  const h = Math.sqrt(h2);
  const t0 = Math.max(0, t - h);
  const t1 = Math.min(l, t + h);
  return t0 < t1 ? [t0, t1] : null;
}

/** How far the end of a bond at `at` (towards `other`) stops short: past every ink box of the
 *  atom's label (inflated by the margin) that one of its lines crosses in its first half, the
 *  lines being the axis shifted by each of `laterals`. 0 without boxes. */
export function endTrim(at: Vec, other: Vec, boxes: readonly Box[], laterals: readonly number[], margin: number): number {
  if (boxes.length === 0) return 0;
  const dx = other[0] - at[0];
  const dy = other[1] - at[1];
  const l = Math.hypot(dx, dy) || 1;
  const n: Vec = [-dy / l, dx / l];
  let trim = 0;
  for (const k of laterals) {
    const p: Vec = [at[0] + n[0] * k, at[1] + n[1] * k];
    const q: Vec = [other[0] + n[0] * k, other[1] + n[1] * k];
    for (const b of boxes) {
      const iv = segInRoundedBox(p, q, b, margin);
      if (iv && iv[0] <= l / 2) trim = Math.max(trim, iv[1]);
    }
  }
  return trim;
}

/** The absolute ink boxes of an atom's label (empty for a bare vertex). */
export function atomInk(s: AtomScene, M: DrawMetrics): Box[] {
  return s.label ? labelGlyphs(s.label, M).map((g) => shiftBox(glyphInk(g), s.at)) : [];
}

export interface Overlaps {
  labelLabel: { a: number; b: number; area: number }[];
  labelSelf: { atom: number; area: number }[];
  labelBond: { atom: number; bond: number; length: number }[];
}

/** The strokes of a drawn bond as segments with their half-widths (a wedge as its outline). */
function strokes(b: Scene["bonds"][number], M: DrawMetrics): { p: Vec; q: Vec; half: number }[] {
  const w = b.bond.order === 1 && b.bond.stereo === "hash" ? M.hashWidth : M.line;
  const out = b.shape.lines.map((l) => ({ p: l.p, q: l.q, half: w / 2 }));
  for (const poly of b.shape.wedges ?? []) poly.forEach((v, i) => out.push({ p: v, q: poly[(i + 1) % poly.length], half: 0 }));
  return out;
}

/** The box metric of the tests: every overlap between labels, within a label and between a
 *  label and a drawn bond (strokes with their width; any positive length counts). */
export function sceneOverlaps(scene: Scene): Overlaps {
  const M = scene.metrics;
  const ink = scene.atoms.map((s) => ({ id: s.atom.id, glyphs: s.label ? labelGlyphs(s.label, M) : [], boxes: atomInk(s, M) }));
  const out: Overlaps = { labelLabel: [], labelSelf: [], labelBond: [] };
  ink.forEach((a, i) => {
    for (let x = 0; x < a.boxes.length; x++) {
      for (let y = x + 1; y < a.boxes.length; y++) {
        const area = adjacent(a.glyphs[x], a.glyphs[y]) ? 0 : overlapArea(a.boxes[x], a.boxes[y]);
        if (area > 1e-6) out.labelSelf.push({ atom: a.id, area });
      }
    }
    for (const b of ink.slice(i + 1)) {
      const area = a.boxes.reduce((s, p) => s + b.boxes.reduce((t, q) => t + overlapArea(p, q), 0), 0);
      if (area > 1e-6) out.labelLabel.push({ a: a.id, b: b.id, area });
    }
    for (const bs of scene.bonds) {
      const length = strokes(bs, M).reduce((s, st) => s + a.boxes.reduce((t, bx) => t + segBoxLength(st.p, st.q, inflate(bx, st.half)), 0), 0);
      if (length > 1e-6) out.labelBond.push({ atom: a.id, bond: bs.bond.id, length });
    }
  });
  return out;
}
