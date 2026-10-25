// What the canvas draws, in screen pixels: atom labels (ChemDraw-style: carbon is a bare vertex
// unless charged, isolated or flagged explicitH; heteroatoms carry their implicit H count),
// bond strokes (double bonds offset into rings, centred at terminal ends, triple bonds, wedge
// and hash bonds), all as plain data. Pure: SketchCanvas only turns it into SVG elements.
import type { SketchAtom, SketchBond, SketchModel } from "./model";
import type { Vec, View } from "./geometry";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { atomById, bondOrderSum, neighbors, smallestRingThrough } from "./model.ts";
// @ts-ignore TS5097: see above
import { add, len, scale, sideOf, sub, toScreen } from "./geometry.ts";
// @ts-ignore TS5097: see above
import { chargeLabel, fallbackHCount } from "./elements.ts";

export const FONT_PX = 13;
const DOUBLE_GAP_PX = 4.5;
const INNER_SHORTEN = 0.14;
const WEDGE_HALF_PX = 3.6;

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
  /** Filled wedge triangle (wedge bonds). */
  wedge?: Vec[];
}

export interface AtomScene {
  atom: SketchAtom;
  at: Vec;
  label: AtomLabel | null;
}

export interface BondScene {
  bond: SketchBond;
  shape: BondShape;
  /** Full-length centre line (hit target, hover and selection halo). */
  axis: Seg;
}

const H_LEFT_WHEN_ALONE = new Set(["O", "S", "Se", "F", "Cl", "Br", "I"]);

/** The label of an atom, or null for a bare carbon vertex. */
export function atomLabel(m: SketchModel, atom: SketchAtom, hCount: number, view: View): AtomLabel | null {
  const nbrs = neighbors(m, atom.id);
  const shown = atom.el !== "C" || atom.charge !== 0 || atom.explicitH === true || nbrs.length === 0;
  if (!shown) return null;
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

/** Pixels a bond stops short of a labelled atom. */
export function labelTrim(label: AtomLabel | null): number {
  if (!label) return 0;
  return label.el.length > 1 ? FONT_PX * 0.72 : FONT_PX * 0.55;
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
  if (b.order === 1 && b.stereo === "wedge") return { lines: [], wedge: wedgePolygon(main) };
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

/** Everything to draw, in screen pixels. `hCounts` (RDKit's, by atom id) wins over the default
 *  valence fallback. */
export function buildScene(
  m: SketchModel,
  view: View,
  hCounts: ReadonlyMap<number, number> | null,
): { atoms: AtomScene[]; bonds: BondScene[] } {
  const at = new Map<number, Vec>(m.atoms.map((a) => [a.id, toScreen(view, [a.x, a.y])]));
  const atoms: AtomScene[] = m.atoms.map((atom) => {
    const h = hCounts?.get(atom.id) ?? fallbackHCount(atom.el, atom.charge, bondOrderSum(m, atom.id));
    return { atom, at: at.get(atom.id) as Vec, label: atomLabel(m, atom, h, view) };
  });
  const labels = new Map(atoms.map((s) => [s.atom.id, s.label]));
  const pos = (id: number) => at.get(id) ?? ([0, 0] as Vec);
  const bonds: BondScene[] = m.bonds.map((bond) => {
    const p = pos(bond.a);
    const q = pos(bond.b);
    const side = bond.order === 2 ? doubleBondSide(m, bond, pos) : 0;
    const shape = bondShape(p, q, bond, side, labelTrim(labels.get(bond.a) ?? null), labelTrim(labels.get(bond.b) ?? null));
    return { bond, shape, axis: { p, q } };
  });
  return { atoms, bonds };
}
