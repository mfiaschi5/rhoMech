// The Lewis structure B of a pool state, read from its depiction (GET /pool/{state}/depiction):
// atoms by pool atom index (explicit H included), formal charges, bond orders and lone pairs,
// plus the workbench sketch model the 2D views draw. Pure (node --test runs it).
//
// Pool atoms (docs/serve_v9_contract.md): each molblock atom's `pool_atom` is R's index;
// depiction_idx, global_site and map numbers are other index spaces (a product keeps its
// parent's maps in a new atom order, so maps never give pool atoms). The H atoms a depiction
// hides are listed per atom in `hidden_h_atoms`: they are atoms of the structure (bonded to
// their parent, counted in n) and drawn only when asked (`show`: the ones an R entry or an arrow
// touches, or all), placed in the widest free gaps around their parent.
//
// Lone pairs: B_ii = V - q - sum_j B_ij - (implicit H) - (unpaired electrons), in pairs; V is the
// element's valence-electron count (null for elements outside VALENCE: no lone-pair mask).
import type { DepictionOut } from "../../api/types.gen";
import type { SketchAtom, SketchBond, SketchModel } from "../workbench/sketch/model";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { fromMolblock } from "../workbench/sketch/molfile.ts";
// @ts-ignore TS5097: see above
import { BOND_LENGTH } from "../workbench/sketch/model.ts";
import type { BondContext } from "./rmatrix";
import type { ArrowReadingOut, MatrixIO } from "./v9types";

export const VALENCE: Readonly<Record<string, number>> = {
  H: 1, He: 2, Li: 1, Be: 2, B: 3, C: 4, N: 5, O: 6, F: 7, Ne: 8,
  Na: 1, Mg: 2, Al: 3, Si: 4, P: 5, S: 6, Cl: 7, Ar: 8,
  K: 1, Ca: 2, Ga: 3, Ge: 4, As: 5, Se: 6, Br: 7, Kr: 8,
  Rb: 1, Sr: 2, In: 3, Sn: 4, Sb: 5, Te: 6, I: 7, Xe: 8, Cs: 1, Ba: 2,
};

/** A drawn hidden H sits this far from its parent (model units). */
export const HIDDEN_H_BOND = 0.8 * BOND_LENGTH;

export interface PoolAtom {
  /** Pool atom index (R's index). */
  index: number;
  /** The sketch atom id, or null for a hidden H that is not drawn. */
  sketchId: number | null;
  el: string;
  charge: number;
  radical: number;
  /** H on this atom that are no pool atoms (implicit; 0 for a pool state). */
  implicitH: number;
  /** The parent of a hidden H (its pool atom), else null. */
  parent: number | null;
  /** Model coordinates (bond length normalized to the sketch's 1.5); NaN when not drawn. */
  x: number;
  y: number;
}

export interface PoolBond {
  i: number;
  j: number;
  order: number;
  /** The sketch bond id, or null for the bond of an undrawn hidden H. */
  sketchId: number | null;
}

export interface PoolStructure extends BondContext {
  /** Every pool atom, hidden H included. */
  atoms: PoolAtom[];
  /** The atoms of the drawing. */
  drawn: PoolAtom[];
  bonds: PoolBond[];
  /** The drawing as a workbench sketch. */
  model: SketchModel;
  /** Implicit H by sketch atom id (hidden H not drawn), for the sketch canvas labels. */
  hCounts: Map<number, number>;
  atom(i: number): PoolAtom | undefined;
  /** "C3", "O7": element and pool index. */
  label(i: number): string;
}

/** The pool atom index of a depiction atom (molblock position for a server without it). */
export function poolIndexOf(a: { depiction_idx: number; pool_atom?: number }): number {
  return a.pool_atom ?? a.depiction_idx;
}

function lonePairsOf(a: PoolAtom, bondSum: number): number | null {
  const v = VALENCE[a.el];
  if (v === undefined) return null;
  return Math.max(0, Math.floor((v - a.charge - bondSum - a.implicitH - a.radical) / 2));
}

/** The middle of the widest gap between the directions `angles` (radians); up when none. */
function widestGap(angles: number[]): number {
  if (!angles.length) return Math.PI / 2;
  const s = [...angles].sort((x, y) => x - y);
  let best = s[0] + Math.PI;
  let gap = -1;
  s.forEach((a, k) => {
    const b = k + 1 < s.length ? s[k + 1] : s[0] + 2 * Math.PI;
    if (b - a > gap) [gap, best] = [b - a, (a + b) / 2];
  });
  return best;
}

/** Sketch atoms and bonds for the shown hidden H of one parent, in its widest free gaps. */
function placeHydrogens(model: SketchModel, parentId: number, hs: number[], nextId: number): { atoms: SketchAtom[]; bonds: SketchBond[]; ids: Map<number, number> } {
  const p = model.atoms.find((a) => a.id === parentId)!;
  const angles = model.bonds.filter((b) => b.a === parentId || b.b === parentId).map((b) => {
    const q = model.atoms.find((a) => a.id === (b.a === parentId ? b.b : b.a))!;
    return Math.atan2(q.y - p.y, q.x - p.x);
  });
  const out = { atoms: [] as SketchAtom[], bonds: [] as SketchBond[], ids: new Map<number, number>() };
  for (const h of hs) {
    const t = widestGap(angles);
    angles.push(t);
    const id = nextId++;
    out.atoms.push({ id, el: "H", x: p.x + HIDDEN_H_BOND * Math.cos(t), y: p.y + HIDDEN_H_BOND * Math.sin(t), charge: 0 });
    out.bonds.push({ id: nextId++, a: parentId, b: id, order: 1, stereo: "none" });
    out.ids.set(h, id);
  }
  return out;
}

/** The hidden H pool atoms an R or a reading touches. */
export function touchedHydrogens(dep: DepictionOut, matrix: MatrixIO | null, arrows: readonly ArrowReadingOut[] = []): Set<number> {
  const hidden = new Set(dep.atoms.flatMap((a) => a.hidden_h_atoms ?? []));
  const out = new Set<number>();
  const take = (i: number) => hidden.has(i) && out.add(i);
  for (const [i, j] of matrix?.bonds ?? []) [i, j].forEach(take);
  for (const [i] of matrix?.lone ?? []) take(i);
  for (const a of arrows) [...a.source, ...a.target].forEach(take);
  return out;
}

export function poolStructure(dep: DepictionOut, show: ReadonlySet<number> | "all" = new Set()): PoolStructure {
  const base: SketchModel = fromMolblock(dep.molblock, { normalize: true });
  const byPos = new Map(dep.atoms.map((a) => [a.depiction_idx, a] as const));
  const atoms: PoolAtom[] = [];
  const bonds: PoolBond[] = [];
  let model: SketchModel = base;
  const indexOfSketch = new Map<number, number>();
  base.atoms.forEach((s, pos) => {
    const d = byPos.get(pos);
    const index = d ? poolIndexOf(d) : pos;
    const hidden = d?.hidden_h_atoms ?? [];
    indexOfSketch.set(s.id, index);
    atoms.push({ index, sketchId: s.id, el: s.el, charge: s.charge, radical: s.radical ?? 0, implicitH: Math.max(0, (d?.hidden_h ?? 0) - hidden.length), parent: null, x: s.x, y: s.y });
    const drawnHs = hidden.filter((h) => show === "all" || show.has(h));
    const placed = placeHydrogens(model, s.id, drawnHs, model.nextId);
    model = { ...model, atoms: [...model.atoms, ...placed.atoms], bonds: [...model.bonds, ...placed.bonds], nextId: model.nextId + 2 * drawnHs.length };
    for (const h of hidden) {
      const sid = placed.ids.get(h) ?? null;
      const sa = placed.atoms.find((a) => a.id === sid);
      atoms.push({ index: h, sketchId: sid, el: "H", charge: 0, radical: 0, implicitH: 0, parent: index, x: sa?.x ?? Number.NaN, y: sa?.y ?? Number.NaN });
      bonds.push({ i: index, j: h, order: 1, sketchId: sid === null ? null : (placed.bonds.find((b) => b.b === sid)?.id ?? null) });
    }
  });
  for (const b of base.bonds) bonds.push({ i: indexOfSketch.get(b.a)!, j: indexOfSketch.get(b.b)!, order: b.order, sketchId: b.id });
  return assemble(atoms, bonds, model);
}

function assemble(atoms: PoolAtom[], bonds: PoolBond[], model: SketchModel): PoolStructure {
  const byIndex = new Map(atoms.map((a) => [a.index, a] as const));
  const orders = new Map<string, number>();
  const sums = new Map<number, number>();
  for (const b of bonds) {
    orders.set(b.i < b.j ? `${b.i}-${b.j}` : `${b.j}-${b.i}`, b.order);
    sums.set(b.i, (sums.get(b.i) ?? 0) + b.order);
    sums.set(b.j, (sums.get(b.j) ?? 0) + b.order);
  }
  const drawn = atoms.filter((a) => a.sketchId !== null);
  const hCounts = new Map<number, number>();
  for (const a of drawn) {
    const undrawn = atoms.filter((h) => h.parent === a.index && h.sketchId === null).length;
    hCounts.set(a.sketchId!, a.implicitH + undrawn);
  }
  return {
    atoms,
    drawn,
    bonds,
    model,
    n: atoms.length,
    hCounts,
    atom: (i) => byIndex.get(i),
    label: (i) => `${byIndex.get(i)?.el ?? "?"}${i}`,
    bondOrder: (i, j) => orders.get(i < j ? `${i}-${j}` : `${j}-${i}`) ?? 0,
    lonePairs: (i) => {
      const a = byIndex.get(i);
      return a ? lonePairsOf(a, sums.get(i) ?? 0) : null;
    },
  };
}
