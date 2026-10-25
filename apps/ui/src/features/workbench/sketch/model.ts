// The sketch model: an immutable atom/bond graph in 2D, with pure operations only (no DOM, no
// RDKit) so that it runs under `node --experimental-strip-types --test`.
//
// Coordinates are molfile units (Angstrom-like, y up); a standard bond is BOND_LENGTH long.
// Atom and bond ids come from one counter (`nextId`) and are never reused within a model's
// history, so a highlight or a selection keeps pointing at the same object across edits. The
// array order of `atoms` and `bonds` is the molfile order (RDKit atom index = position).
//
// `explicitH` is a display flag: the atom is drawn with its label and implicit H count even
// when it is a carbon (ChemDraw's "show element symbol"). It is not written to the molfile.

export type BondOrder = 1 | 2 | 3;
export type BondStereo = "none" | "wedge" | "hash";

export interface SketchAtom {
  readonly id: number;
  readonly el: string;
  readonly x: number;
  readonly y: number;
  readonly charge: number;
  readonly explicitH?: boolean;
}

/** A bond from `a` to `b`; for wedge and hash bonds `a` is the narrow end (the stereocentre). */
export interface SketchBond {
  readonly id: number;
  readonly a: number;
  readonly b: number;
  readonly order: BondOrder;
  readonly stereo: BondStereo;
}

export interface SketchModel {
  readonly atoms: readonly SketchAtom[];
  readonly bonds: readonly SketchBond[];
  readonly nextId: number;
}

export const BOND_LENGTH = 1.5;
/** Two atoms closer than this (in bond lengths) are the same place: new atoms merge. */
export const MERGE_TOLERANCE = 0.25;

export function emptySketch(): SketchModel {
  return { atoms: [], bonds: [], nextId: 1 };
}

export function atomById(m: SketchModel, id: number): SketchAtom | undefined {
  return m.atoms.find((a) => a.id === id);
}

export function bondById(m: SketchModel, id: number): SketchBond | undefined {
  return m.bonds.find((b) => b.id === id);
}

export function findBond(m: SketchModel, a: number, b: number): SketchBond | undefined {
  return m.bonds.find((x) => (x.a === a && x.b === b) || (x.a === b && x.b === a));
}

export function neighbors(m: SketchModel, id: number): number[] {
  const out: number[] = [];
  for (const b of m.bonds) {
    if (b.a === id) out.push(b.b);
    else if (b.b === id) out.push(b.a);
  }
  return out;
}

export function incidentBonds(m: SketchModel, id: number): SketchBond[] {
  return m.bonds.filter((b) => b.a === id || b.b === id);
}

/** Sum of the bond orders at an atom. */
export function bondOrderSum(m: SketchModel, id: number): number {
  return incidentBonds(m, id).reduce((s, b) => s + b.order, 0);
}

export function addAtom(
  m: SketchModel,
  el: string,
  x: number,
  y: number,
  extra: { charge?: number; explicitH?: boolean } = {},
): [SketchModel, number] {
  const id = m.nextId;
  const atom: SketchAtom = { id, el, x, y, charge: extra.charge ?? 0, ...(extra.explicitH ? { explicitH: true } : {}) };
  return [{ ...m, atoms: [...m.atoms, atom], nextId: id + 1 }, id];
}

/** Add a bond a-b; an existing bond between them is returned unchanged (no duplicates). */
export function addBond(
  m: SketchModel,
  a: number,
  b: number,
  order: BondOrder = 1,
  stereo: BondStereo = "none",
): [SketchModel, number] {
  if (a === b) throw new Error(`addBond: self bond on atom ${a}`);
  const existing = findBond(m, a, b);
  if (existing) return [m, existing.id];
  const id = m.nextId;
  return [{ ...m, bonds: [...m.bonds, { id, a, b, order, stereo }], nextId: id + 1 }, id];
}

function mapAtoms(m: SketchModel, ids: Iterable<number>, fn: (a: SketchAtom) => SketchAtom): SketchModel {
  const set = new Set(ids);
  let changed = false;
  const atoms = m.atoms.map((a) => {
    if (!set.has(a.id)) return a;
    const next = fn(a);
    if (next !== a) changed = true;
    return next;
  });
  return changed ? { ...m, atoms } : m;
}

function mapBond(m: SketchModel, id: number, fn: (b: SketchBond) => SketchBond): SketchModel {
  let changed = false;
  const bonds = m.bonds.map((b) => {
    if (b.id !== id) return b;
    const next = fn(b);
    if (next !== b) changed = true;
    return next;
  });
  return changed ? { ...m, bonds } : m;
}

export function setElement(m: SketchModel, ids: Iterable<number>, el: string): SketchModel {
  return mapAtoms(m, ids, (a) => (a.el === el ? a : { ...a, el }));
}

export function setCharge(m: SketchModel, id: number, charge: number): SketchModel {
  return mapAtoms(m, [id], (a) => (a.charge === charge ? a : { ...a, charge }));
}

export function adjustCharge(m: SketchModel, ids: Iterable<number>, delta: number): SketchModel {
  return mapAtoms(m, ids, (a) => ({ ...a, charge: Math.max(-4, Math.min(4, a.charge + delta)) }));
}

export function toggleExplicitH(m: SketchModel, ids: Iterable<number>): SketchModel {
  return mapAtoms(m, ids, (a) => {
    const { explicitH, ...rest } = a;
    return explicitH ? rest : { ...rest, explicitH: true };
  });
}

/** Set a bond's order; a multiple bond loses its wedge/hash. */
export function setBondOrder(m: SketchModel, id: number, order: BondOrder): SketchModel {
  return mapBond(m, id, (b) =>
    b.order === order ? b : { ...b, order, stereo: order === 1 ? b.stereo : "none" },
  );
}

/** 1 -> 2 -> 3 -> 1 (ChemDraw's click on a bond with the bond tool). */
export function cycleBondOrder(m: SketchModel, id: number): SketchModel {
  const b = bondById(m, id);
  if (!b) return m;
  return setBondOrder(m, id, b.order === 3 ? 1 : ((b.order + 1) as BondOrder));
}

/** Make a bond a single wedge or hash bond. Applying the same stereo again flips its direction
 *  (the narrow end moves to the other atom), as in ChemDraw. */
export function setBondStereo(m: SketchModel, id: number, stereo: BondStereo): SketchModel {
  return mapBond(m, id, (b) => {
    if (stereo === "none") return b.stereo === "none" ? b : { ...b, stereo };
    if (b.stereo === stereo && b.order === 1) return { ...b, a: b.b, b: b.a };
    return { ...b, order: 1, stereo };
  });
}

/** Remove atoms and every bond touching them. */
export function removeAtoms(m: SketchModel, ids: Iterable<number>): SketchModel {
  const set = new Set(ids);
  if (set.size === 0) return m;
  const atoms = m.atoms.filter((a) => !set.has(a.id));
  if (atoms.length === m.atoms.length) return m;
  return { ...m, atoms, bonds: m.bonds.filter((b) => !set.has(b.a) && !set.has(b.b)) };
}

/** Remove bonds; an end atom left without bonds goes too when it is a plain carbon (it was only
 *  a vertex of the deleted bond), as ChemDraw's eraser does. */
export function removeBonds(m: SketchModel, ids: Iterable<number>): SketchModel {
  const set = new Set(ids);
  const gone = m.bonds.filter((b) => set.has(b.id));
  if (gone.length === 0) return m;
  const next: SketchModel = { ...m, bonds: m.bonds.filter((b) => !set.has(b.id)) };
  const orphans = new Set<number>();
  for (const b of gone) {
    for (const id of [b.a, b.b]) {
      const atom = atomById(next, id);
      if (atom && isPlainCarbon(atom) && neighbors(next, id).length === 0) orphans.add(id);
    }
  }
  return removeAtoms(next, orphans);
}

export function isPlainCarbon(a: SketchAtom): boolean {
  return a.el === "C" && a.charge === 0 && !a.explicitH;
}

/** Remove a selection: the given atoms (with their bonds) and the given bonds. */
export function removeSelection(m: SketchModel, atomIds: Iterable<number>, bondIds: Iterable<number>): SketchModel {
  return removeAtoms(removeBonds(m, bondIds), atomIds);
}

export function moveAtoms(m: SketchModel, ids: Iterable<number>, dx: number, dy: number): SketchModel {
  if (dx === 0 && dy === 0) return m;
  return mapAtoms(m, ids, (a) => ({ ...a, x: a.x + dx, y: a.y + dy }));
}

/** Replace coordinates by atom id (atoms not in the map keep theirs). */
export function setCoords(m: SketchModel, coords: ReadonlyMap<number, readonly [number, number]>): SketchModel {
  return mapAtoms(m, coords.keys(), (a) => {
    const c = coords.get(a.id);
    return c ? { ...a, x: c[0], y: c[1] } : a;
  });
}

/** Merge atom `drop` into `keep`: drop's bonds move to keep (duplicates and the bond between the
 *  two vanish), then drop is removed. */
export function mergeAtoms(m: SketchModel, keep: number, drop: number): SketchModel {
  if (keep === drop || !atomById(m, keep) || !atomById(m, drop)) return m;
  const bonds: SketchBond[] = [];
  for (const b of m.bonds) {
    const a = b.a === drop ? keep : b.a;
    const c = b.b === drop ? keep : b.b;
    if (a === c) continue;
    if (bonds.some((x) => (x.a === a && x.b === c) || (x.a === c && x.b === a))) continue;
    bonds.push(a === b.a && c === b.b ? b : { ...b, a, b: c });
  }
  return { ...m, atoms: m.atoms.filter((a) => a.id !== drop), bonds };
}

/** The atom closest to (x, y) within `tol` model units, excluding `exclude`. */
export function atomAt(
  m: SketchModel,
  x: number,
  y: number,
  tol = MERGE_TOLERANCE * BOND_LENGTH,
  exclude: ReadonlySet<number> = new Set(),
): SketchAtom | undefined {
  let best: SketchAtom | undefined;
  let bestD = tol;
  for (const a of m.atoms) {
    if (exclude.has(a.id)) continue;
    const d = Math.hypot(a.x - x, a.y - y);
    if (d <= bestD) {
      best = a;
      bestD = d;
    }
  }
  return best;
}

/** The atoms of the smallest ring through bond a-b (at most `maxSize` atoms), in ring order
 *  starting at a, or null when the bond is acyclic. */
export function smallestRingThrough(m: SketchModel, a: number, b: number, maxSize = 8): number[] | null {
  const prev = new Map<number, number>([[b, b]]);
  let frontier = [b];
  for (let depth = 1; depth < maxSize && frontier.length; depth++) {
    const next: number[] = [];
    for (const u of frontier) {
      for (const v of neighbors(m, u)) {
        if (u === b && v === a) continue; // the bond itself
        if (v === a) return pathBack(prev, u).concat(a).reverse();
        if (prev.has(v)) continue;
        prev.set(v, u);
        next.push(v);
      }
    }
    frontier = next;
  }
  return null;
}

function pathBack(prev: Map<number, number>, end: number): number[] {
  const path = [end];
  let cur = end;
  while (prev.get(cur) !== cur) {
    cur = prev.get(cur) as number;
    path.push(cur);
  }
  return path; // end ... b
}

/** Mean bond length (BOND_LENGTH for a model without bonds). */
export function meanBondLength(m: SketchModel): number {
  if (m.bonds.length === 0) return BOND_LENGTH;
  let sum = 0;
  for (const b of m.bonds) {
    const p = atomById(m, b.a);
    const q = atomById(m, b.b);
    if (p && q) sum += Math.hypot(p.x - q.x, p.y - q.y);
  }
  return sum / m.bonds.length || BOND_LENGTH;
}

/** Scale the drawing about its centroid so the mean bond is BOND_LENGTH (imports, clean-up). */
export function normalizeBondLength(m: SketchModel, centre?: readonly [number, number]): SketchModel {
  if (m.atoms.length === 0) return m;
  const k = BOND_LENGTH / meanBondLength(m);
  const [cx, cy] = centroid(m);
  const [tx, ty] = centre ?? [cx, cy];
  return { ...m, atoms: m.atoms.map((a) => ({ ...a, x: tx + (a.x - cx) * k, y: ty + (a.y - cy) * k })) };
}

export function centroid(m: SketchModel): [number, number] {
  if (m.atoms.length === 0) return [0, 0];
  let x = 0;
  let y = 0;
  for (const a of m.atoms) {
    x += a.x;
    y += a.y;
  }
  return [x / m.atoms.length, y / m.atoms.length];
}

/** Translate the drawing so its centroid is at (x, y). */
export function centreAt(m: SketchModel, x: number, y: number): SketchModel {
  const [cx, cy] = centroid(m);
  return moveAtoms(m, m.atoms.map((a) => a.id), x - cx, y - cy);
}
