// Drawing operations that need geometry: growing a bond at the best free angle, ring templates
// (on empty canvas, spiro on an atom, fused on a bond) and zigzag chains. Pure; new atoms that
// land on an existing atom (within MERGE_TOLERANCE bond lengths) reuse it.
import type { BondOrder, BondStereo, SketchModel } from "./model";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { BOND_LENGTH, addAtom, addBond, atomAt, atomById, bondById, findBond, neighbors } from "./model.ts";
// @ts-ignore TS5097: see above
import { add, fromAngle, growthAngle, largestGapDirection, polygonAt, polygonFromVertex, polygonOnEdge, sideOf } from "./geometry.ts";
import type { Vec } from "./geometry";

export type RingKind = "benzene" | "cyclohexane" | "cyclopentane";
export const RING_SIZE: Record<RingKind, number> = { benzene: 6, cyclohexane: 6, cyclopentane: 5 };

export type Placement = { kind: "empty"; at: Vec } | { kind: "atom"; id: number } | { kind: "bond"; id: number; at?: Vec };

const pos = (m: SketchModel, id: number): Vec => {
  const a = atomById(m, id);
  if (!a) throw new Error(`no atom ${id}`);
  return [a.x, a.y];
};

/** The atom at p if one is there, else a new atom of element `el` at p. */
export function atomAtOrNew(m: SketchModel, p: Vec, el = "C"): [SketchModel, number] {
  const hit = atomAt(m, p[0], p[1]);
  return hit ? [m, hit.id] : addAtom(m, el, p[0], p[1]);
}

/** Where a bond grown from atom `id` would end. */
export function growthEnd(m: SketchModel, id: number, length = BOND_LENGTH): Vec {
  const c = pos(m, id);
  const nbrs = neighbors(m, id).map((n) => pos(m, n));
  const others = m.atoms.filter((a) => a.id !== id).map((a): Vec => [a.x, a.y]);
  return add(c, fromAngle(growthAngle(c, nbrs, others, length), length));
}

export interface Grown {
  model: SketchModel;
  atom: number;
  bond: number;
}

/** Grow a bond from atom `id` at the best free angle; the far end joins an atom already there.
 *  Growing onto an atom that is already bonded only changes nothing. */
export function growBond(
  m: SketchModel,
  id: number,
  order: BondOrder = 1,
  stereo: BondStereo = "none",
  el = "C",
): Grown {
  const [m1, end] = atomAtOrNew(m, growthEnd(m, id), el);
  if (end === id) return { model: m, atom: id, bond: -1 };
  const [m2, bond] = addBond(m1, id, end, order, stereo);
  return { model: m2, atom: end, bond };
}

/** A new bond on empty canvas: from p at 30 degrees, one standard bond long. */
export function bondFromPoint(m: SketchModel, p: Vec, order: BondOrder = 1, stereo: BondStereo = "none"): Grown {
  const [m1, a] = atomAtOrNew(m, p);
  return growBond(m1, a, order, stereo);
}

/** A bond from atom `from` (or a new atom at a point) to `to`, joining existing atoms. */
export function bondBetweenPoints(
  m: SketchModel,
  from: number | Vec,
  to: Vec,
  order: BondOrder = 1,
  stereo: BondStereo = "none",
): Grown {
  const [m1, a] = typeof from === "number" ? [m, from] : atomAtOrNew(m, from);
  const [m2, b] = atomAtOrNew(m1, to);
  if (a === b) return { model: m, atom: a, bond: -1 };
  const [m3, bond] = addBond(m2, a, b, order, stereo);
  return { model: m3, atom: b, bond };
}

/** The ring vertices for a placement; `fixed` gives the atoms that vertices 0 (and 1) already
 *  are. */
export function ringVertices(m: SketchModel, n: number, at: Placement): { pts: Vec[]; fixed: number[] } {
  if (at.kind === "empty") return { pts: polygonAt(at.at, n, BOND_LENGTH), fixed: [] };
  if (at.kind === "atom") {
    const p = pos(m, at.id);
    const nbrs = neighbors(m, at.id).map((x) => pos(m, x));
    return { pts: polygonFromVertex(p, largestGapDirection(p, nbrs), n, BOND_LENGTH), fixed: [at.id] };
  }
  const b = bondById(m, at.id);
  if (!b) throw new Error(`no bond ${at.id}`);
  const p = pos(m, b.a);
  const q = pos(m, b.b);
  return { pts: polygonOnEdge(p, q, n, fusedSide(m, b.a, b.b, at.at)), fixed: [b.a, b.b] };
}

/** The side of bond a-b for a fused ring: away from the substituents already on the bond, or
 *  the clicked side when both sides are equally busy. */
function fusedSide(m: SketchModel, a: number, b: number, click?: Vec): 1 | -1 {
  const p = pos(m, a);
  const q = pos(m, b);
  let busy = 0;
  for (const id of [a, b]) {
    for (const n of neighbors(m, id)) if (n !== a && n !== b) busy += sideOf(p, q, pos(m, n));
  }
  if (busy !== 0) return busy > 0 ? -1 : 1;
  return click && sideOf(p, q, click) === -1 ? -1 : 1;
}

/** Add a ring template. Returns the model and the ring atoms in ring order. */
export function addRing(m: SketchModel, kind: RingKind, at: Placement): { model: SketchModel; atoms: number[] } {
  const n = RING_SIZE[kind];
  const { pts, fixed } = ringVertices(m, n, at);
  let model = m;
  const ids: number[] = [];
  pts.forEach((p, k) => {
    if (k < fixed.length) ids.push(fixed[k]);
    else {
      const [next, id] = atomAtOrNew(model, p);
      model = next;
      ids.push(id);
    }
  });
  const before = model;
  for (let k = 0; k < n; k++) {
    const u = ids[k];
    const v = ids[(k + 1) % n];
    if (u !== v) model = addBond(model, u, v)[0];
  }
  if (kind === "benzene") model = kekulizeRing(model, before, ids);
  return { model, atoms: ids };
}

/** Put alternating double bonds on a new benzene ring: of the two alternation patterns, the one
 *  that can place the most doubles without giving an atom a second multiple bond (an existing
 *  shared bond keeps its order). `before` is the model before the ring bonds were added. */
function kekulizeRing(m: SketchModel, before: SketchModel, ring: number[]): SketchModel {
  const n = ring.length;
  const edge = (k: number) => [ring[k], ring[(k + 1) % n]] as const;
  const busy = (id: number) => before.bonds.some((b) => (b.a === id || b.b === id) && b.order > 1);
  const plan = (start: number) => {
    const doubles: number[] = [];
    let score = 0;
    for (let k = start; k < n; k += 2) {
      const [u, v] = edge(k);
      const old = findBond(before, u, v);
      if (old) score += old.order === 2 ? 1 : 0;
      else if (!busy(u) && !busy(v)) {
        doubles.push(k);
        score += 1;
      }
    }
    return { doubles, score };
  };
  const a = plan(1);
  const b = plan(0);
  const best = b.score > a.score ? b : a;
  let model = m;
  for (const k of best.doubles) {
    const [u, v] = edge(k);
    const bond = findBond(model, u, v);
    if (bond) model = { ...model, bonds: model.bonds.map((x) => (x.id === bond.id ? { ...x, order: 2 } : x)) };
  }
  return model;
}

/** Add a zigzag chain of `pts` from atom `from` (or a new atom at a point). */
export function addChain(m: SketchModel, from: number | Vec, pts: readonly Vec[]): { model: SketchModel; atoms: number[] } {
  let [model, prev] = typeof from === "number" ? [m, from] : atomAtOrNew(m, from);
  const atoms = [prev];
  for (const p of pts) {
    const [next, id] = atomAtOrNew(model, p);
    model = id === prev ? next : addBond(next, prev, id)[0];
    prev = id;
    atoms.push(id);
  }
  return { model, atoms };
}
