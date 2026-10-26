// The two sides of an elementary step as drawable layouts keyed by the SOURCE pool atom (pure;
// node --test runs it), shared by the step viewport variants (StepTextbook, StepAnimated,
// StepOrbital).
//
// Identity across a step: a product state's pool atoms come in a new order (serve contract §1),
// but every atom keeps its map number, so the k-th bracket atom of `product_mapped_smiles` is the
// source pool atom whose bracket atom in `source_mapped_smiles` carries the same map
// (productIndexMap). A side (sideOf) is a drawn PoolStructure read as atoms (element, charge, H
// count, lone pairs, model position) and bonds, renamed into source pool atoms.
//
// Product placement (alignProduct): each product fragment is fitted onto the source positions
// of its atoms by a 2D Procrustes fit (rotation, optionally a reflection, translation; least
// squares), so what does not change stays where it was; fragments that then crowd each other
// are pushed apart along the line between their centres (separate) until `gap` model units
// separate their boxes. The textbook scheme shifts the result to the right of the reactants; the
// animated step morphs into it in place.
import type { Vec } from "./arrowPath";
import type { PoolStructure } from "./poolStructure";

export type XY = Map<number, Vec>;

export interface SideAtom {
  /** Source pool atom index. */
  index: number;
  el: string;
  charge: number;
  /** H on the atom that are not drawn as atoms (implicit plus hidden, undrawn H). */
  hCount: number;
  /** Lone pairs (null: element without a valence count). */
  lone: number | null;
  /** Drawn with its symbol even when a carbon (an atom R touches). */
  labelled: boolean;
  x: number;
  y: number;
}

export interface SideBond {
  i: number;
  j: number;
  order: number;
}

export interface Side {
  atoms: Map<number, SideAtom>;
  bonds: Map<string, SideBond>;
}

export const pairKey = (i: number, j: number): string => (i < j ? `${i}-${j}` : `${j}-${i}`);

/** The map numbers of a mapped SMILES in atom order; null when an atom carries no map (an
 *  organic-subset atom outside brackets, or a bracket atom without ":n"). */
export function mappedAtomMaps(smiles: string): number[] | null {
  const out: number[] = [];
  let k = 0;
  while (k < smiles.length) {
    const c = smiles[k];
    if (c === "[") {
      const end = smiles.indexOf("]", k);
      if (end < 0) return null;
      const m = /:(\d+)$/.exec(smiles.slice(k + 1, end));
      if (!m) return null;
      out.push(Number(m[1]));
      k = end + 1;
    } else if (/[BCNOPSFIbcnops]/.test(c)) {
      return null;
    } else k += 1;
  }
  return out;
}

/** Product pool atom -> source pool atom, by map number; null when either SMILES cannot be read
 *  or a product map has no source atom. */
export function productIndexMap(sourceSmiles: string, productSmiles: string): Map<number, number> | null {
  const src = mappedAtomMaps(sourceSmiles);
  const prod = mappedAtomMaps(productSmiles);
  if (!src || !prod) return null;
  const byMap = new Map(src.map((m, i) => [m, i] as const));
  const out = new Map<number, number>();
  for (const [k, m] of prod.entries()) {
    const i = byMap.get(m);
    if (i === undefined) return null;
    out.set(k, i);
  }
  return out;
}

/** The drawn atoms and bonds of a structure, renamed by `rename` (identity when absent). */
export function sideOf(s: PoolStructure, rename?: ReadonlyMap<number, number> | null): Side {
  const name = (i: number) => rename?.get(i) ?? i;
  const flagged = new Set(s.model.atoms.filter((a) => a.explicitH).map((a) => a.id));
  const atoms = new Map<number, SideAtom>();
  const undrawn = new Map<number, number>();
  for (const h of s.atoms) if (h.parent !== null && h.sketchId === null) undrawn.set(h.parent, (undrawn.get(h.parent) ?? 0) + 1);
  for (const a of s.drawn) {
    const id = a.sketchId!;
    const hCount = a.implicitH + (undrawn.get(a.index) ?? 0);
    atoms.set(name(a.index), { index: name(a.index), el: a.el, charge: a.charge, hCount, lone: s.lonePairs(a.index), labelled: flagged.has(id), x: a.x, y: a.y });
  }
  const bonds = new Map<string, SideBond>();
  for (const b of s.bonds) {
    if (b.sketchId === null) continue;
    const [i, j] = [name(b.i), name(b.j)];
    bonds.set(pairKey(i, j), { i: Math.min(i, j), j: Math.max(i, j), order: b.order });
  }
  return { atoms, bonds };
}

export const xyOf = (side: Side): XY => new Map([...side.atoms.values()].map((a) => [a.index, [a.x, a.y] as Vec]));

/** Connected components of a side (lists of atom indices). */
export function sideFragments(side: Side): number[][] {
  const parent = new Map([...side.atoms.keys()].map((i) => [i, i] as const));
  const find = (x: number): number => {
    while (parent.get(x) !== x) x = parent.get(x)!;
    return x;
  };
  for (const b of side.bonds.values()) if (parent.has(b.i) && parent.has(b.j)) parent.set(find(b.i), find(b.j));
  const groups = new Map<number, number[]>();
  for (const i of side.atoms.keys()) groups.set(find(i), [...(groups.get(find(i)) ?? []), i]);
  return [...groups.values()];
}

export type Transform = (p: Vec) => Vec;

const mean = (ps: readonly Vec[]): Vec => [ps.reduce((s, p) => s + p[0], 0) / ps.length, ps.reduce((s, p) => s + p[1], 0) / ps.length];

/** The rigid motion (rotation, a reflection when it fits better, translation) taking `from`
 *  onto `to` in the least-squares sense; a translation for one point. */
export function procrustes(from: readonly Vec[], to: readonly Vec[]): Transform {
  if (!from.length) return (p) => p;
  const [ca, cb] = [mean(from), mean(to)];
  if (from.length === 1) return (p) => [p[0] - ca[0] + cb[0], p[1] - ca[1] + cb[1]];
  const fit = (flip: number) => {
    let sxx = 0;
    let sxy = 0;
    from.forEach((p, k) => {
      const [ax, ay] = [flip * (p[0] - ca[0]), p[1] - ca[1]];
      const [bx, by] = [to[k][0] - cb[0], to[k][1] - cb[1]];
      sxx += ax * bx + ay * by;
      sxy += ax * by - ay * bx;
    });
    const t = Math.atan2(sxy, sxx);
    const [c, s] = [Math.cos(t), Math.sin(t)];
    const f: Transform = (p) => {
      const [ax, ay] = [flip * (p[0] - ca[0]), p[1] - ca[1]];
      return [c * ax - s * ay + cb[0], s * ax + c * ay + cb[1]];
    };
    const res = from.reduce((r, p, k) => r + (f(p)[0] - to[k][0]) ** 2 + (f(p)[1] - to[k][1]) ** 2, 0);
    return { f, res };
  };
  const [keep, flip] = [fit(1), fit(-1)];
  return flip.res < keep.res - 1e-9 ? flip.f : keep.f;
}

interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

function boxOf(xy: XY, ids: readonly number[], pad: number): Box {
  const ps = ids.map((i) => xy.get(i)).filter((p): p is Vec => !!p);
  return { x0: Math.min(...ps.map((p) => p[0])) - pad, x1: Math.max(...ps.map((p) => p[0])) + pad, y0: Math.min(...ps.map((p) => p[1])) - pad, y1: Math.max(...ps.map((p) => p[1])) + pad };
}

/** Push crowding fragments apart (boxes grown by gap / 2) along the axis of least overlap, the
 *  smaller fragment moving; at most 60 passes. */
export function separate(xy0: XY, frags: readonly number[][], gap: number): XY {
  const xy = new Map(xy0);
  const move = (ids: readonly number[], dx: number, dy: number) => ids.forEach((i) => xy.set(i, [xy.get(i)![0] + dx, xy.get(i)![1] + dy]));
  for (let pass = 0; pass < 60; pass++) {
    let moved = false;
    for (let a = 0; a < frags.length; a++)
      for (let b = a + 1; b < frags.length; b++) {
        const [A, B] = [boxOf(xy, frags[a], gap / 2), boxOf(xy, frags[b], gap / 2)];
        const ox = Math.min(A.x1, B.x1) - Math.max(A.x0, B.x0);
        const oy = Math.min(A.y1, B.y1) - Math.max(A.y0, B.y0);
        if (ox <= 1e-9 || oy <= 1e-9) continue;
        const small = frags[a].length <= frags[b].length ? a : b;
        const [S, O] = small === a ? [A, B] : [B, A];
        const dir = (u0: number, u1: number, v0: number, v1: number) => (u0 + u1 >= v0 + v1 ? 1 : -1);
        if (ox <= oy) move(frags[small], dir(S.x0, S.x1, O.x0, O.x1) * (ox + 1e-6), 0);
        else move(frags[small], 0, dir(S.y0, S.y1, O.y0, O.y1) * (oy + 1e-6));
        moved = true;
      }
    if (!moved) break;
  }
  return xy;
}

/** Fragments side by side (same row: overlapping vertical ranges) moved right until `gap`
 *  model units separate their label extents (`half(i)`: an atom's half-width); left to right,
 *  each move carrying everything to its right. Never moves fragments closer. */
export function spreadFragments(xy0: XY, frags: readonly number[][], half: (i: number) => number, gap: number): XY {
  const xy = new Map(xy0);
  const ext = (f: readonly number[]) => {
    const ps = f.filter((i) => xy.has(i));
    return {
      l: Math.min(...ps.map((i) => xy.get(i)![0] - half(i))),
      r: Math.max(...ps.map((i) => xy.get(i)![0] + half(i))),
      y0: Math.min(...ps.map((i) => xy.get(i)![1])) - 0.5,
      y1: Math.max(...ps.map((i) => xy.get(i)![1])) + 0.5,
    };
  };
  const order = frags.filter((f) => f.some((i) => xy.has(i))).sort((a, b) => ext(a).l + ext(a).r - ext(b).l - ext(b).r);
  for (let k = 1; k < order.length; k++) {
    const b = ext(order[k]);
    let need = 0;
    for (let m = 0; m < k; m++) {
      const a = ext(order[m]);
      if (a.y1 < b.y0 || b.y1 < a.y0) continue;
      need = Math.max(need, a.r + gap - b.l);
    }
    if (need > 1e-9) for (const f of order.slice(k)) for (const i of f) if (xy.has(i)) xy.set(i, [xy.get(i)![0] + need, xy.get(i)![1]]);
  }
  return xy;
}

/** The product positions fitted onto the source (see the header): `prod` and `frags` in source
 *  pool atoms. */
export function alignProduct(src: XY, prod: XY, frags: readonly number[][], gap: number): XY {
  const out: XY = new Map();
  for (const f of frags) {
    const shared = f.filter((i) => src.has(i) && prod.has(i));
    const T = procrustes(shared.map((i) => prod.get(i)!), shared.map((i) => src.get(i)!));
    for (const i of f) if (prod.has(i)) out.set(i, T(prod.get(i)!));
  }
  return separate(out, frags, gap);
}

export interface RelaxOptions {
  /** Standard bond length (model units). */
  bond: number;
  iterations: number;
}

/** The product drawn by moving the reactant drawing as little as it takes (the animated step's
 *  target): from `start`, springs pull every product bond to one bond length, the neighbours of an
 *  atom push apart until they make the angle its neighbour count asks (120° for up to three, 90°
 *  for four or more, 180° at a triple bond or an allene-like centre; a chain atom is pulled to
 *  120° both ways), and atoms that are neither
 *  bonded nor share a neighbour push apart when closer than 1.8 bond lengths. Plain gradient steps, each move
 *  capped at a third of a bond. */
export function relaxLayout(start: XY, side: Side, o: RelaxOptions): XY {
  const L = o.bond;
  const ids = [...start.keys()];
  const nb = new Map<number, number[]>(ids.map((i) => [i, []]));
  const linear = new Set<number>();
  const order = new Map<number, number[]>();
  for (const b of side.bonds.values()) {
    if (!nb.has(b.i) || !nb.has(b.j)) continue;
    nb.get(b.i)!.push(b.j);
    nb.get(b.j)!.push(b.i);
    order.set(b.i, [...(order.get(b.i) ?? []), b.order]);
    order.set(b.j, [...(order.get(b.j) ?? []), b.order]);
  }
  for (const [i, os] of order) if (os.length === 2 && (os.includes(3) || os.every((x) => x === 2))) linear.add(i);
  const springs: [number, number, number][] = [];
  const spread: [number, number, number, boolean][] = [];
  const near = new Set<string>();
  for (const b of side.bonds.values()) if (nb.has(b.i) && nb.has(b.j)) springs.push([b.i, b.j, L]), near.add(pairKey(b.i, b.j));
  for (const [c, ns] of nb) {
    const t = linear.has(c) ? 2 * L : ns.length >= 4 ? Math.SQRT2 * L : Math.sqrt(3) * L;
    // a chain atom (two neighbours) bends to 120° both ways; crowded centres only spread
    const two = ns.length === 2 && !linear.has(c);
    for (let a = 0; a < ns.length; a++) for (let b = a + 1; b < ns.length; b++) spread.push([ns[a], ns[b], t, two]), near.add(pairKey(ns[a], ns[b]));
  }
  // a hair of deterministic jitter, so a straight row can still bend
  const xy = new Map([...start].map(([i, p]) => [i, [p[0], p[1] + (0.004 * (((i * 7919) % 13) - 6)) / 6] as [number, number]]));
  for (let it = 0; it < o.iterations; it++) {
    const f = new Map(ids.map((i) => [i, [0, 0] as [number, number]]));
    const pull = (i: number, j: number, target: number, k: number) => {
      const [p, q] = [xy.get(i)!, xy.get(j)!];
      const [dx, dy] = [q[0] - p[0], q[1] - p[1]];
      const d = Math.hypot(dx, dy) || 1e-6;
      const g = (k * (d - target)) / d;
      f.get(i)![0] += g * dx;
      f.get(i)![1] += g * dy;
      f.get(j)![0] -= g * dx;
      f.get(j)![1] -= g * dy;
    };
    for (const [i, j, t] of springs) pull(i, j, t, 0.8);
    // neighbours of one atom only push apart (opposite ones of a cross may sit farther)
    for (const [i, j, t, two] of spread) if (two || Math.hypot(xy.get(j)![0] - xy.get(i)![0], xy.get(j)![1] - xy.get(i)![1]) < t) pull(i, j, t, 0.35);
    for (let a = 0; a < ids.length; a++)
      for (let b = a + 1; b < ids.length; b++) {
        const [i, j] = [ids[a], ids[b]];
        if (near.has(pairKey(i, j))) continue;
        const [p, q] = [xy.get(i)!, xy.get(j)!];
        if (Math.hypot(q[0] - p[0], q[1] - p[1]) < 1.8 * L) pull(i, j, 1.8 * L, 0.25);
      }
    for (const i of ids) {
      const [fx, fy] = f.get(i)!;
      const m = Math.hypot(fx, fy);
      const k = m > L / 3 ? L / 3 / m : 1;
      xy.get(i)![0] += fx * k * 0.5;
      xy.get(i)![1] += fy * k * 0.5;
    }
  }
  return new Map([...xy].map(([i, p]) => [i, [p[0], p[1]] as Vec]));
}

/** The bounding box of positions (null when empty). */
export function bounds(xys: readonly XY[]): Box | null {
  const ps = xys.flatMap((m) => [...m.values()]);
  if (!ps.length) return null;
  return { x0: Math.min(...ps.map((p) => p[0])), x1: Math.max(...ps.map((p) => p[0])), y0: Math.min(...ps.map((p) => p[1])), y1: Math.max(...ps.map((p) => p[1])) };
}

/** Positions shifted by (dx, dy). */
export const shifted = (xy: XY, dx: number, dy: number): XY => new Map([...xy].map(([i, p]) => [i, [p[0] + dx, p[1] + dy] as Vec]));
