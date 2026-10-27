// Reaction matrices R = ΔBE on the pool atoms (spec "Curved arrows as bond--electron matrix
// edits"), pure: no React, no DOM, so `node --experimental-strip-types --test` runs it.
//
// A matrix is the contract's MatrixIO, kept canonical by normalize(): bonds [i, j, delta] with
// i < j, lone [i, delta], no zero entries, sorted. Entries are summed when arrows are added, so a
// value can leave the contract's range (bonds ±1, lone ±2); localIssues() reports that.
//
// Arrow -> R (the spec's three elementary matrices):
//   lone pair on a -> bond (a, b):           R_aa -2, R_ab +1
//   bond (a, c) -> bond (a, b), new at a:    R_ac -1, R_ab +1
//   bond (a, c) -> lone pair on k (k in a,c): R_ac -1, R_kk +2
// arrowFromGesture() turns a drag (atom or bond -> atom or bond) into one of these readings.
//
// Formal charges follow from R: q'_i = q_i - R_ii - sum_j R_ij (spec, the BE matrix).
import type { ArrowReadingOut, MatrixIO } from "../v9types";

export type Entry = { kind: "bond"; i: number; j: number; delta: number } | { kind: "lone"; i: number; delta: number };

export const EMPTY_MATRIX: MatrixIO = { bonds: [], lone: [] };

/** The structure facts R is checked against (PoolStructure satisfies it). */
export interface BondContext {
  /** Formal bond order of (i, j) in the reactant B; 0 when not bonded. */
  bondOrder(i: number, j: number): number;
  /** Lone pairs on atom i in B; null when the element's valence is unknown (no mask). */
  lonePairs(i: number): number | null;
  /** The number of pool atoms. */
  readonly n: number;
}

export const MAX_BOND_ORDER = 3;

export const pairKey = (i: number, j: number): string => (i < j ? `${i}-${j}` : `${j}-${i}`);

export function entryKey(e: Entry): string {
  return e.kind === "bond" ? `b:${pairKey(e.i, e.j)}` : `l:${e.i}`;
}

/** Merge duplicates (summing), order every pair i < j, drop zeros, sort. */
export function normalize(m: MatrixIO): MatrixIO {
  const bonds = new Map<string, [number, number, number]>();
  for (const [a, b, d] of m.bonds) {
    if (a === b) continue;
    const [i, j] = a < b ? [a, b] : [b, a];
    const k = pairKey(i, j);
    const prev = bonds.get(k);
    bonds.set(k, [i, j, (prev?.[2] ?? 0) + d]);
  }
  const lone = new Map<number, number>();
  for (const [i, d] of m.lone) lone.set(i, (lone.get(i) ?? 0) + d);
  return {
    bonds: [...bonds.values()].filter((e) => e[2] !== 0).sort((x, y) => x[0] - y[0] || x[1] - y[1]),
    lone: [...lone.entries()].filter(([, d]) => d !== 0).sort((x, y) => x[0] - y[0]),
  };
}

/** a + sign * b, normalized. */
export function addMatrix(a: MatrixIO, b: MatrixIO, sign: 1 | -1 = 1): MatrixIO {
  return normalize({
    bonds: [...a.bonds, ...b.bonds.map(([i, j, d]) => [i, j, sign * d] as [number, number, number])],
    lone: [...a.lone, ...b.lone.map(([i, d]) => [i, sign * d] as [number, number])],
  });
}

export function isEmptyMatrix(m: MatrixIO): boolean {
  return m.bonds.length === 0 && m.lone.length === 0;
}

export function sameMatrix(a: MatrixIO, b: MatrixIO): boolean {
  return JSON.stringify(normalize(a)) === JSON.stringify(normalize(b));
}

export function bondDelta(m: MatrixIO, i: number, j: number): number {
  const [a, b] = i < j ? [i, j] : [j, i];
  return m.bonds.find((e) => e[0] === a && e[1] === b)?.[2] ?? 0;
}

export function loneDelta(m: MatrixIO, i: number): number {
  return m.lone.find((e) => e[0] === i)?.[1] ?? 0;
}

export function entriesOf(m: MatrixIO): Entry[] {
  return [
    ...m.bonds.map(([i, j, delta]): Entry => ({ kind: "bond", i, j, delta })),
    ...m.lone.map(([i, delta]): Entry => ({ kind: "lone", i, delta })),
  ];
}

/** The matrix with one entry set to `value` (0 removes it). */
export function setEntry(m: MatrixIO, e: Entry, value: number): MatrixIO {
  if (e.kind === "bond") {
    const [i, j] = e.i < e.j ? [e.i, e.j] : [e.j, e.i];
    const bonds = m.bonds.filter((b) => !(b[0] === i && b[1] === j));
    return normalize({ bonds: [...bonds, [i, j, value]], lone: m.lone });
  }
  return normalize({ bonds: m.bonds, lone: [...m.lone.filter((l) => l[0] !== e.i), [e.i, value]] });
}

export function removeEntries(m: MatrixIO, keys: ReadonlySet<string>): MatrixIO {
  return normalize({
    bonds: m.bonds.filter(([i, j]) => !keys.has(`b:${pairKey(i, j)}`)),
    lone: m.lone.filter(([i]) => !keys.has(`l:${i}`)),
  });
}

/** Every atom an entry of R touches, sorted. */
export function matrixAtoms(m: MatrixIO): number[] {
  const s = new Set<number>();
  for (const [i, j] of m.bonds) s.add(i).add(j);
  for (const [i] of m.lone) s.add(i);
  return [...s].sort((a, b) => a - b);
}

/** Δq_i = -(R_ii + sum_j R_ij), the non-zero ones. */
export function chargeDeltas(m: MatrixIO): Map<number, number> {
  const d = new Map<number, number>();
  const add = (i: number, v: number) => d.set(i, (d.get(i) ?? 0) + v);
  for (const [i, v] of m.lone) add(i, -v);
  for (const [i, j, v] of m.bonds) {
    add(i, -v);
    add(j, -v);
  }
  for (const [i, v] of [...d]) if (v === 0) d.delete(i);
  return d;
}

/** Electrons created by R: sum_i R_ii + 2 sum_{i<j} R_ij (0 for a conserving step). */
export function electronBalance(m: MatrixIO): number {
  return m.lone.reduce((s, [, v]) => s + v, 0) + 2 * m.bonds.reduce((s, [, , v]) => s + v, 0);
}

// ---- arrows

export type Compiled = { ok: true; matrix: MatrixIO } | { ok: false; error: string };

const fail = (error: string): Compiled => ({ ok: false, error });

/** The elementary matrix of one curved arrow (spec rules above). */
export function compileArrow(a: ArrowReadingOut): Compiled {
  const [s0, s1] = a.source;
  const [t0, t1] = a.target;
  if (a.kind === "lone_pair_to_bond") {
    if (a.source.length !== 1 || a.target.length !== 2) return fail("lone_pair_to_bond needs source [a] and target [a, b]");
    if (t0 !== s0 || t1 === s0) return fail("the forming bond must start at the lone pair's atom");
    return { ok: true, matrix: normalize({ bonds: [[s0, t1, 1]], lone: [[s0, -2]] }) };
  }
  if (a.source.length !== 2 || s0 === s1) return fail(`${a.kind} needs a bond source [a, c]`);
  if (a.kind === "bond_to_bond") {
    if (a.target.length !== 2 || t0 !== s0 || t1 === s0 || t1 === s1) return fail("bond_to_bond needs target [a, b] with the new bond at the source's atom a");
    return { ok: true, matrix: normalize({ bonds: [[s0, s1, -1], [s0, t1, 1]], lone: [] }) };
  }
  if (a.target.length !== 1 || (t0 !== s0 && t0 !== s1)) return fail("bond_to_lone_pair needs target [k] on the breaking bond");
  return { ok: true, matrix: normalize({ bonds: [[s0, s1, -1]], lone: [[t0, 2]] }) };
}

/** The sum of several arrows' matrices (the first failure is returned). */
export function compileArrows(arrows: readonly ArrowReadingOut[]): Compiled {
  let m = EMPTY_MATRIX;
  for (const a of arrows) {
    const c = compileArrow(a);
    if (!c.ok) return c;
    m = addMatrix(m, c.matrix);
  }
  return { ok: true, matrix: m };
}

/** An end of an arrow gesture: an atom, or a bond with the atom the pointer was nearer to. */
export type Endpoint = { kind: "atom"; i: number } | { kind: "bond"; i: number; j: number; near?: number };

export type Gesture = { ok: true; arrow: ArrowReadingOut } | { ok: false; error: string };

const bad = (error: string): Gesture => ({ ok: false, error });

function fromAtom(i: number, to: Endpoint, ctx: BondContext | null): Gesture {
  const lp = ctx?.lonePairs(i);
  if (lp === 0) return bad(`atom ${i} has no lone pair`);
  let b: number;
  if (to.kind === "atom") {
    if (to.i === i) return bad("an arrow needs two different ends");
    b = to.i;
  } else {
    if (to.i !== i && to.j !== i) return bad("a lone pair goes to an atom, or to a bond of its own atom");
    b = to.i === i ? to.j : to.i;
  }
  if (ctx && ctx.bondOrder(i, b) >= MAX_BOND_ORDER) return bad(`bond ${i}-${b} is already triple`);
  return { ok: true, arrow: { kind: "lone_pair_to_bond", source: [i], target: [i, b] } };
}

function fromBond(src: Extract<Endpoint, { kind: "bond" }>, to: Endpoint, ctx: BondContext | null): Gesture {
  if (ctx && ctx.bondOrder(src.i, src.j) < 1) return bad(`atoms ${src.i} and ${src.j} are not bonded`);
  if (to.kind === "atom") {
    if (to.i === src.i || to.i === src.j) {
      return { ok: true, arrow: { kind: "bond_to_lone_pair", source: [src.i, src.j], target: [to.i] } };
    }
    const a = src.near === src.j ? src.j : src.i;
    const c = a === src.i ? src.j : src.i;
    if (ctx && ctx.bondOrder(a, to.i) >= MAX_BOND_ORDER) return bad(`bond ${a}-${to.i} is already triple`);
    return { ok: true, arrow: { kind: "bond_to_bond", source: [a, c], target: [a, to.i] } };
  }
  const shared = [src.i, src.j].filter((x) => x === to.i || x === to.j);
  if (shared.length === 2) return bad("an arrow needs two different ends");
  if (shared.length === 0) return bad("bond-to-bond arrows need two bonds that share an atom");
  const a = shared[0];
  const c = a === src.i ? src.j : src.i;
  const b = a === to.i ? to.j : to.i;
  if (ctx && ctx.bondOrder(a, b) >= MAX_BOND_ORDER) return bad(`bond ${a}-${b} is already triple`);
  return { ok: true, arrow: { kind: "bond_to_bond", source: [a, c], target: [a, b] } };
}

/** The reading of a drag from `from` to `to` (masked by the reactant structure when given). */
export function arrowFromGesture(from: Endpoint, to: Endpoint, ctx: BondContext | null = null): Gesture {
  return from.kind === "atom" ? fromAtom(from.i, to, ctx) : fromBond(from, to, ctx);
}

/** "lp O3 → C5–O3"-style text of one arrow; `label(i)` names an atom. */
export function arrowText(a: ArrowReadingOut, label: (i: number) => string = String): string {
  const side = (xs: number[]) => (xs.length === 2 ? `${label(xs[0])}–${label(xs[1])}` : `lp ${label(xs[0])}`);
  return `${side(a.source)} → ${side(a.target)}`;
}

// ---- local checks (before the server's /step/check)

export interface LocalIssue {
  code: string;
  message: string;
  atoms: number[];
}

/** The contract's allowed deltas: bonds ±1 (±2, ±3 accepted), lone pairs ±2 (±4, ±6). */
const BOND_DELTAS = new Set([1, 2, 3]);
const LONE_DELTAS = new Set([2, 4, 6]);

function rangeIssues(m: MatrixIO): LocalIssue[] {
  const out: LocalIssue[] = [];
  for (const [i, j, d] of m.bonds) {
    if (!BOND_DELTAS.has(Math.abs(d))) out.push({ code: "MATRIX_MALFORMED", message: `bond ${i}-${j}: ${d > 0 ? "+" : ""}${d} is no allowed bond change`, atoms: [i, j] });
  }
  for (const [i, d] of m.lone) {
    if (!LONE_DELTAS.has(Math.abs(d))) out.push({ code: "MATRIX_MALFORMED", message: `atom ${i}: ${d > 0 ? "+" : ""}${d} lone-pair electrons (±2 per pair)`, atoms: [i] });
  }
  return out;
}

function boundIssues(m: MatrixIO, ctx: BondContext): LocalIssue[] {
  const out: LocalIssue[] = [];
  for (const [i, j, d] of m.bonds) {
    const before = ctx.bondOrder(i, j);
    const o = before + d;
    if (before === 0 && d < 0) out.push({ code: "BREAK_UNBONDED", message: `atoms ${i} and ${j} are not bonded`, atoms: [i, j] });
    else if (o < 0 || o > MAX_BOND_ORDER) out.push({ code: "BOND_ORDER_RANGE", message: `bond ${i}-${j} would have order ${o}`, atoms: [i, j] });
  }
  for (const [i, d] of m.lone) {
    const lp = ctx.lonePairs(i);
    if (lp !== null && 2 * lp + d < 0) out.push({ code: "NO_LONE_PAIR", message: `atom ${i} has no lone pair to give`, atoms: [i] });
  }
  for (const i of matrixAtoms(m)) {
    if (i < 0 || i >= ctx.n) out.push({ code: "MATRIX_MALFORMED", message: `atom ${i} is not in the pool`, atoms: [] });
  }
  return out;
}

/** What is wrong with R before the server sees it, with the server's issue codes: deltas the
 *  contract does not allow (MATRIX_MALFORMED), bond orders outside 0..3 (BOND_ORDER_RANGE,
 *  BREAK_UNBONDED), a lone pair taken from an atom without one (NO_LONE_PAIR), a step that
 *  creates or destroys electrons (CHARGE_NOT_CONSERVED). */
export function localIssues(m: MatrixIO, ctx: BondContext | null): LocalIssue[] {
  const out = [...rangeIssues(m), ...(ctx ? boundIssues(m, ctx) : [])];
  const bal = electronBalance(m);
  if (bal !== 0) out.push({ code: "CHARGE_NOT_CONSERVED", message: `R ${bal > 0 ? "creates" : "destroys"} ${Math.abs(bal)} electron${Math.abs(bal) === 1 ? "" : "s"}`, atoms: matrixAtoms(m) });
  return out;
}

/** Only deltas the contract accepts: what may be sent to the server. */
export function inContractRange(m: MatrixIO): boolean {
  return rangeIssues(m).length === 0;
}
