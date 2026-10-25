// Tetrahedral stereo of a 2D drawing as signed volumes, and the choice of the wedge or hash
// that draws a wanted configuration. Pure (no RDKit); rdkitSketch.ts checks the result with
// RDKit and repairs a centre RDKit reads otherwise.
//
// A centre's configuration is the sign of a volume over its drawn neighbours in ascending id
// order: three neighbours (an implicit H is the fourth) det(v1, v2, v3), four neighbours
// det(v2 - v1, v3 - v1, v4 - v1), with v the neighbour minus the centre. The true sign comes
// from x, y and the depth z of a projection; the drawn sign from x, y and a pseudo-depth of
// +-|v| on each neighbour behind a wedge (+) or hash (-) whose narrow end is the centre (the
// pseudo-3D reading of a wedge drawing).
//
// assignStereo puts one mark per centre, on the single bond that draws the wanted sign and
// scores best: acyclic bond (+4), terminal neighbour (+2), neighbour not itself a centre (+3,
// so no bond carries two narrow ends), a clearly non-degenerate volume (+|V|), and with depth
// the neighbour farthest in z (+2 |dz| / L) marked as it sits (wedge in front, hash behind:
// +3, else -3); the bond that already carries this centre's mark keeps it (+5; not after a
// projection, whose old marks belong to another layout).
import type { BondStereo, SketchBond, SketchModel } from "./model";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { BOND_LENGTH, neighbors, setBondMarks, smallestRingThrough } from "./model.ts";

type V3 = [number, number, number];

/** Below this normalized |volume| a drawing does not define the configuration. */
export const MIN_VOLUME = 0.08;

const det3 = (a: V3, b: V3, c: V3): number =>
  a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
const sub3 = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm3 = (a: V3): number => Math.hypot(a[0], a[1], a[2]);

/** The normalized signed volume of neighbour vectors (3 or 4 of them), 0 otherwise. */
export function chiralVolume(vs: readonly V3[]): number {
  if (vs.length === 3) {
    const l = norm3(vs[0]) * norm3(vs[1]) * norm3(vs[2]);
    return l > 1e-12 ? det3(vs[0], vs[1], vs[2]) / l : 0;
  }
  if (vs.length === 4) {
    const l = (norm3(vs[0]) + norm3(vs[1]) + norm3(vs[2]) + norm3(vs[3])) / 4;
    const v = det3(sub3(vs[1], vs[0]), sub3(vs[2], vs[0]), sub3(vs[3], vs[0]));
    return l > 1e-12 ? v / l ** 3 : 0;
  }
  return 0;
}

/** Drawn neighbours of a centre in ascending id order (the order of every volume). */
export const stereoNeighbours = (m: SketchModel, c: number): number[] => [...neighbors(m, c)].sort((a, b) => a - b);

function bondBetween(m: SketchModel, a: number, b: number): SketchBond | undefined {
  return m.bonds.find((x) => (x.a === a && x.b === b) || (x.a === b && x.b === a));
}

/** The true volume from x, y and the depth z (null without depth on the centre or a
 *  neighbour). */
export function trueVolume(m: SketchModel, c: number): number | null {
  const at = new Map(m.atoms.map((a) => [a.id, a]));
  const centre = at.get(c);
  if (!centre || centre.z === undefined) return null;
  const vs: V3[] = [];
  for (const n of stereoNeighbours(m, c)) {
    const a = at.get(n);
    if (!a || a.z === undefined) return null;
    vs.push([a.x - centre.x, a.y - centre.y, a.z - centre.z]);
  }
  return chiralVolume(vs);
}

/** The pseudo-depth sign of bond b seen from centre c: +1 wedge, -1 hash (narrow end at c). */
function markSign(b: Pick<SketchBond, "a" | "stereo" | "order">, c: number): number {
  if (b.order !== 1 || b.a !== c) return 0;
  return b.stereo === "wedge" ? 1 : b.stereo === "hash" ? -1 : 0;
}

/** The volume as drawn (2D + wedges); `override` replaces the marks at c by one mark. */
export function drawnVolume(m: SketchModel, c: number, override?: { bond: number; sign: number }): number {
  const at = new Map(m.atoms.map((a) => [a.id, a]));
  const centre = at.get(c);
  if (!centre) return 0;
  const vs: V3[] = [];
  for (const n of stereoNeighbours(m, c)) {
    const a = at.get(n);
    const b = bondBetween(m, c, n);
    if (!a || !b) return 0;
    const dx = a.x - centre.x;
    const dy = a.y - centre.y;
    const s = override ? (b.id === override.bond ? override.sign : 0) : markSign(b, c);
    vs.push([dx, dy, s * Math.hypot(dx, dy)]);
  }
  return chiralVolume(vs);
}

/** Atoms that are the narrow end of a wedge or hash. */
export function markedCentres(m: SketchModel): number[] {
  const out = new Set<number>();
  for (const b of m.bonds) if (b.order === 1 && b.stereo !== "none") out.add(b.a);
  return m.atoms.map((a) => a.id).filter((id) => out.has(id));
}

/** The drawn configuration (+1 / -1) of each marked centre that defines one. */
export function drawnParities(m: SketchModel): Map<number, number> {
  const out = new Map<number, number>();
  for (const c of markedCentres(m)) {
    const v = drawnVolume(m, c);
    if (Math.abs(v) >= MIN_VOLUME) out.set(c, Math.sign(v));
  }
  return out;
}

/** The true configuration of each centre from the depth (centres without depth are left out). */
export function depthParities(m: SketchModel, centres: Iterable<number>): Map<number, number> {
  const out = new Map<number, number>();
  for (const c of centres) {
    const v = trueVolume(m, c);
    if (v !== null && Math.abs(v) > 1e-6) out.set(c, Math.sign(v));
  }
  return out;
}

/** A mark on c-n of a four-neighbour centre is ambiguous when the three other neighbours lie
 *  in one half-plane (within 170 deg): the centre is then outside their triangle, and RDKit's
 *  pseudo-3D reading need not follow the volume. */
export function ambiguousMark(m: SketchModel, c: number, n: number): boolean {
  const nbrs = stereoNeighbours(m, c);
  if (nbrs.length !== 4) return false;
  const at = new Map(m.atoms.map((a) => [a.id, a]));
  const centre = at.get(c);
  if (!centre) return true;
  const t = nbrs
    .filter((k) => k !== n)
    .map((k) => {
      const a = at.get(k);
      return a ? Math.atan2(a.y - centre.y, a.x - centre.x) : 0;
    })
    .sort((p, q) => p - q);
  const gaps = [t[1] - t[0], t[2] - t[1], t[0] + 2 * Math.PI - t[2]];
  return Math.max(...gaps) > (190 * Math.PI) / 180;
}

interface Choice {
  bond: number;
  stereo: BondStereo;
  score: number;
}

/** The best mark at centre c for configuration `sign`, or null when no single bond is free. */
function chooseMark(m: SketchModel, c: number, sign: number, centres: ReadonlySet<number>, taken: ReadonlySet<number>, opts: AssignOptions): Choice | null {
  const useDepth = opts.useDepth ?? false;
  const at = new Map(m.atoms.map((a) => [a.id, a]));
  const zc = at.get(c)?.z;
  let best: Choice | null = null;
  for (const n of stereoNeighbours(m, c)) {
    const b = bondBetween(m, c, n);
    if (!b || b.order !== 1 || taken.has(b.id)) continue;
    const v = drawnVolume(m, c, { bond: b.id, sign: 1 });
    const stereo: BondStereo = Math.sign(v) === sign ? "wedge" : "hash";
    const clear = Math.abs(v) >= MIN_VOLUME && !ambiguousMark(m, c, n);
    let score = Math.min(Math.abs(v), 1) + (clear ? 0 : -20);
    if (!smallestRingThrough(m, c, n)) score += 4;
    if (neighbors(m, n).length === 1) score += 2;
    if (!centres.has(n)) score += 3;
    if (!opts.clearOthers && b.stereo !== "none" && b.a === c) score += 5;
    const zn = at.get(n)?.z;
    if (useDepth && zc !== undefined && zn !== undefined) {
      const dz = zn - zc;
      score += (2 * Math.abs(dz)) / BOND_LENGTH;
      if (Math.abs(dz) > 1e-6) score += (dz > 0) === (stereo === "wedge") ? 3 : -3;
    }
    if (!best || score > best.score + 1e-9) best = { bond: b.id, stereo, score };
  }
  return best;
}

export interface AssignOptions {
  /** Prefer marks that follow the depth (projection). */
  useDepth?: boolean;
  /** Clear every mark not chosen here and ignore the old ones (projection); else only the
   *  marks at the centres are redrawn, the bond that had one first. */
  clearOthers?: boolean;
}

/** Marks that draw configuration targets.get(c) at every centre c (one mark per centre). */
export function assignStereo(m: SketchModel, targets: ReadonlyMap<number, number>, opts: AssignOptions = {}): SketchModel {
  const centres = new Set(targets.keys());
  const marks = new Map<number, { stereo: BondStereo; narrow: number }>();
  for (const b of m.bonds) {
    if (b.stereo === "none") continue;
    if (opts.clearOthers || centres.has(b.a)) marks.set(b.id, { stereo: "none", narrow: b.a });
  }
  const taken = new Set<number>();
  for (const a of m.atoms) {
    const sign = targets.get(a.id);
    if (!sign) continue;
    const pick = chooseMark(m, a.id, sign, centres, taken, opts);
    if (!pick) continue;
    taken.add(pick.bond);
    marks.set(pick.bond, { stereo: pick.stereo, narrow: a.id });
  }
  return setBondMarks(m, marks);
}
