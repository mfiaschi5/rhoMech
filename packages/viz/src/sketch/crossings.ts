// Bond crossings of a 2D drawing: pairs of bonds (no shared atom) whose segments cross, and
// which of the two is in front. With a depth on all four atoms (a 3D projection) the bond whose
// interpolated z at the crossing is larger is in front; otherwise (or on a depth tie) the bond
// drawn later (higher index in model.bonds) is. Pure, model units; the drawing turns the
// crossing points into gaps on the bond behind (drawing.ts, ChemDraw's crossing breaks).
import type { SketchModel } from "./model";
import type { Vec } from "./geometry";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { cross, sub } from "./geometry.ts";

export interface Crossing {
  /** The bond in front and the one behind (ids). */
  front: number;
  back: number;
  /** The crossing point (model units) and its parameter along each bond (0 at a, 1 at b). */
  at: Vec;
  tFront: number;
  tBack: number;
}

/** Parameters (s along p->q, t along r->u) of a proper crossing, or null. Endpoints within
 *  `eps` (fraction of the length) do not count, so bonds that merely touch are not crossings. */
export function segmentCrossing(p: Vec, q: Vec, r: Vec, u: Vec, eps = 0.02): [number, number] | null {
  const d1 = sub(q, p);
  const d2 = sub(u, r);
  const den = cross(d1, d2);
  if (Math.abs(den) < 1e-12) return null;
  const w = sub(r, p);
  const s = cross(w, d2) / den;
  const t = cross(w, d1) / den;
  if (s <= eps || s >= 1 - eps || t <= eps || t >= 1 - eps) return null;
  return [s, t];
}

const depthAt = (za: number | undefined, zb: number | undefined, t: number): number | null =>
  za === undefined || zb === undefined || !Number.isFinite(za) || !Number.isFinite(zb) ? null : za + (zb - za) * t;

/** Every crossing of the drawing, in bond order (i < j). */
export function findCrossings(m: SketchModel): Crossing[] {
  const atom = new Map(m.atoms.map((a) => [a.id, a]));
  const segs = m.bonds.map((b) => {
    const pa = atom.get(b.a);
    const pb = atom.get(b.b);
    return pa && pb ? { b, p: [pa.x, pa.y] as Vec, q: [pb.x, pb.y] as Vec, za: pa.z, zb: pb.z } : null;
  });
  const out: Crossing[] = [];
  for (let i = 0; i < segs.length; i++) {
    const s1 = segs[i];
    if (!s1) continue;
    for (let j = i + 1; j < segs.length; j++) {
      const s2 = segs[j];
      if (!s2 || s1.b.a === s2.b.a || s1.b.a === s2.b.b || s1.b.b === s2.b.a || s1.b.b === s2.b.b) continue;
      const hit = segmentCrossing(s1.p, s1.q, s2.p, s2.q);
      if (!hit) continue;
      const [s, t] = hit;
      const z1 = depthAt(s1.za, s1.zb, s);
      const z2 = depthAt(s2.za, s2.zb, t);
      const firstInFront = z1 !== null && z2 !== null && Math.abs(z1 - z2) > 1e-9 ? z1 > z2 : false;
      const at: Vec = [s1.p[0] + (s1.q[0] - s1.p[0]) * s, s1.p[1] + (s1.q[1] - s1.p[1]) * s];
      out.push(
        firstInFront
          ? { front: s1.b.id, back: s2.b.id, at, tFront: s, tBack: t }
          : { front: s2.b.id, back: s1.b.id, at, tFront: t, tBack: s },
      );
    }
  }
  return out;
}
