// 3D -> 2D projection of the sketch: the record geometry of each drawn atom, rotated into the 3D
// viewer's current frame, becomes the 2D drawing (x right, y up) plus a depth z per atom
// (positive towards the viewer). Pure: the page reads the viewer's quaternion and hands the
// rotated coordinates over as a Projection3D, so this module never imports 3Dmol.
//
// Frame (3Dmol GLViewer): a model point p is drawn at R(q) (p + t) + (0, 0, rz), q the
// rotationGroup quaternion of getView() [.., qx, qy, qz, qw], t the modelGroup position; the
// camera sits on +z looking down -z, so view x is screen right, view y screen up and view z
// points at the viewer. The projection is orthographic (3Dmol's perspective, fov 20 deg, only
// changes the apparent size with depth; the page checks the residual against modelToScreen).
import type { SketchModel } from "./model";
import type { Vec } from "./geometry";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { BOND_LENGTH, centroid, setBreakCrossings, setPositions } from "./model.ts";

export type Vec3 = readonly [number, number, number];
/** A unit quaternion [x, y, z, w] (3Dmol / three.js order). */
export type Quat = readonly [number, number, number, number];

/** Rotated view-frame coordinates of drawn atoms (Angstrom): ids[i] is at xy[i], depth z[i]. */
export interface Projection3D {
  ids: number[];
  xy: Vec[];
  z: number[];
}

/** v' = q v q* (three.js Vector3.applyQuaternion, which 3Dmol uses). */
export function rotateByQuat(v: Vec3, q: Quat): Vec3 {
  const [x, y, z] = v;
  const [qx, qy, qz, qw] = q;
  const tx = 2 * (qy * z - qz * y);
  const ty = 2 * (qz * x - qx * z);
  const tz = 2 * (qx * y - qy * x);
  return [x + qw * tx + (qy * tz - qz * ty), y + qw * ty + (qz * tx - qx * tz), z + qw * tz + (qx * ty - qy * tx)];
}

/** The quaternion of a rotation by `angle` (radians) about `axis`. */
export function quatFromAxisAngle(axis: Vec3, angle: number): Quat {
  const l = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const s = Math.sin(angle / 2) / l;
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}

/** Model points in the view frame: x right, y up, z towards the viewer (translation dropped). */
export function toViewFrame(points: readonly Vec3[], q: Quat): Vec3[] {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  const u: Quat = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
  return points.map((p) => rotateByQuat(p, u));
}

/** The Projection3D of drawn atoms `ids` at model points `points`, seen through `q`. */
export function projectPoints(ids: readonly number[], points: readonly Vec3[], q: Quat): Projection3D {
  const v = toViewFrame(points, q);
  return { ids: [...ids], xy: v.map((p): Vec => [p[0], p[1]]), z: v.map((p) => p[2]) };
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const k = s.length >> 1;
  return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2;
}

/** The median 3D length of the drawn bonds whose two atoms are projected. */
export function medianBond3D(m: SketchModel, p: Projection3D): number {
  const at = new Map(p.ids.map((id, i) => [id, i]));
  const ls: number[] = [];
  for (const b of m.bonds) {
    const i = at.get(b.a);
    const j = at.get(b.b);
    if (i === undefined || j === undefined) continue;
    ls.push(Math.hypot(p.xy[i][0] - p.xy[j][0], p.xy[i][1] - p.xy[j][1], p.z[i] - p.z[j]));
  }
  return median(ls);
}

/** The drawing laid out as the projection: scaled so the median 3D bond is BOND_LENGTH (the
 *  same factor for the depth, so x, y, z stay a rotated copy of the record), centred on the old
 *  centroid, with crossing breaks on. Null when an atom is missing from the projection. */
export function applyProjection(m: SketchModel, p: Projection3D): SketchModel | null {
  const at = new Map(p.ids.map((id, i) => [id, i]));
  if (m.atoms.length === 0 || m.atoms.some((a) => !at.has(a.id))) return null;
  const med = medianBond3D(m, p);
  const k = med > 1e-6 ? BOND_LENGTH / med : 1;
  const n = p.ids.length;
  const cx = p.xy.reduce((s, v) => s + v[0], 0) / n;
  const cy = p.xy.reduce((s, v) => s + v[1], 0) / n;
  const cz = p.z.reduce((s, v) => s + v, 0) / n;
  const [ox, oy] = centroid(m);
  const pos = new Map<number, readonly [number, number, number]>();
  p.ids.forEach((id, i) => pos.set(id, [ox + (p.xy[i][0] - cx) * k, oy + (p.xy[i][1] - cy) * k, (p.z[i] - cz) * k]));
  return setBreakCrossings(setPositions(m, pos), true);
}
