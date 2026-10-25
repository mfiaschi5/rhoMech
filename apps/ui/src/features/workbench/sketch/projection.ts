// The 3D viewer's frame, shared with the server: the quaternion convention the sketcher's 3D->2D
// sends to POST /api/depict2d (mode match3d; serve/depiction2d.view_matrix is the same rotation)
// and the plain orthographic projection the page uses to check that convention against 3Dmol's
// own screen positions (viewer/viewProjection.ts). Pure: this module never imports 3Dmol.
//
// Frame (3Dmol GLViewer): a model point p is drawn at R(q) (p + t) + (0, 0, rz), q the
// rotationGroup quaternion of getView() [.., qx, qy, qz, qw], t the modelGroup position; the
// camera sits on +z looking down -z, so view x is screen right, view y screen up and view z
// points at the viewer. The projection is orthographic (3Dmol's perspective, fov 20 deg, only
// changes the apparent size with depth; the page checks the residual against modelToScreen).
import type { Vec } from "./geometry";

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
