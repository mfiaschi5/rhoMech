// The 3D view as the sketcher's 3D->2D source: the viewer's current quaternion (getView()[4..7],
// 3Dmol's rotationGroup; sketch/projection.ts has the frame), which the sketcher sends to POST
// /api/depict2d (match3d), and the check of that convention: the record coordinates of the drawn
// atoms (through the POST's sketch_to_record) rotated by the quaternion and projected
// orthographically must match what 3Dmol draws: modelToScreen of the same atoms, fitted by a
// scale and a translation only (no rotation, no mirror), leaves a small residual.
import type { GLViewer } from "3dmol";
import type { MoleculeGeometryOut, SketchMappingOut } from "../../../api/types.gen";
import { projectPoints, type Projection3D, type Quat, type Vec3 } from "../sketch/projection";
import type { SketchModel } from "../sketch";

/** The viewer's rotation quaternion [x, y, z, w], or null when it has none. */
export function viewQuaternion(viewer: GLViewer): Quat | null {
  try {
    const v = viewer.getView() as number[];
    const q: Quat = [v[4], v[5], v[6], v[7]];
    return q.every((x) => Number.isFinite(x)) ? q : null;
  } catch {
    return null;
  }
}

/** Record index of each drawn atom (sketch atom i is molblock atom i), or null when the mapping
 *  does not cover the drawing or the geometry lacks an atom. */
export function recordIndices(model: SketchModel, mapping: SketchMappingOut, geometry: MoleculeGeometryOut): number[] | null {
  if (mapping.sketch_to_record.length !== model.atoms.length) return null;
  const n = geometry.atoms.length;
  const idx = mapping.sketch_to_record.map((r) => (Number.isInteger(r) && r >= 0 && r < n ? r : -1));
  return idx.includes(-1) ? null : idx;
}

/** The record point of each record index (geometry.atoms is in record order; record_index is
 *  checked so a reordered payload still works). */
function recordPoints(geometry: MoleculeGeometryOut, idx: readonly number[]): Vec3[] {
  const byIndex = new Map(geometry.atoms.map((a) => [a.record_index, a]));
  return idx.map((r) => {
    const a = byIndex.get(r) ?? geometry.atoms[r];
    return [a.x, a.y, a.z];
  });
}

/** The drawn atoms in the viewer's current frame. */
export function viewProjection(viewer: GLViewer, model: SketchModel, mapping: SketchMappingOut, geometry: MoleculeGeometryOut): Projection3D | null {
  const q = viewQuaternion(viewer);
  const idx = recordIndices(model, mapping, geometry);
  if (!q || !idx) return null;
  return projectPoints(
    model.atoms.map((a) => a.id),
    recordPoints(geometry, idx),
    q,
  );
}

/** RMS residual of fitting 3Dmol's screen positions (x right, y flipped to up) by s * xy + t,
 *  relative to the RMS spread of the screen points: ~0 when the projection is what the viewer
 *  shows (3Dmol's perspective adds a few percent), ~1 or more for a mirror or a wrong turn. */
export function screenResidual(viewer: GLViewer, geometry: MoleculeGeometryOut, model: SketchModel, mapping: SketchMappingOut, p: Projection3D): number | null {
  const idx = recordIndices(model, mapping, geometry);
  if (!idx || idx.length < 3) return null;
  let screen: { x: number; y: number }[];
  try {
    const pts = recordPoints(geometry, idx).map(([x, y, z]) => ({ x, y, z }));
    screen = viewer.modelToScreen(pts) as { x: number; y: number }[];
  } catch {
    return null;
  }
  const S = screen.map((s): [number, number] => [s.x, -s.y]);
  const n = S.length;
  const mean = (xs: [number, number][]) => xs.reduce((a, v) => [a[0] + v[0] / n, a[1] + v[1] / n], [0, 0]);
  const [sx, sy] = mean(S);
  const [px, py] = mean(p.xy.map((v): [number, number] => [v[0], v[1]]));
  let num = 0;
  let den = 0;
  let spread = 0;
  for (let i = 0; i < n; i++) {
    const a = [p.xy[i][0] - px, p.xy[i][1] - py];
    const b = [S[i][0] - sx, S[i][1] - sy];
    num += a[0] * b[0] + a[1] * b[1];
    den += a[0] * a[0] + a[1] * a[1];
    spread += b[0] * b[0] + b[1] * b[1];
  }
  if (den < 1e-12 || spread < 1e-12) return null;
  const s = num / den;
  if (s <= 0) return 99; // a half turn (s < 0) is no match either
  let res = 0;
  for (let i = 0; i < n; i++) {
    const ex = S[i][0] - sx - s * (p.xy[i][0] - px);
    const ey = S[i][1] - sy - s * (p.xy[i][1] - py);
    res += ex * ex + ey * ey;
  }
  return Math.sqrt(res / spread);
}
