// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
import test from "node:test";
import assert from "node:assert/strict";
import { projectPoints, quatFromAxisAngle, rotateByQuat, toViewFrame } from "./projection.ts";
import { addAtom, addBond, emptySketch, sketchExtras, withExtras } from "@ksmech/viz/sketch/model";
import { fromMolblock, toMolblock } from "@ksmech/viz/sketch/molfile";
import { applyDepiction } from "./depiction.ts";

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const closeV = (p, q, eps = 1e-9) => p.forEach((x, i) => close(x, q[i], eps));

/** 3Dmol's Matrix4.setRotationFromQuaternion (column-major te[], what the renderer applies to
 *  the rotationGroup), as rows. */
function threeDmolMatrix([x, y, z, w]) {
  const [x2, y2, z2] = [x + x, y + y, z + z];
  const [xx, xy, xz, yy, yz, zz, wx, wy, wz] = [x * x2, x * y2, x * z2, y * y2, y * z2, z * z2, w * x2, w * y2, w * z2];
  return [
    [1 - (yy + zz), xy - wz, xz + wy],
    [xy + wz, 1 - (xx + zz), yz - wx],
    [xz - wy, yz + wx, 1 - (xx + yy)],
  ];
}
const apply = (M, v) => M.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);

test("rotateByQuat: known rotations, right-handed, equal to 3Dmol's rotation matrix", () => {
  closeV(rotateByQuat([1, 0, 0], quatFromAxisAngle([0, 0, 1], Math.PI / 2)), [0, 1, 0]);
  closeV(rotateByQuat([0, 1, 0], quatFromAxisAngle([1, 0, 0], Math.PI / 2)), [0, 0, 1]);
  closeV(rotateByQuat([0, 0, 1], quatFromAxisAngle([0, 1, 0], Math.PI / 2)), [1, 0, 0]);
  const q = quatFromAxisAngle([0.3, -1, 0.7], 2.2);
  const M = threeDmolMatrix(q);
  for (const v of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.4, -2, 1.3]]) closeV(rotateByQuat(v, q), apply(M, v));
  // a proper rotation keeps the handedness: x' cross y' = z'
  const [ex, ey, ez] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map((v) => rotateByQuat(v, q));
  closeV([ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]], ez);
});

test("toViewFrame normalizes the quaternion; identity keeps x right, y up, z to the viewer", () => {
  const pts = [[1, 2, 3], [-1, 0.5, -2]];
  assert.deepEqual(toViewFrame(pts, [0, 0, 0, 1]), pts);
  closeV(toViewFrame([[1, 0, 0]], [0, 0, 2, 2])[0], [0, 1, 0]); // 90 deg about z, unnormalized
});

test("projectPoints: a 90 deg turn about y puts +x into the depth (towards the viewer is -x)", () => {
  const p = projectPoints([7, 8], [[1, 0, 0], [0, 1, 0]], quatFromAxisAngle([0, 1, 0], Math.PI / 2));
  assert.deepEqual(p.ids, [7, 8]);
  closeV(p.xy[0], [0, 0]);
  close(p.z[0], -1);
  closeV(p.xy[1], [0, 1]);
  close(p.z[1], 0);
});

function propane() {
  let m = emptySketch();
  const ids = [];
  for (const [x, y] of [[0, 0], [1.3, 0.75], [2.6, 0]]) {
    let id;
    [m, id] = addAtom(m, "C", x, y);
    ids.push(id);
  }
  [m] = addBond(m, ids[0], ids[1]);
  [m] = addBond(m, ids[1], ids[2]);
  return { m, ids };
}

test("depth is kept by the session extras and never written to the molfile", () => {
  const { m } = propane();
  const out = applyDepiction(m, { coords: [[0, 0], [1.27, 0.7], [2.54, 0]], depth: [0, 0.9, 0], wedges: [] });
  const mb = toMolblock(out);
  assert.match(mb, /2D/);
  for (const line of mb.split("\n").slice(4, 7)) assert.equal(Number.parseFloat(line.slice(20, 30)), 0);
  const back = withExtras(fromMolblock(mb), JSON.parse(JSON.stringify(sketchExtras(out))));
  assert.deepEqual(back.atoms.map((a) => a.z), out.atoms.map((a) => a.z));
  assert.equal(back.breakCrossings, true);
});
