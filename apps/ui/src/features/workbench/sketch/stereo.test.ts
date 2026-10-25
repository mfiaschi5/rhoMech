// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts). Loads the
// real RDKit.js wasm build (skipped when it is not installed), like rdkit.test.ts.
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { drawnStereo, importSketch, repairStereo, stereoFromDepth } from "./rdkitSketch.ts";
import { applyProjection, projectPoints, quatFromAxisAngle } from "./projection.ts";
import { assignStereo, depthParities, drawnParities } from "./stereo.ts";
import { projectedLayout } from "./projectLayout.ts";
import { FIXTURES_3D } from "./projection.fixtures.ts";

const wasmJs = fileURLToPath(new URL("../../../../node_modules/@rdkit/rdkit/dist/RDKit_minimal.js", import.meta.url));
const available = existsSync(wasmJs);
let rdkitPromise = null;
const rdkit = () => (rdkitPromise ??= createRequire(import.meta.url)(wasmJs)());
const opts = { skip: available ? false : "RDKit.js not installed (npm --prefix apps/ui install)" };

/** Six view orientations: identity, the three axes by 90 deg, two oblique ones. */
const ORIENTATIONS = [
  [[0, 0, 1], 0],
  [[1, 0, 0], Math.PI / 2],
  [[0, 1, 0], Math.PI / 2],
  [[0, 0, 1], Math.PI / 2],
  [[1, 1, 0], 2.1],
  [[0.3, -1, 0.7], -1.3],
].map(([axis, t]) => quatFromAxisAngle(axis, t));

/** The sketch of a fixture projected through q, its wedges drawn from the depth (as the
 *  Sketcher's projection does before the layout fix). */
function project(rd, fx, q) {
  const sketch = importSketch(rd, fx.smiles);
  const proj = projectPoints(sketch.atoms.map((a) => a.id), fx.xyz, q);
  const m = applyProjection(sketch, proj);
  const target = stereoFromDepth(rd, m);
  const marked = assignStereo(m, depthParities(m, target.centres.keys()), { useDepth: true, clearOthers: true });
  return { m: marked, target };
}

test("3D stereo read from the projection equals the record's in every orientation", opts, async () => {
  const rd = await rdkit();
  for (const fx of FIXTURES_3D) {
    for (const q of ORIENTATIONS) {
      const { target } = project(rd, fx, q);
      assert.equal(target.smiles, fx.isomeric, `${fx.name}: a rotation keeps the configuration`);
    }
  }
});

test("projected wedges reproduce the record's isomeric SMILES, before and after the fix", opts, async () => {
  const rd = await rdkit();
  let repaired = 0;
  for (const fx of FIXTURES_3D) {
    ORIENTATIONS.forEach((q, k) => {
      const { m, target } = project(rd, fx, q);
      const drawn = drawnStereo(rd, m);
      assert.equal(drawn.smiles, fx.isomeric, `${fx.name} orientation ${k}: wedges from depth`);
      const fixed = projectedLayout(m, target.centres.keys());
      assert.ok(fixed.stereoKept, `${fx.name} orientation ${k}: configurations kept by the fix`);
      const a = fixed.after;
      assert.equal(a.labelOverlaps + a.closeContacts + a.atomOnBond, 0, `${fx.name} orientation ${k}: ${JSON.stringify(a)}`);
      assert.ok(a.bondLengthRms < 0.08, `${fx.name} orientation ${k}: bond RMS ${a.bondLengthRms}`);
      const after = repairStereo(rd, fixed.model, target);
      if (after.model !== fixed.model) repaired++;
      assert.ok(after.ok, `${fx.name} orientation ${k}: ${drawnStereo(rd, fixed.model).smiles} != ${fx.isomeric}`);
    });
  }
  assert.equal(repaired, 0, "the geometric wedge choice needs no RDKit repair on the fixtures");
});

test("one mark per stereocentre, wedge towards the viewer, hash away", opts, async () => {
  const rd = await rdkit();
  for (const fx of FIXTURES_3D) {
    const { m, target } = project(rd, fx, ORIENTATIONS[4]);
    const marks = m.bonds.filter((b) => b.stereo !== "none");
    assert.equal(marks.length, target.centres.size, fx.name);
    assert.deepEqual(new Set(marks.map((b) => b.a)), new Set(target.centres.keys()), `${fx.name}: narrow ends at the centres`);
    const z = new Map(m.atoms.map((a) => [a.id, a.z]));
    const agree = marks.filter((b) => (z.get(b.b) > z.get(b.a)) === (b.stereo === "wedge")).length;
    assert.ok(agree >= marks.length - 1, `${fx.name}: ${agree}/${marks.length} marks follow the depth`);
  }
});

test("L-alanine and (R)-CHFClBr: the drawn parity is the 3D parity", opts, async () => {
  const rd = await rdkit();
  for (const name of ["alanine", "chfclbr"]) {
    const fx = FIXTURES_3D.find((f) => f.name === name);
    for (const q of ORIENTATIONS) {
      const { m, target } = project(rd, fx, q);
      const centre = [...target.centres.keys()][0];
      assert.deepEqual(drawnParities(m), depthParities(m, [centre]));
    }
  }
});
