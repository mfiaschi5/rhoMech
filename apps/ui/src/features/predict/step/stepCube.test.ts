// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The orbital variant's 3D scene: cube headers, the species of a cube, offsets, pool atoms in 3D.
import test from "node:test";
import assert from "node:assert/strict";
import { BOHR, cubeAtoms, cubeEnergy, offsetCube, offsetMolblock, poolAtoms3D, speciesOfCube } from "./stepCube.ts";

const CUBE = [
  "ksmech-serve cube v1 state=s token=28 kind=virt E_r=-1.918554",
  "units bohr; values bohr^-3/2; phase arbitrary (Pipek-Mezey)",
  "    2   -8.535707   -8.559472   -9.205221",
  "   52    0.377945    0.000000    0.000000",
  "   49    0.000000    0.377945    0.000000",
  "   50    0.000000    0.000000    0.377945",
  "    6    6.000000   -0.303737    0.008820    0.003888",
  "   17   17.000000    3.081179   -0.089596   -0.039169",
  "  5.69426E-17  1.43683E-16",
].join("\n");

const GEO = {
  state_id: "s",
  species: [
    { fragment: 0, token_key: "C[O-]", species_key: "a", atoms: [{ global_site: 0, record_index: 0, map: 1, Z: 6, symbol: "C", x: 0, y: 0, z: 0 }, { global_site: 1, record_index: 1, map: 2, Z: 8, symbol: "O", x: 1.3, y: 0, z: 0 }, { global_site: 2, record_index: 2, map: 3, Z: 1, symbol: "H", x: -0.4, y: 0.9, z: 0 }, { global_site: 3, record_index: 3, map: 4, Z: 1, symbol: "H", x: -0.4, y: -0.9, z: 0 }], bonds: [[0, 1, 1], [0, 2, 1], [0, 3, 1]], xyz: "", molblock: "", display_offset: [0, 0, 0], frame_note: "" },
    { fragment: 1, token_key: "CCl", species_key: "b", atoms: [{ global_site: 5, record_index: 0, map: 6, Z: 6, symbol: "C", x: -0.303737 * BOHR, y: 0.00882 * BOHR, z: 0.003888 * BOHR }, { global_site: 6, record_index: 1, map: 7, Z: 17, symbol: "Cl", x: 3.081179 * BOHR, y: -0.089596 * BOHR, z: -0.039169 * BOHR }], bonds: [[0, 1, 1]], xyz: "", molblock: "", display_offset: [4.8, 0, 0], frame_note: "" },
  ],
};

test("cube header: the LMO energy and the atoms in Å", () => {
  assert.equal(cubeEnergy(CUBE), -1.918554);
  assert.equal(cubeEnergy("no energy here\n"), null);
  const atoms = cubeAtoms(CUBE);
  assert.equal(atoms.length, 2);
  assert.equal(atoms[1].z, 17);
  assert.ok(Math.abs(atoms[1].at[0] - 3.081179 * BOHR) < 1e-9);
});

test("the species of a cube, by its atoms", () => {
  assert.equal(speciesOfCube(CUBE, GEO), 1);
  assert.equal(speciesOfCube(CUBE.replace("    6    6.000000", "    7    7.000000"), GEO), null);
});

test("offsets move the cube's origin and atoms, and a molblock's atoms", () => {
  const moved = offsetCube(CUBE, [4.8, 0, 0]);
  const origin = Number(moved.split("\n")[2].trim().split(/\s+/)[1]);
  assert.ok(Math.abs(origin - (-8.535707 + 4.8 / BOHR)) < 1e-5);
  assert.ok(Math.abs(cubeAtoms(moved)[0].at[0] - (-0.303737 * BOHR + 4.8)) < 1e-5);
  assert.equal(moved.split("\n").at(-1), CUBE.split("\n").at(-1), "the values are untouched");
  assert.equal(offsetCube(CUBE, [0, 0, 0]), CUBE);
  const mol = "\n     RDKit          3D\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n    0.0102   -0.0010    0.0009 C   0  0  0  0\nM  END\n";
  assert.equal(offsetMolblock(mol, [1, 2, 3]).split("\n")[4], "    1.0102    1.9990    3.0009 C   0  0  0  0");
});

test("pool atoms in 3D: heavy atoms by global site, hidden H by their parent, offsets applied", () => {
  const dep = { atoms: [{ pool_atom: 0, global_site: 0, hidden_h_atoms: [7, 8] }, { pool_atom: 1, global_site: 1, hidden_h_atoms: [] }, { pool_atom: 5, global_site: 5, hidden_h_atoms: [] }] };
  const m = poolAtoms3D(dep, GEO);
  assert.deepEqual(m.get(1), { species: 0, index: 1, at: [1.3, 0, 0] });
  assert.equal(m.get(7).index, 2);
  assert.equal(m.get(8).index, 3);
  assert.ok(Math.abs(m.get(5).at[0] - (-0.303737 * BOHR + 4.8)) < 1e-9);
});
