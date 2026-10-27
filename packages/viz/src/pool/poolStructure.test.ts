// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The pool structure of a depiction: pool atoms from `pool_atom` (not the molblock order, not
// map numbers), the hidden H as bonded atoms (bond orders, lone pairs, atom count), drawn only
// when shown (placed beside their parent, the parent's H label reduced), and the hidden H an R
// or a reading touches.
import test from "node:test";
import assert from "node:assert/strict";
import { HIDDEN_H_BOND, poolStructure, touchedHydrogens } from "./poolStructure.ts";

/** Methanol drawn as O (molblock 1) and C (molblock 2): pool atoms O1 and C0, H 2-4 on C and
 *  H 5 on O, all hidden; maps deliberately unrelated to the pool atoms (a product's). */
const MOLBLOCK = [
  "",
  "  test",
  "",
  "  2  1  0  0  0  0  0  0  0  0999 V2000",
  "    1.5000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0",
  "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
  "  1  2  1  0",
  "M  END",
].join("\n");
const DEP = {
  ref: "s0",
  molblock: MOLBLOCK,
  atoms: [
    { depiction_idx: 0, pool_atom: 1, hidden_h_atoms: [5], global_site: null, map: 7, element: "O", fragment: 0, hidden_h: 1 },
    { depiction_idx: 1, pool_atom: 0, hidden_h_atoms: [2, 3, 4], global_site: null, map: 3, element: "C", fragment: 0, hidden_h: 3 },
  ],
  server_svg: null,
  server_coords: null,
};

test("pool atoms come from pool_atom; hidden H are bonded atoms of the structure", () => {
  const s = poolStructure(DEP);
  assert.equal(s.n, 6);
  assert.equal(s.label(1), "O1");
  assert.equal(s.label(0), "C0");
  assert.equal(s.label(5), "H5");
  assert.equal(s.bondOrder(0, 1), 1);
  assert.equal(s.bondOrder(0, 3), 1);
  assert.equal(s.bondOrder(1, 5), 1);
  assert.equal(s.bondOrder(1, 3), 0);
  assert.equal(s.lonePairs(1), 2, "O: 6 - 2 bonds = 4 free electrons");
  assert.equal(s.lonePairs(0), 0);
  assert.equal(s.lonePairs(4), 0);
  assert.deepEqual(s.drawn.map((a) => a.index).sort(), [0, 1]);
  const c = s.atom(0);
  assert.equal(s.hCounts.get(c.sketchId), 3, "undrawn H stay in the label");
});

test("a shown hidden H is drawn beside its parent and leaves the parent's label", () => {
  const s = poolStructure(DEP, { show: new Set([3]) });
  assert.deepEqual(s.drawn.map((a) => a.index).sort(), [0, 1, 3]);
  const c = s.atom(0);
  const h = s.atom(3);
  assert.ok(h.sketchId !== null);
  assert.ok(Math.abs(Math.hypot(h.x - c.x, h.y - c.y) - HIDDEN_H_BOND) < 1e-9);
  assert.ok(h.x < c.x, "in the free gap, away from O");
  assert.equal(s.hCounts.get(c.sketchId), 2);
  assert.equal(s.model.atoms.length, 3);
  assert.equal(s.model.bonds.length, 2);
  assert.equal(poolStructure(DEP, { show: "all" }).drawn.length, 6);
});

test("the hidden H an R or its arrows touch", () => {
  const m = { bonds: [[1, 5, -1], [0, 1, 1]], lone: [[3, 2]] };
  assert.deepEqual([...touchedHydrogens(DEP, m)].sort(), [3, 5]);
  assert.deepEqual([...touchedHydrogens(DEP, null, [{ kind: "bond_to_lone_pair", source: [0, 2], target: [2] }])], [2]);
});
