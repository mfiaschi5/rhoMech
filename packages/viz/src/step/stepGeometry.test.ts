// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// Step sides: map numbers across a step, the Procrustes fit, fragment separation and spreading,
// and a side read from a real pool depiction (MeO- + MeCl).
import test from "node:test";
import assert from "node:assert/strict";
import { alignProduct, mappedAtomMaps, procrustes, productIndexMap, relaxLayout, separate, sideFragments, sideOf, spreadFragments } from "./stepGeometry.ts";
import { poolStructure } from "../pool/poolStructure.ts";

const close = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol;

const SRC = "[C:1]([O-:2])([H:3])([H:4])[H:5].[C:6]([Cl:7])([H:8])([H:9])[H:10]";
const PROD = "[C:1]([O:2][C:6]([H:8])([H:9])[H:10])([H:3])([H:4])[H:5].[Cl-:7]";

test("mapped SMILES: maps in atom order; an unmapped atom gives null", () => {
  assert.deepEqual(mappedAtomMaps(SRC), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.deepEqual(mappedAtomMaps(PROD), [1, 2, 6, 8, 9, 10, 3, 4, 5, 7]);
  assert.equal(mappedAtomMaps("C[O-:2]"), null);
  assert.equal(mappedAtomMaps("[CH3][O-:2]"), null);
  assert.deepEqual(mappedAtomMaps("[c:1]1[c:2][c:3]1"), [1, 2, 3]);
});

test("product pool atom -> source pool atom by map", () => {
  const m = productIndexMap(SRC, PROD);
  // product: C1 O2 C6 H8 H9 H10 H3 H4 H5 Cl7 -> source indices 0 1 5 7 8 9 2 3 4 6
  assert.deepEqual([...m.entries()].sort((a, b) => a[0] - b[0]).map(([, i]) => i), [0, 1, 5, 7, 8, 9, 2, 3, 4, 6]);
  assert.equal(productIndexMap(SRC, "[C:1][O:99]"), null);
});

test("procrustes recovers a rotation, a reflection and a translation", () => {
  const from = [[0, 0], [1, 0], [1, 2]];
  const rot = (p) => [Math.cos(0.7) * p[0] - Math.sin(0.7) * p[1] + 3, Math.sin(0.7) * p[0] + Math.cos(0.7) * p[1] - 1];
  const T = procrustes(from, from.map(rot));
  for (const p of from) assert.ok(close(T(p)[0], rot(p)[0], 1e-9) && close(T(p)[1], rot(p)[1], 1e-9));
  const mirror = (p) => [-p[0] + 5, p[1]];
  const M = procrustes(from, from.map(mirror));
  for (const p of from) assert.ok(close(M(p)[0], mirror(p)[0], 1e-9) && close(M(p)[1], mirror(p)[1], 1e-9));
  const one = procrustes([[1, 1]], [[4, 5]]);
  assert.deepEqual(one([2, 2]), [5, 6]);
});

test("separate: crowding fragments end at least the gap apart; apart ones stay", () => {
  const xy = new Map([[0, [0, 0]], [1, [1, 0]], [2, [1.2, 0.1]]]);
  const out = separate(xy, [[0, 1], [2]], 1);
  assert.deepEqual(out.get(0), [0, 0], "the bigger fragment holds still");
  const b = out.get(2);
  assert.ok(b[0] - 1 >= 1 - 1e-6 || Math.abs(b[1] - 0) >= 1 - 1e-6);
  const far = new Map([[0, [0, 0]], [1, [10, 0]]]);
  assert.deepEqual(separate(far, [[0], [1]], 1), far);
});

test("spread: labels keep the gap; fragments never come closer", () => {
  const xy = new Map([[0, [0, 0]], [1, [2, 0]], [2, [10, 0]]]);
  const out = spreadFragments(xy, [[0], [1], [2]], () => 0.5, 1.5);
  assert.ok(out.get(1)[0] - 0.5 - (out.get(0)[0] + 0.5) >= 1.5 - 1e-9);
  assert.equal(out.get(2)[0] >= 10, true);
  // another row (no vertical overlap) is not pushed
  const rows = new Map([[0, [0, 0]], [1, [0.5, 5]]]);
  assert.deepEqual(spreadFragments(rows, [[0], [1]], () => 0.5, 1.5).get(1), [0.5, 5]);
});

test("align: each product fragment lands on its atoms' source positions", () => {
  const src = new Map([[0, [0, 0]], [1, [1.5, 0]], [5, [4.5, 0]], [6, [6, 0]]]);
  // the product drawn elsewhere, turned by 90°: C0-O1-C5 and Cl6 alone
  const prod = new Map([[0, [10, 10]], [1, [10, 11.5]], [5, [11.3, 12.25]], [6, [20, 20]]]);
  const out = alignProduct(src, prod, [[0, 1, 5], [6]], 1.5);
  assert.ok(Math.hypot(out.get(1)[0] - 1.5, out.get(1)[1]) < 1.6, "O stays near its reactant place");
  // Cl alone is translated onto its source position, then pushed clear of the other fragment
  const cl = out.get(6);
  assert.ok(Math.abs(cl[1]) < 3 && cl[0] > 4);
});

const DEP = {
  ref: "s",
  molblock: "\n     RDKit          2D\n\n  4  2  0  0  0  0  0  0  0  0999 V2000\n   -0.5000   -0.5000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.5000   -0.5000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.5000    0.5000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.5000    0.5000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0\n  1  2  1  0\n  3  4  1  0\nM  CHG  1   2  -1\nM  END\n",
  atoms: [
    { depiction_idx: 0, pool_atom: 0, hidden_h_atoms: [2, 3, 4], global_site: 0, map: 1, element: "C", fragment: 0, hidden_h: 3 },
    { depiction_idx: 1, pool_atom: 1, hidden_h_atoms: [], global_site: 1, map: 2, element: "O", fragment: 0, hidden_h: 0 },
    { depiction_idx: 2, pool_atom: 5, hidden_h_atoms: [7, 8, 9], global_site: 5, map: 6, element: "C", fragment: 1, hidden_h: 3 },
    { depiction_idx: 3, pool_atom: 6, hidden_h_atoms: [], global_site: 6, map: 7, element: "Cl", fragment: 1, hidden_h: 0 },
  ],
  server_svg: null,
  server_coords: null,
};

test("a side from a pool depiction: elements, charges, real H counts, lone pairs, bonds", () => {
  const s = poolStructure(DEP, { label: new Set([1, 5, 6]), show: new Set([7]) });
  const side = sideOf(s);
  assert.deepEqual([...side.atoms.keys()].sort((a, b) => a - b), [0, 1, 5, 6, 7]);
  assert.equal(side.atoms.get(1).lone, 3);
  assert.equal(side.atoms.get(1).charge, -1);
  assert.equal(side.atoms.get(0).hCount, 3);
  assert.equal(side.atoms.get(5).hCount, 2, "a labelled carbon keeps its H count (one H drawn)");
  assert.equal(side.atoms.get(5).labelled, true);
  assert.deepEqual([...side.bonds.keys()].sort(), ["0-1", "5-6", "5-7"]);
  assert.equal(sideFragments(side).length, 2);
});

const sd = (bonds) => ({ atoms: new Map(), bonds: new Map(bonds.map(([i, j, order = 1]) => [`${Math.min(i, j)}-${Math.max(i, j)}`, { i: Math.min(i, j), j: Math.max(i, j), order }])) });
const d = (xy, i, j) => Math.hypot(xy.get(i)[0] - xy.get(j)[0], xy.get(i)[1] - xy.get(j)[1]);

test("relax: a drawing that already fits its bonds holds still", () => {
  const L = 1.5;
  const start = new Map([[0, [0, 0]], [1, [L, 0]], [2, [L + L * Math.cos(Math.PI / 3), L * Math.sin(Math.PI / 3)]]]);
  const out = relaxLayout(start, sd([[0, 1], [1, 2]]), { bond: L, iterations: 200 });
  for (const i of [0, 1, 2]) assert.ok(Math.hypot(out.get(i)[0] - start.get(i)[0], out.get(i)[1] - start.get(i)[1]) < 0.05);
});

test("relax: an SN2 pulls O and C to a bond and pushes the leaving Cl away", () => {
  const L = 1.5;
  // O1 ... C5-Cl6 nearly in a row (the reactant face to face), product bonds C0-O1, O1-C5
  const start = new Map([[0, [0, 0.2]], [1, [L, 0]], [5, [4 * L, 0]], [6, [5 * L, 0.1]]]);
  const out = relaxLayout(start, sd([[0, 1], [1, 5]]), { bond: L, iterations: 400 });
  assert.ok(Math.abs(d(out, 1, 5) - L) < 0.1, "the new bond has a bond's length");
  assert.ok(d(out, 5, 6) >= 1.75 * L, "the chloride ends clear of the carbon");
  assert.ok(out.get(6)[0] > out.get(5)[0], "and stays on its side");
});

test("relax: four neighbours spread to right angles; a triple bond stays straight", () => {
  const L = 1.5;
  const start = new Map([[0, [0, 0]], [1, [L, 0.1]], [2, [-L, 0.2]], [3, [0.1, L]], [4, [3 * L, -0.4]]]);
  const out = relaxLayout(start, sd([[0, 1], [0, 2], [0, 3], [0, 4]]), { bond: L, iterations: 600 });
  for (const j of [1, 2, 3, 4]) assert.ok(Math.abs(d(out, 0, j) - L) < 0.15);
  const ang = (j) => Math.atan2(out.get(j)[1] - out.get(0)[1], out.get(j)[0] - out.get(0)[0]);
  const as = [1, 2, 3, 4].map(ang).sort((a, b) => a - b);
  const gaps = as.map((a, k) => (k + 1 < as.length ? as[k + 1] - a : as[0] + 2 * Math.PI - a));
  for (const g of gaps) assert.ok(Math.abs(g - Math.PI / 2) < 0.25, `gap ${g}`);
  const nitrile = relaxLayout(new Map([[0, [0, 0]], [1, [L, 0.3]], [2, [2 * L, 0.9]]]), sd([[0, 1], [1, 2, 3]]), { bond: L, iterations: 600 });
  assert.ok(Math.abs(d(nitrile, 0, 2) - 2 * L) < 0.12, "C-C#N is linear");
});
