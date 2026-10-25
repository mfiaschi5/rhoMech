// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The arrow -> R compiler (the spec's three elementary matrices and the SN2 worked example),
// gestures to readings with their masks, normalization, charge deltas and the local issues.
import test from "node:test";
import assert from "node:assert/strict";
import {
  addMatrix,
  arrowFromGesture,
  chargeDeltas,
  compileArrow,
  compileArrows,
  electronBalance,
  entriesOf,
  localIssues,
  normalize,
  removeEntries,
  setEntry,
} from "./rmatrix.ts";

/** Methoxide + chloromethane, atoms O0 C1 Cl2 (the spec's worked example, no H). */
const SN2 = {
  n: 3,
  bondOrder: (i, j) => ((i === 1 && j === 2) || (i === 2 && j === 1) ? 1 : 0),
  lonePairs: (i) => [3, 0, 3][i],
};

test("lone pair -> bond: R_aa -2, R_ab +1", () => {
  const c = compileArrow({ kind: "lone_pair_to_bond", source: [0], target: [0, 1] });
  assert.deepEqual(c, { ok: true, matrix: { bonds: [[0, 1, 1]], lone: [[0, -2]] } });
});

test("bond -> bond with the new bond at a: R_ac -1, R_ab +1 (pairs kept i < j)", () => {
  const c = compileArrow({ kind: "bond_to_bond", source: [5, 2], target: [5, 9] });
  assert.deepEqual(c, { ok: true, matrix: { bonds: [[2, 5, -1], [5, 9, 1]], lone: [] } });
});

test("bond -> lone pair: R_ac -1, R_kk +2", () => {
  const c = compileArrow({ kind: "bond_to_lone_pair", source: [1, 2], target: [2] });
  assert.deepEqual(c, { ok: true, matrix: { bonds: [[1, 2, -1]], lone: [[2, 2]] } });
});

test("malformed readings are refused", () => {
  assert.equal(compileArrow({ kind: "lone_pair_to_bond", source: [0], target: [1, 2] }).ok, false);
  assert.equal(compileArrow({ kind: "bond_to_bond", source: [0, 1], target: [0, 1] }).ok, false);
  assert.equal(compileArrow({ kind: "bond_to_lone_pair", source: [0, 1], target: [3] }).ok, false);
  assert.equal(compileArrow({ kind: "bond_to_bond", source: [0], target: [0, 1] }).ok, false);
});

test("the SN2 step is the sum of its two arrows and conserves electrons", () => {
  const c = compileArrows([
    { kind: "lone_pair_to_bond", source: [0], target: [0, 1] },
    { kind: "bond_to_lone_pair", source: [1, 2], target: [2] },
  ]);
  assert.ok(c.ok);
  assert.deepEqual(c.matrix, { bonds: [[0, 1, 1], [1, 2, -1]], lone: [[0, -2], [2, 2]] });
  assert.equal(electronBalance(c.matrix), 0);
  assert.deepEqual([...chargeDeltas(c.matrix)].sort(), [[0, 1], [2, -1]]);
  assert.deepEqual(localIssues(c.matrix, SN2), []);
});

test("normalize merges, orders, drops zeros and sorts", () => {
  const m = normalize({ bonds: [[3, 1, 1], [1, 3, -1], [2, 0, 1], [4, 4, 1]], lone: [[2, 2], [2, -2], [1, -2]] });
  assert.deepEqual(m, { bonds: [[0, 2, 1]], lone: [[1, -2]] });
  assert.deepEqual(addMatrix(m, m, -1), { bonds: [], lone: [] });
});

test("setEntry and removeEntries edit single entries", () => {
  let m = setEntry({ bonds: [], lone: [] }, { kind: "bond", i: 4, j: 2, delta: 0 }, 1);
  m = setEntry(m, { kind: "lone", i: 4, delta: 0 }, -2);
  assert.deepEqual(m, { bonds: [[2, 4, 1]], lone: [[4, -2]] });
  assert.deepEqual(entriesOf(m).map((e) => e.kind), ["bond", "lone"]);
  assert.deepEqual(removeEntries(m, new Set(["b:2-4"])), { bonds: [], lone: [[4, -2]] });
});

test("gestures: atom -> atom is a lone pair forming a bond", () => {
  const g = arrowFromGesture({ kind: "atom", i: 0 }, { kind: "atom", i: 1 }, SN2);
  assert.deepEqual(g, { ok: true, arrow: { kind: "lone_pair_to_bond", source: [0], target: [0, 1] } });
  assert.equal(arrowFromGesture({ kind: "atom", i: 1 }, { kind: "atom", i: 0 }, SN2).ok, false, "C1 has no lone pair");
  assert.equal(arrowFromGesture({ kind: "atom", i: 0 }, { kind: "atom", i: 0 }, SN2).ok, false);
});

test("gestures: a bond dropped on its own atom is a leaving group", () => {
  const g = arrowFromGesture({ kind: "bond", i: 1, j: 2 }, { kind: "atom", i: 2 }, SN2);
  assert.deepEqual(g, { ok: true, arrow: { kind: "bond_to_lone_pair", source: [1, 2], target: [2] } });
  assert.equal(arrowFromGesture({ kind: "bond", i: 0, j: 1 }, { kind: "atom", i: 2 }, SN2).ok, false, "O0-C1 is no bond");
});

test("gestures: a bond to another atom forms the bond at the nearer end", () => {
  const ctx = { n: 4, bondOrder: (i, j) => (Math.min(i, j) === 0 && Math.max(i, j) === 1 ? 2 : 0), lonePairs: () => 0 };
  const near1 = arrowFromGesture({ kind: "bond", i: 0, j: 1, near: 1 }, { kind: "atom", i: 3 }, ctx);
  assert.deepEqual(near1.arrow, { kind: "bond_to_bond", source: [1, 0], target: [1, 3] });
  const near0 = arrowFromGesture({ kind: "bond", i: 0, j: 1 }, { kind: "atom", i: 3 }, ctx);
  assert.deepEqual(near0.arrow, { kind: "bond_to_bond", source: [0, 1], target: [0, 3] });
});

test("gestures: bond -> bond needs a shared atom; lone pair -> own bond raises its order", () => {
  const ctx = { n: 4, bondOrder: (i, j) => (Math.abs(i - j) === 1 ? 1 : 0), lonePairs: () => 1 };
  const g = arrowFromGesture({ kind: "bond", i: 1, j: 2 }, { kind: "bond", i: 0, j: 1 }, ctx);
  assert.deepEqual(g.arrow, { kind: "bond_to_bond", source: [1, 2], target: [1, 0] });
  assert.equal(arrowFromGesture({ kind: "bond", i: 0, j: 1 }, { kind: "bond", i: 2, j: 3 }, ctx).ok, false);
  const lp = arrowFromGesture({ kind: "atom", i: 2 }, { kind: "bond", i: 1, j: 2 }, ctx);
  assert.deepEqual(lp.arrow, { kind: "lone_pair_to_bond", source: [2], target: [2, 1] });
  assert.equal(arrowFromGesture({ kind: "atom", i: 0 }, { kind: "bond", i: 2, j: 3 }, ctx).ok, false);
});

test("gestures are masked by the triple-bond bound", () => {
  const ctx = { n: 3, bondOrder: (i, j) => (i + j === 1 ? 3 : 0), lonePairs: () => 1 };
  assert.equal(arrowFromGesture({ kind: "atom", i: 0 }, { kind: "atom", i: 1 }, ctx).ok, false);
});

test("local issues: ranges, bounds, missing lone pair, electron balance", () => {
  const codes = (m) => localIssues(m, SN2).map((i) => i.code).sort();
  assert.deepEqual(codes({ bonds: [[1, 2, 4]], lone: [[0, -3]] }), ["BOND_ORDER_RANGE", "CHARGE_NOT_CONSERVED", "MATRIX_MALFORMED", "MATRIX_MALFORMED"]);
  assert.deepEqual(codes({ bonds: [[0, 1, 2]], lone: [[0, -4]] }), [], "±2 bonds and ±4 lone electrons are allowed");
  assert.deepEqual(codes({ bonds: [[0, 1, -1]], lone: [[0, 2]] }), ["BREAK_UNBONDED"]);
  assert.deepEqual(codes({ bonds: [[1, 2, -2]], lone: [[1, 2], [2, 2]] }), ["BOND_ORDER_RANGE"]);
  assert.deepEqual(codes({ bonds: [], lone: [[1, -2], [2, 2]] }), ["NO_LONE_PAIR"]);
  assert.deepEqual(codes({ bonds: [[0, 1, 1]], lone: [] }), ["CHARGE_NOT_CONSERVED"]);
  assert.deepEqual(codes({ bonds: [[0, 7, 1]], lone: [[0, -2]] }), ["MATRIX_MALFORMED"]);
});
