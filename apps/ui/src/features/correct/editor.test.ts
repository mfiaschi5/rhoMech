// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The R editor reducer: the bond and lone-pair cycles with their masks, two-atom bond forming,
// the arrow tool (compiled and kept for drawing), the eraser, the selection, undo / redo, clear.
import test from "node:test";
import assert from "node:assert/strict";
import { canRedo, canUndo, editorReducer, initEditor, keyAction, nextBondDelta, nextLoneDelta } from "./editor.ts";

/** O0 C1 Cl2 with a C1-Cl2 single bond, a C1=O3 double bond and a C4#N5 triple bond. */
const ORDERS = { "1-2": 1, "1-3": 2, "4-5": 3 };
const CTX = {
  n: 6,
  bondOrder: (i, j) => ORDERS[i < j ? `${i}-${j}` : `${j}-${i}`] ?? 0,
  lonePairs: (i) => [3, 0, 3, 2, 0, 1][i],
};
const run = (s, ...actions) => actions.reduce((st, a) => editorReducer(st, a, CTX), s);

test("bond cycles 0 -> -1 -> +1 -> 0, masked by the order bounds", () => {
  assert.deepEqual([0, -1, 1].map((v) => nextBondDelta(v, 1)), [-1, 1, 0]);
  assert.deepEqual([0, 1].map((v) => nextBondDelta(v, 0)), [1, 0], "no bond: no -1");
  assert.deepEqual([0, -1].map((v) => nextBondDelta(v, 3)), [-1, 0], "triple: no +1");
  let s = run(initEditor(), { type: "clickBond", i: 1, j: 2 });
  assert.deepEqual(s.matrix.bonds, [[1, 2, -1]]);
  s = run(s, { type: "clickBond", i: 2, j: 1 });
  assert.deepEqual(s.matrix.bonds, [[1, 2, 1]]);
  s = run(s, { type: "clickBond", i: 1, j: 2 });
  assert.deepEqual(s.matrix.bonds, []);
  s = run(s, { type: "clickBond", i: 4, j: 5 }, { type: "clickBond", i: 4, j: 5 });
  assert.deepEqual(s.matrix.bonds, [], "triple bond: -1 then back to 0");
});

test("two atoms clicked with the bond tool form a bond; the same atom twice cancels", () => {
  let s = run(initEditor(), { type: "clickAtom", i: 0 });
  assert.equal(s.pending, 0);
  s = run(s, { type: "clickAtom", i: 1 });
  assert.equal(s.pending, null);
  assert.deepEqual(s.matrix.bonds, [[0, 1, 1]]);
  s = run(s, { type: "clickAtom", i: 1 }, { type: "clickAtom", i: 0 });
  assert.deepEqual(s.matrix.bonds, [], "a non-bonded pair cycles 0 <-> +1");
  s = run(s, { type: "clickAtom", i: 2 }, { type: "clickAtom", i: 2 });
  assert.equal(s.pending, null);
  assert.equal(s.past.length, 2, "the cancelled pair is not an undo step");
});

test("lone pair cycles 0 -> -2 -> +2 -> 0; -2 is masked without a lone pair", () => {
  assert.deepEqual([0, -2, 2].map((v) => nextLoneDelta(v, 2)), [-2, 2, 0]);
  assert.deepEqual([0, 2].map((v) => nextLoneDelta(v, 0)), [2, 0]);
  assert.deepEqual([0, -2, 2].map((v) => nextLoneDelta(v, null)), [-2, 2, 0], "unknown element: unmasked");
  let s = run(initEditor(), { type: "tool", tool: "lone" }, { type: "clickAtom", i: 0 });
  assert.deepEqual(s.matrix.lone, [[0, -2]]);
  s = run(s, { type: "clickAtom", i: 1 });
  assert.deepEqual(s.matrix.lone, [[0, -2], [1, 2]], "C1 has no lone pair: straight to +2");
  assert.match(s.notice, /no lone pair/);
});

test("the arrow tool adds the compiled matrix and keeps the arrows while they add up to R", () => {
  let s = run(
    initEditor(),
    { type: "tool", tool: "arrow" },
    { type: "arrow", arrow: { kind: "lone_pair_to_bond", source: [0], target: [0, 1] } },
    { type: "arrow", arrow: { kind: "bond_to_lone_pair", source: [1, 2], target: [2] } },
  );
  assert.deepEqual(s.matrix, { bonds: [[0, 1, 1], [1, 2, -1]], lone: [[0, -2], [2, 2]] });
  assert.equal(s.arrows.length, 2);
  s = run(s, { type: "tool", tool: "lone" }, { type: "clickAtom", i: 3 });
  assert.equal(s.arrows.length, 0, "an edit outside the arrows drops them (R is the truth)");
  s = run(s, { type: "undo" });
  assert.equal(s.arrows.length, 2, "undo brings them back");
  const bad = run(s, { type: "arrow", arrow: { kind: "lone_pair_to_bond", source: [0], target: [1, 2] } });
  assert.deepEqual(bad.matrix, s.matrix);
  assert.ok(bad.notice);
});

test("the eraser removes an atom's entries, a bond's entry, or an arrow's matrix", () => {
  let s = run(
    initEditor(),
    { type: "tool", tool: "arrow" },
    { type: "arrow", arrow: { kind: "lone_pair_to_bond", source: [0], target: [0, 1] } },
    { type: "arrow", arrow: { kind: "bond_to_lone_pair", source: [1, 2], target: [2] } },
    { type: "tool", tool: "eraser" },
  );
  const arrowErased = run(s, { type: "clickArrow", index: 1 });
  assert.deepEqual(arrowErased.matrix, { bonds: [[0, 1, 1]], lone: [[0, -2]] });
  assert.equal(arrowErased.arrows.length, 1);
  s = run(s, { type: "clickAtom", i: 0 });
  assert.deepEqual(s.matrix, { bonds: [[1, 2, -1]], lone: [[2, 2]] });
  s = run(s, { type: "clickBond", i: 1, j: 2 });
  assert.deepEqual(s.matrix, { bonds: [], lone: [[2, 2]] });
});

test("selection, Delete, setEntry and clear", () => {
  let s = run(initEditor({ bonds: [[0, 1, 1], [1, 2, -1]], lone: [[0, -2], [2, 2]] }), { type: "select", keys: ["b:0-1", "l:2"] });
  assert.deepEqual([...s.selection].sort(), ["b:0-1", "l:2"]);
  s = run(s, { type: "select", keys: ["l:2"], additive: true });
  assert.deepEqual([...s.selection], ["b:0-1"], "additive select toggles");
  s = run(s, { type: "deleteSelection" });
  assert.deepEqual(s.matrix, { bonds: [[1, 2, -1]], lone: [[0, -2], [2, 2]] });
  assert.equal(s.selection.size, 0);
  s = run(s, { type: "setEntry", entry: { kind: "lone", i: 0, delta: -2 }, value: 0 });
  assert.deepEqual(s.matrix.lone, [[2, 2]]);
  s = run(s, { type: "clear" });
  assert.deepEqual(s.matrix, { bonds: [], lone: [] });
  assert.equal(s.past.length, 3);
});

test("undo / redo walk the history; a new edit drops the redo branch", () => {
  let s = run(initEditor(), { type: "clickBond", i: 1, j: 2 }, { type: "clickBond", i: 1, j: 3 });
  assert.ok(canUndo(s));
  s = run(s, { type: "undo" });
  assert.deepEqual(s.matrix.bonds, [[1, 2, -1]]);
  assert.ok(canRedo(s));
  s = run(s, { type: "redo" });
  assert.deepEqual(s.matrix.bonds, [[1, 2, -1], [1, 3, -1]]);
  s = run(s, { type: "undo" }, { type: "undo" });
  assert.deepEqual(s.matrix.bonds, []);
  assert.ok(!canUndo(s));
  s = run(s, { type: "undo" });
  assert.deepEqual(s.matrix.bonds, [], "undo on an empty history is a no-op");
  s = run(s, { type: "redo" }, { type: "clickBond", i: 4, j: 5 });
  assert.ok(!canRedo(s));
  assert.deepEqual(s.matrix.bonds, [[1, 2, -1], [4, 5, -1]]);
});

test("load replaces the matrix and resets the history; tools keep it", () => {
  let s = run(initEditor(), { type: "clickBond", i: 1, j: 2 }, { type: "tool", tool: "lone" });
  assert.equal(s.past.length, 1);
  s = run(s, { type: "load", matrix: { bonds: [[2, 1, 1]], lone: [] } });
  assert.deepEqual(s.matrix.bonds, [[1, 2, 1]]);
  assert.equal(s.past.length, 0);
  assert.equal(s.tool, "lone");
});

test("keys: tools, undo / redo, Delete, Enter; modified letters are not tools", () => {
  const k = (key, mods = {}) => keyAction({ key, ctrlKey: false, metaKey: false, shiftKey: false, altKey: false, ...mods });
  assert.deepEqual(["b", "l", "a", "e"].map((x) => k(x).tool), ["bond", "lone", "arrow", "eraser"]);
  assert.deepEqual(k("z", { ctrlKey: true }), { type: "undo" });
  assert.deepEqual(k("Z", { ctrlKey: true, shiftKey: true }), { type: "redo" });
  assert.deepEqual(k("y", { metaKey: true }), { type: "redo" });
  assert.deepEqual(k("Delete"), { type: "deleteSelection" });
  assert.deepEqual(k("Backspace"), { type: "deleteSelection" });
  assert.equal(k("Enter"), "apply");
  assert.equal(k("b", { ctrlKey: true }), null);
  assert.equal(k("x"), null);
});
