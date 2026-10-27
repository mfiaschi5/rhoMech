// @ts-nocheck -- run by `node --experimental-strip-types --test` (tsconfig has no node types
// and no allowImportingTsExtensions); node strips the types without checking them.
import test from "node:test";
import assert from "node:assert/strict";
import {
  BOND_LENGTH,
  addAtom,
  addBond,
  cycleBondOrder,
  emptySketch,
  mergeAtoms,
  neighbors,
  removeBonds,
  setBondStereo,
  smallestRingThrough,
} from "@ksmech/viz/sketch/model";
import { initHistory, pushHistory, undo, redo, HISTORY_CAP, canUndo, canRedo } from "./history.ts";
import { applyClick, applyKey, NO_SELECTION, hitTest } from "./tools.ts";
import { fallbackHCount, hillFormula } from "@ksmech/viz/sketch/elements";

const ethane = () => {
  let [m, a] = addAtom(emptySketch(), "C", 0, 0);
  let b;
  [m, b] = addAtom(m, "C", BOND_LENGTH, 0);
  [m] = addBond(m, a, b);
  return { m, a, b };
};

test("ids come from one counter and are not reused", () => {
  const { m, a, b } = ethane();
  assert.deepEqual([a, b, m.bonds[0].id, m.nextId], [1, 2, 3, 4]);
  assert.deepEqual(neighbors(m, a), [b]);
});

test("addBond does not duplicate and refuses self bonds", () => {
  const { m, a, b } = ethane();
  const [m2, id] = addBond(m, b, a, 2);
  assert.equal(m2, m);
  assert.equal(id, m.bonds[0].id);
  assert.throws(() => addBond(m, a, a));
});

test("bond order cycles 1 -> 2 -> 3 -> 1 and stereo flips on the second click", () => {
  const { m } = ethane();
  const id = m.bonds[0].id;
  let x = cycleBondOrder(m, id);
  assert.equal(x.bonds[0].order, 2);
  x = cycleBondOrder(cycleBondOrder(x, id), id);
  assert.equal(x.bonds[0].order, 1);
  const w = setBondStereo(m, id, "wedge");
  assert.equal(w.bonds[0].stereo, "wedge");
  const flipped = setBondStereo(w, id, "wedge");
  assert.deepEqual([flipped.bonds[0].a, flipped.bonds[0].b], [w.bonds[0].b, w.bonds[0].a]);
});

test("erasing a bond drops orphan plain carbons but keeps heteroatoms", () => {
  let [m, a] = addAtom(emptySketch(), "C", 0, 0);
  let o;
  [m, o] = addAtom(m, "O", 1.5, 0);
  [m] = addBond(m, a, o);
  const after = removeBonds(m, [m.bonds[0].id]);
  assert.deepEqual(after.atoms.map((x) => x.el), ["O"]);
});

test("mergeAtoms moves bonds and drops duplicates", () => {
  let { m, a, b } = ethane();
  let c;
  [m, c] = addAtom(m, "C", 3, 0);
  [m] = addBond(m, b, c);
  [m] = addBond(m, a, c); // triangle a-b-c
  const merged = mergeAtoms(m, a, c);
  assert.equal(merged.atoms.length, 2);
  assert.equal(merged.bonds.length, 1);
});

test("smallest ring through a bond", () => {
  let m = applyClick(emptySketch(), "cyclohexane", "C", null, [0, 0]);
  const b = m.bonds[0];
  const ring = smallestRingThrough(m, b.a, b.b);
  assert.equal(ring.length, 6);
  assert.deepEqual(ring.slice(0, 2), [b.a, b.b]);
  const { m: chain } = ethane();
  assert.equal(smallestRingThrough(chain, 1, 2), null);
});

test("history: undo/redo, redo stack dropped on push, cap", () => {
  let h = initHistory(0);
  h = pushHistory(h, 1);
  h = pushHistory(h, 2);
  h = undo(h);
  assert.equal(h.present, 1);
  assert.ok(canRedo(h));
  h = redo(h);
  assert.equal(h.present, 2);
  h = undo(h);
  h = pushHistory(h, 5);
  assert.equal(h.present, 5);
  assert.ok(!canRedo(h));
  assert.equal(pushHistory(h, 5), h, "pushing the present is a no-op");
  for (let i = 0; i < HISTORY_CAP + 50; i++) h = pushHistory(h, 100 + i);
  assert.equal(h.past.length, HISTORY_CAP);
  let n = 0;
  while (canUndo(h)) {
    h = undo(h);
    n++;
  }
  assert.equal(n, HISTORY_CAP);
});

test("ethanol by clicks: bond on empty, bond on the end atom, o key", () => {
  let m = applyClick(emptySketch(), "bond", "C", null, [0, 0]);
  assert.equal(m.atoms.length, 2);
  const end = m.atoms[1].id;
  m = applyClick(m, "bond", "C", { kind: "atom", id: end }, [0, 0]);
  assert.equal(m.atoms.length, 3);
  m = applyKey(m, "o", { kind: "atom", id: m.atoms[2].id }, NO_SELECTION);
  assert.deepEqual(m.atoms.map((a) => a.el), ["C", "C", "O"]);
  assert.equal(applyKey(m, "q", { kind: "atom", id: 1 }, NO_SELECTION), null);
});

test("keys on a hovered bond set its order; Delete erases", () => {
  let m = applyClick(emptySketch(), "bond", "C", null, [0, 0]);
  const hit = { kind: "bond", id: m.bonds[0].id };
  assert.equal(applyKey(m, "3", hit, NO_SELECTION).bonds[0].order, 3);
  assert.equal(applyKey(m, "Delete", hit, NO_SELECTION).atoms.length, 0);
  assert.equal(applyKey(m, "+", { kind: "atom", id: 1 }, NO_SELECTION).atoms[0].charge, 1);
});

test("hit test prefers atoms over bonds", () => {
  const { m } = ethane();
  const view = { s: 20, ox: 100, oy: 100 };
  assert.deepEqual(hitTest(m, view, [101, 99]), { kind: "atom", id: 1 });
  assert.deepEqual(hitTest(m, view, [115, 102]), { kind: "bond", id: 3 });
  assert.equal(hitTest(m, view, [115, 130]), null);
});

test("fallback H counts and Hill formulas", () => {
  assert.equal(fallbackHCount("C", 0, 1), 3);
  assert.equal(fallbackHCount("N", 1, 1), 3);
  assert.equal(fallbackHCount("O", -1, 1), 0);
  assert.equal(fallbackHCount("C", 1, 2), 1);
  assert.equal(fallbackHCount("Fe", 0, 0), 0);
  assert.equal(hillFormula(new Map([["O", 1], ["H", 6], ["C", 2]])), "C2H6O");
  assert.equal(hillFormula(new Map([["Cl", 1], ["H", 1]])), "ClH");
  assert.equal(hillFormula(new Map([["C", 2], ["H", 3], ["O", 2]]), -1), "C2H3O2-");
});
