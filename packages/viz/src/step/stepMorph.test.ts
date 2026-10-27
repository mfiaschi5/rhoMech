// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The morph of the animated step: endpoints, easing, bond fades and cross-fades, the timeline.
import test from "node:test";
import assert from "node:assert/strict";
import { PHASES, bondAlpha, changedBonds, ease, morphFrame, morphPlan, phase } from "./stepMorph.ts";

const atom = (index, x, y, extra = {}) => ({ index, el: "C", charge: 0, hCount: 0, lone: 0, labelled: false, x, y, ...extra });
const side = (atoms, bonds) => ({ atoms: new Map(atoms.map((a) => [a.index, a])), bonds: new Map(bonds.map(([i, j, order]) => [`${i}-${j}`, { i, j, order }])) });

// O1 ... C5-Cl6 -> O1-C5 ... Cl6 (an SN2), plus C0-O1 kept
const before = side([atom(0, 0, 0), atom(1, 1.5, 0, { el: "O", charge: -1 }), atom(5, 4.5, 0), atom(6, 6, 0, { el: "Cl" })], [[0, 1, 1], [5, 6, 1]]);
const after = side([atom(0, 0, 0), atom(1, 1.5, 0, { el: "O" }), atom(5, 3, 0), atom(6, 6, 0, { el: "Cl", charge: -1 })], [[0, 1, 1], [1, 5, 1]]);
const to = new Map([[0, [0, 0]], [1, [1.5, 0]], [5, [3, 0.5]], [6, [6.5, 0]]]);
const plan = morphPlan(before, after, to);

test("plan: the union of both sides' bonds with their orders", () => {
  const byKey = Object.fromEntries(plan.bonds.map((b) => [b.key, [b.before, b.after]]));
  assert.deepEqual(byKey, { "0-1": [1, 1], "5-6": [1, 0], "1-5": [0, 1] });
  assert.deepEqual(changedBonds(plan).map((b) => b.key).sort(), ["1-5", "5-6"]);
  assert.deepEqual(plan.atoms, [0, 1, 5, 6]);
});

test("frame: t = 0 is the reactant, t = 1 the product, positions in between stay on the segment", () => {
  const f0 = morphFrame(plan, 0);
  const f1 = morphFrame(plan, 1);
  for (const i of plan.atoms) {
    assert.deepEqual(f0.xy.get(i), plan.from.get(i));
    assert.deepEqual(f1.xy.get(i), plan.to.get(i));
  }
  const mid = morphFrame(plan, 0.55).xy.get(5);
  assert.ok(mid[0] < 4.5 && mid[0] > 3 && mid[1] > 0 && mid[1] < 0.5);
  // outside [0, 1] clamps
  assert.deepEqual(morphFrame(plan, -1).xy.get(5), plan.from.get(5));
  assert.deepEqual(morphFrame(plan, 2).xy.get(5), plan.to.get(5));
});

test("frame: atoms hold still while the arrows draw, then move monotonically", () => {
  const atRest = morphFrame(plan, PHASES.move[0] - 0.01).xy.get(5);
  assert.deepEqual(atRest, plan.from.get(5));
  let last = Infinity;
  for (let t = 0; t <= 1.0001; t += 0.05) {
    const x = morphFrame(plan, t).xy.get(5)[0];
    assert.ok(x <= last + 1e-12);
    last = x;
  }
  assert.equal(morphFrame(plan, PHASES.arrows[1]).arrowDraw, 1);
  assert.equal(morphFrame(plan, 0).arrowDraw, 0);
});

test("bond fades: unchanged solid, broken out, formed in, order change cross-fades", () => {
  assert.deepEqual(bondAlpha({ before: 1, after: 1 }, 0.5), { before: 1, after: 0 });
  assert.equal(bondAlpha({ before: 1, after: 0 }, 0).before, 1);
  assert.equal(bondAlpha({ before: 1, after: 0 }, 1).before, 0);
  assert.equal(bondAlpha({ before: 0, after: 1 }, 0).after, 0);
  assert.equal(bondAlpha({ before: 0, after: 1 }, 1).after, 1);
  for (const t of [0, 0.3, 0.5, 0.7, 1]) {
    const a = bondAlpha({ before: 2, after: 1 }, t);
    assert.ok(Math.abs(a.before + a.after - 1) < 1e-9, "a cross-fade keeps the total opacity");
  }
});

test("labels switch to the product's at the charge time; lone pairs fade out then in", () => {
  assert.equal(morphFrame(plan, PHASES.charge - 0.01).productLabels, false);
  assert.equal(morphFrame(plan, PHASES.charge).productLabels, true);
  assert.equal(morphFrame(plan, 0).loneOut, 1);
  assert.equal(morphFrame(plan, 1).loneOut, 0);
  assert.equal(morphFrame(plan, 0).loneIn, 0);
  assert.equal(morphFrame(plan, 1).loneIn, 1);
  assert.equal(morphFrame(plan, 1).arrowAlpha, 0);
});

test("ease and phase", () => {
  assert.equal(ease(0), 0);
  assert.equal(ease(1), 1);
  assert.equal(ease(0.5), 0.5);
  assert.equal(phase(0.1, [0.2, 0.4]), 0);
  assert.ok(Math.abs(phase(0.3, [0.2, 0.4]) - 0.5) < 1e-12);
  assert.equal(phase(0.9, [0.2, 0.4]), 1);
});

test("an atom on one side only stays put", () => {
  const p = morphPlan(side([atom(0, 0, 0)], []), side([atom(0, 0, 0), atom(9, 2, 2)], []), new Map([[0, [1, 1]], [9, [2, 2]]]));
  assert.deepEqual(p.from.get(9), [2, 2]);
  assert.deepEqual(morphFrame(p, 0.5).xy.get(9), [2, 2]);
});
