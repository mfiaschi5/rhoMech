// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
import test from "node:test";
import assert from "node:assert/strict";
import { ANCHOR_HAND, ANCHOR_PROJECTION, fixLayout, layoutMetrics } from "./layoutFix.ts";
import { projectedLayout, tilt } from "./projectLayout.ts";
import { BOND_LENGTH as L, addAtom, addBond, emptySketch, setBondStereo } from "./model.ts";
import { addChain, addRing } from "./templates.ts";
import { drawnParities } from "./stereo.ts";
import { applyClick } from "./tools.ts";
import { dist } from "./geometry.ts";

/** Atoms [el, x, y] and bonds [i, j, order] (indices) as a sketch. */
function sketch(atoms, bonds) {
  let m = emptySketch();
  const ids = [];
  for (const [el, x, y] of atoms) {
    let id;
    [m, id] = addAtom(m, el, x * L, y * L);
    ids.push(id);
  }
  for (const [i, j, o = 1] of bonds) [m] = addBond(m, ids[i], ids[j], o);
  return { m, ids };
}

/** What must never change: atoms (id, element, charge) and bonds (unordered ends, order). */
const topology = (m) => ({
  atoms: m.atoms.map((a) => [a.id, a.el, a.charge]),
  bonds: m.bonds.map((b) => [b.id, Math.min(b.a, b.b), Math.max(b.a, b.b), b.order]),
});

const defects = (x) => x.labelOverlaps + x.closeContacts + x.atomOnBond + x.smallAngles;

const BAD = {
  // ethylene glycol folded so that the two OH labels sit on each other
  glycolFold: sketch(
    [["O", 0, 0], ["C", 0.86, 0.5], ["C", 1.72, 0], ["O", 0.15, 0.08]],
    [[0, 1], [1, 2], [2, 3]],
  ),
  // propane with a methylamine N lying on the C1-C2 bond
  nOnBond: sketch(
    [["C", 0, 0], ["C", 1, 0], ["C", 1.5, 0.87], ["N", 0.5, 0.03], ["C", 0.5, -1]],
    [[0, 1], [1, 2], [3, 4]],
  ),
  // a bond that points along the view axis (zero length) and two bonds 10 deg apart
  collapsed: sketch(
    [["C", 0, 0], ["C", 0, 0], ["C", 1, 0], ["Cl", 0.98, 0.17]],
    [[0, 1], [0, 2], [0, 3]],
  ),
  // a charged nitrogen whose label box (NH3+) covers a neighbouring chain bond
  labelOnBond: sketch(
    [["C", 0, 0], ["C", 1, 0], ["C", 2, 0], ["N", 1, 0.12], ["C", 1.6, 0.8]],
    [[0, 1], [1, 2], [3, 4]],
  ),
};
BAD.labelOnBond.m = { ...BAD.labelOnBond.m, atoms: BAD.labelOnBond.m.atoms.map((a) => (a.el === "N" ? { ...a, charge: 1 } : a)) };

test("constructed bad layouts: every defect fixed, bonds near BOND_LENGTH, topology kept", () => {
  for (const [name, { m }] of Object.entries(BAD)) {
    const r = fixLayout(m, { anchor: ANCHOR_HAND });
    assert.ok(defects(r.before) > 0, `${name}: the case is bad to begin with`);
    assert.equal(r.after.labelOverlaps, 0, `${name}: label overlaps ${JSON.stringify(r.after)}`);
    assert.equal(r.after.closeContacts, 0, `${name}: close contacts`);
    assert.equal(r.after.atomOnBond, 0, `${name}: atom on bond`);
    assert.equal(r.after.smallAngles, 0, `${name}: small angles`);
    assert.ok(r.after.bondLengthRms < 0.05, `${name}: bond RMS ${r.after.bondLengthRms}`);
    assert.ok(r.iterations <= 300);
    assert.deepEqual(topology(r.model), topology(m), `${name}: topology`);
  }
});

test("the collapsed bond opens to a full bond, deterministically", () => {
  const { m, ids } = BAD.collapsed;
  const a = fixLayout(m, { anchor: ANCHOR_PROJECTION });
  const b = fixLayout(m, { anchor: ANCHOR_PROJECTION });
  assert.deepEqual(a.model, b.model);
  const p = a.model.atoms.find((x) => x.id === ids[0]);
  const q = a.model.atoms.find((x) => x.id === ids[1]);
  assert.ok(Math.abs(dist([p.x, p.y], [q.x, q.y]) - L) < 0.08 * L);
});

test("a clean drawing is left (almost) where it is", () => {
  const m = addRing(emptySketch(), "benzene", { kind: "empty", at: [0, 0] }).model;
  const r = fixLayout(m, { anchor: ANCHOR_HAND });
  assert.equal(defects(r.before), 0);
  const moved = Math.max(...r.model.atoms.map((a, i) => dist([a.x, a.y], [m.atoms[i].x, m.atoms[i].y])));
  assert.ok(moved < 0.02 * L, `moved ${moved}`);
});

test("Fix layout on a hand drawing: a bridge drawn across a ring, wedge meaning kept", () => {
  // cyclohexane with a two-carbon chain drawn from the top atom across the ring onto the bottom
  // atom (which becomes O: a flat-drawn oxabicycle) and a wedged methyl placed right on top of a
  // ring atom
  let m = applyClick(emptySketch(), "cyclohexane", "C", null, [0, 0]);
  const top = m.atoms.reduce((p, a) => (a.y > p.y ? a : p));
  const ch = addChain(m, top.id, [[0.3, 0.4], [-0.2, -0.6], [0.25, -1.7]]);
  m = ch.model;
  const end = ch.atoms[ch.atoms.length - 1];
  m = { ...m, atoms: m.atoms.map((a) => (a.id === end ? { ...a, el: "O" } : a)) };
  // a stereocentre: the top atom gets a wedge to a methyl hidden on the ring atom at (1.3, 0.75)
  const c1 = ch.atoms[0];
  let me;
  [m, me] = addAtom(m, "C", 1.4, 0.7);
  [m] = addBond(m, c1, me);
  m = setBondStereo(m, m.bonds[m.bonds.length - 1].id, "wedge");
  const before = layoutMetrics(m);
  assert.ok(before.closeContacts + before.atomOnBond > 0, JSON.stringify(before));
  const r = fixLayout(m, { anchor: ANCHOR_HAND });
  assert.equal(defects(r.after), 0, JSON.stringify(r.after));
  assert.ok(r.after.bondLengthRms < 0.06);
  assert.ok(r.stereoKept);
  assert.deepEqual(drawnParities(r.model), drawnParities(m));
  assert.deepEqual(topology(r.model), topology(m));
});

test("tilt is a proper rotation of x, y and depth; untilted views are untouched", () => {
  let { m } = sketch([["C", 0, 0], ["C", 1, 0]], [[0, 1]]);
  m = { ...m, atoms: m.atoms.map((a, i) => ({ ...a, z: i ? 0.5 : -0.5 })) };
  assert.equal(tilt(m, 0, 0), m);
  const t = tilt(m, 10, -12);
  const len3 = (x) => Math.hypot(x.atoms[0].x - x.atoms[1].x, x.atoms[0].y - x.atoms[1].y, x.atoms[0].z - x.atoms[1].z);
  assert.ok(Math.abs(len3(t) - len3(m)) < 1e-9);
});

test("projectedLayout keeps the exact view when it has no defect", () => {
  let { m } = sketch([["C", 0, 0], ["C", 0.87, 0.5], ["O", 1.74, 0]], [[0, 1], [1, 2]]);
  m = { ...m, atoms: m.atoms.map((a) => ({ ...a, z: 0 })) };
  const r = projectedLayout(m, []);
  assert.deepEqual(r.tilt, [0, 0]);
  assert.equal(defects(r.after), 0);
});
