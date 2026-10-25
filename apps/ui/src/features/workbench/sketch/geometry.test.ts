// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
import test from "node:test";
import assert from "node:assert/strict";
import { DEG, angleOf, chainPoints, dist, growthAngle, largestGapDirection, normAngle, polygonOnEdge, sub } from "./geometry.ts";
import { BOND_LENGTH, addAtom, addBond, atomById, emptySketch, findBond, neighbors } from "./model.ts";
import { addChain, addRing, bondBetweenPoints, bondFromPoint, growBond } from "./templates.ts";

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const deg = (t) => Math.round(normAngle(t) / DEG);
const xy = (m, id) => {
  const a = atomById(m, id);
  return [a.x, a.y];
};

test("growth angle: isolated 30 deg, 120 deg from one neighbour, zigzag trans", () => {
  assert.equal(deg(growthAngle([0, 0], [], [], 1.5)), 30);
  // neighbour to the left: +-120 deg are equally horizontal, the upward one wins
  assert.equal(deg(growthAngle([0, 0], [[-1.5, 0]], [], 1.5)), 60);
  // propane zigzag: growing from the end of a 30-deg bond goes to -30 deg (trans)
  let g = bondFromPoint(emptySketch(), [0, 0]);
  assert.equal(deg(angleOf(sub(xy(g.model, g.atom), [0, 0]))), 30);
  const second = growBond(g.model, g.atom);
  const v = sub(xy(second.model, second.atom), xy(second.model, g.atom));
  assert.equal(deg(angleOf(v)), 330);
  const third = growBond(second.model, second.atom);
  const w = sub(xy(third.model, third.atom), xy(third.model, second.atom));
  assert.equal(deg(angleOf(w)), 30, "trans to the first carbon");
  for (const b of third.model.bonds) close(dist(xy(third.model, b.a), xy(third.model, b.b)), BOND_LENGTH);
});

test("growth from two neighbours bisects the largest gap", () => {
  // neighbours at 90 and 210 deg: the free gap 210..450 has its bisector at 330 deg
  const t = growthAngle([0, 0], [[0, 1.5], [-1.3, -0.75]], [], 1.5);
  assert.equal(deg(t), 330);
  assert.equal(deg(largestGapDirection([0, 0], [[1, 0]])), 180);
});

test("growing from a centre with three neighbours fills the fourth position", () => {
  let m = bondFromPoint(emptySketch(), [0, 0]).model;
  const c = m.atoms[0].id;
  m = growBond(m, c).model;
  m = growBond(m, c).model;
  const g = growBond(m, c);
  assert.equal(neighbors(g.model, c).length, 4);
  const pts = neighbors(g.model, c).map((n) => xy(g.model, n));
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++) assert.ok(dist(pts[i], pts[j]) > 1.0, "no overlapping substituents");
});

test("a bond dragged onto an existing atom closes the ring instead of stacking atoms", () => {
  // cyclohexane minus one bond: growing from one end goes outward (no auto-closure), a drag
  // released near the other end joins it
  let m = addRing(emptySketch(), "cyclohexane", { kind: "empty", at: [0, 0] }).model;
  const last = m.bonds[m.bonds.length - 1];
  m = { ...m, bonds: m.bonds.filter((b) => b.id !== last.id) };
  assert.equal(growBond(m, last.a).model.atoms.length, 7);
  const [bx, by] = xy(m, last.b);
  const g = bondBetweenPoints(m, last.a, [bx + 0.1, by - 0.1]);
  assert.equal(g.model.atoms.length, 6);
  assert.ok(findBond(g.model, last.a, last.b));
});

test("polygon on an edge keeps the edge and the side", () => {
  const pts = polygonOnEdge([0, 0], [1.5, 0], 6, 1);
  assert.deepEqual(pts.slice(0, 2), [[0, 0], [1.5, 0]]);
  assert.ok(pts.slice(2).every((p) => p[1] > 0));
  for (let k = 0; k < 6; k++) close(dist(pts[k], pts[(k + 1) % 6]), 1.5);
  assert.ok(polygonOnEdge([0, 0], [1.5, 0], 5, -1).slice(2).every((p) => p[1] < 0));
});

test("benzene on empty canvas: 6 atoms, alternating 3 double bonds", () => {
  const { model } = addRing(emptySketch(), "benzene", { kind: "empty", at: [0, 0] });
  assert.equal(model.atoms.length, 6);
  assert.deepEqual(model.bonds.map((b) => b.order).sort(), [1, 1, 1, 2, 2, 2]);
  for (const a of model.atoms) assert.equal(model.bonds.filter((b) => (b.a === a.id || b.b === a.id) && b.order === 2).length, 1);
});

test("ring fusion: naphthalene from benzene + benzene on a single bond", () => {
  let { model } = addRing(emptySketch(), "benzene", { kind: "empty", at: [0, 0] });
  const shared = model.bonds.find((b) => b.order === 1);
  const fused = addRing(model, "benzene", { kind: "bond", id: shared.id });
  const m = fused.model;
  assert.equal(m.atoms.length, 10);
  assert.equal(m.bonds.length, 11);
  assert.equal(m.bonds.filter((b) => b.order === 2).length, 5);
  for (const a of m.atoms) {
    const doubles = m.bonds.filter((b) => (b.a === a.id || b.b === a.id) && b.order === 2).length;
    assert.equal(doubles, 1, `atom ${a.id} has ${doubles} double bonds`);
  }
  // the new ring is on the other side of the shared bond: no atom lands on an old one
  for (const a of m.atoms) for (const b of m.atoms) if (a.id < b.id) assert.ok(dist([a.x, a.y], [b.x, b.y]) > 1.0);
});

test("ring fusion on a bond with a substituent goes away from it; spiro on an atom", () => {
  let { model } = addRing(emptySketch(), "cyclohexane", { kind: "empty", at: [0, 0] });
  const fused = addRing(model, "cyclopentane", { kind: "bond", id: model.bonds[0].id });
  assert.equal(fused.model.atoms.length, 9);
  assert.equal(fused.model.bonds.length, 10);
  const spiro = addRing(model, "cyclopentane", { kind: "atom", id: model.atoms[0].id });
  assert.equal(spiro.model.atoms.length, 10);
  assert.equal(neighbors(spiro.model, model.atoms[0].id).length, 4);
  for (const a of spiro.model.atoms)
    for (const b of spiro.model.atoms) if (a.id < b.id) assert.ok(dist([a.x, a.y], [b.x, b.y]) > 1.0);
});

test("ring on a substituted atom points away from the substituent", () => {
  let [m, a] = addAtom(emptySketch(), "C", 0, 0);
  let b;
  [m, b] = addAtom(m, "O", -1.5, 0);
  [m] = addBond(m, a, b);
  const r = addRing(m, "benzene", { kind: "atom", id: a });
  const xs = r.atoms.map((id) => xy(r.model, id)[0]);
  assert.ok(xs.every((x) => x >= -1e-9), "ring to the right of the O substituent");
});

test("chain points zigzag at bond length along the snapped drag", () => {
  const pts = chainPoints([0, 0], [6.1, 0.3], 1.5);
  assert.equal(pts.length, 5);
  let prev = [0, 0];
  for (const p of pts) {
    close(dist(prev, p), 1.5);
    prev = p;
  }
  const { model, atoms } = addChain(emptySketch(), [0, 0], pts);
  assert.equal(atoms.length, 6);
  assert.equal(model.bonds.length, 5);
});
