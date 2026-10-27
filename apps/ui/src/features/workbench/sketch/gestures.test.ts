// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
import test from "node:test";
import assert from "node:assert/strict";
import { DEG, angleOf, dist, normAngle, sub } from "@ksmech/viz/sketch/geometry";
import { BOND_LENGTH, addAtom, addBond, emptySketch, findBond } from "@ksmech/viz/sketch/model";
import { anchorOf, dragModel, dragSnap, finishMove, FREE, snapEnd } from "./gestures.ts";
import { NO_SELECTION } from "./tools.ts";

const start = (base, tool, hit, at) => ({ tool, hit, at, base, selection: NO_SELECTION });
const xy = (m, id) => {
  const a = m.atoms.find((x) => x.id === id);
  return [a.x, a.y];
};

test("bond drag: one bond length at a 30-degree-snapped angle, free with Alt / Shift", () => {
  const [m, a] = addAtom(emptySketch(), "C", 0, 0);
  const out = dragModel(start(m, "bond", { kind: "atom", id: a }, [0, 0]), [3, 1.0], null);
  const end = out.atoms[1];
  assert.equal(Math.round(normAngle(angleOf([end.x, end.y])) / DEG), 30, "18.4 deg snaps to 30");
  assert.ok(Math.abs(dist([0, 0], [end.x, end.y]) - BOND_LENGTH) < 1e-9);
  const free = dragModel(start(m, "bond", { kind: "atom", id: a }, [0, 0]), [3, 1.0], null, { length: true, angle: false });
  assert.ok(Math.abs(angleOf([free.atoms[1].x, free.atoms[1].y]) - Math.atan2(1, 3)) < 1e-9);
});

test("bond drag onto an atom bonds to it; wedge drag keeps the narrow end at the start", () => {
  let [m, a] = addAtom(emptySketch(), "C", 0, 0);
  let b;
  [m, b] = addAtom(m, "O", 4, 0);
  const out = dragModel(start(m, "wedge", { kind: "atom", id: a }, [0, 0]), [3.9, 0.1], { kind: "atom", id: b });
  assert.equal(out.atoms.length, 2);
  const bond = findBond(out, a, b);
  assert.deepEqual([bond.a, bond.b, bond.stereo], [a, b, "wedge"]);
  assert.equal(dragModel(start(m, "bond", { kind: "bond", id: 99 }, [0, 0]), [1, 1], null), null);
});

test("select drag moves one atom; dropping it on another atom merges them", () => {
  let [m, a] = addAtom(emptySketch(), "C", 0, 0);
  let b;
  [m, b] = addAtom(m, "N", 3, 0);
  const d = start(m, "select", { kind: "atom", id: a }, [0, 0]);
  const moved = dragModel(d, [1, 1], null);
  assert.deepEqual(xy(moved, a), [1, 1]);
  assert.deepEqual(xy(moved, b), [3, 0]);
  const dropped = finishMove(d, dragModel(d, [3, 0], null), { kind: "atom", id: b });
  assert.deepEqual(dropped.atoms.map((x) => x.el), ["N"]);
});

test("chain drag makes a zigzag from the pressed atom", () => {
  const [m, a] = addAtom(emptySketch(), "C", 0, 0);
  const out = dragModel(start(m, "chain", { kind: "atom", id: a }, [0, 0]), [5.2, 0], null);
  assert.equal(out.bonds.length, 4);
  for (const bd of out.bonds) assert.ok(Math.abs(dist(xy(out, bd.a), xy(out, bd.b)) - BOND_LENGTH) < 1e-9);
  const ends = out.atoms.map((x) => sub([x.x, x.y], [0, 0]));
  assert.ok(ends.every((v) => v[0] >= -1e-9), "along the drag");
});

// ---- fixed lengths and fixed angles (ChemDraw's Object menu)

const angleDeg = (p, q) => Math.round(normAngle(angleOf(sub(q, p))) / DEG);
const FIXED = { length: true, angle: true };

test("snapEnd: fixed length and 30-degree steps; each constraint on its own; free", () => {
  const e = snapEnd([0, 0], [3, 2], FIXED);
  assert.equal(angleDeg([0, 0], e), 30);
  assert.ok(Math.abs(dist([0, 0], e) - BOND_LENGTH) < 1e-9);
  const lenOnly = snapEnd([0, 0], [3, 2], { length: true, angle: false });
  assert.ok(Math.abs(dist([0, 0], lenOnly) - BOND_LENGTH) < 1e-9);
  assert.ok(Math.abs(angleOf(lenOnly) - Math.atan2(2, 3)) < 1e-9);
  const angOnly = snapEnd([0, 0], [3, 2], { length: false, angle: true });
  assert.equal(angleDeg([0, 0], angOnly), 30);
  assert.ok(Math.abs(dist([0, 0], angOnly) - Math.hypot(3, 2)) < 1e-9);
  assert.deepEqual(snapEnd([0, 0], [3, 2], FREE), [3, 2]);
  for (const [x, y, deg] of [[-1, 0.2, 180], [0.1, -2, 270], [1, 1.2, 60], [-1, -0.5, 210]]) assert.equal(angleDeg([0, 0], snapEnd([0, 0], [x, y], FIXED)), deg);
});

test("dragSnap: Alt or Shift frees the drag; else the toggles", () => {
  assert.deepEqual(dragSnap(FIXED, { altKey: true, shiftKey: false }), FREE);
  assert.deepEqual(dragSnap(FIXED, { altKey: false, shiftKey: true }), FREE);
  const toggles = { length: false, angle: true };
  assert.equal(dragSnap(toggles, { altKey: false, shiftKey: false }), toggles);
});

test("bond drag from empty canvas: standard length at 30-degree steps; free length when off", () => {
  const d = start(emptySketch(), "bond", null, [0, 0]);
  const out = dragModel(d, [0.2, 4], null, FIXED);
  const [p, q] = out.atoms.map((a) => [a.x, a.y]);
  assert.equal(angleDeg(p, q), 90);
  assert.ok(Math.abs(dist(p, q) - BOND_LENGTH) < 1e-9);
  const long = dragModel(d, [0, 4], null, { length: false, angle: true });
  assert.ok(Math.abs(dist(...long.atoms.map((a) => [a.x, a.y])) - 4) < 1e-9);
});

test("select drag of a bond end: it snaps to its neighbour; free with FREE; joins an atom it is dropped on", () => {
  let [m, a] = addAtom(emptySketch(), "C", 0, 0);
  let b;
  let c;
  [m, b] = addAtom(m, "O", 1.5, 0);
  [m, c] = addAtom(m, "N", 0, 3);
  [m] = addBond(m, a, b);
  assert.equal(anchorOf(m, [b]), a);
  assert.equal(anchorOf(m, [a, b]), null);
  const d = start(m, "select", { kind: "atom", id: b }, [1.5, 0]);
  const snapped = dragModel(d, [1.1, 2.3], null, FIXED);
  assert.equal(angleDeg([0, 0], xy(snapped, b)), 60);
  assert.ok(Math.abs(dist([0, 0], xy(snapped, b)) - BOND_LENGTH) < 1e-9);
  assert.deepEqual(xy(dragModel(d, [1.1, 2.3], null, FREE), b), [1.1, 2.3]);
  const onto = dragModel(d, [0.1, 2.9], { kind: "atom", id: c }, FIXED);
  assert.deepEqual(xy(onto, b), [0, 3], "lands exactly on the target");
  const joined = finishMove(d, onto, { kind: "atom", id: c });
  assert.equal(joined.atoms.length, 2);
  assert.ok(findBond(joined, a, c), "the bond now ends on the target");
  const lone = start(m, "select", { kind: "atom", id: c }, [0, 3]);
  assert.deepEqual(xy(dragModel(lone, [0.3, 3.3], null, FIXED), c), [0.3, 3.3], "an atom without bonds moves freely");
});
