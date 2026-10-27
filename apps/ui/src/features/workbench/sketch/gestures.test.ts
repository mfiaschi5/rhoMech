// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
import test from "node:test";
import assert from "node:assert/strict";
import { DEG, angleOf, dist, normAngle, sub } from "@ksmech/viz/sketch/geometry";
import { BOND_LENGTH, addAtom, emptySketch, findBond } from "@ksmech/viz/sketch/model";
import { dragModel, finishMove } from "./gestures.ts";
import { NO_SELECTION } from "./tools.ts";

const start = (base, tool, hit, at) => ({ tool, hit, at, base, selection: NO_SELECTION });
const xy = (m, id) => {
  const a = m.atoms.find((x) => x.id === id);
  return [a.x, a.y];
};

test("bond drag: one bond length at a 30-degree-snapped angle, free with Shift", () => {
  const [m, a] = addAtom(emptySketch(), "C", 0, 0);
  const out = dragModel(start(m, "bond", { kind: "atom", id: a }, [0, 0]), [3, 1.0], null);
  const end = out.atoms[1];
  assert.equal(Math.round(normAngle(angleOf([end.x, end.y])) / DEG), 30, "18.4 deg snaps to 30");
  assert.ok(Math.abs(dist([0, 0], [end.x, end.y]) - BOND_LENGTH) < 1e-9);
  const free = dragModel(start(m, "bond", { kind: "atom", id: a }, [0, 0]), [3, 1.0], null, true);
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
