// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
import test from "node:test";
import assert from "node:assert/strict";
import { applyDepiction } from "./depiction.ts";
import { addAtom, addBond, centroid, emptySketch, setBondStereo } from "@ksmech/viz/sketch/model";

/** CH(OH)(F)CH3 drawn with ids that are not positions, and one old wedge. */
function sketch() {
  let m = emptySketch();
  const ids = [];
  for (const [el, x, y] of [["C", 5, 5], ["O", 6, 6], ["F", 4, 6], ["C", 5, 3.5]]) {
    let id;
    [m, id] = addAtom(m, el, x, y);
    ids.push(id);
  }
  const bonds = [];
  for (const j of [1, 2, 3]) {
    let b;
    [m, b] = addBond(m, ids[0], ids[j]);
    bonds.push(b);
  }
  m = setBondStereo(m, bonds[1], "hash");
  return { m, ids, bonds };
}

const OUT = {
  coords: [[0, 0], [1.3, 0.75], [-1.3, 0.75], [0, -1.5]],
  depth: null,
  wedges: [{ bond: 0, narrow: 0, kind: "wedge" }],
};

test("applyDepiction: coordinates by position, centroid kept, wedges replaced by bond index", () => {
  const { m, ids, bonds } = sketch();
  const out = applyDepiction(m, OUT);
  assert.deepEqual(out.atoms.map((a) => a.id), ids, "atom order and ids untouched");
  const c0 = centroid(m);
  const c1 = centroid(out);
  assert.ok(Math.abs(c0[0] - c1[0]) < 1e-9 && Math.abs(c0[1] - c1[1]) < 1e-9);
  const [a, o] = out.atoms;
  assert.ok(Math.abs(o.x - a.x - 1.3) < 1e-9 && Math.abs(o.y - a.y - 0.75) < 1e-9);
  const marks = out.bonds.map((b) => [b.id, b.a, b.stereo]);
  assert.deepEqual(marks, [
    [bonds[0], ids[0], "wedge"],
    [bonds[1], ids[0], "none"],
    [bonds[2], ids[0], "none"],
  ]);
  assert.equal(out.breakCrossings ?? false, false);
  assert.ok(out.atoms.every((x) => x.z === undefined));
});

test("applyDepiction: a narrow end on the far atom swaps the bond's ends; depth turns breaks on", () => {
  const { m, ids, bonds } = sketch();
  const out = applyDepiction(m, { ...OUT, depth: [0, 1, -1, 0.5], wedges: [{ bond: 1, narrow: 2, kind: "hash" }] });
  const b = out.bonds.find((x) => x.id === bonds[1]);
  assert.deepEqual([b.a, b.b, b.stereo], [ids[2], ids[0], "hash"]);
  assert.deepEqual(out.atoms.map((x) => x.z), [0, 1, -1, 0.5]);
  assert.equal(out.breakCrossings, true);
});

test("applyDepiction refuses an answer for another drawing", () => {
  const { m } = sketch();
  assert.equal(applyDepiction(m, { ...OUT, coords: OUT.coords.slice(0, 3) }), null);
  assert.equal(applyDepiction(m, { ...OUT, wedges: [{ bond: 7, narrow: 0, kind: "wedge" }] }), null);
  assert.equal(applyDepiction(m, { ...OUT, wedges: [{ bond: 0, narrow: 3, kind: "wedge" }] }), null);
});
