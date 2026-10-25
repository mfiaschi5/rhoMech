// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
import test from "node:test";
import assert from "node:assert/strict";
import { findCrossings, segmentCrossing } from "./crossings.ts";
import { BREAK_GAP_PX, buildScene } from "./drawing.ts";
import { addAtom, addBond, emptySketch, setBondStereo, setBreakCrossings } from "./model.ts";
import { dist } from "./geometry.ts";

/** Two bonds crossing at the origin: a-b horizontal (first), c-d vertical (second); z by id. */
function cross(z = null, orders = [1, 1]) {
  let m = emptySketch();
  const at = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  const ids = [];
  at.forEach(([x, y], i) => {
    let id;
    [m, id] = addAtom(m, "C", x, y);
    ids.push(id);
    if (z) m = { ...m, atoms: m.atoms.map((a) => (a.id === id ? { ...a, z: z[i] } : a)) };
  });
  [m] = addBond(m, ids[0], ids[1], orders[0]);
  [m] = addBond(m, ids[2], ids[3], orders[1]);
  return { m, ids, h: m.bonds[0].id, v: m.bonds[1].id };
}

const VIEW = { s: 40, ox: 100, oy: 100 };

test("segmentCrossing: proper crossings only", () => {
  assert.deepEqual(segmentCrossing([-1, 0], [1, 0], [0, -1], [0, 1]), [0.5, 0.5]);
  assert.equal(segmentCrossing([-1, 0], [1, 0], [0, 0], [0, 1]), null, "touching at an end");
  assert.equal(segmentCrossing([-1, 0], [1, 0], [-1, 1], [1, 1]), null, "parallel");
  assert.equal(segmentCrossing([-1, 0], [1, 0], [2, -1], [2, 1]), null, "beyond the end");
});

test("crossings skip bonds sharing an atom; front is the later bond without depth", () => {
  const { m, h, v } = cross();
  const c = findCrossings(m);
  assert.equal(c.length, 1);
  assert.deepEqual([c[0].front, c[0].back], [v, h]);
  assert.deepEqual(c[0].at, [0, 0]);
  // a star: three bonds from one atom never cross each other
  let s = emptySketch();
  let o;
  [s, o] = addAtom(s, "C", 0, 0);
  for (const [x, y] of [[1, 0], [-1, 0.2], [0, -1]]) {
    let id;
    [s, id] = addAtom(s, "C", x, y);
    [s] = addBond(s, o, id);
  }
  assert.equal(findCrossings(s).length, 0);
});

test("with depth the bond nearer the viewer at the crossing is in front", () => {
  const front = cross([1, 1, -1, -1]); // horizontal bond at z = 1, vertical at -1
  assert.deepEqual([findCrossings(front.m)[0].front, findCrossings(front.m)[0].back], [front.h, front.v]);
  const back = cross([-1, -1, 1, 1]);
  assert.equal(findCrossings(back.m)[0].front, back.v);
  // interpolated: the vertical bond runs from z = -3 to z = 5, so z = 1 at the crossing; the
  // horizontal bond at 0.5 is behind it
  const mixed = cross([0.5, 0.5, -3, 5]);
  assert.equal(findCrossings(mixed.m)[0].front, mixed.v);
});

test("the bond behind is drawn with a gap around the front bond; off without breakCrossings", () => {
  const { m, h, v } = cross();
  const off = buildScene(m, VIEW, null);
  assert.ok(off.bonds.every((b) => b.broken === 0 && b.shape.lines.length === 1));
  const on = buildScene(setBreakCrossings(m, true), VIEW, null);
  const back = on.bonds.find((b) => b.bond.id === h);
  const front = on.bonds.find((b) => b.bond.id === v);
  assert.equal(back.broken, 1);
  assert.equal(front.broken, 0);
  assert.equal(front.shape.lines.length, 1);
  assert.equal(back.shape.lines.length, 2);
  const [l, r] = back.shape.lines;
  const centre = [100, 100];
  assert.ok(dist(l.q, centre) >= BREAK_GAP_PX && dist(r.p, centre) >= BREAK_GAP_PX, "gap on each side");
  assert.ok(dist(l.q, centre) < BREAK_GAP_PX + 3, "a narrow gap, not a missing bond");
});

test("multiple bonds break every stroke, wedges split, hashes lose the ticks at the crossing", () => {
  const dbl = cross(null, [2, 1]);
  const s = buildScene(setBreakCrossings(dbl.m, true), VIEW, null).bonds.find((b) => b.bond.id === dbl.h);
  assert.equal(s.shape.lines.length, 4, "both strokes of the double bond broken");
  const triple = cross(null, [3, 2]);
  const t = buildScene(setBreakCrossings(triple.m, true), VIEW, null).bonds.find((b) => b.bond.id === triple.h);
  assert.ok(t.shape.lines.length >= 5, "the triple bond's strokes broken (the short outer ones may vanish)");
  const w = cross();
  const wedge = setBreakCrossings(setBondStereo(w.m, w.h, "wedge"), true);
  const ws = buildScene(wedge, VIEW, null).bonds.find((b) => b.bond.id === w.h);
  assert.equal(ws.shape.wedges.length, 2);
  const hash = setBreakCrossings(setBondStereo(w.m, w.h, "hash"), true);
  const plain = buildScene(setBondStereo(w.m, w.h, "hash"), VIEW, null).bonds.find((b) => b.bond.id === w.h);
  const hs = buildScene(hash, VIEW, null).bonds.find((b) => b.bond.id === w.h);
  assert.ok(hs.shape.lines.length < plain.shape.lines.length && hs.shape.lines.length > 0);
  assert.equal(hs.broken, 1);
});
