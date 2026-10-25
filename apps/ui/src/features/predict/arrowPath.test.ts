// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The Bézier arrows of a reading: end points (lone pair, bond midpoint, forming bond), the bow
// away from the centroid, the trimmed ends, the head along the tangent, alternating sides.
import test from "node:test";
import assert from "node:assert/strict";
import {
  arrowEnds,
  bezierArrow,
  cubicPoint,
  cubicTangent,
  formingBondPoint,
  freeDirection,
  layoutReading,
  lonePairPoint,
  subCubic,
  tAtArcLength,
} from "./arrowPath.ts";

const close = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol;
const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);

/** O0 at (0,0) bonded to nothing, C1 at (100,0) bonded to Cl2 at (140,0); 40 px bonds. */
const POS = [[0, 0], [100, 0], [140, 0]];
const NB = { 0: [], 1: [2], 2: [1] };
const CTX = { pos: (i) => POS[i], neighbours: (i) => (NB[i] ?? []).map((k) => POS[k]), centroid: [80, 20], bondPx: 40 };

test("free direction: opposite a single bond, the widest gap, away from the centroid alone", () => {
  const d1 = freeDirection([0, 0], [[10, 0]], [0, 0]);
  assert.ok(close(d1[0], -1) && close(d1[1], 0, 1e-9));
  const d2 = freeDirection([0, 0], [[10, 0], [0, 10]], [0, 0]);
  assert.ok(d2[0] < 0 && d2[1] < 0, "between the two bonds' far side");
  const d0 = freeDirection([0, 0], [], [0, 10]);
  assert.ok(close(d0[0], 0) && close(d0[1], -1));
});

test("lone pair point sits 0.38 bond lengths out along the free direction", () => {
  const lp = lonePairPoint(CTX, 1);
  assert.ok(close(dist(lp.at, POS[1]), 0.38 * 40));
  assert.ok(lp.at[0] < 100, "C1's free side is away from Cl2");
});

test("forming bond point: midpoint when near, short of the target when far", () => {
  assert.deepEqual(formingBondPoint([0, 0], [40, 0], 40), [20, 0]);
  const far = formingBondPoint([0, 0], [200, 0], 40);
  assert.ok(close(far[0], 200 - 18) && close(far[1], 0));
});

test("arrow ends per kind", () => {
  const lp = arrowEnds({ kind: "lone_pair_to_bond", source: [0], target: [0, 1] }, CTX);
  assert.ok(lp.tailDir, "a lone-pair tail has a direction");
  assert.deepEqual(lp.head, formingBondPoint(POS[0], POS[1], 40));
  const leave = arrowEnds({ kind: "bond_to_lone_pair", source: [1, 2], target: [2] }, CTX);
  assert.deepEqual(leave.tail, [120, 0]);
  assert.equal(leave.tailDir, null);
  assert.ok(leave.head[0] > 140, "the new lone pair on Cl2 is on its free side");
  assert.equal(arrowEnds({ kind: "bond_to_bond", source: [1, 9], target: [1, 0] }, CTX), null, "an unplaced atom");
});

test("the curve bows away from the centroid and both ends are trimmed", () => {
  const tail = [0, 0];
  const head = [100, 0];
  const p = bezierArrow(tail, head, { centroid: [50, 40], gapTail: 4, gapHead: 5, headLength: 9 });
  assert.ok(p.c1[1] < 0 && p.c2[1] < 0, "the control points are on the far side of the chord");
  assert.ok(p.mid[1] < 0);
  const start = p.d.match(/^M ([-\d.]+) ([-\d.]+)/).slice(1).map(Number);
  assert.ok(close(dist(start, tail), 4, 0.3), `tail trimmed by 4 px (got ${dist(start, tail)})`);
  assert.ok(close(dist(p.tip, head), 5, 0.3), "tip 5 px short of the head point");
  const flipped = bezierArrow(tail, head, { centroid: [50, 40], side: -1 });
  assert.ok(flipped.mid[1] > 0, "side -1 bows towards the centroid");
});

test("the head's axis is the curve's tangent at the tip", () => {
  const p = bezierArrow([0, 0], [80, 60], { centroid: [0, 60], headLength: 10, headWidth: 8 });
  const [tip, l, r] = p.head;
  const base = [(l[0] + r[0]) / 2, (l[1] + r[1]) / 2];
  const axis = [tip[0] - base[0], tip[1] - base[1]];
  const t1 = tAtArcLength([0, 0], p.c1, p.c2, [80, 60], 4, true);
  const tan = cubicTangent([0, 0], p.c1, p.c2, [80, 60], t1);
  const cos = (axis[0] * tan[0] + axis[1] * tan[1]) / (Math.hypot(...axis) * Math.hypot(...tan));
  assert.ok(cos > 0.9999, `head along the tangent (cos ${cos})`);
  assert.ok(close(Math.hypot(...axis), 10, 1e-6));
  assert.ok(close(dist(l, r), 8, 1e-6));
});

test("subCubic reproduces the original curve on its interval", () => {
  const P = [[0, 0], [10, 30], [60, 30], [70, 0]];
  const [a, b, c, d] = subCubic(...P, 0.2, 0.7);
  for (const u of [0, 0.25, 0.5, 1]) {
    const q = cubicPoint(a, b, c, d, u);
    const o = cubicPoint(...P, 0.2 + 0.5 * u);
    assert.ok(dist(q, o) < 1e-9);
  }
});

test("arc-length parameters: the start, the end and a point between", () => {
  const P = [[0, 0], [33, 0], [66, 0], [99, 0]];
  assert.ok(close(cubicPoint(...P, tAtArcLength(...P, 10))[0], 10, 0.05));
  assert.ok(close(cubicPoint(...P, tAtArcLength(...P, 10, true))[0], 89, 0.05));
});

test("a reading's duplicate arrows alternate sides; unplaced ones are null", () => {
  const a = { kind: "lone_pair_to_bond", source: [0], target: [0, 1] };
  const [p, q, r] = layoutReading([a, a, { kind: "bond_to_bond", source: [1, 7], target: [1, 0] }], CTX);
  assert.ok(p && q);
  assert.notEqual(Math.sign(p.mid[1] - 0), Math.sign(q.mid[1] - 0));
  assert.equal(r, null);
});
