// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// Textbook geometry: lone-pair slots, the dots of a pair, and curved arrows routed without
// crossings, clear of atoms and bonds.
import test from "node:test";
import assert from "node:assert/strict";
import { angDist, arrowCrossings, boxReach, candidate, crossings, lonePairSlots, pairDots, polyline, routeArrows, segmentsCross } from "./stepRoute.ts";

const DEG = Math.PI / 180;
const deg = (a) => Math.round(a / DEG);

test("slots: the sides first, clear of the bonds and of the charge", () => {
  // a terminal O bonded to the left (180°): top, bottom, right
  assert.deepEqual(lonePairSlots([180 * DEG], [], 3).map(deg).sort((a, b) => a - b), [-90, 0, 90]);
  // a charge at the upper right (-40°) keeps the right side and the top free of pairs
  const s = lonePairSlots([180 * DEG], [-40 * DEG], 3).map(deg);
  for (const a of s) assert.ok(angDist(a * DEG, -40 * DEG) >= 35 * DEG - 1e-9, `slot ${a} too close to the charge`);
  // pairs never sit on a bond
  for (const a of lonePairSlots([0, 90 * DEG], [], 2)) assert.ok(angDist(a, 0) >= 40 * DEG && angDist(a, 90 * DEG) >= 40 * DEG);
});

test("slots: the preferred direction is served first", () => {
  const s = lonePairSlots([180 * DEG], [], 3, 10 * DEG);
  assert.equal(deg(s[0]), 0);
  const t = lonePairSlots([180 * DEG], [], 3, 100 * DEG);
  assert.equal(deg(t[0]), 90);
});

test("slots: a hard-blocked slot is skipped", () => {
  const s = lonePairSlots([180 * DEG], [], 2, null, [-90 * DEG]).map(deg);
  assert.ok(!s.includes(-90));
});

test("box reach and pair dots", () => {
  const box = { x0: -10, x1: 10, y0: -5, y1: 5 };
  assert.equal(boxReach(box, 0), 10);
  assert.equal(boxReach(box, Math.PI / 2), 5);
  const [a, b] = pairDots([0, 0], 0, 12, 6);
  assert.deepEqual([a[0], b[0]], [12, 12]);
  assert.equal(Math.abs(a[1] - b[1]), 6);
});

test("segment crossings", () => {
  assert.equal(segmentsCross([0, 0], [10, 10], [0, 10], [10, 0]), true);
  assert.equal(segmentsCross([0, 0], [10, 0], [0, 5], [10, 5]), false);
  assert.equal(crossings([[0, 0], [10, 10]], [[0, 10], [10, 0]]), 1);
});

test("candidate: the bow side and size; a lone-pair tail leaves along its direction", () => {
  const spec = { tail: [0, 0], head: [100, 0], ends: [] };
  const up = candidate(spec, 1, 0.42, 24);
  const down = candidate(spec, -1, 0.42, 24);
  assert.ok(up.c1[1] > 0 && down.c1[1] < 0);
  const lp = candidate({ ...spec, tailDir: [0, -1] }, 1, 0.28, 24);
  assert.ok(lp.c1[1] < 0, "the first control point follows the lone pair (up)");
  const pts = polyline(up);
  assert.deepEqual(pts[0], [0, 0]);
  assert.deepEqual(pts.at(-1), [100, 0]);
});

test("route: two arrows whose default sides would cross are routed apart", () => {
  // tails and heads interleaved on a line: A 0 -> 100, B 40 -> 150; both bowing the same way
  // cross once, opposite ways never
  const A = { tail: [0, 0], head: [100, 0], ends: [] };
  const B = { tail: [40, 0], head: [150, 0], ends: [] };
  assert.ok(arrowCrossings([candidate(A, 1, 0.28, 24), candidate(B, 1, 0.28, 24)]) > 0);
  const r = routeArrows([A, B], { atoms: [], bonds: [] }, [75, 0]);
  assert.equal(arrowCrossings(r), 0);
  assert.notEqual(r[0].side, r[1].side);
});

test("route: an arrow bows away from an atom in its way, and away from a bond", () => {
  const A = { tail: [0, 0], head: [100, 0], ends: [] };
  // an atom just above the chord's middle (screen y down: y < 0 is up), too big to bow over
  const r = routeArrows([A], { atoms: [{ i: 9, at: [50, -25], r: 40 }], bonds: [] }, [50, 200]);
  const mid = polyline(r[0])[12];
  assert.ok(mid[1] > 0, "bows downwards, away from the atom");
  // a bond crossing the upper side
  const r2 = routeArrows([A], { atoms: [], bonds: [[[50, 5], [50, 60]]] }, [50, -200]);
  assert.ok(polyline(r2[0])[12][1] < 0, "bows upwards, away from the bond below");
});

test("route: end atoms are not obstacles of their own arrow", () => {
  const A = { tail: [0, 0], head: [100, 0], ends: [9] };
  const r = routeArrows([A], { atoms: [{ i: 9, at: [50, 25], r: 18 }], bonds: [] }, [50, -200]);
  // with the atom ignored, the outward preference (away from the centroid above) wins: down
  assert.ok(polyline(r[0])[12][1] > 0);
});

test("route: a side -1 alternative spec is used when the arrow bows that way", () => {
  const A = { tail: [0, 0], head: [100, 0], ends: [], alt: { tail: [0, 0], head: [100, 10], ends: [] } };
  const r = routeArrows([A], { atoms: [{ i: 9, at: [50, 25], r: 30 }], bonds: [] }, [50, 0]);
  assert.equal(r[0].side, -1);
  assert.deepEqual(r[0].head, [100, 10]);
});
