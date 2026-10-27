// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// Badge placement: the preferred spot when free, turned away from an obstacle, and two badges
// of one atom never on top of each other.
import test from "node:test";
import assert from "node:assert/strict";
import { boxAround, placeBadges, segmentBoxes } from "./badges.ts";

const overlaps = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

test("a free badge takes its preferred direction on the nearest ring", () => {
  const at = placeBadges([{ key: "a", at: [100, 100], w: 20, h: 16, prefer: 0 }], [], 50).get("a");
  assert.ok(Math.abs(at[0] - 130) < 1e-9 && Math.abs(at[1] - 100) < 1e-9);
});

test("an obstacle turns the badge aside; two badges of one atom do not overlap", () => {
  const bond = segmentBoxes([100, 100], [160, 100]);
  const pos = placeBadges(
    [
      { key: "lone", at: [100, 100], w: 20, h: 16, prefer: 0 },
      { key: "charge", at: [100, 100], w: 18, h: 18, prefer: 0 },
    ],
    [boxAround([100, 100], 10, 8), ...bond],
    50,
  );
  const a = boxAround(pos.get("lone"), 10, 8);
  const b = boxAround(pos.get("charge"), 9, 9);
  for (const o of bond) assert.ok(!overlaps(a, o) && !overlaps(b, o), "clear of the bond");
  assert.ok(!overlaps(a, b), "clear of each other");
  assert.ok(!overlaps(a, boxAround([100, 100], 10, 8)), "clear of the label");
});
