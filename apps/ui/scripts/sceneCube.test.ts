// @ts-nocheck -- run by `npm test` (node --experimental-strip-types --test).
import test from "node:test";
import assert from "node:assert/strict";
import { strideCube } from "./sceneCube.ts";

// a 3 x 2 x 5 grid whose value at (i, j, k) is "i.j.k" written as a number 100i + 10j + k
const values = [];
for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 5; k++) values.push(String(100 * i + 10 * j + k));
const rows = [];
for (let r = 0; r < values.length; r += 5) rows.push(values.slice(r, r + 5).join(" "));
const cube = ["title", "units", "    1    0.000000    0.000000    0.000000", "    3    0.500000    0.000000    0.000000", "    2    0.000000    0.500000    0.000000", "    5    0.000000    0.000000    0.500000", "    6    6.000000    0.000000    0.000000    0.000000", ...rows].join("\n");

test("stride 2 keeps every other point and doubles the axis vectors", () => {
  const out = strideCube(cube, 2).trim().split("\n");
  assert.equal(out[1], "units; grid stride 2 (scene-export)");
  assert.equal(out[3].trim(), "2    1.000000    0.000000    0.000000");
  assert.equal(out[4].trim(), "1    0.000000    1.000000    0.000000");
  assert.equal(out[5].trim(), "3    0.000000    0.000000    1.000000");
  assert.equal(out[6].trim(), "6    6.000000    0.000000    0.000000    0.000000");
  assert.deepEqual(out.slice(7).map((l) => l.trim().split(/\s+/)), [["0", "2", "4"], ["200", "202", "204"]]);
});

test("stride 1 is the identity; a wrong value count is refused", () => {
  assert.equal(strideCube(cube, 1), cube);
  assert.throws(() => strideCube(cube.replace(/ 4$/m, ""), 2), /values for a 3x2x5 grid/);
});
