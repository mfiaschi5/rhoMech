// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The level diagram's pure layout (levels.ts): regrouping the canonical MOs by a degeneracy
// threshold (the server's rule: a new level more than the threshold above the level's first
// MO), the HOMO / LUMO blocks at that threshold, the threshold's bounds and its persistence in
// the layout preferences, the zoom and pan of the energy range, and the frontier labels kept
// apart.
import test from "node:test";
import assert from "node:assert/strict";
import { DEGENERACY, clampRange, degeneracies, degenerateLevels, energyWindow, frontierBlocks, groupLevels, groupLmoLevels, levelSlots, MIN_SPAN, panRange, sanitizeDegeneracy, spreadLabels, zoomRange } from "@ksmech/viz/orbitals/levels";
import { DEFAULT_PREFS, sanitizePrefs, updatePrefs } from "../layout/prefs.ts";

const row = (index, energy_ev, occupation, level = null, frontier = null) => ({ index, energy_ev, occupation, level, frontier, label: "", kind: occupation > 0.5 ? "occupied" : "virtual", centres: [], centroid: null });

/** A benzene-like spectrum: a near-degenerate e1g HOMO pair (0.03 eV apart), an exactly
 *  degenerate e2u LUMO pair, and pairs split by 0.2 eV and 0.4 eV further out. */
const MOS = [
  row(0, -20.0, 2, 0),
  row(1, -15.2, 2, 1),
  row(2, -15.0, 2, 2),
  row(3, -12.4, 2, 3),
  row(4, -12.0, 2, 4),
  row(5, -9.23, 2, 5, "homo"),
  row(6, -9.2, 2, 5, "homo"),
  row(7, -1.1, 0, 6, "lumo"),
  row(8, -1.1, 0, 6, "lumo"),
  row(9, 2.0, 0, 7, "lumo_plus_1"),
];

test("the threshold regroups the levels: 0.05 eV reproduces the server, larger values merge split pairs", () => {
  const server = groupLevels(MOS);
  const at005 = groupLevels(MOS, 0.05);
  assert.deepEqual(at005.map((g) => g.rows.map((r) => r.index)), server.map((g) => g.rows.map((r) => r.index)));
  assert.equal(at005.length, 8);
  assert.equal(groupLevels(MOS, 0.001).length, 9, "0.001 eV splits the 0.03 eV HOMO pair");
  assert.equal(groupLevels(MOS, 0.25).length, 7, "0.25 eV merges the 0.2 eV pair");
  assert.equal(groupLevels(MOS, 0.5).length, 6, "0.5 eV merges the 0.4 eV pair too");
  const ids = degenerateLevels(MOS, 0.05);
  assert.equal(ids.get(5), ids.get(6));
  assert.notEqual(ids.get(1), ids.get(2));
  const sorted = [...ids.values()];
  assert.equal(Math.max(...sorted), 7, "levels are numbered in ascending energy");
});

test("a level grows from its first MO, not by chaining (the server's rule)", () => {
  const chain = [row(0, 0, 2), row(1, 0.04, 2), row(2, 0.08, 2), row(3, 0.12, 2)];
  const g = groupLevels(chain, 0.05);
  assert.deepEqual(g.map((x) => x.rows.map((r) => r.index)), [[0, 1], [2, 3]]);
});

test("the HOMO / LUMO blocks follow the threshold", () => {
  const f005 = frontierBlocks(MOS, 0.05);
  assert.deepEqual([5, 6].map((i) => f005.get(i)), ["homo", "homo"]);
  assert.deepEqual([7, 8].map((i) => f005.get(i)), ["lumo", "lumo"]);
  assert.deepEqual([3, 4].map((i) => f005.get(i)), [undefined, "homo_minus_1"], "HOMO-1: the next distinct level");
  assert.equal(f005.get(9), "lumo_plus_1");
  MOS.filter((r) => r.frontier).forEach((r) => assert.equal(f005.get(r.index), r.frontier, "same as the server at 0.05 eV"));
  const f0001 = frontierBlocks(MOS, 0.001);
  assert.equal(f0001.get(6), "homo");
  assert.equal(f0001.get(5), "homo_minus_1", "split: the lower MO of the pair becomes HOMO-1");
  const f05 = frontierBlocks(MOS, 0.5);
  assert.deepEqual([3, 4].map((i) => f05.get(i)), ["homo_minus_1", "homo_minus_1"]);
});

test("the threshold is clamped to 0.001-0.5 eV and persisted with the layout preferences", () => {
  assert.equal(DEGENERACY.dflt, 0.05);
  assert.equal(sanitizeDegeneracy(0), 0.001);
  assert.equal(sanitizeDegeneracy(3), 0.5);
  assert.equal(sanitizeDegeneracy("0.12"), 0.12);
  assert.equal(sanitizeDegeneracy("x"), 0.05);
  assert.equal(DEFAULT_PREFS.degeneracyEv, 0.05);
  assert.equal(sanitizePrefs({ degeneracyEv: 0.2 }).degeneracyEv, 0.2);
  assert.equal(sanitizePrefs({ degeneracyEv: 9 }).degeneracyEv, 0.5);
  assert.equal(sanitizePrefs({ degeneracyEv: "no" }).degeneracyEv, 0.05);
  assert.equal(updatePrefs(DEFAULT_PREFS, { degeneracyEv: 0.0123 }).degeneracyEv, 0.012);
});

test("zoom about a point keeps that point fixed; pan moves the range; both stay near the spectrum", () => {
  const limit = energyWindow(MOS, "all");
  const r = [-14, -4];
  const z = zoomRange(r, 0.5, -9.2, limit);
  assert.ok(Math.abs(z[1] - z[0] - 5) < 1e-9, "half the span");
  assert.ok(Math.abs((-9.2 - z[0]) / (z[1] - z[0]) - (-9.2 - r[0]) / (r[1] - r[0])) < 1e-9, "the cursor energy stays put");
  const deep = Array.from({ length: 60 }, () => 0).reduce((acc) => zoomRange(acc, 0.5, -9.2, limit), r);
  assert.ok(deep[1] - deep[0] >= MIN_SPAN - 1e-12, "not below the smallest span");
  const out = Array.from({ length: 60 }, () => 0).reduce((acc) => zoomRange(acc, 2, -9.2, limit), r);
  assert.ok(out[1] - out[0] <= 4 * (limit[1] - limit[0]) + 1e-9, "not beyond 4x the spectrum");
  assert.deepEqual(panRange(r, 1, limit), [-13, -3]);
  const far = panRange(r, 1000, limit);
  assert.ok((far[0] + far[1]) / 2 <= limit[1] + 1e-9, "the centre stays inside the spectrum");
  assert.deepEqual(clampRange([-30, -28], [-10, 0]), [-11, -9]);
});

test("frontier labels are spread at least the gap apart, staying as near their levels as possible", () => {
  const ys = spreadLabels([100, 102, 300, 104], 15, 0, 400);
  const sorted = [...ys].sort((a, b) => a - b);
  for (let i = 1; i < sorted.length; i++) assert.ok(sorted[i] - sorted[i - 1] >= 15 - 1e-9);
  assert.equal(ys[2], 300, "a label with room stays on its level");
  const edge = spreadLabels([395, 398], 15, 0, 400);
  assert.ok(Math.max(...edge) <= 400 && Math.abs(edge[0] - edge[1]) >= 15 - 1e-9, "kept inside at the edge");
});

test("LMOs group at the threshold, occupied and virtual apart; degeneracies sizes the groups", () => {
  const lmos = [row(0, -14.1, 2), row(1, -14.12, 2), row(2, -14.08, 2), row(3, -11.0, 2), row(4, 3.0, 0), row(5, 3.02, 0), row(6, 3.01, 2)];
  const g = groupLmoLevels(lmos, 0.05);
  assert.deepEqual(g.map((x) => x.rows.map((r) => r.index)), [[0, 1, 2], [3], [6], [4, 5]], "the occupied LMO 6 never joins the virtual pair");
  assert.equal(new Set(g.map((x) => x.level)).size, g.length, "level ids are unique across the blocks");
  assert.equal(groupLmoLevels(lmos, 0.001).length, 7);
  const d = degeneracies(lmos, 0.05, true);
  assert.deepEqual([0, 3, 4, 6].map((i) => d.get(i)), [3, 1, 2, 1]);
  assert.equal(degeneracies(MOS, 0.05, false).get(5), 2, "canonical: the HOMO pair");
});

test("levelSlots: groups a few px apart share a row side by side, far groups get the full width", () => {
  // the toluene IBO C-H bonds: two groups 0.13 eV apart, then one far away
  const lmos = [row(7, -13.7228, 2), row(8, -13.7227, 2), row(10, -13.6916, 2), row(12, -13.5954, 2), row(13, -13.5458, 2), row(15, -11.478, 2)];
  const groups = groupLmoLevels(lmos, 0.05);
  assert.equal(groups.length, 3);
  const y = (e) => -e * 10; // 10 px per eV: the first two groups are 1.3 px apart
  const slots = levelSlots(groups, y, 0, 100, 100, 4);
  const xs = [7, 8, 10, 12, 13].map((i) => slots.get(i));
  assert.equal(new Set(xs.map((s) => s.cluster)).size, 1, "one row");
  for (let k = 1; k < xs.length; k++) assert.ok(xs[k].x >= xs[k - 1].x + xs[k - 1].w, "no overlap");
  assert.equal(slots.get(15).w, 100, "a lone level spans the column");
  assert.notEqual(slots.get(15).cluster, xs[0].cluster);
  const zoomed = levelSlots(groups, (e) => -e * 1000, 0, 100, 100, 4);
  assert.notEqual(zoomed.get(7).cluster, zoomed.get(12).cluster, "zoomed in, the groups separate");
  const crowd = groupLevels(Array.from({ length: 40 }, (_, i) => row(i, i * 0.06, 2)), 0.05);
  const packed = levelSlots(crowd, y, 0, 100, 100, 10);
  assert.ok([...packed.values()].every((s) => s.w >= 1 && s.x >= 0 && s.x + s.w <= 100.0001), "a crowded row stays in the column");
});
