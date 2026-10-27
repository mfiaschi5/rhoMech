// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The network viewports' pure layouts: layered columns by depth with barycentre ordering, back
// edges, step strengths, step types and readings, keyboard neighbours, the pathway trie with
// merged prefixes and its packed columns, and the scheme layout (no card overlaps another, on
// real jobs and on a synthetic 200-state network).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { backEdges, edgeStrength, intersects, layeredLayout, mappedSymbols, neighbour, pathwayTree, pathwayWeights, readingLabel, schemeLayout, stepType, treeRows } from "./netLayout.ts";

/** The networks of two real jobs (fixtures/: trimmed GET /api/jobs/{id} results). */
const REAL = ["j7c319c35065fadc2", "j729d164ec27f1100"].map((id) => [id, JSON.parse(readFileSync(new URL(`./fixtures/${id}.json`, import.meta.url), "utf8"))]);

const node = (id, depth, best_logp = 0) => ({ id, depth, best_logp, species: [], display_smiles: id, status: "expanded", is_root: depth === 0 });
const edge = (id, source, target, weight = -1, extra = {}) => ({ id, source, target, weight, kind: "predicted", recurrence: false, arrows: [], ...extra });
const path = (id, rank, edge_ids, logp, probability = null) => ({ id, rank, edge_ids, node_ids: [], logp, probability });

/** r -> a, b; a -> c; b -> d; d -> r (a cycle back to the root). */
const NET = {
  root: "r",
  nodes: [node("r", 0), node("a", 1, -0.5), node("b", 1, -1), node("c", 2), node("d", 2)],
  edges: [edge("e1", "r", "a", -0.5), edge("e2", "r", "b", -1.5), edge("e3", "a", "c"), edge("e4", "b", "d"), edge("e5", "d", "r", -3, { recurrence: true })],
  pathways: [path("p1", 1, ["e1", "e3"], -1), path("p2", 2, ["e2", "e4"], -2), path("p3", 3, ["e1"], -3)],
};

test("layered layout: one column per depth, centred, children follow their parents", () => {
  const { pos, layers } = layeredLayout(NET, { dx: 100, dy: 50 });
  assert.deepEqual(layers.map((l) => l.length), [1, 2, 2]);
  assert.deepEqual([pos.get("r").x, pos.get("a").x, pos.get("c").x], [0, 100, 200]);
  assert.equal(pos.get("r").y, 0);
  assert.equal(pos.get("a").y + pos.get("b").y, 0, "a column is centred");
  assert.ok(pos.get("a").y < pos.get("b").y, "the better state first");
  assert.ok(pos.get("c").y < pos.get("d").y, "c follows a, d follows b: no crossing");
});

test("a wide layer becomes a brick of sub-columns; later layers shift right", () => {
  const wide = { root: "r", nodes: [node("r", 0), ...Array.from({ length: 8 }, (_, k) => node(`n${k}`, 1, -k)), node("z", 2)], edges: [...Array.from({ length: 8 }, (_, k) => edge(`e${k}`, "r", `n${k}`)), edge("ez", "n0", "z")], pathways: [] };
  const { pos, columns } = layeredLayout(wide, { dx: 100, dy: 50, maxPerColumn: 6, subDx: 60 });
  assert.deepEqual(columns.map((c) => [c.layer, c.x0, c.x1, c.count]), [[0, 0, 0, 1], [1, 100, 160, 8], [2, 260, 260, 1]]);
  const xs = Array.from({ length: 8 }, (_, k) => pos.get(`n${k}`).x);
  assert.deepEqual([...new Set(xs)].sort((a, b) => a - b), [100, 160], "two sub-columns");
  const ys = Array.from({ length: 8 }, (_, k) => pos.get(`n${k}`).y);
  assert.ok(Math.max(...ys) - Math.min(...ys) <= 7 * 25 + 1e-9, "rows dy / 2 apart");
  for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
    const [a, b] = [pos.get(`n${i}`), pos.get(`n${j}`)];
    assert.ok(a.x !== b.x || Math.abs(a.y - b.y) >= 50 - 1e-9, "cards in one sub-column keep dy apart");
  }
  assert.equal(pos.get("z").x, 260);
});

test("back edges, strengths, keyboard neighbours", () => {
  assert.deepEqual([...backEdges(NET)], ["e5"]);
  const s = edgeStrength(NET);
  assert.equal(s.get("e1"), 1);
  assert.ok(Math.abs(s.get("e2") - Math.exp(-1)) < 1e-9);
  const { pos } = layeredLayout(NET);
  assert.equal(neighbour(NET, pos, "r", "right"), "a");
  assert.equal(neighbour(NET, pos, "a", "down"), "b");
  assert.equal(neighbour(NET, pos, "b", "up"), "a");
  assert.equal(neighbour(NET, pos, "d", "left"), "b");
  assert.equal(neighbour(NET, pos, "c", "right"), null);
});

test("step types and readings with element symbols", () => {
  const sym = mappedSymbols("[C:1]([O-:2])([H:3])[H:4].[C:5]([Cl:6])[c:7][Br:8]");
  assert.deepEqual(sym, ["C", "O", "H", "H", "C", "Cl", "C", "Br"]);
  const sn2 = [{ kind: "lone_pair_to_bond", source: [1], target: [1, 4] }, { kind: "bond_to_lone_pair", source: [4, 5], target: [5] }];
  assert.equal(stepType(sn2, sym), "substitution");
  assert.equal(readingLabel(sn2, sym), "lp O1 → C4 · C4–Cl5 → lp Cl5");
  assert.equal(stepType([sn2[0]], sym), "addition");
  assert.equal(stepType([sn2[1]], sym), "heterolysis");
  const pt = [{ kind: "lone_pair_to_bond", source: [1], target: [1, 2] }, { kind: "bond_to_lone_pair", source: [2, 0], target: [0] }];
  assert.equal(stepType(pt, sym), "proton transfer");
});

test("pathway trie: shared prefixes merged, weights summed, lanes by rank", () => {
  const w = pathwayWeights(NET);
  const t = pathwayTree(NET);
  assert.equal(t.children.length, 2, "p1 and p3 share e1");
  const e1 = t.children.find((c) => c.edgeId === "e1");
  assert.deepEqual(e1.ranks, [1, 3]);
  assert.ok(Math.abs(e1.weight - (w.get("p1") + w.get("p3"))) < 1e-12);
  assert.ok(Math.abs(t.weight - 1) < 1e-9);
  const rows = treeRows(t);
  const rowOf = (key) => rows.find((r) => r.node.key === key).row;
  assert.equal(rowOf(""), 0);
  assert.equal(rowOf("e1"), 0);
  assert.equal(rowOf("e1,e3"), 0, "the best pathway is one straight lane");
  assert.equal(rowOf("e2"), 1);
  const folded = treeRows(t, new Set(["e2"]));
  assert.ok(!folded.some((r) => r.node.key === "e2,e4"));
  assert.equal(folded.find((r) => r.node.key === "e2").hidden, 1);
});

test("probabilities, when given, weight the pathways directly", () => {
  const net = { ...NET, pathways: [path("p1", 1, ["e1"], -1, 0.6), path("p2", 2, ["e2"], -2, 0.2)] };
  const w = pathwayWeights(net);
  assert.ok(Math.abs(w.get("p1") - 0.75) < 1e-12);
});

/** Uneven card sizes, deterministic per id (cards and their captions are one rectangle). */
const sizeOf = (id) => {
  let h = 0;
  for (const c of id) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return { w: 140 + (h % 180), h: 90 + ((h >> 8) % 70) };
};

function assertNoOverlap(scheme, label) {
  const rs = [...scheme.rects];
  for (let i = 0; i < rs.length; i++) for (let j = i + 1; j < rs.length; j++) assert.ok(!intersects(rs[i][1], rs[j][1]), `${label}: ${rs[i][0]} overlaps ${rs[j][0]}`);
}

test("scheme layout: no card overlaps another on the real jobs; columns uniform, centred", () => {
  for (const [id, net] of REAL) {
    const sch = schemeLayout(net, sizeOf);
    assert.equal(sch.rects.size, net.nodes.length);
    assertNoOverlap(sch, id);
    for (const c of sch.columns) {
      const cards = net.nodes.filter((n) => (n.id === net.root ? 0 : n.depth) === c.layer).map((n) => sch.rects.get(n.id));
      assert.ok(cards.every((r) => r.w === c.width), "one width per column");
      const top = Math.min(...cards.map((r) => r.y));
      const bottom = Math.max(...cards.map((r) => r.y + r.h));
      assert.ok(Math.abs(top + bottom) < cards[0].h + 1e-6, "centred on y = 0 (within a card)");
    }
    for (let k = 1; k < sch.columns.length; k++) assert.ok(sch.columns[k].x0 >= sch.columns[k - 1].x1 + 90, "columns apart");
  }
});

test("scheme layout: a synthetic 200-state network has no overlap and splits tall columns", () => {
  const nodes = [{ id: "r", depth: 0, best_logp: 0 }];
  const edges = [];
  for (let k = 0; k < 199; k++) {
    const depth = 1 + (k % 4);
    nodes.push({ id: `n${k}`, depth, best_logp: -k / 50 });
    const parents = nodes.filter((n) => n.depth === depth - 1);
    edges.push({ id: `e${k}`, source: parents[k % parents.length].id, target: `n${k}`, weight: -1 });
  }
  const sch = schemeLayout({ root: "r", nodes, edges, pathways: [] }, sizeOf);
  assertNoOverlap(sch, "200 states");
  assert.ok(sch.columns.some((c) => c.x1 - c.x0 > c.width), "tall columns become sub-columns");
  assert.ok(sch.bounds.h < 7 * 180 + 6 * 18 + 1e-6, "no column taller than seven cards");
});

test("pathway columns of the real jobs are packed: every column's rows run 0..n-1", () => {
  for (const [, net] of REAL) {
    const rows = treeRows(pathwayTree(net));
    const byDepth = new Map();
    for (const r of rows) byDepth.set(r.node.depth, [...(byDepth.get(r.node.depth) ?? []), r.row]);
    for (const list of byDepth.values()) assert.deepEqual([...list].sort((a, b) => a - b), list.map((_, k) => k));
  }
});
