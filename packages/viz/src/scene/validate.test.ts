// @ts-nocheck -- run by `npm --prefix apps/ui test` (node --experimental-strip-types --test).
// validateScene on small hand-made documents of each type: what passes, and the JSON paths of
// what does not.
import test from "node:test";
import assert from "node:assert/strict";
import { isRelativeAssetPath, validateScene } from "./validate.ts";

const SHA = "a".repeat(64);
const asset = (path = "x/orbital-3.cube") => ({ path, bytes: 10, sha256: SHA, media_type: "chemical/x-cube" });
const provenance = { exporter: "test", exported_at: "2026-10-27T00:00:00Z", server: {}, request: {}, notes: [] };
const doc = (scene) => ({ kind: "ksmech-scene", version: 1, scene, provenance });
const depiction = { ref: "s0", molblock: "M  END", atoms: [{ depiction_idx: 0, pool_atom: 0, map: 1, element: "O", hidden_h_atoms: [] }], server_svg: null, server_coords: null };
const reading = { i: 0, j: 1, g_atom: -1, donor_token: 3, acceptor_token: 7, g_rs: -0.3, term: -0.2 };
const edge = (extra = {}) => ({ id: "e1", kind: "predicted", source_state_id: "s0", product_state_id: "s1", source_mapped_smiles: "[O-:1]", product_mapped_smiles: "[O-:1]", matrix: { bonds: [[0, 1, 1]], lone: [[0, -2]] }, anchor: null, arrows: [{ kind: "lone_pair_to_bond", source: [0], target: [0, 1] }], readings: [reading], score: 1.2, logp: null, predictor: "floor", rank: 1, product_smiles: ["CO"], sigma_keys: ["k"], formal_charges: [[0, 0]], ...extra });
const step = (extra = {}) => ({ type: "step", edge: edge(), probabilities: false, source: { depiction, geometry: { state_id: "s0", species: [{ molblock: "M  END", display_offset: [0, 0, 0] }] }, cubes: [{ index: 3, asset: asset() }] }, product: { depiction }, ...extra });
const row = (index, kind) => ({ index, kind, occupation: kind === "occupied" ? 2 : 0, energy_ev: -index });
const molecule = (extra = {}) => ({ type: "molecule", smiles: "C=O", molecule: { molecule_id: "m1" }, geometry: { molblock: "M  END", atoms: [] }, orbitals: { set: "canonical", homo_index: 1, orbitals: [row(1, "occupied"), row(2, "virtual")] }, cubes: [{ index: 2, asset: asset("m/mo-2.cube") }], selected: 1, ...extra });
const node = (id) => ({ id, display_smiles: "C", depth: 0, species: [] });
const network = (extra = {}) => ({ type: "network", network: { root: "n0", nodes: [node("n0"), node("n1")], edges: [{ ...edge(), source: "n0", target: "n1" }], pathways: [{ node_ids: ["n0", "n1"], edge_ids: ["e1"] }], scores_are_probabilities: false }, ...extra });

const errors = (d) => {
  const v = validateScene(d);
  return v.ok ? [] : v.errors;
};

test("a step, a molecule and a network scene pass", () => {
  for (const s of [step(), molecule(), network(), step({ product: null, variants: ["orbital"] }), molecule({ orbitals: null, cubes: [], selected: null })]) {
    assert.deepEqual(errors(doc(s)), [], s.type);
  }
});

test("the envelope: kind, version, type, provenance", () => {
  assert.deepEqual(errors({ ...doc(step()), kind: "scene" }), ['kind: expected "ksmech-scene", got "scene"']);
  assert.deepEqual(errors({ ...doc(step()), version: 2 }), ["version: expected version 1, got 2"]);
  assert.match(errors(doc({ type: "movie" }))[0], /^scene.type: expected one of molecule, step, network/);
  assert.deepEqual(errors({ ...doc(step()), provenance: { ...provenance, notes: "x" } }), ['provenance.notes: expected an array, got "x"']);
  assert.deepEqual(errors(null), ["(document): expected an object, got null"]);
});

test("assets: a relative path inside the scene's folder, a byte count, a sha256", () => {
  for (const p of ["../x.cube", "/x.cube", "https://e.org/x.cube", "a//b", "a\\b", ""]) assert.equal(isRelativeAssetPath(p), false, p);
  assert.equal(isRelativeAssetPath("sn2-step/orbital-16.cube"), true);
  const bad = step();
  bad.source.cubes[0].asset = { path: "../etc/passwd", bytes: -1, sha256: "ABC", media_type: "x" };
  assert.deepEqual(errors(doc(bad)), [
    'scene.source.cubes[0].asset.path: expected a relative path inside the scene\'s folder, got "../etc/passwd"',
    "scene.source.cubes[0].asset.bytes: expected a byte count, got -1",
    'scene.source.cubes[0].asset.sha256: expected a lowercase hex sha256, got "ABC"',
  ]);
});

test("ids must point into the document", () => {
  const s = step();
  s.source.cubes[0].index = 99;
  assert.deepEqual(errors(doc(s)), ["scene.source.cubes[0].index: expected a donor or acceptor token of scene.edge.readings, got 99"]);
  assert.deepEqual(errors(doc(molecule({ selected: 5 }))), ["scene.selected: expected an orbital index of scene.orbitals, got 5"]);
  const n = network();
  n.network.edges[0].target = "n9";
  assert.deepEqual(errors(doc(n)), ['scene.network.edges[0].target: expected the id of a node, got "n9"']);
  assert.deepEqual(errors(doc(network({ network: { ...network().network, root: "zz" } }))), ['scene.network.root: expected the id of a node, got "zz"']);
});

test("the fields the views read are typed", () => {
  const s = step({ edge: edge({ matrix: { bonds: [[0, "1", 1]], lone: [] }, score: "high", arrows: [{ kind: "push", source: [0], target: [1] }] }) });
  assert.deepEqual(errors(doc(s)), [
    'scene.edge.matrix.bonds[0]: expected an integer triple, got [0,"1",1]',
    'scene.edge.arrows[0].kind: expected one of lone_pair_to_bond, bond_to_bond, bond_to_lone_pair, got "push"',
    'scene.edge.score: expected a number or null, got "high"',
  ]);
  const g = step();
  g.source.geometry.species[0].display_offset = [0, 0];
  assert.deepEqual(errors(doc(g)), ["scene.source.geometry.species[0].display_offset: expected an [x, y, z] offset, got [0,0]"]);
});

test("scene.schema.json agrees with validateScene on the envelope, the types and asset paths", async () => {
  const { readFileSync } = await import("node:fs");
  const schema = JSON.parse(readFileSync(new URL("../../scene.schema.json", import.meta.url), "utf8"));
  assert.equal(schema.properties.kind.const, "ksmech-scene");
  assert.equal(schema.properties.version.const, 1);
  const types = schema.properties.scene.oneOf.map((r) => schema.$defs[r.$ref.split("/").pop()].properties.type.const);
  assert.deepEqual(types, ["molecule", "step", "network"]);
  const pattern = new RegExp(schema.$defs.assetRef.properties.path.pattern);
  for (const p of ["a/b.cube", "sn2-step/orbital-16.cube", "x.cube", "../x.cube", "/x.cube", "https://e.org/x", "a//b", "a/../b", "a\\b", "a/"]) {
    assert.equal(pattern.test(p), isRelativeAssetPath(p), p);
  }
  // every scene the validator accepts has the schema's required keys
  for (const [s, def] of [[step(), "stepScene"], [molecule(), "moleculeScene"], [network(), "networkScene"]]) {
    for (const k of schema.$defs[def].required) assert.ok(k in s, `${def}.${k}`);
  }
});
