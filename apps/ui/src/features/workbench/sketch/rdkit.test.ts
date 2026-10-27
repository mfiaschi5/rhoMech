// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts). Loads the
// real RDKit.js wasm build from apps/ui/node_modules (skipped when it is not installed).
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { analyzeSketch, badAtomIndices, cleanUpSketch, exportSketch, importSketch } from "./rdkitSketch.ts";
import { applyClick, applyKey, NO_SELECTION } from "./tools.ts";
import { BOND_LENGTH, emptySketch, meanBondLength, setBondStereo } from "@ksmech/viz/sketch/model";
import { toMolblock } from "@ksmech/viz/sketch/molfile";

const wasmJs = fileURLToPath(new URL("../../../../node_modules/@rdkit/rdkit/dist/RDKit_minimal.js", import.meta.url));
const available = existsSync(wasmJs);
let rdkitPromise = null;
const rdkit = () => (rdkitPromise ??= createRequire(import.meta.url)(wasmJs)());

const opts = { skip: available ? false : "RDKit.js not installed (npm --prefix apps/ui install)" };

function ethanol() {
  let m = applyClick(emptySketch(), "bond", "C", null, [0, 0]);
  m = applyClick(m, "bond", "C", { kind: "atom", id: m.atoms[1].id }, [0, 0]);
  return applyKey(m, "o", { kind: "atom", id: m.atoms[2].id }, NO_SELECTION);
}

test("ethanol drawn by clicks is CCO with RDKit's H counts and descriptors", opts, async () => {
  const rd = await rdkit();
  const m = ethanol();
  const info = analyzeSketch(rd, m);
  assert.equal(info.valid, true);
  assert.equal(info.smiles, "CCO");
  assert.equal(info.formula, "C2H6O");
  assert.match(info.inchi, /^InChI=1S\/C2H6O\//);
  assert.deepEqual([...info.hCounts.values()], [3, 2, 1]);
  assert.ok(Math.abs(info.descriptors.amw - 46.069) < 0.01);
  assert.equal(info.descriptors.NumHBD, 1);
  assert.equal(exportSketch(rd, m, "smiles"), "CCO");
});

test("a pentavalent carbon is invalid and its atom is marked", opts, async () => {
  const rd = await rdkit();
  let m = applyClick(emptySketch(), "bond", "C", null, [0, 0]);
  const c = m.atoms[0].id;
  for (let i = 0; i < 4; i++) m = applyKey(m, "1", { kind: "atom", id: c }, NO_SELECTION);
  const info = analyzeSketch(rd, m);
  assert.equal(info.valid, false);
  assert.deepEqual(info.badAtoms, [c]);
  assert.match(info.error, /valence/i);
  assert.equal(info.smiles, "");
  assert.deepEqual(badAtomIndices("[10:00:00] Can't kekulize mol.  Unkekulized atoms: 0 1 2\n"), [0, 1, 2]);
});

test("charges reach RDKit through M  CHG; empty sketch is not valid", opts, async () => {
  const rd = await rdkit();
  let m = ethanol();
  m = applyKey(m, "-", { kind: "atom", id: m.atoms[2].id }, NO_SELECTION);
  const info = analyzeSketch(rd, m);
  assert.equal(info.smiles, "CC[O-]");
  assert.equal(info.formula, "C2H5O-");
  const empty = analyzeSketch(rd, emptySketch());
  assert.deepEqual([empty.empty, empty.valid, empty.smiles], [true, false, ""]);
});

test("wedge bonds give a stereocentre and survive the clean-up", opts, async () => {
  const rd = await rdkit();
  // C[C@@H](O)N style centre: C with Me, OH, NH2 and one wedge
  let m = applyClick(emptySketch(), "bond", "C", null, [0, 0]);
  const centre = m.atoms[1].id;
  m = applyClick(m, "bond", "C", { kind: "atom", id: centre }, [0, 0]);
  m = applyKey(m, "o", { kind: "atom", id: m.atoms[2].id }, NO_SELECTION);
  m = applyKey(m, "1", { kind: "atom", id: centre }, NO_SELECTION);
  m = applyKey(m, "n", { kind: "atom", id: m.atoms[3].id }, NO_SELECTION);
  const wedgeBond = m.bonds.find((b) => b.a === centre && m.atoms.find((a) => a.id === b.b).el === "N");
  m = setBondStereo(m, wedgeBond.id, "wedge");
  const before = analyzeSketch(rd, m).smiles;
  assert.match(before, /@/);
  const clean = cleanUpSketch(rd, m);
  assert.ok(clean);
  assert.ok(Math.abs(meanBondLength(clean) - BOND_LENGTH) < 1e-6);
  assert.deepEqual(clean.atoms.map((a) => a.id), m.atoms.map((a) => a.id));
  assert.equal(analyzeSketch(rd, clean).smiles, before, "same stereocentre after the new layout");
});

test("import: SMILES (aromatic, kekulized), molfile with coordinates, errors", opts, async () => {
  const rd = await rdkit();
  const benz = importSketch(rd, "c1ccccc1O");
  assert.equal(benz.atoms.length, 7);
  assert.deepEqual(benz.bonds.map((b) => b.order).filter((o) => o === 2).length, 3);
  assert.equal(analyzeSketch(rd, benz).smiles, "Oc1ccccc1");
  assert.ok(Math.abs(meanBondLength(benz) - BOND_LENGTH) < 1e-6);
  const chiral = importSketch(rd, "C[C@H](N)C(=O)O");
  assert.equal(analyzeSketch(rd, chiral).smiles, "C[C@H](N)C(=O)O");
  const fromMol = importSketch(rd, toMolblock(ethanol()));
  assert.equal(analyzeSketch(rd, fromMol).smiles, "CCO");
  assert.throws(() => importSketch(rd, "C1CC"), /./);
  assert.throws(() => importSketch(rd, "   "), /nothing/);
});

test("clean-up lays out an invalid drawing too", opts, async () => {
  const rd = await rdkit();
  let m = applyClick(emptySketch(), "bond", "C", null, [0, 0]);
  for (let i = 0; i < 4; i++) m = applyKey(m, "1", { kind: "atom", id: m.atoms[0].id }, NO_SELECTION);
  const clean = cleanUpSketch(rd, m);
  assert.ok(clean);
  assert.equal(clean.atoms.length, m.atoms.length);
});
