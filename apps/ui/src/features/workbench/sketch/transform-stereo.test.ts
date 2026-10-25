// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts). Loads the
// real RDKit.js wasm build (skipped when it is not installed): rotations, scaling and flips keep
// the canonical isomeric SMILES of the drawing, i.e. the same stereoisomer; a bare mirror image
// (coordinates only, marks not swapped) would draw the enantiomer.
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { analyzeSketch, importSketch } from "./rdkitSketch.ts";
import { applyTransform, TRANSFORM_ACTIONS, transformTargets, flipAtoms } from "./transform.ts";
import { NO_SELECTION } from "./tools.ts";

const wasmJs = fileURLToPath(new URL("../../../../node_modules/@rdkit/rdkit/dist/RDKit_minimal.js", import.meta.url));
const available = existsSync(wasmJs);
let rdkitPromise = null;
const rdkit = () => (rdkitPromise ??= createRequire(import.meta.url)(wasmJs)());
const opts = { skip: available ? false : "RDKit.js not installed (npm --prefix apps/ui install)" };

const CASES = [
  "C[C@H](N)C(=O)O", // L-alanine
  "C[C@@]12CC[C@@H](C1(C)C)CC2=O", // camphor: two centres in a bicycle
  "O[C@@H]1[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O", // an inositol: six centres
  "C/C=C/[C@H](Cl)CC", // E double bond next to a centre
];

/** The coordinates only, mirrored left-right (what a naive flip would do). */
const bareMirror = (m) => ({ ...m, atoms: m.atoms.map((a) => ({ ...a, x: -a.x })) });

test("every transform of the whole drawing keeps the canonical isomeric SMILES", opts, async () => {
  const rd = await rdkit();
  for (const smi of CASES) {
    const m = importSketch(rd, smi);
    const before = analyzeSketch(rd, m).smiles;
    assert.match(before, /@/, smi);
    const ids = transformTargets(m, NO_SELECTION);
    for (const a of TRANSFORM_ACTIONS) assert.equal(analyzeSketch(rd, applyTransform(m, ids, a)).smiles, before, `${smi} ${a}`);
    const chained = ["flip-h", "rotate-cw15", "flip-v", "scale-up", "rotate-ccw90"].reduce((acc, a) => applyTransform(acc, ids, a), m);
    assert.equal(analyzeSketch(rd, chained).smiles, before, `${smi} chained`);
  }
});

test("a bare mirror of the coordinates gives the enantiomer (why the flip swaps the marks)", opts, async () => {
  const rd = await rdkit();
  const m = importSketch(rd, "C[C@H](N)C(=O)O");
  const before = analyzeSketch(rd, m).smiles;
  const mirrored = analyzeSketch(rd, bareMirror(m)).smiles;
  assert.notEqual(mirrored, before);
  assert.equal(mirrored, "C[C@@H](N)C(=O)O");
  assert.equal(analyzeSketch(rd, flipAtoms(m, transformTargets(m, NO_SELECTION), "h")).smiles, before);
});

test("flipping one whole fragment of a two-fragment drawing keeps both", opts, async () => {
  const rd = await rdkit();
  const m = importSketch(rd, "C[C@H](N)C(=O)O.C[C@@H](O)CC");
  const before = analyzeSketch(rd, m).smiles;
  const first = m.atoms.slice(0, 6).map((a) => a.id); // the alanine atoms come first
  const sel = { atoms: new Set(first), bonds: new Set() };
  const ids = transformTargets(m, sel);
  assert.deepEqual(ids, first);
  for (const a of ["flip-h", "flip-v", "rotate-cw90", "scale-down"]) assert.equal(analyzeSketch(rd, applyTransform(m, ids, a)).smiles, before, a);
});
