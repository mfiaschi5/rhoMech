// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts). Loads the
// real RDKit.js wasm build (skipped when it is not installed).
// CIP labels of drawings imported from SMILES (RDKit's 2D layout with its wedges, then our
// molblock back to RDKit): (R)- and (S)-CHFClBr, L-alanine S, (R)-limonene, (E)- and
// (Z)-2-butene, meso-tartaric acid (R,S) and the pseudo-asymmetric r/s of 1,4-dimethyl-
// cyclohexane, cholesterol's eight centres against the known 3S,8S,9S,10R,13R,14S,17R,20R, "?"
// for an undefined centre, the same labels after a flip, a rotation and a wedge swap (the
// enantiomer), the info-bar list, and the scene marks.
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { importSketch, stereoLabels } from "./rdkitSketch.ts";
import { cipItems, cipText, parseStereoTags } from "./cip.ts";
import { applyTransform } from "./transform.ts";
import { buildScene } from "./drawing.ts";
import { setBondStereo } from "./model.ts";

const wasmJs = fileURLToPath(new URL("../../../../node_modules/@rdkit/rdkit/dist/RDKit_minimal.js", import.meta.url));
const available = existsSync(wasmJs);
let rdkitPromise = null;
const rdkit = () => (rdkitPromise ??= createRequire(import.meta.url)(wasmJs)());
const opts = { skip: available ? false : "RDKit.js not installed (npm --prefix apps/ui install)" };

/** The sorted multiset of atom codes and the bond codes of a SMILES drawn in the sketcher. */
async function codes(smiles) {
  const rd = await rdkit();
  const m = importSketch(rd, smiles);
  const s = stereoLabels(rd, m);
  return { m, s, atoms: [...s.atoms.values()].sort().join(""), bonds: [...s.bonds.values()].sort().join("") };
}

/** The code of the atom drawn from SMILES atom `i` (import keeps the SMILES atom order). */
const codeAt = (m, s, i) => s.atoms.get(m.atoms[i].id);

test("CHFClBr, alanine, limonene", opts, async () => {
  let r = await codes("F[C@H](Cl)Br");
  assert.equal(codeAt(r.m, r.s, 1), "R");
  r = await codes("F[C@@H](Cl)Br");
  assert.equal(codeAt(r.m, r.s, 1), "S");
  r = await codes("C[C@H](N)C(=O)O"); // L-alanine
  assert.equal(codeAt(r.m, r.s, 1), "S");
  assert.equal(r.atoms, "S");
  r = await codes("C[C@@H](N)C(=O)O"); // D-alanine
  assert.equal(r.atoms, "R");
  r = await codes("CC1=CC[C@@H](CC1)C(=C)C"); // (R)-(+)-limonene, PubChem CID 440917
  assert.equal(codeAt(r.m, r.s, 4), "R");
  assert.equal(r.atoms, "R", "the ring C=C has no E/Z");
  assert.equal(r.bonds, "");
});

test("2-butene E / Z; meso compounds; pseudo-asymmetric r / s", opts, async () => {
  const e = await codes("C/C=C/C");
  assert.equal(e.bonds, "E");
  assert.equal(cipText(cipItems(e.m, e.s)), "C2=C3 E");
  assert.equal((await codes("C/C=C\\C")).bonds, "Z");
  assert.equal((await codes("O[C@H](C(O)=O)[C@@H](O)C(O)=O")).atoms, "RS", "meso-tartaric acid");
  assert.equal((await codes("C[C@H](O)[C@@H](C)O")).atoms, "RS", "meso-2,3-butanediol");
  assert.equal((await codes("C[C@H](O)[C@H](C)O")).atoms, "SS", "(S,S)-2,3-butanediol");
  assert.equal((await codes("C[C@@H]1CC[C@H](C)CC1")).atoms, "ss", "cis-1,4-dimethylcyclohexane");
  assert.equal((await codes("C[C@H]1CC[C@H](C)CC1")).atoms, "rr", "trans-1,4-dimethylcyclohexane");
});

test("cholesterol: 3S, 8S, 9S, 10R, 13R, 14S, 17R, 20R", opts, async () => {
  // PubChem CID 5997; SMILES atom index -> steroid position
  const smi = "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C";
  const known = { 1: ["20", "R"], 8: ["17", "R"], 11: ["14", "S"], 12: ["13", "R"], 15: ["9", "S"], 16: ["8", "S"], 20: ["10", "R"], 23: ["3", "S"] };
  const r = await codes(smi);
  assert.equal(r.s.atoms.size, 8);
  for (const [i, [pos, c]] of Object.entries(known)) assert.equal(codeAt(r.m, r.s, Number(i)), c, `C${pos}`);
});

test("'?' for a possible stereocentre without wedges; labels by id in drawing order", opts, async () => {
  const r = await codes("CC(O)CC");
  assert.equal(r.atoms, "?");
  const items = cipItems(r.m, r.s);
  assert.equal(items.length, 1);
  assert.equal(items[0].kind, "atom");
  assert.match(cipText(items), /^C\d \?$/);
  assert.deepEqual([...parseStereoTags(r.m, "not json").atoms], []);
  assert.deepEqual([...parseStereoTags(r.m, '{"CIP_atoms":[[99,"(R)"],[1,"(Q)"]]}').atoms], []);
});

test("a flip or a rotation keeps the labels; swapping the wedge gives the enantiomer", opts, async () => {
  const rd = await rdkit();
  const r = await codes("C[C@H](N)C(=O)O");
  const ids = r.m.atoms.map((a) => a.id);
  for (const a of ["flip-h", "flip-v", "rotate-cw15", "rotate-ccw90", "scale-up"]) {
    const t = applyTransform(r.m, ids, a);
    assert.deepEqual([...stereoLabels(rd, t).atoms], [...r.s.atoms], a);
  }
  const mark = r.m.bonds.find((b) => b.stereo !== "none");
  const other = setBondStereo(r.m, mark.id, mark.stereo === "wedge" ? "hash" : "wedge");
  assert.equal([...stereoLabels(rd, other).atoms.values()].join(""), "R");
});

test("scene: CIP marks near their atom and bond, none without labels", opts, async () => {
  const r = await codes("C/C=C/[C@H](F)Cl");
  const view = { s: 25, ox: 200, oy: 150 };
  const scene = buildScene(r.m, view, null, { cip: r.s });
  const centre = scene.atoms.find((a) => a.cip);
  assert.ok(centre, "the stereocentre has a mark");
  assert.match(centre.cip.text, /^\([RS]\)$/);
  const d = Math.hypot(centre.cip.at[0] - centre.at[0], centre.cip.at[1] - centre.at[1]);
  assert.ok(d > 8 && d < 30, `near the atom (${d.toFixed(1)} px)`);
  const dbl = scene.bonds.find((b) => b.cip);
  assert.equal(dbl.cip.text, "(E)");
  assert.equal(buildScene(r.m, view, null).atoms.some((a) => a.cip), false);
});
