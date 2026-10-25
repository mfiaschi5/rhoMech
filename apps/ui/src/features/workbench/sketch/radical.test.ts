// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
// Radicals: the model edits and tools, the "." key (no collision with any other shortcut), the
// V2000 `M  RAD` round trip (ours and real RDKit.js: SMILES C[CH2] and [CH2], H counts), the
// dots placed in the largest free angle, and radicals kept by a server layout (by atom id).
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { addAtom, addBond, cycleRadical, emptySketch, radicalElectrons, removeBonds, setRadical, toggleRadical } from "./model.ts";
import { fromMolblock, toMolblock } from "./molfile.ts";
import { applyClick, applyKey, KEY_ELEMENTS, NO_SELECTION, RADICAL_KEY, TOOLS } from "./tools.ts";
import { transformKey } from "./transform.ts";
import { analyzeSketch, exportSketch, importSketch } from "./rdkitSketch.ts";
import { buildScene } from "./drawing.ts";
import { freeAngle, radicalDots, boxExit } from "./marks.ts";
import { applyDepiction } from "./depiction.ts";
import { fallbackHCount } from "./elements.ts";

const wasmJs = fileURLToPath(new URL("../../../../node_modules/@rdkit/rdkit/dist/RDKit_minimal.js", import.meta.url));
const available = existsSync(wasmJs);
let rdkitPromise = null;
const rdkit = () => (rdkitPromise ??= createRequire(import.meta.url)(wasmJs)());
const opts = { skip: available ? false : "RDKit.js not installed (npm --prefix apps/ui install)" };

/** Ethyl: C1-C2 with one unpaired electron on C2. */
function ethyl(n = 1) {
  let [m, a] = addAtom(emptySketch(), "C", 0, 0);
  let b;
  [m, b] = addAtom(m, "C", 1.5, 0);
  [m] = addBond(m, a, b);
  return { m: setRadical(m, [b], n), a, b };
}

test("model: set, toggle (second click clears), cycle, count; a radical C is not an orphan vertex", () => {
  const { m, a, b } = ethyl(0);
  assert.equal(m.atoms[1].radical, undefined);
  const one = toggleRadical(m, [b], 1);
  assert.equal(one.atoms[1].radical, 1);
  assert.equal(toggleRadical(one, [b], 1).atoms[1].radical, undefined, "second click clears");
  assert.equal(toggleRadical(one, [b], 2).atoms[1].radical, 2, "the other tool replaces");
  let c = m;
  const seen = [];
  for (let i = 0; i < 3; i++) {
    c = cycleRadical(c, [b]);
    seen.push(c.atoms[1].radical ?? 0);
  }
  assert.deepEqual(seen, [1, 2, 0]);
  assert.equal(radicalElectrons(setRadical(one, [a], 2)), 3);
  assert.equal(setRadical(m, [b], 0), m, "no-op keeps the model");
  const gone = removeBonds(one, [one.bonds[0].id]);
  assert.deepEqual(gone.atoms.map((x) => x.id), [b], "the plain C goes, the radical C stays");
});

test("tools and the '.' key; '.' collides with no element, bond, charge, transform or editor key", () => {
  assert.ok(TOOLS.includes("radical") && TOOLS.includes("diradical"));
  const { m, b } = ethyl(0);
  const hit = { kind: "atom", id: b };
  assert.equal(applyClick(m, "radical", "C", hit, [0, 0]).atoms[1].radical, 1);
  assert.equal(applyClick(m, "diradical", "C", hit, [0, 0]).atoms[1].radical, 2);
  assert.equal(applyClick(m, "radical", "C", null, [5, 5]), m, "empty canvas: nothing");
  assert.equal(applyKey(m, RADICAL_KEY, hit, NO_SELECTION).atoms[1].radical, 1);
  const sel = { atoms: new Set([b]), bonds: new Set() };
  assert.equal(applyKey(m, ".", null, sel).atoms[1].radical, 1, "on the selection");
  assert.equal(RADICAL_KEY in KEY_ELEMENTS, false);
  assert.ok(!["1", "2", "3", "+", "=", "-", "_", "Delete", "Backspace", "Escape", "r"].includes(RADICAL_KEY));
  const key = (k) => ({ key: k, shiftKey: false, ctrlKey: false, metaKey: false, altKey: false });
  assert.equal(transformKey(key(RADICAL_KEY)), null);
});

test("molfile: M  RAD 2 (doublet) and 3 (triplet) round trip; RAD 1 and old charge code 4 read", () => {
  for (const n of [1, 2]) {
    const { m } = ethyl(n);
    const mb = toMolblock(m);
    assert.match(mb, new RegExp(`^M  RAD  1   2   ${n === 1 ? 2 : 3}$`, "m"));
    const back = fromMolblock(mb);
    assert.equal(back.atoms[1].radical, n);
    assert.equal(back.atoms[0].radical, undefined);
  }
  const none = toMolblock(ethyl(0).m);
  assert.doesNotMatch(none, /M {2}RAD/);
  const singlet = toMolblock(ethyl(1).m).replace(/^M {2}RAD {2}1 {3}2 {3}2$/m, "M  RAD  1   2   1");
  assert.equal(fromMolblock(singlet).atoms[1].radical, 2);
  const old = toMolblock(ethyl(0).m).split("\n");
  old[5] = old[5].slice(0, 36) + "  4" + old[5].slice(39);
  assert.equal(fromMolblock(old.join("\n")).atoms[1].radical, 1, "atom-block charge code 4 = doublet");
  // nine radicals: two M  RAD lines of at most eight
  let m = emptySketch();
  const ids = [];
  for (let i = 0; i < 9; i++) {
    let id;
    [m, id] = addAtom(m, "C", i * 1.5, 0);
    ids.push(id);
  }
  const many = toMolblock(setRadical(m, ids, 2));
  assert.equal((many.match(/^M {2}RAD/gm) ?? []).length, 2);
  assert.ok(fromMolblock(many).atoms.every((a) => a.radical === 2));
});

test("fallback H counts drop by the unpaired electrons", () => {
  assert.equal(fallbackHCount("C", 0, 1, 1), 2);
  assert.equal(fallbackHCount("C", 0, 0, 2), 2);
  assert.equal(fallbackHCount("O", 0, 1, 1), 0);
});

test("RDKit: our M  RAD gives C[CH2] and [CH2]; SMILES imports keep the radicals; H counts drop", opts, async () => {
  const rd = await rdkit();
  const eth = analyzeSketch(rd, ethyl(1).m);
  assert.equal(eth.valid, true);
  assert.equal(eth.smiles, "[CH2]C");
  assert.equal(eth.formula, "C2H5");
  assert.deepEqual([...eth.hCounts.values()], [3, 2]);
  const carbene = analyzeSketch(rd, setRadical(addAtom(emptySketch(), "C", 0, 0)[0], [1], 2));
  assert.equal(carbene.smiles, "[CH2]");
  assert.equal(carbene.formula, "CH2");
  for (const [smi, rads, canon] of [["C[CH2]", [0, 1], "[CH2]C"], ["[CH2]", [2], "[CH2]"], ["C[O]", [0, 1], "C[O]"], ["CC(C)[N]C", [0, 0, 0, 1, 0], "C[N]C(C)C"]]) {
    const m = importSketch(rd, smi);
    assert.deepEqual(m.atoms.map((a) => a.radical ?? 0), rads, smi);
    assert.equal(exportSketch(rd, m, "smiles"), canon, smi);
    assert.equal(exportSketch(rd, fromMolblock(toMolblock(m)), "smiles"), canon, `${smi} round trip`);
  }
});

const VIEW = { s: 20, ox: 100, oy: 100 };

test("dots: in the largest free angle, clear of the label; two dots side by side", () => {
  // one bond to the right (screen angle 0): the free angle is straight left
  assert.ok(Math.abs(freeAngle([0]) - Math.PI) < 1e-9);
  // bonds at 0 and 90 deg (down): the largest gap is centred at 225 deg (up-left)
  assert.ok(Math.abs(freeAngle([0, Math.PI / 2]) - (5 * Math.PI) / 4) < 1e-9);
  assert.ok(Math.abs(freeAngle([]) + Math.PI / 2) < 1e-9, "alone: above");
  assert.equal(boxExit({ x0: -5, x1: 5, y0: -6, y1: 6 }, 0), 5);
  const two = radicalDots([0, 0], null, -Math.PI / 2, 2);
  assert.equal(two.length, 2);
  assert.ok(Math.abs(two[0][1] - two[1][1]) < 1e-9 && Math.abs(two[0][0] - two[1][0]) > 4, "side by side across an upward direction");
  const s = buildScene(ethyl(1).m, VIEW, null);
  const c2 = s.atoms[1];
  assert.equal(c2.label, null, "a radical carbon stays a vertex");
  assert.equal(c2.dots.length, 1);
  assert.ok(c2.dots[0][0] > c2.at[0] + 4, "the dot is away from the bond (which goes left)");
  const carbene = buildScene(setRadical(addAtom(emptySketch(), "C", 0, 0)[0], [1], 2), VIEW, null).atoms[0];
  assert.ok(carbene.label, "an isolated carbon is labelled CH2");
  assert.equal(carbene.dots.length, 2);
  assert.ok(carbene.dots.every((d) => d[0] < carbene.at[0] - 6), "left of the label, away from its H2");
});

test("a server layout keeps the radicals by atom id", () => {
  const { m, b } = ethyl(2);
  const next = applyDepiction(m, { coords: [[0, 0], [1.3, 0.75]], depth: null, wedges: [] });
  assert.equal(next.atoms.find((a) => a.id === b).radical, 2);
});
