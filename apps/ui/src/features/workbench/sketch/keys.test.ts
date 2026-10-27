// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
// The ChemDraw-like key map: hover keys on atoms and bonds (tools.applyKey), the ring hotkeys,
// the label editor's parser, the Ctrl+Shift flips, and the cheat sheet table (shortcuts.ts).
import test from "node:test";
import assert from "node:assert/strict";
import { addAtom, addBond, emptySketch, setCharge } from "@ksmech/viz/sketch/model";
import { applyKey, applyRingKey, KEY_ELEMENTS, NO_SELECTION, parseAtomLabel, relabelAtom, ringKey } from "./tools.ts";
import { transformKey } from "./transform.ts";
import { SHORTCUTS, shortcutsOf } from "./shortcuts.ts";

function ethane() {
  let [m, a] = addAtom(emptySketch(), "C", 0, 0);
  let b;
  [m, b] = addAtom(m, "C", 1.5, 0);
  let bond;
  [m, bond] = addBond(m, a, b);
  return { m, a, b, bond };
}

const key = (k, o = {}) => ({ key: k, code: o.code, shiftKey: !!o.shift, ctrlKey: !!o.ctrl, metaKey: !!o.meta, altKey: !!o.alt });

test("bond hover keys: w wedge, h hash, again flips the narrow end; 1 2 3 the order", () => {
  const { m, a, b, bond } = ethane();
  const hit = { kind: "bond", id: bond };
  const w = applyKey(m, "w", hit, NO_SELECTION);
  assert.deepEqual([w.bonds[0].stereo, w.bonds[0].a], ["wedge", a]);
  const flipped = applyKey(w, "W", hit, NO_SELECTION);
  assert.deepEqual([flipped.bonds[0].stereo, flipped.bonds[0].a], ["wedge", b]);
  assert.equal(applyKey(m, "h", hit, NO_SELECTION).bonds[0].stereo, "hash");
  assert.equal(applyKey(m, "2", hit, NO_SELECTION).bonds[0].order, 2);
});

test("atom hover keys: h is still hydrogen on an atom; Space makes a plain carbon", () => {
  const { m, a } = ethane();
  const hit = { kind: "atom", id: a };
  assert.equal(applyKey(m, "h", hit, NO_SELECTION).atoms[0].el, "H");
  const charged = setCharge(applyKey(m, "n", hit, NO_SELECTION), a, 1);
  const carbon = applyKey(charged, " ", hit, NO_SELECTION).atoms[0];
  assert.deepEqual([carbon.el, carbon.charge], ["C", 0]);
  assert.equal(applyKey(m, " ", null, NO_SELECTION), null, "Space with nothing hovered is no model edit");
});

test("w / h on a bond selection set the stereo of the selected bonds", () => {
  const { m, bond } = ethane();
  const sel = { atoms: new Set(), bonds: new Set([bond]) };
  assert.equal(applyKey(m, "w", null, sel).bonds[0].stereo, "wedge");
});

test("ring hotkeys: Shift+3 ... Shift+8 cycloalkanes, Shift+A benzene, by physical key", () => {
  assert.equal(ringKey(key("#", { shift: true, code: "Digit3" })), "cyclopropane");
  assert.equal(ringKey(key("&", { shift: true, code: "Digit7" })), "cycloheptane");
  assert.equal(ringKey(key("*", { shift: true, code: "Digit8" })), "cyclooctane");
  assert.equal(ringKey(key("A", { shift: true, code: "KeyA" })), "benzene");
  assert.equal(ringKey(key("!", { shift: true, code: "Digit1" })), null);
  assert.equal(ringKey(key("3", { code: "Digit3" })), null, "plain 3 stays the bond order");
  assert.equal(ringKey(key("a", { shift: true, ctrl: true, code: "KeyA" })), null);
  const { m, a, bond } = ethane();
  assert.equal(applyRingKey(m, "cyclobutane", { kind: "atom", id: a }).atoms.length, 5, "spiro: 3 new atoms");
  const fused = applyRingKey(m, "benzene", { kind: "bond", id: bond });
  assert.equal(fused.atoms.length, 6);
  assert.equal(fused.bonds.filter((x) => x.order === 2).length, 3);
  assert.equal(applyRingKey(m, "benzene", null), null);
});

test("label editor: element symbols with an optional charge, case-insensitive", () => {
  assert.deepEqual(parseAtomLabel("n+"), { el: "N", charge: 1 });
  assert.deepEqual(parseAtomLabel(" cl "), { el: "Cl", charge: 0 });
  assert.deepEqual(parseAtomLabel("Fe3+"), { el: "Fe", charge: 3 });
  assert.deepEqual(parseAtomLabel("O−"), { el: "O", charge: -1 });
  assert.equal(parseAtomLabel("Xx"), null);
  assert.equal(parseAtomLabel("Me"), null, "no abbreviations");
  const { m, a } = ethane();
  const out = relabelAtom(m, a, { el: "S", charge: -1 }).atoms[0];
  assert.deepEqual([out.el, out.charge], ["S", -1]);
});

test("Ctrl+Shift+H / Ctrl+Shift+V flip (ChemDraw), Alt+H / Alt+V kept; Ctrl+V alone is paste", () => {
  assert.equal(transformKey(key("H", { ctrl: true, shift: true, code: "KeyH" })), "flip-h");
  assert.equal(transformKey(key("V", { meta: true, shift: true, code: "KeyV" })), "flip-v");
  assert.equal(transformKey(key("h", { alt: true, code: "KeyH" })), "flip-h");
  assert.equal(transformKey(key("v", { ctrl: true, code: "KeyV" })), null);
  assert.equal(transformKey(key("K", { ctrl: true, shift: true, code: "KeyK" })), null, "Clean is no transform");
  for (const k of ["w", " ", "Enter", "?"]) assert.equal(transformKey(key(k)), null, k);
});

test("the cheat sheet lists every hover element key and marks the ChemDraw differences", () => {
  const atomKeys = shortcutsOf("atom").map((s) => s.keys.toLowerCase()).join(" ");
  for (const k of Object.keys(KEY_ELEMENTS)) assert.ok(atomKeys.includes(k), k);
  assert.ok(shortcutsOf("bond").some((s) => s.keys === "W"));
  for (const s of SHORTCUTS) assert.ok(["chemdraw", "standard", "chemrho"].includes(s.origin), s.keys);
  assert.ok(SHORTCUTS.filter((s) => s.note).every((s) => s.origin !== "standard"));
});
