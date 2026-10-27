// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// Prose of the step viewport: diagnostic symbols, the reading caption, alternatives.
import test from "node:test";
import assert from "node:assert/strict";
import { captionText, diagSymbol, dominantReading, entryWords, stepAlternative } from "./stepText.ts";

const R = { i: 1, j: 5, g_atom: -1.2801, donor_token: 16, acceptor_token: 28, g_rs: -0.35, term: -0.219, donor_diag: "lone_pair(O)", acceptor_diag: "sigma_star(C-Cl)", canonical: { k: 6, m: 7, e_k_ev: -8.5479, e_m_ev: -4.2121, k_label: "HOMO", m_label: "LUMO", term: -0.28, share: 0.79, index_source: "qm" } };

test("diagnostic symbols", () => {
  assert.equal(diagSymbol("lone_pair(O)"), "n(O)");
  assert.equal(diagSymbol("sigma_star(C-Cl)"), "σ*(C–Cl)");
  assert.equal(diagSymbol("pi_star(C-O)"), "π*(C–O)");
  assert.equal(diagSymbol("empty_p(B)"), "p(B)");
  assert.equal(diagSymbol("delocalized_pi(C-C)"), "π(deloc., C–C)");
  assert.equal(diagSymbol(null), null);
  assert.equal(diagSymbol("weird"), "weird");
});

test("the caption of a reading", () => {
  assert.equal(captionText(R), "n(O) → σ*(C–Cl) · G^atom −1.28 eV⁻¹ · HOMO → LUMO, −8.55 / −4.21 eV");
  assert.equal(captionText({ ...R, donor_diag: null, canonical: null }), "orbital 16 → σ*(C–Cl) · G^atom −1.28 eV⁻¹");
  assert.match(captionText({ ...R, canonical: { ...R.canonical, k_label: "HOMO-1" } }), /HOMO−1 → LUMO/);
});

test("the dominant reading has the largest |term|", () => {
  assert.equal(dominantReading([R, { ...R, term: -0.5 }, { ...R, term: 0.3 }]), 1);
  assert.equal(dominantReading([]), -1);
});

test("entries in words", () => {
  assert.equal(entryWords("bond", 1, 0), "bond forms");
  assert.equal(entryWords("bond", -1, 1), "bond breaks");
  assert.equal(entryWords("bond", -1, 2), "bond order 2 → 1");
  assert.equal(entryWords("lone", -2, 0), "gives up a lone pair");
});

test("alternatives cycle through the edge (-1) and its n alternatives", () => {
  assert.equal(stepAlternative(-1, 3, 1), 0);
  assert.equal(stepAlternative(2, 3, 1), -1);
  assert.equal(stepAlternative(-1, 3, -1), 2);
  assert.equal(stepAlternative(0, 3, -1), -1);
  assert.equal(stepAlternative(-1, 0, 1), -1);
});
