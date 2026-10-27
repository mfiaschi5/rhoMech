// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
// Collision-aware labels (labelLayout.ts) and "tidy fragments" (fragments.ts): the Boc-alanine
// ammonium salt, a zwitterion, a crowded polyol (glucose, ring and open chain) and a sulfonate
// salt in the Screen, ACS 1996 and Total synthesis styles have zero label-label, label-self and
// label-bond overlaps by the box metric (sceneOverlaps: ink boxes at real glyph metrics, bond
// strokes with their width), both from RDKit.js's layout and from the server's Clean layout
// (coordinates recorded from POST /api/depict2d, textbook style, on 2026-10-25). The tidy moves
// whole fragments only (bonds, wedges and RDKit's canonical SMILES unchanged), the charge slot
// and the H side leave a collision, a crowded isolated label is displaced with a leader or
// reported, and the export draws the same placement.
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { addAtom, addBond, emptySketch, setBondStereo, setCharge } from "@ksmech/viz/sketch/model";
import { analyzeSketch, importSketch } from "./rdkitSketch.ts";
import { buildScene } from "@ksmech/viz/sketch/drawing";
import { sceneOverlaps, segInBox, endTrim } from "@ksmech/viz/sketch/labelLayout";
import { fragments, fragmentsTouch, tidyFragments } from "./fragments.ts";
import { PRESETS, drawMetrics, pxPerPt } from "@ksmech/viz/sketch/style";
import { exportLayout } from "@ksmech/viz/sketch/exportImage";
import { toChemfig } from "./latex.ts";

const wasmJs = fileURLToPath(new URL("../../../../node_modules/@rdkit/rdkit/dist/RDKit_minimal.js", import.meta.url));
const available = existsSync(wasmJs);
let rdkitPromise = null;
const rdkit = () => (rdkitPromise ??= createRequire(import.meta.url)(wasmJs)());
const opts = { skip: available ? false : "RDKit.js not installed (npm --prefix apps/ui install)" };

const SMILES = {
  "Boc-Ala NH4+": "C[C@H](NC(=O)OC(C)(C)C)C(=O)[O-].[NH4+]",
  zwitterion: "C[C@@H](C(=O)[O-])[NH3+]",
  glucose: "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
  "open glucose": "OC[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)C=O",
  "Na tosylate": "Cc1ccc(cc1)S(=O)(=O)[O-].[Na+]",
};

/** The server's Clean layouts: [element, x, y, charge] atoms, [a, b, order, stereo] bonds. */
const SERVER = {
  "Boc-Ala NH4+": {
    atoms: [["C", -3.473, 1.556, 0], ["C", -2.302, 0.62, 0], ["N", -0.905, 1.166, 0], ["C", 0.267, 0.23, 0], ["O", 0.042, -1.253, 0], ["O", 1.664, 0.776, 0], ["C", 2.836, -0.16, 0], ["C", 3.772, 1.011, 0], ["C", 1.899, -1.332, 0], ["C", 4.007, -1.097, 0], ["C", -2.527, -0.863, 0], ["O", -3.924, -1.41, 0], ["O", -1.355, -1.8, -1], ["N", 0, 2.556, 1]],
    bonds: [[1, 0, 1, "hash"], [1, 2, 1, "none"], [2, 3, 1, "none"], [3, 4, 2, "none"], [3, 5, 1, "none"], [5, 6, 1, "none"], [6, 7, 1, "none"], [6, 8, 1, "none"], [6, 9, 1, "none"], [1, 10, 1, "none"], [10, 11, 2, "none"], [10, 12, 1, "none"]],
  },
  zwitterion: {
    atoms: [["C", 1.5, 1.299, 0], ["C", 0.75, 0, 0], ["C", -0.75, 0, 0], ["O", -1.5, 1.299, 0], ["O", -1.5, -1.299, -1], ["N", 1.5, -1.299, 1]],
    bonds: [[1, 0, 1, "hash"], [1, 2, 1, "none"], [2, 3, 2, "none"], [2, 4, 1, "none"], [1, 5, 1, "none"]],
  },
  glucose: {
    atoms: [["O", 3.655, 1.144, 0], ["C", 2.813, -0.097, 0], ["C", 1.317, 0.012, 0], ["O", 0.663, 1.363, 0], ["C", -0.833, 1.472, 0], ["O", -1.486, 2.822, 0], ["C", -1.675, 0.231, 0], ["O", -3.171, 0.34, 0], ["C", -1.022, -1.119, 0], ["O", -1.864, -2.36, 0], ["C", 0.474, -1.229, 0], ["O", 1.128, -2.579, 0]],
    bonds: [[0, 1, 1, "none"], [2, 1, 1, "hash"], [2, 3, 1, "none"], [3, 4, 1, "none"], [4, 5, 1, "none"], [4, 6, 1, "none"], [6, 7, 1, "wedge"], [6, 8, 1, "none"], [8, 9, 1, "hash"], [8, 10, 1, "none"], [10, 11, 1, "wedge"], [10, 2, 1, "none"]],
  },
  "Na tosylate": {
    atoms: [["C", 3.955, -0.208, 0], ["C", 2.455, -0.208, 0], ["C", 1.705, 1.091, 0], ["C", 0.205, 1.091, 0], ["C", -0.545, -0.208, 0], ["C", 0.205, -1.507, 0], ["C", 1.705, -1.507, 0], ["S", -2.045, -0.208, 0], ["O", -2.045, 1.292, 0], ["O", -2.045, -1.708, 0], ["O", -3.545, -0.208, -1], ["Na", 0, 2.292, 1]],
    bonds: [[0, 1, 1, "none"], [1, 2, 1, "none"], [2, 3, 2, "none"], [3, 4, 1, "none"], [4, 5, 2, "none"], [5, 6, 1, "none"], [4, 7, 1, "none"], [7, 8, 2, "none"], [7, 9, 2, "none"], [7, 10, 1, "none"], [6, 1, 2, "none"]],
  },
};

function fromLayout(f) {
  let m = emptySketch();
  const ids = [];
  for (const [el, x, y, c] of f.atoms) {
    let id;
    [m, id] = addAtom(m, el, x, y);
    if (c) m = setCharge(m, id, c);
    ids.push(id);
  }
  for (const [a, b, o, st] of f.bonds) {
    let id;
    [m, id] = addBond(m, ids[a], ids[b], o);
    if (st !== "none") m = setBondStereo(m, id, st);
  }
  return m;
}

const STYLES = Object.entries(PRESETS);

function sceneIn(m, h, style, zoom = 25) {
  const view = { s: zoom, ox: 400, oy: 300 };
  return buildScene(m, view, h, { metrics: drawMetrics(style, pxPerPt(style, zoom)) });
}

function assertClean(scene, what) {
  const o = sceneOverlaps(scene);
  assert.deepEqual(o.labelLabel, [], `${what}: label-label`);
  assert.deepEqual(o.labelSelf, [], `${what}: label-self`);
  assert.deepEqual(o.labelBond, [], `${what}: label-bond`);
}

test("zero label overlaps: the four test molecules in all three styles, RDKit.js and server layouts, after the tidy", opts, async () => {
  const rd = await rdkit();
  const layouts = [
    ...Object.entries(SMILES).map(([name, smi]) => [`${name} (RDKit.js)`, importSketch(rd, smi)]),
    ...Object.entries(SERVER).map(([name, f]) => [`${name} (server)`, fromLayout(f)]),
  ];
  for (const [name, m0] of layouts) {
    const h = analyzeSketch(rd, m0).hCounts;
    for (const [sid, style] of STYLES) {
      const m = tidyFragments(m0, h, { style });
      for (const zoom of [18, 25, 40]) assertClean(sceneIn(m, h, style, zoom), `${name} ${sid} ${zoom}`);
    }
  }
});

test("tidy fragments: the server put NH4+ against the NH; the tidy moves only the ion, rigidly, keeping SMILES and stereo", opts, async () => {
  const rd = await rdkit();
  for (const [name, f] of [["Boc-Ala NH4+", SERVER["Boc-Ala NH4+"]], ["Na tosylate", SERVER["Na tosylate"]]]) {
    const m = fromLayout(f);
    const info = analyzeSketch(rd, m);
    for (const [sid, style] of STYLES) {
      // the Screen style's fixed 13 px labels are small at 100 %: there the ions may be clear
      if (sid !== "screen") assert.equal(fragmentsTouch(m, info.hCounts, { style }), true, `${name} ${sid}: the ion touches`);
      const t = tidyFragments(m, info.hCounts, { style });
      if (t === m) continue;
      assert.notEqual(t, m);
      const [big, ion] = fragments(m);
      assert.equal(ion.length, 1);
      const moved = (id) => {
        const a = m.atoms.find((x) => x.id === id);
        const b = t.atoms.find((x) => x.id === id);
        return [b.x - a.x, b.y - a.y];
      };
      big.forEach((id) => assert.deepEqual(moved(id), [0, 0], "the largest fragment stays"));
      assert.ok(Math.hypot(...moved(ion[0])) > 0.05, "the ion moved");
      assert.deepEqual(t.bonds, m.bonds, "bonds, orders and wedges unchanged");
      const after = analyzeSketch(rd, t);
      assert.equal(after.smiles, info.smiles, "same molecule, same stereo");
      assert.equal(fragmentsTouch(t, info.hCounts, { style }), false, "nothing touches after");
      assert.equal(tidyFragments(t, info.hCounts, { style }), t, "idempotent");
    }
  }
});

test("a tidy on the Screen style with the publication presets also clears ACS 1996 and Total synthesis", opts, async () => {
  const rd = await rdkit();
  const m = fromLayout(SERVER["Boc-Ala NH4+"]);
  const h = analyzeSketch(rd, m).hCounts;
  const t = tidyFragments(m, h, { style: PRESETS.screen, alsoFor: [PRESETS.acs1996, PRESETS["total-synthesis"]] });
  for (const [sid, style] of STYLES) {
    assert.equal(fragmentsTouch(t, h, { style }), false, sid);
    const sc = sceneIn(t, h, style);
    assertClean(sc, sid);
    assert.deepEqual(sc.issues, [], `${sid}: nothing left to report`);
  }
});

test("a single fragment and far-apart fragments are left alone", opts, async () => {
  const rd = await rdkit();
  const one = fromLayout(SERVER.glucose);
  assert.equal(tidyFragments(one, analyzeSketch(rd, one).hCounts, { style: PRESETS.acs1996 }), one);
  let m = emptySketch();
  let a;
  [m, a] = addAtom(m, "Na", 0, 0);
  m = setCharge(m, a, 1);
  [m, a] = addAtom(m, "Cl", 6, 0);
  m = setCharge(m, a, -1);
  assert.equal(tidyFragments(m, null, { style: PRESETS["total-synthesis"] }), m);
});

/** A C-O- and a N placed (in font sizes of `style` at zoom 25) where the O's default
 *  upper-right charge would touch it. */
function chargeCrowd(style = PRESETS.acs1996) {
  const f = drawMetrics(style, pxPerPt(style, 25)).font / 25;
  let m = emptySketch();
  let o, c, n;
  [m, c] = addAtom(m, "C", -1.5, 0);
  [m, o] = addAtom(m, "O", 0, 0);
  [m] = addBond(m, c, o, 1);
  m = setCharge(m, o, -1);
  [m, n] = addAtom(m, "N", 1.05 * f, 0.85 * f);
  return { m, o, n };
}

test("the charge slot leaves a collision; without the placement the default slot collides", () => {
  for (const [sid, style] of STYLES) {
    const { m, o } = chargeCrowd(style);
    const view = { s: 25, ox: 300, oy: 300 };
    const M = drawMetrics(style, pxPerPt(style, 25));
    const raw = buildScene(m, view, null, { metrics: M, placeLabels: false });
    assert.equal(raw.atoms.find((a) => a.atom.id === o).label.chargeSlot, "ur");
    assert.ok(sceneOverlaps(raw).labelLabel.length > 0, `${sid}: the default collides`);
    const sc = buildScene(m, view, null, { metrics: M });
    assert.notEqual(sc.atoms.find((a) => a.atom.id === o).label.chargeSlot, "ur", sid);
    assertClean(sc, sid);
  }
});

/** An NH in a chain (H above) with a chloride just above it (in font sizes of `style`). */
function nhUnderChloride(style) {
  const f = drawMetrics(style, pxPerPt(style, 25)).font / 25;
  let m = emptySketch();
  let c1, n, c2, cl;
  [m, c1] = addAtom(m, "C", -1.3, -0.75);
  [m, n] = addAtom(m, "N", 0, 0);
  [m, c2] = addAtom(m, "C", 1.3, -0.75);
  [m] = addBond(m, c1, n, 1);
  [m] = addBond(m, n, c2, 1);
  [m, cl] = addAtom(m, "Cl", 0.2 * f, 1.6 * f);
  m = setCharge(m, cl, -1);
  return { m, n };
}

test("an NH whose H would hit a neighbour's label moves its H to another side", () => {
  const view = { s: 25, ox: 300, oy: 300 };
  for (const [sid, style] of STYLES) {
    const { m, n } = nhUnderChloride(style);
    const M = drawMetrics(style, pxPerPt(style, 25));
    const raw = buildScene(m, view, null, { metrics: M, placeLabels: false });
    assert.equal(raw.atoms.find((a) => a.atom.id === n).label.hSide, "above");
    const sc = buildScene(m, view, null, { metrics: M });
    assert.notEqual(sc.atoms.find((a) => a.atom.id === n).label.hSide, "above", sid);
    assertClean(sc, sid);
  }
});

test("two isolated labels on top of each other: displaced with a leader, or reported", () => {
  let m = emptySketch();
  let a, b;
  [m, a] = addAtom(m, "N", 0, 0);
  m = setCharge(m, a, 1);
  [m, b] = addAtom(m, "Cl", 0.15, 0.05);
  m = setCharge(m, b, -1);
  for (const [sid, style] of STYLES) {
    const sc = sceneIn(m, null, style);
    const displaced = sc.atoms.filter((s) => s.label.offset);
    const o = sceneOverlaps(sc);
    if (o.labelLabel.length) assert.ok(sc.issues.some((i) => i.kind === "label-label"), `${sid}: an overlap left is reported`);
    else assert.ok(displaced.length > 0 && displaced.every((s) => s.leader || Math.hypot(...s.label.offset) < 3 * sc.metrics.line), `${sid}: displaced with a leader`);
    const lay = exportLayout(m, null, style, {});
    const leaders = lay.prims.filter((p) => p.k === "line");
    assert.equal(leaders.length, sc.atoms.filter((s) => s.leader).length, `${sid}: the export draws the same leaders`);
  }
});

test("a bond end clears every glyph box of its label across its whole width", () => {
  const box = { x0: -5, x1: 5, y0: -4, y1: 4 };
  const t = endTrim([0, 0], [40, 10], [box], [-2, 0, 2], 1.5);
  for (const k of [-2, 0, 2]) {
    const n = [-10 / Math.hypot(40, 10), 40 / Math.hypot(40, 10)];
    const u = [40 / Math.hypot(40, 10), 10 / Math.hypot(40, 10)];
    const p = [n[0] * k + u[0] * t, n[1] * k + u[1] * t];
    assert.equal(segInBox(p, [p[0] + u[0] * 30, p[1] + u[1] * 30], { x0: -5.01, x1: 5.01, y0: -4.01, y1: 4.01 }), null, `line ${k} starts outside the box`);
  }
});

test("the export and the chemfig hints follow the placement", opts, async () => {
  const rd = await rdkit();
  const { m, o } = chargeCrowd();
  const style = PRESETS.acs1996;
  const sc = buildScene(m, { s: style.bondLengthPt / 1.5, ox: 0, oy: 0 }, null, { metrics: { ...drawMetrics(style, 1), metricText: true } });
  const slot = sc.atoms.find((a) => a.atom.id === o).label.chargeSlot;
  const lay = exportLayout(m, null, style, {});
  const minus = lay.prims.find((p) => p.k === "text" && p.text === "−");
  const oText = lay.prims.find((p) => p.k === "text" && p.text === "O");
  assert.ok(minus && oText);
  const below = minus.y > oText.y - 1;
  assert.equal(below, slot[0] === "l" || slot === "d", `the export draws the ${slot} slot`);
  const cf = toChemfig(m, null, style);
  if (slot !== "ur" && slot !== "r") assert.match(cf, /\\charge\{-?\d+=\$\\scriptstyle -\$\}\{O\}/);
  const z = importSketch(rd, SMILES.zwitterion);
  assert.match(toChemfig(z, analyzeSketch(rd, z).hCounts, style), /NH_3\^\{\+\}|H_3N\^\{\+\}/);
});
