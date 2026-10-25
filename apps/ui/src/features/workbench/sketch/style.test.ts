// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
// Drawing styles (style.ts): the presets' documented values, the point -> pixel mapping, the
// double-bond spacing as a % of the bond length, the Screen style reproducing the app's former
// pixel constants, sanitizing, and switching styles leaving the model untouched.
import test from "node:test";
import assert from "node:assert/strict";
import {
  ACS_1996_STYLE,
  CSS_PX_PER_PT,
  PRESETS,
  SCREEN_STYLE,
  TOTAL_SYNTHESIS_STYLE,
  actualSizeScale,
  drawMetrics,
  pxPerPt,
  resolveStyle,
  sanitizeStyle,
  screenMetrics,
} from "./style.ts";
import { BREAK_GAP_PX, buildScene, wedgeHalf } from "./drawing.ts";
import { addAtom, addBond, emptySketch, setBondStereo } from "./model.ts";
import { dist, toScreen } from "./geometry.ts";
import { DEFAULT_PREFS, sanitizePrefs } from "../layout/prefs.ts";

test("ACS Document 1996 has ChemDraw's published values", () => {
  const s = ACS_1996_STYLE;
  assert.equal(s.bondLengthPt, 14.4);
  assert.ok(Math.abs((s.bondLengthPt / 72) * 2.54 - 0.508) < 1e-9, "14.4 pt = 0.508 cm");
  assert.equal(s.lineWidthPt, 0.6);
  assert.equal(s.boldWidthPt, 2.0);
  assert.equal(s.marginPt, 1.6);
  assert.equal(s.hashSpacingPt, 2.5);
  assert.equal(s.bondSpacingPct, 18);
  assert.equal(s.chainAngleDeg, 120);
  assert.equal(s.fontSizePt, 10);
  assert.equal(s.fontFamily, "arial");
  assert.equal(s.colour, false);
  assert.equal(s.scaleWithZoom, true);
});

test("Total synthesis: bold lines, tapered wedges, 20 % spacing, Arial 11 pt, black and white", () => {
  const s = TOTAL_SYNTHESIS_STYLE;
  assert.deepEqual(
    [s.bondLengthPt, s.lineWidthPt, s.boldWidthPt, s.hashSpacingPt, s.hashWidthPt, s.bondSpacingPct, s.marginPt, s.fontSizePt],
    [15, 1.1, 3.2, 2.6, 1.0, 20, 2.2, 11],
  );
  assert.equal(s.wedgeShape, "tapered");
  assert.equal(s.colour, false);
  assert.ok(s.lineWidthPt / s.bondLengthPt > ACS_1996_STYLE.lineWidthPt / ACS_1996_STYLE.bondLengthPt, "bolder than ACS");
  assert.deepEqual(Object.keys(PRESETS), ["screen", "acs1996", "total-synthesis"]);
});

test("Screen reproduces the former pixel constants (13 px font, 1.5 px lines, 4.5 px gap)", () => {
  const M = screenMetrics();
  const near = (a, b) => Math.abs(a - b) < 1e-9;
  assert.ok(near(M.font, 13) && near(M.line, 1.5) && near(M.doubleGap, 4.5) && near(M.bold / 2, 3.6));
  assert.ok(near(M.breakGap, BREAK_GAP_PX) && near(M.hashSpacing, 4.5) && near(M.bond, 37.5));
  assert.equal(pxPerPt(SCREEN_STYLE, 25), CSS_PX_PER_PT);
  assert.equal(pxPerPt(SCREEN_STYLE, 80), CSS_PX_PER_PT, "Screen sizes do not follow the zoom");
});

test("points map to pixels through the zoom: a standard bond is the fixed bond length", () => {
  for (const s of [ACS_1996_STYLE, TOTAL_SYNTHESIS_STYLE]) {
    for (const zoom of [10, 25, 60]) {
      const k = pxPerPt(s, zoom);
      const M = drawMetrics(s, k);
      assert.ok(Math.abs(M.bond - 1.5 * zoom) < 1e-9, "1.5 model units on screen = bondLengthPt");
      assert.ok(Math.abs(M.line / M.bond - s.lineWidthPt / s.bondLengthPt) < 1e-12, "proportions kept at any zoom");
      assert.ok(Math.abs(M.font - s.fontSizePt * k) < 1e-9);
    }
  }
  assert.ok(Math.abs(actualSizeScale(ACS_1996_STYLE) * 1.5 - 14.4 * CSS_PX_PER_PT) < 1e-9, "actual size: 19.2 px bonds");
});

/** An isolated C=C (centred) and a benzene-like ring bond (offset), horizontal. */
function doubles() {
  let m = emptySketch();
  let a, b;
  [m, a] = addAtom(m, "C", 0, 0);
  [m, b] = addAtom(m, "C", 1.5, 0);
  [m] = addBond(m, a, b, 2);
  return m;
}

test("the double-bond spacing is the style's % of the fixed bond length", () => {
  const view = { s: 25, ox: 100, oy: 100 };
  for (const s of [ACS_1996_STYLE, TOTAL_SYNTHESIS_STYLE]) {
    const M = drawMetrics(s, pxPerPt(s, view.s));
    const [l1, l2] = buildScene(doubles(), view, null, { metrics: M }).bonds[0].shape.lines;
    const gap = Math.abs(l1.p[1] - l2.p[1]);
    assert.ok(Math.abs(gap - (s.bondSpacingPct / 100) * M.bond) < 1e-9, `${s.bondSpacingPct} % of ${M.bond} px`);
  }
});

test("wedges: classic grows from a point to the bold width, tapered starts at the line width", () => {
  const M = drawMetrics(ACS_1996_STYLE, 2);
  assert.ok(wedgeHalf(0, M) < M.line / 2 && Math.abs(wedgeHalf(1, M) - M.bold / 2) < 1e-9);
  const T = drawMetrics(TOTAL_SYNTHESIS_STYLE, 2);
  assert.ok(Math.abs(wedgeHalf(0, T) - T.line / 2) < 1e-9 && Math.abs(wedgeHalf(0.8, T) - T.bold / 2) < 1e-9);
});

test("sanitizeStyle clamps numbers and drops unknown values; resolveStyle applies the screen prefs", () => {
  const s = sanitizeStyle({ lineWidthPt: 99, bondLengthPt: "x", fontFamily: "comic", wedgeShape: "tapered", colour: "yes" });
  assert.equal(s.lineWidthPt, 5);
  assert.equal(s.bondLengthPt, ACS_1996_STYLE.bondLengthPt);
  assert.equal(s.fontFamily, ACS_1996_STYLE.fontFamily);
  assert.equal(s.wedgeShape, "tapered");
  assert.equal(s.colour, ACS_1996_STYLE.colour);
  const scr = resolveStyle("screen", null, { fontPx: 16, bondWidth: 2, colourAtoms: false });
  assert.ok(Math.abs(scr.fontSizePt * CSS_PX_PER_PT - 16) < 1e-9 && Math.abs(scr.lineWidthPt * CSS_PX_PER_PT - 2) < 1e-9 && scr.colour === false);
  assert.equal(resolveStyle("acs1996", null, { fontPx: 20 }), ACS_1996_STYLE, "publication styles ignore the screen prefs");
  const custom = { ...TOTAL_SYNTHESIS_STYLE, lineWidthPt: 2 };
  assert.equal(resolveStyle("custom", custom).lineWidthPt, 2);
});

test("the preferences persist the style id and a sanitized custom style", () => {
  assert.equal(DEFAULT_PREFS.drawingStyle, "screen");
  const p = sanitizePrefs({ drawingStyle: "acs1996", customStyle: { lineWidthPt: -3, wedgeShape: "tapered" } });
  assert.equal(p.drawingStyle, "acs1996");
  assert.equal(p.customStyle.lineWidthPt, 0.1);
  assert.equal(p.customStyle.wedgeShape, "tapered");
  assert.equal(sanitizePrefs({ drawingStyle: "rsc" }).drawingStyle, "screen");
});

test("switching styles is render-only: the model and the atom positions do not change", () => {
  let m = emptySketch();
  const ids = [];
  for (const [x, y, el] of [[0, 0, "C"], [1.3, 0.75, "C"], [2.6, 0, "O"], [1.3, 2.25, "N"]]) {
    let id;
    [m, id] = addAtom(m, el, x, y);
    ids.push(id);
  }
  [m] = addBond(m, ids[0], ids[1], 1);
  [m] = addBond(m, ids[1], ids[2], 2);
  let b;
  [m, b] = addBond(m, ids[1], ids[3], 1);
  m = setBondStereo(m, b, "wedge");
  const before = JSON.stringify(m);
  const view = { s: 30, ox: 50, oy: 200 };
  const scenes = [SCREEN_STYLE, ACS_1996_STYLE, TOTAL_SYNTHESIS_STYLE].map((s) => buildScene(m, view, null, { metrics: drawMetrics(s, pxPerPt(s, view.s)) }));
  assert.equal(JSON.stringify(m), before);
  for (const sc of scenes) {
    sc.atoms.forEach((a) => assert.ok(dist(a.at, toScreen(view, [a.atom.x, a.atom.y])) < 1e-9));
    assert.equal(sc.bonds.length, 3);
  }
  assert.notDeepEqual(scenes[1].bonds[2].shape.wedges, scenes[2].bonds[2].shape.wedges, "the wedge shape differs");
});
