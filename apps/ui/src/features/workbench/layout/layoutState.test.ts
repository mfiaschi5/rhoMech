// @ts-nocheck -- run by `node --experimental-strip-types --test` (see sketch/model.test.ts).
// The layout reducers: presets and reset, moves within and across columns and the wide row
// (new column, empty column removed), hide / show / collapse, column splitter clamping, row
// heights, the orbital info toggle, the "custom" mark, the grid template, sanitizing whatever
// storage holds, and the preferences.
import test from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_LAYOUT,
  HEIGHTS,
  MAX_COLUMNS,
  MIN_FRACTION,
  PANELS,
  PRESETS,
  canMove,
  gridTemplate,
  layoutReducer,
  presetLayout,
  sanitizeLayout,
  visibleColumns,
} from "./layoutState.ts";
import { DEFAULT_PREFS, sanitizePrefs, themeAttribute, updatePrefs } from "./prefs.ts";

/** Two columns with two panels each and the properties below (the moves' fixture). */
const B = { ...DEFAULT_LAYOUT, columns: [["sketch", "orbitals"], ["viewer", "surfaces"]], wide: ["properties"] };
const all = (s) => [...s.columns.flat(), ...s.wide].sort();
const run = (s, ...actions) => actions.reduce(layoutReducer, s);
const sum = (xs) => xs.reduce((a, b) => a + b, 0);

test("presets hold every panel once; reset is the default", () => {
  for (const p of PRESETS) {
    const s = presetLayout(p);
    assert.deepEqual(all(s), [...PANELS].sort(), p);
    assert.equal(s.widths.length, s.columns.length);
    assert.ok(Math.abs(sum(s.widths) - 1) < 1e-9);
    assert.equal(s.preset, p);
  }
  assert.deepEqual(DEFAULT_LAYOUT.columns, [["sketch"], ["viewer", "surfaces"]]);
  assert.deepEqual(DEFAULT_LAYOUT.wide, ["orbitals", "properties"], "the orbitals full width: diagram and list side by side");
  const messy = run(DEFAULT_LAYOUT, { type: "hide", panel: "viewer" }, { type: "move", panel: "sketch", dir: "right" });
  assert.deepEqual(run(messy, { type: "reset" }), DEFAULT_LAYOUT);
  assert.equal(run(DEFAULT_LAYOUT, { type: "preset", name: "3d" }).columns[0][0], "viewer");
});

test("up / down swap within a column; down from the bottom goes to the wide row and back up", () => {
  let s = run(B, { type: "move", panel: "orbitals", dir: "up" });
  assert.deepEqual(s.columns[0], ["orbitals", "sketch"]);
  assert.equal(s.preset, "custom");
  assert.equal(canMove(B, "sketch", "up"), false);
  s = run(B, { type: "move", panel: "orbitals", dir: "down" });
  assert.deepEqual(s.columns[0], ["sketch"]);
  assert.deepEqual(s.wide, ["orbitals", "properties"]);
  s = run(s, { type: "move", panel: "orbitals", dir: "up" });
  assert.deepEqual(s.columns[0], ["sketch", "orbitals"], "back to the column with the fewest panels");
  assert.equal(canMove(B, "properties", "down"), false);
});

test("left / right across columns; a new column on the right, an emptied column removed", () => {
  let s = run(B, { type: "move", panel: "orbitals", dir: "right" });
  assert.deepEqual(s.columns, [["sketch"], ["viewer", "orbitals", "surfaces"]]);
  assert.equal(canMove(B, "sketch", "left"), false);
  s = run(B, { type: "move", panel: "surfaces", dir: "right" });
  assert.equal(s.columns.length, 3);
  assert.deepEqual(s.columns[2], ["surfaces"]);
  assert.ok(Math.abs(sum(s.widths) - 1) < 1e-9);
  assert.equal(canMove(s, "surfaces", "right"), false, "alone in the last column");
  s = run(s, { type: "move", panel: "surfaces", dir: "left" });
  assert.equal(s.columns.length, 2, "the empty column goes");
  assert.equal(s.widths.length, 2);
  // a lone panel moving left empties its column
  s = run(B, { type: "move", panel: "viewer", dir: "left" }, { type: "move", panel: "surfaces", dir: "left" });
  assert.deepEqual(s.columns, [["surfaces", "viewer", "sketch", "orbitals"]], "each keeps its row index");
  assert.equal(s.columns.length, 1);
  assert.deepEqual(s.widths, [1]);
  // the wide row: left to the first column, right to the last
  s = run(B, { type: "move", panel: "properties", dir: "right" });
  assert.deepEqual(s.columns[1], ["viewer", "surfaces", "properties"]);
  assert.deepEqual(s.wide, []);
  const three = run(B, { type: "move", panel: "surfaces", dir: "right" }, { type: "move", panel: "orbitals", dir: "right" }, { type: "move", panel: "orbitals", dir: "right" });
  assert.equal(three.columns.length, MAX_COLUMNS);
  assert.equal(canMove(three, "orbitals", "right"), false, `at most ${MAX_COLUMNS} columns`);
});

test("moves skip hidden panels; hide / show keep the place; collapse toggles", () => {
  let s = run(B, { type: "move", panel: "surfaces", dir: "left" });
  assert.deepEqual(s.columns[0], ["sketch", "surfaces", "orbitals"]);
  s = run(s, { type: "hide", panel: "surfaces" }, { type: "move", panel: "orbitals", dir: "up" });
  assert.deepEqual(s.columns[0], ["orbitals", "surfaces", "sketch"], "swapped with the visible sketch");
  s = run(s, { type: "show", panel: "surfaces" });
  assert.deepEqual(s.hidden, []);
  assert.equal(run(s, { type: "show", panel: "surfaces" }), s, "showing a shown panel is a no-op");
  s = run(B, { type: "collapse", panel: "viewer" });
  assert.deepEqual(s.collapsed, ["viewer"]);
  assert.deepEqual(run(s, { type: "collapse", panel: "viewer" }).collapsed, []);
  assert.deepEqual(run(s, { type: "collapse", panel: "viewer", collapsed: true }).collapsed, ["viewer"]);
});

test("column splitter: delta moves width between the pair, clamped; heights clamped", () => {
  const [a, b] = DEFAULT_LAYOUT.widths;
  let s = run(DEFAULT_LAYOUT, { type: "resizeColumns", left: 0, right: 1, from: [a, b], delta: 0.1 });
  assert.ok(Math.abs(s.widths[0] - (a + 0.1)) < 1e-9);
  assert.ok(Math.abs(sum(s.widths) - 1) < 1e-9);
  s = run(DEFAULT_LAYOUT, { type: "resizeColumns", left: 0, right: 1, from: [a, b], delta: -5 });
  assert.ok(Math.abs(s.widths[0] - MIN_FRACTION) < 1e-9);
  s = run(DEFAULT_LAYOUT, { type: "setHeight", panel: "viewer", px: 10_000 });
  assert.equal(s.heights.viewer, HEIGHTS.viewer.max);
  assert.equal(run(DEFAULT_LAYOUT, { type: "setHeight", panel: "sketch", px: 12 }).heights.sketch, HEIGHTS.sketch.min);
  assert.equal(run(DEFAULT_LAYOUT, { type: "setHeight", panel: "sketch", px: DEFAULT_LAYOUT.heights.sketch }), DEFAULT_LAYOUT);
  const info = run(DEFAULT_LAYOUT, { type: "orbitalInfo" });
  assert.equal(info.orbitalInfo, false);
  assert.equal(info.preset, "default", "the info toggle is not a layout change");
});

test("visible columns and the grid template", () => {
  const s = run(DEFAULT_LAYOUT, { type: "hide", panel: "viewer" }, { type: "hide", panel: "surfaces" });
  const cols = visibleColumns(s);
  assert.equal(cols.length, 1);
  assert.equal(cols[0].fraction, 1);
  assert.equal(gridTemplate(cols), "minmax(0, 1.0000fr)");
  assert.equal(gridTemplate(visibleColumns(DEFAULT_LAYOUT)), "minmax(0, 0.4167fr) 8px minmax(0, 0.5833fr)");
});

test("sanitize: junk, duplicates, unknown and missing panels, bad numbers; a round trip is identity", () => {
  for (const junk of [null, undefined, 3, "x", [], {}]) assert.deepEqual(all(sanitizeLayout(junk)), [...PANELS].sort());
  assert.deepEqual(sanitizeLayout(JSON.parse(JSON.stringify(DEFAULT_LAYOUT))), DEFAULT_LAYOUT);
  const custom = run(DEFAULT_LAYOUT, { type: "move", panel: "surfaces", dir: "right" }, { type: "collapse", panel: "sketch" }, { type: "setHeight", panel: "viewer", px: 600 });
  assert.deepEqual(sanitizeLayout(JSON.parse(JSON.stringify(custom))), custom);
  const s = sanitizeLayout({ columns: [["sketch", "sketch", "bogus"], ["viewer"], [], ["orbitals"]], widths: [-1, "a"], heights: { viewer: 1e9, sketch: "big" }, hidden: ["nope", "viewer"], preset: "weird" });
  assert.deepEqual(s.columns, [["sketch"], ["viewer"]], "max three columns, the empty one removed");
  assert.deepEqual(s.wide.sort(), ["orbitals", "properties", "surfaces"]);
  assert.deepEqual(s.widths, [0.5, 0.5]);
  assert.equal(s.heights.viewer, HEIGHTS.viewer.max);
  assert.equal(s.heights.sketch, HEIGHTS.sketch.dflt);
  assert.deepEqual(s.hidden, ["viewer"]);
  assert.equal(s.preset, "custom");
});

test("preferences: defaults, clamping, unknown values, theme attribute", () => {
  assert.deepEqual(sanitizePrefs(null), DEFAULT_PREFS);
  assert.deepEqual(sanitizePrefs(JSON.parse(JSON.stringify(DEFAULT_PREFS))), DEFAULT_PREFS);
  const p = sanitizePrefs({ fontPx: 99, bondWidth: 0.01, colourAtoms: "no", labelStyle: "fancy", rep: "sticks", orbitalIso: -1, theme: "dark" });
  assert.equal(p.fontPx, 22);
  assert.equal(p.bondWidth, 0.6);
  assert.equal(p.colourAtoms, true);
  assert.equal(p.labelStyle, "skeletal");
  assert.equal(p.rep, "sticks");
  assert.equal(p.orbitalIso, 0.005);
  assert.equal(p.theme, "dark");
  assert.equal(updatePrefs(DEFAULT_PREFS, { fontPx: 16.4 }).fontPx, 16);
  assert.equal(themeAttribute("auto"), null);
  assert.equal(themeAttribute("light"), "light");
});
