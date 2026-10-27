// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
// Rotate / flip / scale math, the selection frame and its handle gestures, the key map (no
// collision with the existing shortcuts) and one history step per transform.
import test from "node:test";
import assert from "node:assert/strict";
import { BOND_LENGTH, addAtom, addBond, emptySketch, meanBondLength, setBondStereo } from "@ksmech/viz/sketch/model";
import { DEG, toScreen } from "@ksmech/viz/sketch/geometry";
import { initHistory, pushHistory, undo, redo } from "./history.ts";
import { KEY_ELEMENTS, NO_SELECTION } from "./tools.ts";
import {
  applyTransform,
  pivotOf,
  dragAngle,
  dragScale,
  flipAtoms,
  resetBondLength,
  rotateAtoms,
  scaleAtoms,
  transformKey,
  transformTargets,
  TRANSFORM_ACTIONS,
} from "./transform.ts";
import { frameHandleAt, handleDragModel, selectionFrame } from "./frame.ts";

/** A-B-C chain with a wedge A->B (narrow end A) and a depth on every atom. */
function chain() {
  let m = emptySketch();
  const ids = [];
  for (const [x, y, z] of [[0, 0, 0.5], [1.5, 0, -0.2], [3, 1, 0.1]]) {
    let id;
    [m, id] = addAtom(m, "C", x, y);
    m = { ...m, atoms: m.atoms.map((a) => (a.id === id ? { ...a, z } : a)) };
    ids.push(id);
  }
  [m] = addBond(m, ids[0], ids[1]);
  [m] = addBond(m, ids[1], ids[2]);
  m = setBondStereo(m, m.bonds[0].id, "wedge");
  return { m, ids };
}

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} vs ${b}`);
const xy = (m) => m.atoms.map((a) => [a.x, a.y]);

test("targets: the selection's atoms (bond ends too), else the whole drawing", () => {
  const { m, ids } = chain();
  assert.deepEqual(transformTargets(m, NO_SELECTION), ids);
  assert.deepEqual(transformTargets(m, { atoms: new Set([ids[2]]), bonds: new Set([m.bonds[0].id]) }), ids);
  assert.deepEqual(transformTargets(m, { atoms: new Set([ids[0]]), bonds: new Set() }), [ids[0]]);
  assert.deepEqual(pivotOf(m, ids), [1.5, 1 / 3]);
});

test("rotation about the box centre keeps distances, depth, ids and wedges", () => {
  const { m, ids } = chain();
  const r = rotateAtoms(m, ids, 90 * DEG);
  assert.deepEqual(r.atoms.map((a) => a.id), ids);
  assert.deepEqual(r.bonds, m.bonds);
  assert.deepEqual(r.atoms.map((a) => a.z), m.atoms.map((a) => a.z));
  close(meanBondLength(r), meanBondLength(m));
  // (0,0) about the centroid (1.5,1/3) by +90 deg (counter-clockwise) -> (11/6, -7/6)
  close(r.atoms[0].x, 11 / 6);
  close(r.atoms[0].y, -7 / 6);
  const back = applyTransform(applyTransform(m, ids, "rotate-cw15"), ids, "rotate-ccw15");
  xy(back).forEach((p, i) => (close(p[0], m.atoms[i].x), close(p[1], m.atoms[i].y)));
  assert.equal(rotateAtoms(m, ids, 0), m);
});

test("a partial rotation moves only the targets", () => {
  const { m, ids } = chain();
  const r = rotateAtoms(m, [ids[1], ids[2]], 180 * DEG);
  assert.equal(r.atoms[0], m.atoms[0]);
  close(r.atoms[1].x, 3);
  close(r.atoms[1].y, 1);
  close(r.atoms[2].y, 0);
});

test("uniform scale scales coordinates and depth about the centre; reset bond length", () => {
  const { m, ids } = chain();
  const s = scaleAtoms(m, ids, 2);
  close(meanBondLength(s), 2 * meanBondLength(m));
  pivotOf(s, ids).forEach((v, i) => close(v, pivotOf(m, ids)[i]));
  assert.deepEqual(s.atoms.map((a) => a.z), [1, -0.4, 0.2]);
  assert.deepEqual(s.bonds, m.bonds);
  const up = applyTransform(m, ids, "scale-up");
  close(meanBondLength(up), 1.1 * meanBondLength(m));
  close(meanBondLength(applyTransform(up, ids, "scale-down")), meanBondLength(m));
  close(meanBondLength(resetBondLength(s, ids)), BOND_LENGTH);
  const lone = addAtom(emptySketch(), "C", 1, 1)[0];
  assert.equal(resetBondLength(lone, [lone.atoms[0].id]), lone, "no bond: nothing to reset");
});

test("flip mirrors the targets, negates their depth and swaps the marks at flipped centres", () => {
  const { m, ids } = chain();
  const h = flipAtoms(m, ids, "h");
  assert.deepEqual(xy(h), [[3, 0], [1.5, 0], [0, 1]]);
  assert.deepEqual(h.atoms.map((a) => a.z), [-0.5, 0.2, -0.1]);
  assert.equal(h.bonds[0].stereo, "hash");
  assert.equal(h.bonds[0].a, ids[0], "the narrow end stays the stereocentre");
  assert.equal(h.bonds[1].stereo, "none");
  const v = flipAtoms(m, ids, "v");
  xy(v).flat().forEach((c, i) => close(c, [0, 2 / 3, 1.5, 2 / 3, 3, -1 / 3][i]));
  assert.equal(v.bonds[0].stereo, "hash");
  const twice = flipAtoms(h, ids, "h");
  twice.atoms.forEach((a, i) => (close(a.x, m.atoms[i].x), close(a.y, m.atoms[i].y), close(a.z, m.atoms[i].z)));
  assert.deepEqual(twice.bonds, m.bonds);
  // a mark whose narrow end is not flipped is left alone
  const part = flipAtoms(m, [ids[1], ids[2]], "h");
  assert.equal(part.bonds[0].stereo, "wedge");
});

test("every action on an empty drawing is a no-op", () => {
  const e = emptySketch();
  for (const a of TRANSFORM_ACTIONS) assert.equal(applyTransform(e, [], a), e, a);
});

test("each transform is one history step; undo restores the coordinates", () => {
  const { m, ids } = chain();
  let h = initHistory(m);
  h = pushHistory(h, applyTransform(h.present, ids, "rotate-cw15"));
  h = pushHistory(h, applyTransform(h.present, ids, "flip-h"));
  assert.equal(h.past.length, 2);
  h = undo(h);
  assert.equal(h.present.bonds[0].stereo, "wedge");
  h = undo(h);
  assert.equal(h.present, m);
  assert.equal(redo(h).present.atoms[0].x !== m.atoms[0].x, true);
});

const key = (key, mods = {}) => ({ key, code: mods.code, shiftKey: !!mods.shift, ctrlKey: !!mods.ctrl, metaKey: false, altKey: !!mods.alt });

test("keys: R / Shift+R, Alt+H / Alt+V (by code), Ctrl+= / Ctrl+-", () => {
  assert.equal(transformKey(key("r")), "rotate-cw15");
  assert.equal(transformKey(key("R", { shift: true })), "rotate-ccw15");
  assert.equal(transformKey(key("˙", { alt: true, code: "KeyH" })), "flip-h");
  assert.equal(transformKey(key("v", { alt: true, code: "KeyV" })), "flip-v");
  assert.equal(transformKey(key("=", { ctrl: true })), "scale-up");
  assert.equal(transformKey(key("+", { ctrl: true, shift: true })), "scale-up");
  assert.equal(transformKey(key("-", { ctrl: true })), "scale-down");
});

test("keys: no collision with the element, bond, charge, layout and editor shortcuts", () => {
  const plain = [...Object.keys(KEY_ELEMENTS), "1", "2", "3", "+", "=", "-", "_", "Delete", "Backspace", "Escape"];
  for (const k of plain) assert.equal(transformKey(key(k)), null, k);
  for (const k of ["D", "X"]) assert.equal(transformKey(key(k, { shift: true })), null, `Shift+${k}`);
  for (const k of ["z", "y", "a", "v", "Z"]) assert.equal(transformKey(key(k, { ctrl: true })), null, `Ctrl+${k}`);
  assert.equal(transformKey(key("h")), null, "plain h stays the H element");
  assert.equal(transformKey(key("r", { ctrl: true })), null, "Ctrl+R stays the browser's reload");
});

test("drag angle (Shift: 15 degree steps) and drag scale (clamped; Shift: 5 % steps)", () => {
  close(dragAngle([0, 0], [1, 0], [0, 1], false), 90 * DEG);
  close(dragAngle([0, 0], [1, 0], [Math.cos(0.3), Math.sin(0.3)], true), 15 * DEG);
  close(dragAngle([0, 0], [-1, 0.01], [-1, -0.01], false), 0.02, 1e-3);
  close(dragScale([0, 0], [1, 0], [2, 0], false), 2);
  close(dragScale([0, 0], [1, 0], [1.33, 0], true), 1.35);
  close(dragScale([0, 0], [1, 0], [100, 0], false), 10);
  close(dragScale([0, 0], [1, 0], [0, 0], false), 0.1);
});

const VIEW = { s: 20, ox: 100, oy: 100 };

test("frame: box, handles, hit test; none for fewer than two atoms", () => {
  const { m, ids } = chain();
  assert.equal(selectionFrame(m, [ids[0]], VIEW), null);
  const f = selectionFrame(m, ids, VIEW);
  assert.deepEqual(f.ids, ids);
  assert.deepEqual(f.centre, [1.5, 1 / 3]);
  assert.deepEqual(f.box, { x0: 86, y0: 66, x1: 174, y1: 114 });
  assert.deepEqual(frameHandleAt(f, f.rotate), { kind: "rotate" });
  assert.deepEqual(frameHandleAt(f, [175, 115]), { kind: "scale", corner: 2 });
  assert.equal(frameHandleAt(f, [130, 90]), null);
});

test("handle drags: a quarter turn of the rotate handle, a corner pulled to twice the distance", () => {
  const { m, ids } = chain();
  const f = selectionFrame(m, ids, VIEW);
  const c = toScreen(VIEW, f.centre);
  const from = f.rotate;
  const r = from[1] - c[1]; // negative: above the centre on screen
  const quarter = handleDragModel({ handle: { kind: "rotate" }, ids, centre: f.centre, from, base: m }, VIEW, [c[0] - r, c[1]], false);
  const ref = rotateAtoms(m, ids, -90 * DEG, f.centre); // dragged to the right of the centre: clockwise
  quarter.atoms.forEach((a, i) => (close(a.x, ref.atoms[i].x), close(a.y, ref.atoms[i].y)));
  const corner = f.corners[2];
  const far = [2 * corner[0] - c[0], 2 * corner[1] - c[1]];
  const big = handleDragModel({ handle: { kind: "scale", corner: 2 }, ids, centre: f.centre, from: corner, base: m }, VIEW, far, false);
  close(meanBondLength(big), 2 * meanBondLength(m));
});
