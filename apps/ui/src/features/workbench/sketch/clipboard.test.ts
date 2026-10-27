// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
// Copy / cut / duplicate parts (clipboard.ts): the part of a selection, fresh ids on append, the
// duplicate placed right of the drawing and selected.
import test from "node:test";
import assert from "node:assert/strict";
import { addAtom, addBond, emptySketch, BOND_LENGTH } from "@ksmech/viz/sketch/model";
import { NO_SELECTION } from "./tools.ts";
import { appendPart, duplicateSelection, partOf } from "./clipboard.ts";

function propanol() {
  let m = emptySketch();
  const ids = [];
  for (const [el, x] of [["C", 0], ["C", 1.5], ["O", 3]]) {
    let id;
    [m, id] = addAtom(m, el, x, 0);
    ids.push(id);
  }
  [m] = addBond(m, ids[0], ids[1]);
  [m] = addBond(m, ids[1], ids[2], 1, "wedge");
  return { m, ids };
}

test("partOf: the selected atoms and the bonds among them; everything without a selection", () => {
  const { m, ids } = propanol();
  const part = partOf(m, { atoms: new Set([ids[1], ids[2]]), bonds: new Set() });
  assert.deepEqual(part.atoms.map((a) => a.el), ["C", "O"]);
  assert.equal(part.bonds.length, 1);
  assert.equal(partOf(m, NO_SELECTION).atoms.length, 3);
  const byBond = partOf(m, { atoms: new Set(), bonds: new Set([m.bonds[0].id]) });
  assert.equal(byBond.atoms.length, 2, "a bond brings its ends");
});

test("appendPart: fresh ids, the same elements, bonds and stereo, moved", () => {
  const { m } = propanol();
  const { model, selection } = appendPart(m, m, 10, 1);
  assert.equal(model.atoms.length, 6);
  assert.equal(model.bonds.length, 4);
  assert.equal(new Set([...model.atoms.map((a) => a.id), ...model.bonds.map((b) => b.id)]).size, 10, "ids unique");
  const copy = model.atoms.filter((a) => selection.atoms.has(a.id));
  assert.deepEqual(copy.map((a) => [a.el, a.x, a.y]), [["C", 10, 1], ["C", 11.5, 1], ["O", 13, 1]]);
  assert.equal(model.bonds.filter((b) => selection.bonds.has(b.id) && b.stereo === "wedge").length, 1);
});

test("duplicateSelection: a copy right of the drawing, two bond lengths clear, selected", () => {
  const { m, ids } = propanol();
  const dup = duplicateSelection(m, { atoms: new Set([ids[0]]), bonds: new Set() });
  const added = dup.model.atoms.filter((a) => dup.selection.atoms.has(a.id));
  assert.equal(added.length, 1);
  assert.equal(added[0].x, 3 + 2 * BOND_LENGTH);
  assert.equal(duplicateSelection(emptySketch(), NO_SELECTION), null);
});
