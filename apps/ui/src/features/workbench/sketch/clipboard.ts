// Copy, cut and duplicate of the sketcher as pure model edits (Ctrl+C / Ctrl+X / Ctrl+D). The
// part copied is the selection (its atoms, the ends of its bonds and every bond between them),
// or the whole drawing when nothing is selected; the clipboard gets it as a V2000 molfile in
// text/plain (ChemDraw and this sketcher both paste that) plus its SMILES under
// chemical/x-daylight-smiles (Sketcher's copy listener). Duplicate puts a copy of the part to the
// right of the drawing and selects it.
import type { SketchAtom, SketchBond, SketchModel } from "@ksmech/viz/sketch/model";
import { BOND_LENGTH } from "@ksmech/viz/sketch/model";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { selectionAtoms, type Selection } from "./tools.ts";

/** The part a copy takes: the selection's atoms and the bonds among them, or everything. */
export function partOf(m: SketchModel, sel: Selection): SketchModel {
  const ids = selectionAtoms(m, sel);
  if (ids.size === 0) return { atoms: m.atoms, bonds: m.bonds, nextId: m.nextId };
  return {
    atoms: m.atoms.filter((a) => ids.has(a.id)),
    bonds: m.bonds.filter((b) => ids.has(b.a) && ids.has(b.b)),
    nextId: m.nextId,
  };
}

/** True when the selection is empty (a copy then takes the whole drawing). */
export const isEmptySelection = (sel: Selection): boolean => sel.atoms.size === 0 && sel.bonds.size === 0;

/** `part` added to `m` with fresh ids, moved by (dx, dy); the new atoms and bonds as a
 *  selection. */
export function appendPart(m: SketchModel, part: SketchModel, dx: number, dy: number): { model: SketchModel; selection: Selection } {
  let next = m.nextId;
  const ids = new Map<number, number>();
  const atoms: SketchAtom[] = part.atoms.map((a) => {
    ids.set(a.id, next);
    return { ...a, id: next++, x: a.x + dx, y: a.y + dy };
  });
  const bonds: SketchBond[] = [];
  for (const b of part.bonds) {
    const a = ids.get(b.a);
    const c = ids.get(b.b);
    if (a !== undefined && c !== undefined) bonds.push({ ...b, id: next++, a, b: c });
  }
  const model = { ...m, atoms: [...m.atoms, ...atoms], bonds: [...m.bonds, ...bonds], nextId: next };
  return { model, selection: { atoms: new Set(atoms.map((a) => a.id)), bonds: new Set(bonds.map((b) => b.id)) } };
}

/** A copy of the selection (or of everything) right of the whole drawing, one bond length
 *  clear of it, and selected; null on an empty drawing. */
export function duplicateSelection(m: SketchModel, sel: Selection): { model: SketchModel; selection: Selection } | null {
  const part = partOf(m, sel);
  if (!part.atoms.length) return null;
  const maxX = Math.max(...m.atoms.map((a) => a.x));
  const minX = Math.min(...part.atoms.map((a) => a.x));
  return appendPart(m, part, maxX - minX + 2 * BOND_LENGTH, 0);
}
