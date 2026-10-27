// Drag gestures as pure functions: the model a drag would produce (shown as a preview while the
// pointer moves, committed on release). A click (no drag) goes through tools.applyClick.
//   bond / wedge / hash: a bond from the pressed atom (or a new atom at the press point) towards
//     the pointer, or onto the atom under the pointer;
//   chain: a zigzag chain along the drag (tools.chainPoints);
//   select: moves the pressed object, or the whole selection when the pressed object is in it;
//     a single atom released on another atom merges into it.
// Fixed lengths and fixed angles (ChemDraw's Object menu; the toolbar toggles, Snap): a drawn
// bond is one standard bond long (BOND_LENGTH) at a multiple of 30 degrees (ChemDraw's own
// default step is 15; this sketcher's chain geometry is built on 30). A dragged atom with one
// bonded neighbour (a bond end) lands at the same fixed length and angle from that neighbour,
// or exactly on the atom under the pointer (released there, it joins it). Alt or Shift held
// during the drag turns both off for that drag (ChemDraw: Alt / Option); a toggle turned off
// turns its constraint off for every drag.
import type { SketchModel } from "@ksmech/viz/sketch/model";
import type { Vec } from "@ksmech/viz/sketch/geometry";
import type { Hit, Selection, ToolName } from "./tools";
import { BOND_LENGTH, atomById, bondById, mergeAtoms, moveAtoms, neighbors } from "@ksmech/viz/sketch/model";
import { add, angleOf, chainPoints, fromAngle, sideOf, snapAngle, sub } from "@ksmech/viz/sketch/geometry";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { addChain, bondBetweenPoints } from "./templates.ts";
// @ts-ignore TS5097: see above
import { selectionAtoms } from "./tools.ts";

/** Which constraints a drag keeps. */
export interface Snap {
  length: boolean;
  angle: boolean;
}

export const FIXED: Snap = { length: true, angle: true };
export const FREE: Snap = { length: false, angle: false };

/** The end of a bond from `p` towards `to` under `snap`: BOND_LENGTH long with a fixed length
 *  (else as long as the pointer is far), its angle a multiple of 30 degrees with fixed angles. */
export function snapEnd(p: Vec, to: Vec, snap: Snap, length = BOND_LENGTH): Vec {
  if (!snap.length && !snap.angle) return to;
  const v = sub(to, p);
  const r = Math.hypot(v[0], v[1]);
  if (r < 1e-9) return snap.length ? add(p, [length, 0]) : to;
  const t = snap.angle ? snapAngle(angleOf(v)) : angleOf(v);
  return add(p, fromAngle(t, snap.length ? length : r));
}

/** The snap of a drag event: the toggles, both off while Alt or Shift is held. */
export function dragSnap(toggles: Snap, mods: { altKey: boolean; shiftKey: boolean }): Snap {
  return mods.altKey || mods.shiftKey ? FREE : toggles;
}

export interface DragStart {
  tool: ToolName;
  hit: Hit;
  /** Press point, model frame. */
  at: Vec;
  base: SketchModel;
  selection: Selection;
}

/** The ids a select-drag moves. */
export function movedAtoms(d: DragStart): number[] {
  const { hit, base, selection } = d;
  if (!hit) return [];
  const inSel = hit.kind === "atom" ? selection.atoms.has(hit.id) : selection.bonds.has(hit.id);
  if (inSel) return [...selectionAtoms(base, selection)];
  if (hit.kind === "atom") return [hit.id];
  const b = bondById(base, hit.id);
  return b ? [b.a, b.b] : [];
}

function startPoint(d: DragStart): Vec {
  if (d.hit?.kind === "atom") {
    const a = atomById(d.base, d.hit.id);
    if (a) return [a.x, a.y];
  }
  return d.at;
}

/** The one bonded neighbour of a lone moved atom that stays put (the atom is a bond end), else
 *  null (several atoms move, or the atom has no or several neighbours). */
export function anchorOf(base: SketchModel, ids: readonly number[]): number | null {
  if (ids.length !== 1) return null;
  const nbrs = neighbors(base, ids[0]);
  return nbrs.length === 1 ? nbrs[0] : null;
}

/** A select-drag: the moved atoms follow the pointer; a lone bond end snaps to its neighbour
 *  (or onto the atom under the pointer). */
function moveDrag(d: DragStart, to: Vec, toHit: Hit, snap: Snap): SketchModel | null {
  const ids = movedAtoms(d);
  if (!ids.length) return null;
  const dx = to[0] - d.at[0];
  const dy = to[1] - d.at[1];
  const anchor = anchorOf(d.base, ids);
  const atom = atomById(d.base, ids[0]);
  if (anchor === null || !atom || (!snap.length && !snap.angle)) return moveAtoms(d.base, ids, dx, dy);
  const target = toHit?.kind === "atom" && toHit.id !== ids[0] ? atomById(d.base, toHit.id) : undefined;
  const a = atomById(d.base, anchor);
  const end: Vec = target ? [target.x, target.y] : a ? snapEnd([a.x, a.y], [atom.x + dx, atom.y + dy], snap) : [atom.x + dx, atom.y + dy];
  return moveAtoms(d.base, ids, end[0] - atom.x, end[1] - atom.y);
}

/** The model of a drag from `d` to model point `to` (hit `toHit`) under `snap`, or null when
 *  the tool has no drag action from where it started. */
export function dragModel(d: DragStart, to: Vec, toHit: Hit, snap: Snap = FIXED): SketchModel | null {
  const from = d.hit?.kind === "atom" ? d.hit.id : d.at;
  switch (d.tool) {
    case "select":
      return moveDrag(d, to, toHit, snap);
    case "bond":
    case "wedge":
    case "hash": {
      if (d.hit?.kind === "bond") return null;
      const stereo = d.tool === "bond" ? "none" : d.tool;
      const p = startPoint(d);
      const target = toHit?.kind === "atom" && toHit.id !== from ? atomById(d.base, toHit.id) : undefined;
      const end: Vec = target ? [target.x, target.y] : snapEnd(p, to, snap);
      return bondBetweenPoints(d.base, from, end, 1, stereo).model;
    }
    case "chain": {
      if (d.hit?.kind === "bond") return null;
      const p = startPoint(d);
      const snapped = add(p, fromAngle(snapAngle(angleOf(sub(to, p)))));
      const pts = chainPoints(p, to, BOND_LENGTH, sideOf(p, snapped, to) < 0);
      return addChain(d.base, from, pts).model;
    }
    default:
      return null;
  }
}

/** The model to commit at the end of a select-drag: a lone atom dropped on another atom merges
 *  into it. */
export function finishMove(d: DragStart, moved: SketchModel, dropHit: Hit): SketchModel {
  const ids = movedAtoms(d);
  if (ids.length === 1 && dropHit?.kind === "atom" && dropHit.id !== ids[0]) return mergeAtoms(moved, dropHit.id, ids[0]);
  return moved;
}
