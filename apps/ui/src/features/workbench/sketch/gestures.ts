// Drag gestures as pure functions: the model a drag would produce (shown as a preview while the
// pointer moves, committed on release). A click (no drag) goes through tools.applyClick.
//   bond / wedge / hash: a bond from the pressed atom (or a new atom at the press point) towards
//     the pointer, one standard bond long at a 30-degree-snapped angle (Shift: free angle), or
//     onto the atom under the pointer;
//   chain: a zigzag chain along the drag (tools.chainPoints);
//   select: moves the pressed object, or the whole selection when the pressed object is in it;
//     a single atom released on another atom merges into it.
import type { SketchModel } from "./model";
import type { Vec } from "./geometry";
import type { Hit, Selection, ToolName } from "./tools";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { BOND_LENGTH, atomById, bondById, mergeAtoms, moveAtoms } from "./model.ts";
// @ts-ignore TS5097: see above
import { add, angleOf, chainPoints, fromAngle, sideOf, snapAngle, sub } from "./geometry.ts";
// @ts-ignore TS5097: see above
import { addChain, bondBetweenPoints } from "./templates.ts";
// @ts-ignore TS5097: see above
import { selectionAtoms } from "./tools.ts";

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

/** The model of a drag from `d` to model point `to` (hit `toHit`), or null when the tool has
 *  no drag action from where it started. */
export function dragModel(d: DragStart, to: Vec, toHit: Hit, freeAngle = false): SketchModel | null {
  const from = d.hit?.kind === "atom" ? d.hit.id : d.at;
  switch (d.tool) {
    case "select": {
      const ids = movedAtoms(d);
      return ids.length ? moveAtoms(d.base, ids, to[0] - d.at[0], to[1] - d.at[1]) : null;
    }
    case "bond":
    case "wedge":
    case "hash": {
      if (d.hit?.kind === "bond") return null;
      const stereo = d.tool === "bond" ? "none" : d.tool;
      const p = startPoint(d);
      const target = toHit?.kind === "atom" && toHit.id !== from ? atomById(d.base, toHit.id) : undefined;
      const t = angleOf(sub(to, p));
      const end: Vec = target ? [target.x, target.y] : add(p, fromAngle(freeAngle ? t : snapAngle(t), BOND_LENGTH));
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
