// The selection frame of the sketcher: a box around the selected atoms (screen pixels) with four
// corner handles (drag: scale about the selection's centroid, transform.pivotOf) and a
// rotation handle above the top edge (drag: rotate about the centroid). Pure geometry plus the
// model a handle drag produces; drawn by SketchCanvas, driven by useSketchPointer.
import type { SketchModel } from "./model";
import type { Rect, Vec, View } from "./geometry";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { dist, toModel, toScreen } from "./geometry.ts";
// @ts-ignore TS5097: see above
import { dragAngle, dragScale, pivotOf, rotateAtoms, scaleAtoms } from "./transform.ts";

/** Padding of the box around the atom centres, the rotation stalk and the handle hit radius (px). */
export const FRAME_PAD = 14;
export const ROTATE_STALK = 22;
export const HANDLE_HIT = 8;

export interface SelectionFrame {
  /** Atoms the handles move. */
  ids: number[];
  /** Screen box (padded). */
  box: Rect;
  /** Pivot of both gestures: the atoms' centroid, model frame. */
  centre: Vec;
  /** Corner handles, screen: top-left, top-right, bottom-right, bottom-left. */
  corners: Vec[];
  /** Rotation handle, screen. */
  rotate: Vec;
}

export type FrameHandle = { kind: "rotate" } | { kind: "scale"; corner: number };

/** The frame of the given atoms (at least two: one atom has nothing to rotate or scale). */
export function selectionFrame(m: SketchModel, ids: readonly number[], view: View): SelectionFrame | null {
  const set = new Set(ids);
  const atoms = m.atoms.filter((a) => set.has(a.id));
  const centre = pivotOf(m, set);
  if (atoms.length < 2 || !centre) return null;
  const pts = atoms.map((a) => toScreen(view, [a.x, a.y]));
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const box: Rect = { x0: Math.min(...xs) - FRAME_PAD, y0: Math.min(...ys) - FRAME_PAD, x1: Math.max(...xs) + FRAME_PAD, y1: Math.max(...ys) + FRAME_PAD };
  const corners: Vec[] = [
    [box.x0, box.y0],
    [box.x1, box.y0],
    [box.x1, box.y1],
    [box.x0, box.y1],
  ];
  return { ids: atoms.map((a) => a.id), box, centre, corners, rotate: [(box.x0 + box.x1) / 2, box.y0 - ROTATE_STALK] };
}

/** The handle under screen point p, rotation first. */
export function frameHandleAt(f: SelectionFrame | null, p: Vec): FrameHandle | null {
  if (!f) return null;
  if (dist(p, f.rotate) <= HANDLE_HIT) return { kind: "rotate" };
  const corner = f.corners.findIndex((c) => dist(p, c) <= HANDLE_HIT);
  return corner >= 0 ? { kind: "scale", corner } : null;
}

export interface HandleDrag {
  handle: FrameHandle;
  ids: number[];
  centre: Vec;
  /** Press point, screen. */
  from: Vec;
  base: SketchModel;
}

/** The model of a handle drag to screen point `to`; `shift` snaps the angle to 15 degrees or the
 *  scale to 5 % steps. */
export function handleDragModel(d: HandleDrag, view: View, to: Vec, shift: boolean): SketchModel {
  const from = toModel(view, d.from);
  const at = toModel(view, to);
  if (d.handle.kind === "rotate") return rotateAtoms(d.base, d.ids, dragAngle(d.centre, from, at, shift), d.centre);
  return scaleAtoms(d.base, d.ids, dragScale(d.centre, from, at, shift), d.centre);
}
