// Rotate, flip and scale of a sketch (or of its selection) as pure model edits, the toolbar and
// key actions that name them, and the gesture math of the selection frame's handles.
//
// The targets are the selection's atoms (selected atoms plus the ends of selected bonds), or the
// whole drawing when nothing is selected. The pivot is the targets' centroid: it stays put under
// a rotation, flip or scale about itself, so R then Shift+R (or two flips) gives back the same
// drawing, which the centre of the bounding box would not. Every edit moves coordinates only; ids, order and bonds
// are kept, so a selection and the sketch-to-record mapping still hold.
//
// Stereo: an in-plane rotation and a uniform scale keep every wedge and hash meaning the same.
// A flip is a mirror image: drawn as is it would be the enantiomer. To keep the same
// stereoisomer the flip is read as a 180-degree turn about the in-plane axis (x -> -x for a
// horizontal flip is (x, y, z) -> (-x, y, -z)): every wedge whose narrow end is a flipped atom
// becomes a hash and back, and the stored depth z of the flipped atoms is negated (crossings
// keep their front/back). Uniform scaling scales z with x and y.
// Gap: a flip of a partial selection that cuts through the bonds of a stereocentre mirrors only
// some of its neighbours, which is a geometric edit and may change that centre.
import type { SketchAtom, SketchBond, SketchModel } from "./model";
import type { Selection } from "./tools";
import type { Vec } from "./geometry";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { BOND_LENGTH } from "./model.ts";
// @ts-ignore TS5097: see above
import { DEG, angleOf, dist, rotate, snapAngle, sub } from "./geometry.ts";
// @ts-ignore TS5097: see above
import { selectionAtoms } from "./tools.ts";

export const TRANSFORM_ACTIONS = [
  "rotate-cw15",
  "rotate-ccw15",
  "rotate-cw90",
  "rotate-ccw90",
  "flip-h",
  "flip-v",
  "scale-up",
  "scale-down",
  "reset-bond",
] as const;
export type TransformAction = (typeof TRANSFORM_ACTIONS)[number];

/** One scale step of the toolbar and Ctrl+= / Ctrl+-. */
export const SCALE_STEP = 1.1;
/** Gesture limits on the scale factor and the Shift step of the corner handles. */
export const SCALE_MIN = 0.1;
export const SCALE_MAX = 10;
export const SCALE_FINE_STEP = 0.05;
export const ROTATE_SNAP = 15 * DEG;

/** The atom ids a transform moves: the selection's atoms, or every atom when none is selected. */
export function transformTargets(m: SketchModel, sel: Selection): number[] {
  const picked = selectionAtoms(m, sel);
  const ids = m.atoms.filter((a) => picked.has(a.id)).map((a) => a.id);
  return ids.length ? ids : m.atoms.map((a) => a.id);
}

/** Centroid of the given atoms (model frame): the pivot of every transform; null without atoms. */
export function pivotOf(m: SketchModel, ids: Iterable<number>): Vec | null {
  const set = new Set(ids);
  const pts = m.atoms.filter((a) => set.has(a.id));
  if (!pts.length) return null;
  return [pts.reduce((s, a) => s + a.x, 0) / pts.length, pts.reduce((s, a) => s + a.y, 0) / pts.length];
}

function mapIds(m: SketchModel, ids: Iterable<number>, fn: (a: SketchAtom) => SketchAtom): SketchModel {
  const set = new Set(ids);
  if (!set.size) return m;
  return { ...m, atoms: m.atoms.map((a) => (set.has(a.id) ? fn(a) : a)) };
}

/** Rotate atoms by `angle` (radians, counter-clockwise in the model frame, y up) about `about`
 *  (default: their centroid). Depth and stereo marks are unchanged. */
export function rotateAtoms(m: SketchModel, ids: readonly number[], angle: number, about?: Vec): SketchModel {
  const c = about ?? pivotOf(m, ids);
  if (!c || angle === 0) return m;
  return mapIds(m, ids, (a) => {
    const [x, y] = rotate([a.x, a.y], angle, c);
    return { ...a, x, y };
  });
}

/** Scale atoms uniformly by k about `about` (default: their centroid); z scales too. */
export function scaleAtoms(m: SketchModel, ids: readonly number[], k: number, about?: Vec): SketchModel {
  const c = about ?? pivotOf(m, ids);
  if (!c || k === 1 || !(k > 0)) return m;
  return mapIds(m, ids, (a) => {
    const moved = { ...a, x: c[0] + (a.x - c[0]) * k, y: c[1] + (a.y - c[1]) * k };
    return a.z === undefined ? moved : { ...moved, z: a.z * k };
  });
}

const swapMark = (b: SketchBond): SketchBond => (b.stereo === "wedge" ? { ...b, stereo: "hash" } : b.stereo === "hash" ? { ...b, stereo: "wedge" } : b);

/** Mirror atoms about the vertical ("h": left-right) or horizontal ("v": up-down) line through
 *  `about` (default: their centroid), keeping the drawn stereoisomer: the depth of the moved
 *  atoms is negated and the wedges/hashes whose narrow end moved swap (see the module comment). */
export function flipAtoms(m: SketchModel, ids: readonly number[], axis: "h" | "v", about?: Vec): SketchModel {
  const c = about ?? pivotOf(m, ids);
  if (!c) return m;
  const set = new Set(ids);
  const moved = mapIds(m, set, (a) => {
    const xy = axis === "h" ? { x: 2 * c[0] - a.x } : { y: 2 * c[1] - a.y };
    return a.z === undefined ? { ...a, ...xy } : { ...a, ...xy, z: -a.z };
  });
  const bonds = m.bonds.map((b) => (set.has(b.a) ? swapMark(b) : b));
  return { ...moved, bonds };
}

/** Mean length of the bonds with both ends in `ids`, or null without such a bond. */
export function meanLengthWithin(m: SketchModel, ids: readonly number[]): number | null {
  const set = new Set(ids);
  const pos = new Map(m.atoms.map((a) => [a.id, [a.x, a.y] as Vec]));
  const lens = m.bonds.filter((b) => set.has(b.a) && set.has(b.b)).map((b) => dist(pos.get(b.a) as Vec, pos.get(b.b) as Vec));
  const total = lens.reduce((s, l) => s + l, 0);
  return lens.length && total > 0 ? total / lens.length : null;
}

/** Scale atoms so the mean bond among them is BOND_LENGTH (no-op without a bond among them). */
export function resetBondLength(m: SketchModel, ids: readonly number[]): SketchModel {
  const mean = meanLengthWithin(m, ids);
  return mean === null || Math.abs(mean - BOND_LENGTH) < 1e-9 ? m : scaleAtoms(m, ids, BOND_LENGTH / mean);
}

/** One toolbar / key action on the given atoms. Clockwise is a negative model angle (y up). */
export function applyTransform(m: SketchModel, ids: readonly number[], action: TransformAction): SketchModel {
  switch (action) {
    case "rotate-cw15":
      return rotateAtoms(m, ids, -15 * DEG);
    case "rotate-ccw15":
      return rotateAtoms(m, ids, 15 * DEG);
    case "rotate-cw90":
      return rotateAtoms(m, ids, -90 * DEG);
    case "rotate-ccw90":
      return rotateAtoms(m, ids, 90 * DEG);
    case "flip-h":
      return flipAtoms(m, ids, "h");
    case "flip-v":
      return flipAtoms(m, ids, "v");
    case "scale-up":
      return scaleAtoms(m, ids, SCALE_STEP);
    case "scale-down":
      return scaleAtoms(m, ids, 1 / SCALE_STEP);
    case "reset-bond":
      return resetBondLength(m, ids);
  }
}

export interface KeyLike {
  key: string;
  code?: string;
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  altKey: boolean;
}

/** The transform of a key press, or null: R / Shift+R rotate 15 degrees clockwise /
 *  counter-clockwise, Alt+H / Alt+V flip (by physical key, since Alt changes e.key on some
 *  layouts), Ctrl+= (or +) / Ctrl+- scale up / down by 10 %. */
export function transformKey(e: KeyLike): TransformAction | null {
  const mod = e.ctrlKey || e.metaKey;
  const k = e.key.toLowerCase();
  if (e.altKey && !mod) {
    if (e.code === "KeyH" || (!e.code && k === "h")) return "flip-h";
    if (e.code === "KeyV" || (!e.code && k === "v")) return "flip-v";
    return null;
  }
  if (mod && !e.altKey) {
    if (k === "=" || k === "+") return "scale-up";
    if (k === "-" || k === "_") return "scale-down";
    return null;
  }
  if (!mod && !e.altKey && k === "r") return e.shiftKey ? "rotate-ccw15" : "rotate-cw15";
  return null;
}

/** The rotation angle of a rotate-handle drag about `centre` from `from` to `to` (model frame);
 *  `snap` rounds it to ROTATE_SNAP. */
export function dragAngle(centre: Vec, from: Vec, to: Vec, snap: boolean): number {
  if (dist(to, centre) < 1e-9 || dist(from, centre) < 1e-9) return 0;
  const t = angleOf(sub(to, centre)) - angleOf(sub(from, centre));
  const wrapped = Math.atan2(Math.sin(t), Math.cos(t));
  return snap ? snapAngle(wrapped, ROTATE_SNAP) : wrapped;
}

/** The scale factor of a corner-handle drag: distance ratio to `centre`, clamped to
 *  [SCALE_MIN, SCALE_MAX]; `fine` rounds it to SCALE_FINE_STEP. */
export function dragScale(centre: Vec, from: Vec, to: Vec, fine: boolean): number {
  const d0 = dist(from, centre);
  if (d0 < 1e-9) return 1;
  const raw = Math.min(SCALE_MAX, Math.max(SCALE_MIN, dist(to, centre) / d0));
  return fine ? Math.max(SCALE_MIN, Math.round(raw / SCALE_FINE_STEP) * SCALE_FINE_STEP) : raw;
}
