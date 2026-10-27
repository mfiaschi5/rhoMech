// The server's 2D depiction (POST /api/depict2d, serve/depiction2d.py) applied to the sketch.
// Pure (no fetch, no React): the Sketcher gets a Depictor from the page and hands the answer to
// applyDepiction, so this runs under `node --experimental-strip-types --test`.
//
// The answer is in the molblock's order, which is the model's (toMolblock writes atoms and bonds
// in array order): coords[i] is atom i (bond length 1.5 = BOND_LENGTH), depth[i] its depth (+
// towards the viewer; perspective templates and match3d), wedges name a bond by index and the
// atom at the narrow end. The drawing keeps its centroid; every old wedge is replaced; the depth
// is kept (crossing breaks turned on) or dropped.
import type { Depict2DIn, Depict2DOut } from "../../../api/types.gen";
import type { BondStereo, SketchModel } from "@ksmech/viz/sketch/model";
import { centroid, clearDepth, setBondMarks, setBreakCrossings, setPositions } from "@ksmech/viz/sketch/model";

export type CleanStyle = "textbook" | "perspective";
export const CLEAN_STYLES: readonly CleanStyle[] = ["textbook", "perspective"];

/** What the page offers the sketcher: POST /api/depict2d (rejects on any failure). */
export type Depictor = (req: Depict2DIn, signal?: AbortSignal) => Promise<Depict2DOut>;

export type DepictionLayout = Pick<Depict2DOut, "coords" | "depth" | "wedges">;

/** The drawing laid out as the server's answer, or null when the answer does not fit the
 *  drawing (another atom count, a wedge on a missing or multiple bond). */
export function applyDepiction(m: SketchModel, out: DepictionLayout): SketchModel | null {
  const n = m.atoms.length;
  if (n === 0 || out.coords.length !== n || (out.depth && out.depth.length !== n)) return null;
  const marks = new Map<number, { stereo: BondStereo; narrow: number }>();
  for (const b of m.bonds) if (b.stereo !== "none") marks.set(b.id, { stereo: "none", narrow: b.a });
  for (const w of out.wedges) {
    const bond = m.bonds[w.bond];
    const narrow = m.atoms[w.narrow];
    if (!bond || !narrow || bond.order !== 1 || (bond.a !== narrow.id && bond.b !== narrow.id)) return null;
    marks.set(bond.id, { stereo: w.kind, narrow: narrow.id });
  }
  const [ox, oy] = centroid(m);
  const cx = out.coords.reduce((s, c) => s + c[0], 0) / n;
  const cy = out.coords.reduce((s, c) => s + c[1], 0) / n;
  const pos = new Map<number, readonly [number, number, number?]>();
  m.atoms.forEach((a, i) => {
    const [x, y] = out.coords[i];
    pos.set(a.id, out.depth ? [ox + x - cx, oy + y - cy, out.depth[i]] : [ox + x - cx, oy + y - cy]);
  });
  const placed = setBondMarks(setPositions(out.depth ? m : clearDepth(m), pos), marks);
  return setBreakCrossings(placed, !!out.depth);
}

const STYLE_KEY = "ksmech.sketch.cleanStyle";

/** The clean-up style kept per tab (sessionStorage; textbook when unavailable). */
export function restoreCleanStyle(): CleanStyle {
  try {
    const v = window.sessionStorage.getItem(STYLE_KEY);
    return v === "perspective" ? "perspective" : "textbook";
  } catch {
    return "textbook";
  }
}

export function saveCleanStyle(s: CleanStyle): void {
  try {
    window.sessionStorage.setItem(STYLE_KEY, s);
  } catch {
    // storage unavailable: the style lasts as long as the page
  }
}
