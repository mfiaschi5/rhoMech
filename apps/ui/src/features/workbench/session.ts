// What the workbench keeps in sessionStorage (per tab, survives reloads): the current drawing as
// a molblock, and the drawing that was last computed with its molecule id, so a reload of
// #/workbench?m=<id> can post it again and get the sketch mapping back. Every access is in
// try/catch: storage may be unavailable (private mode, quota), and then nothing persists.
import { fromMolblock, toMolblock, type SketchModel } from "./sketch";

const SKETCH_KEY = "ksmech.workbench.sketch";
const COMPUTED_KEY = "ksmech.workbench.computed";

function read(key: string): string | null {
  try {
    return window.sessionStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string | null): void {
  try {
    if (value === null) window.sessionStorage.removeItem(key);
    else window.sessionStorage.setItem(key, value);
  } catch {
    // storage unavailable: nothing persists
  }
}

export function restoreSketch(): SketchModel | null {
  const text = read(SKETCH_KEY);
  if (!text) return null;
  try {
    return fromMolblock(text);
  } catch {
    return null;
  }
}

export function saveSketch(m: SketchModel): void {
  write(SKETCH_KEY, m.atoms.length ? toMolblock(m) : null);
}

export interface ComputedDrawing {
  m: string;
  molblock: string;
}

export function restoreComputed(): ComputedDrawing | null {
  try {
    const v = JSON.parse(read(COMPUTED_KEY) ?? "null") as Partial<ComputedDrawing> | null;
    return v && typeof v.m === "string" && typeof v.molblock === "string" ? { m: v.m, molblock: v.molblock } : null;
  } catch {
    return null;
  }
}

export function saveComputed(c: ComputedDrawing): void {
  write(COMPUTED_KEY, JSON.stringify(c));
}

/** The chemistry of a drawing, independent of 2D positions: elements, charges, bonds by atom
 *  position, orders and wedges. Equal signatures map to the record the same way. */
export function sketchSignature(m: SketchModel): string {
  const pos = new Map(m.atoms.map((a, i) => [a.id, i]));
  const atoms = m.atoms.map((a) => `${a.el}${a.charge}`).join(",");
  const bonds = m.bonds.map((b) => `${pos.get(b.a)}-${pos.get(b.b)}:${b.order}${b.stereo}`).join(",");
  return `${atoms}|${bonds}`;
}
