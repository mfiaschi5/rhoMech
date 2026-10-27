// What the workbench keeps in sessionStorage (per tab, survives reloads): the current drawing as
// JSON {molblock, extras} (extras: what the molfile does not carry, the depth of a projection,
// explicitH and the crossing breaks; a bare molblock from an older build still reads), and the
// drawing that was last computed with its molecule id, so a reload of
// #/workbench?m=<id> can post it again and get the sketch mapping back. Every access is in
// try/catch: storage may be unavailable (private mode, quota), and then nothing persists.
import { fromMolblock, sketchExtras, toMolblock, withExtras, type SketchExtras, type SketchModel } from "./sketch";

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

/** The stored text of a drawing (see the module comment). */
export function encodeSketch(m: SketchModel): string {
  return JSON.stringify({ molblock: toMolblock(m), extras: sketchExtras(m) });
}

export function decodeSketch(text: string): SketchModel | null {
  try {
    if (!text.trimStart().startsWith("{")) return fromMolblock(text);
    const v = JSON.parse(text) as { molblock?: unknown; extras?: SketchExtras };
    return typeof v.molblock === "string" ? withExtras(fromMolblock(v.molblock), v.extras ?? null) : null;
  } catch {
    return null;
  }
}

export function restoreSketch(): SketchModel | null {
  const text = read(SKETCH_KEY);
  return text ? decodeSketch(text) : null;
}

export function saveSketch(m: SketchModel): void {
  write(SKETCH_KEY, m.atoms.length ? encodeSketch(m) : null);
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

/** The constitution of a drawing, independent of positions, depth and wedges: elements,
 *  charges, unpaired electrons, bonds by unordered atom position and their orders. Equal topologies map onto the
 *  record the same way (the POST's mapping is atom to atom). */
export function sketchTopology(m: SketchModel): string {
  const pos = new Map(m.atoms.map((a, i) => [a.id, i]));
  const atoms = m.atoms.map((a) => `${a.el}${a.charge}${a.radical ? `r${a.radical}` : ""}`).join(",");
  const bonds = m.bonds
    .map((b) => {
      const [i, j] = [pos.get(b.a) ?? -1, pos.get(b.b) ?? -1].sort((x, y) => x - y);
      return `${i}-${j}:${b.order}`;
    })
    .join(",");
  return `${atoms}|${bonds}`;
}

/** The wedges of a drawing by atom position (narrow end first). */
export function sketchWedges(m: SketchModel): string {
  const pos = new Map(m.atoms.map((a, i) => [a.id, i]));
  return m.bonds
    .filter((b) => b.order === 1 && b.stereo !== "none")
    .map((b) => `${pos.get(b.a)}>${pos.get(b.b)}:${b.stereo}`)
    .sort()
    .join(",");
}

export interface StereoSmiles {
  /** RDKit canonical isomeric SMILES of the current drawing, of the computed one and of the
   *  record's 3D geometry ("" when unknown). */
  drawn: string;
  computed: string;
  record: string;
}

/** Whether the drawing has changed since the compute in a way that unlinks it: another
 *  topology, or other wedges that draw neither the computed drawing's stereo nor the record's
 *  (unknown counts as changed). Coordinates, depth and wedges that redraw the record's own
 *  configuration (a 3D -> 2D projection) do not. */
export function sketchChanged(current: SketchModel, computed: SketchModel | null, stereo: () => StereoSmiles | null): boolean {
  if (!computed) return false;
  if (sketchTopology(current) !== sketchTopology(computed)) return true;
  if (sketchWedges(current) === sketchWedges(computed)) return false;
  const s = stereo();
  if (!s || !s.drawn) return true;
  return s.drawn !== s.record && s.drawn !== s.computed;
}
