// V2000 molfile writer and reader for the sketch model (pure, no RDKit).
//
// Writer: atoms in model order with 2D coordinates (z = 0 always: the drawing's depth from a
// perspective template or a 3D->2D is display data), charges as `M  CHG` lines (the atom-block
// charge field stays 0), unpaired electrons as `M  RAD` lines (2 = doublet for one, 3 = triplet
// for two), bond stereo 1 = wedge and 6 = hash with the narrow end first. The server's 2D
// depiction (POST /api/depict2d) answers in this atom and bond order.
// Reader: the first record of a molfile or SDF; `M  CHG` replaces the atom-block charges when
// present (as the format says), `M  RAD` likewise the atom-block doublet radical (old charge
// code 4); RAD 1 (singlet) and 3 (triplet) are two unpaired electrons, RAD 2 one. Bond types
// other than 1-3 (aromatic 4, query types) and V3000
// are refused with a message; importSketch() sends those through RDKit first (kekulized V2000).
import type { BondOrder, BondStereo, RadicalCount, SketchAtom, SketchBond, SketchModel } from "./model";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { normalizeBondLength } from "./model.ts";

const f10 = (v: number) => v.toFixed(4).padStart(10);
const i3 = (v: number) => String(v).padStart(3);

export function toMolblock(m: SketchModel, name = ""): string {
  const index = new Map(m.atoms.map((a, i) => [a.id, i + 1]));
  const lines = [name, "  ksmech            2D", ""];
  lines.push(`${i3(m.atoms.length)}${i3(m.bonds.length)}  0  0  0  0  0  0  0  0999 V2000`);
  for (const a of m.atoms) {
    lines.push(`${f10(a.x)}${f10(a.y)}${f10(0)} ${a.el.padEnd(3)} 0${"  0".repeat(11)}`);
  }
  for (const b of m.bonds) {
    const stereo = b.order === 1 ? (b.stereo === "wedge" ? 1 : b.stereo === "hash" ? 6 : 0) : 0;
    lines.push(`${i3(index.get(b.a) ?? 0)}${i3(index.get(b.b) ?? 0)}${i3(b.order)}${i3(stereo)}`);
  }
  propertyLines(lines, "CHG", m.atoms.map((a, i) => [i + 1, a.charge] as const).filter(([, c]) => c !== 0));
  propertyLines(lines, "RAD", m.atoms.flatMap((a, i) => (a.radical ? [[i + 1, a.radical === 1 ? 2 : 3] as const] : [])));
  lines.push("M  END");
  return lines.join("\n") + "\n";
}

/** `M  <tag>` lines of (atom, value) pairs, eight per line. */
function propertyLines(lines: string[], tag: string, pairs: readonly (readonly [number, number])[]): void {
  for (let k = 0; k < pairs.length; k += 8) {
    const chunk = pairs.slice(k, k + 8);
    lines.push(`M  ${tag}${i3(chunk.length)}${chunk.map(([i, c]) => `${String(i).padStart(4)}${String(c).padStart(4)}`).join("")}`);
  }
}

const OLD_CHARGE: Record<number, number> = { 1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3 };

export class MolfileError extends Error {}

function int(s: string | undefined): number {
  const v = Number.parseInt((s ?? "").trim(), 10);
  return Number.isFinite(v) ? v : 0;
}

/** The first record's lines (header through M  END) of a molfile or SDF text. */
function recordLines(text: string): string[] {
  const record = text.split(/^\$\$\$\$\s*$/m)[0] ?? "";
  const lines = record.replace(/\r/g, "").split("\n");
  // An SDF record pasted with leading blank lines: find the counts line instead of trusting
  // the three header lines, when the standard position does not hold one.
  const counts = lines.findIndex((l) => /V[23]000\s*$/.test(l));
  if (counts < 0) throw new MolfileError("not a molfile: no V2000/V3000 counts line");
  if (/V3000\s*$/.test(lines[counts])) throw new MolfileError("V3000 molfile: convert it to V2000 first");
  return counts >= 3 ? lines.slice(counts - 3) : [...Array<string>(3 - counts).fill(""), ...lines];
}

function parseAtom(line: string, i: number): SketchAtom {
  const x = Number.parseFloat(line.slice(0, 10));
  const y = Number.parseFloat(line.slice(10, 20));
  const el = line.slice(31, 34).trim();
  if (!Number.isFinite(x) || !Number.isFinite(y) || !/^[A-Z][a-z]?$/.test(el)) {
    throw new MolfileError(`atom ${i + 1}: cannot read "${line.trim()}"`);
  }
  const code = int(line.slice(36, 39));
  const atom: SketchAtom = { id: i + 1, el, x, y, charge: OLD_CHARGE[code] ?? 0 };
  return code === 4 ? { ...atom, radical: 1 } : atom;
}

function parseBond(line: string, k: number, nAtoms: number, id: number): SketchBond {
  const a = int(line.slice(0, 3));
  const b = int(line.slice(3, 6));
  const type = int(line.slice(6, 9));
  const st = int(line.slice(9, 12));
  if (a < 1 || b < 1 || a > nAtoms || b > nAtoms || a === b) throw new MolfileError(`bond ${k + 1}: bad atoms`);
  if (type < 1 || type > 3) throw new MolfileError(`bond ${k + 1}: type ${type} is not single/double/triple`);
  const stereo: BondStereo = type === 1 && st === 1 ? "wedge" : type === 1 && st === 6 ? "hash" : "none";
  return { id, a, b, order: type as BondOrder, stereo };
}

/** The (atom number -> value) pairs of the `M  <tag>` property lines; null without such lines. */
function readProperty(props: string[], tag: string): Map<number, number> | null {
  const found = props.filter((l) => l.startsWith(`M  ${tag}`));
  if (found.length === 0) return null;
  const out = new Map<number, number>();
  for (const l of found) {
    const n = int(l.slice(6, 9));
    for (let k = 0; k < n; k++) out.set(int(l.slice(9 + 8 * k, 13 + 8 * k)), int(l.slice(13 + 8 * k, 17 + 8 * k)));
  }
  return out;
}

/** Unpaired electrons of an `M  RAD` value: 2 (doublet) is one, 1 (singlet) and 3 (triplet) two. */
const RAD_ELECTRONS: Record<number, RadicalCount> = { 1: 2, 2: 1, 3: 2 };

/** Apply the `M  CHG` and `M  RAD` property lines (each resets the atom-block values when
 *  present). */
function applyPropertyLines(atoms: SketchAtom[], props: string[]): SketchAtom[] {
  const charges = readProperty(props, "CHG");
  const rads = readProperty(props, "RAD");
  return atoms.map((a, i) => {
    const charged = charges ? { ...a, charge: charges.get(i + 1) ?? 0 } : a;
    if (!rads) return charged;
    const { radical: _r, ...rest } = charged;
    const n = RAD_ELECTRONS[rads.get(i + 1) ?? 0] ?? 0;
    return n ? { ...rest, radical: n as 1 | 2 } : rest;
  });
}

export interface ReadOptions {
  /** Rescale so the mean bond is BOND_LENGTH (imports from other programs). */
  normalize?: boolean;
}

/** Read a V2000 molfile (or the first SDF record). Atom ids are 1..n, bond ids n+1..n+m. */
export function fromMolblock(text: string, opts: ReadOptions = {}): SketchModel {
  const lines = recordLines(text);
  const nAtoms = int(lines[3].slice(0, 3));
  const nBonds = int(lines[3].slice(3, 6));
  if (lines.length < 4 + nAtoms + nBonds) throw new MolfileError("truncated molfile");
  const atomLines = lines.slice(4, 4 + nAtoms);
  const bondLines = lines.slice(4 + nAtoms, 4 + nAtoms + nBonds);
  const props = lines.slice(4 + nAtoms + nBonds);
  const atoms = applyPropertyLines(atomLines.map(parseAtom), props);
  const bonds = bondLines.map((l, k) => parseBond(l, k, nAtoms, nAtoms + k + 1));
  const model: SketchModel = { atoms, bonds, nextId: nAtoms + nBonds + 1 };
  return opts.normalize ? normalizeBondLength(model) : model;
}

/** The bonds of a V2000 block as [atom index a, atom index b, stereo] (0-based), in file
 *  order; used to read RDKit's re-wedging after a clean-up. */
export function molblockBondStereo(text: string): Array<[number, number, BondStereo]> {
  const m = fromMolblock(text);
  return m.bonds.map((b) => [b.a - 1, b.b - 1, b.stereo]);
}

/** Does the text look like a molfile / SDF (rather than a SMILES)? */
export function looksLikeMolfile(text: string): boolean {
  return /V[23]000/.test(text) || /^M {2}END/m.test(text);
}
