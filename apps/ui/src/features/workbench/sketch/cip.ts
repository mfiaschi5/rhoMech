// CIP labels of a drawing (the "R/S" toggle): RDKit.js's get_stereo_tags() on our molblock,
// read back by atom and bond order (pure; rdkitSketch.stereoLabels calls RDKit).
//
// RDKit (2026.09, the new CIP labeler) answers {"CIP_atoms": [[i, "(R)"], ...], "CIP_bonds":
// [[i, j, "(E)"], ...]} with atom indices in molblock order, which is the model's order. Atom
// codes are R, S, the pseudo-asymmetric r and s, and "?" for a possible stereocentre whose
// configuration the drawing does not define (no wedge). Bond codes are E and Z; a double bond
// in a 2D drawing always has a configuration (V2000 stereo 0 reads it from the coordinates).
import type { SketchModel } from "./model";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { findBond } from "./model.ts";

export interface StereoLabels {
  /** CIP code by atom id: "R", "S", "r", "s" or "?" (possible, undefined). */
  atoms: ReadonlyMap<number, string>;
  /** CIP code by bond id: "E", "Z" (or "?"). */
  bonds: ReadonlyMap<number, string>;
}

export const NO_STEREO: StereoLabels = { atoms: new Map(), bonds: new Map() };

const code = (s: unknown): string | null => {
  const m = typeof s === "string" ? /^\(?([RSrsEZ?])\)?$/.exec(s.trim()) : null;
  return m ? m[1] : null;
};

/** The labels of get_stereo_tags() JSON for the model it was computed on; malformed entries are
 *  skipped. */
export function parseStereoTags(m: SketchModel, json: string): StereoLabels {
  let v: { CIP_atoms?: unknown[]; CIP_bonds?: unknown[] };
  try {
    v = JSON.parse(json) as typeof v;
  } catch {
    return NO_STEREO;
  }
  const atoms = new Map<number, string>();
  for (const e of Array.isArray(v.CIP_atoms) ? v.CIP_atoms : []) {
    const [i, tag] = Array.isArray(e) ? e : [];
    const atom = typeof i === "number" ? m.atoms[i] : undefined;
    const c = code(tag);
    if (atom && c) atoms.set(atom.id, c);
  }
  const bonds = new Map<number, string>();
  for (const e of Array.isArray(v.CIP_bonds) ? v.CIP_bonds : []) {
    const [i, j, tag] = Array.isArray(e) ? e : [];
    const a = typeof i === "number" ? m.atoms[i] : undefined;
    const b = typeof j === "number" ? m.atoms[j] : undefined;
    const bond = a && b ? findBond(m, a.id, b.id) : undefined;
    const c = code(tag);
    if (bond && c) bonds.set(bond.id, c);
  }
  return { atoms, bonds };
}

export interface CipItem {
  kind: "atom" | "bond";
  /** Atom id, or bond id. */
  id: number;
  /** "C3", or "C1=C2" (1-based atom numbers in drawing order). */
  where: string;
  code: string;
}

/** The labels as a list in drawing order: atoms first, then bonds. */
export function cipItems(m: SketchModel, s: StereoLabels): CipItem[] {
  const num = new Map(m.atoms.map((a, i) => [a.id, i + 1]));
  const out: CipItem[] = [];
  m.atoms.forEach((a, i) => {
    const c = s.atoms.get(a.id);
    if (c) out.push({ kind: "atom", id: a.id, where: `${a.el}${i + 1}`, code: c });
  });
  for (const b of m.bonds) {
    const c = s.bonds.get(b.id);
    const pa = m.atoms.find((a) => a.id === b.a);
    const pb = m.atoms.find((a) => a.id === b.b);
    if (c && pa && pb) out.push({ kind: "bond", id: b.id, where: `${pa.el}${num.get(b.a)}=${pb.el}${num.get(b.b)}`, code: c });
  }
  return out;
}

/** "C3 R · C5 S · C1=C2 E" ("" without labels). */
export const cipText = (items: readonly CipItem[]): string => items.map((x) => `${x.where} ${x.code}`).join(" · ");
