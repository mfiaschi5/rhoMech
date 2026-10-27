// RDKit.js on the sketch model: validation (RDKit is the judge of valence), canonical SMILES,
// InChI, descriptors and implicit H counts; the clean-up layout; import and export. The module
// is passed in (no global, no React), so these run in node with the wasm build as well.
//
// The drawing goes to RDKit as our V2000 molblock, parsed with removeHs: false so RDKit atom i
// is model atom i (H atoms drawn by the user stay atoms). SMILES, InChI and descriptors come
// from a second parse with the H atoms removed. Sanitization errors are read from a captured
// RDKit log ("Explicit valence for atom # 3 N ...", "Unkekulized atoms: 1 2 3") to mark the
// offending atoms. The CIP labels (cip.ts) come from get_stereo_tags() on the first parse, so
// their atom indices are the model's. Radicals reach RDKit as `M  RAD` (molfile.ts).
import type { RDKitModule, RDKitMol } from "@ksmech/viz/chem/rdkit";
import type { SketchModel } from "@ksmech/viz/sketch/model";
import type { StereoLabels } from "@ksmech/viz/sketch/cip";
import { bondOrderSum, centroid, clearDepth, normalizeBondLength } from "@ksmech/viz/sketch/model";
import { fromMolblock, looksLikeMolfile, MolfileError, molblockBondStereo, toMolblock } from "@ksmech/viz/sketch/molfile";
import { fallbackHCount, hillFormula } from "@ksmech/viz/sketch/elements";
import { NO_STEREO, parseStereoTags } from "@ksmech/viz/sketch/cip";

export interface SketchAnalysis {
  /** The model this analysis describes (stale once the drawing changed). */
  forModel: SketchModel;
  empty: boolean;
  valid: boolean;
  /** Canonical SMILES ("" when invalid or empty). */
  smiles: string;
  inchi: string;
  /** RDKit get_descriptors() (null when invalid). */
  descriptors: Record<string, number> | null;
  formula: string;
  error: string | null;
  /** Atom ids RDKit's sanitization complained about. */
  badAtoms: number[];
  /** Implicit H count by atom id (RDKit's when valid, else default valences). */
  hCounts: Map<number, number>;
  /** CIP labels by atom and bond id (empty when invalid). */
  stereo: StereoLabels;
}

const NO_REMOVE_HS = JSON.stringify({ removeHs: false });

function parse(rdkit: RDKitModule, input: string, details: string | null): RDKitMol | null {
  try {
    const mol = details ? rdkit.get_mol(input, details) : rdkit.get_mol(input);
    if (mol && mol.is_valid()) return mol;
    mol?.delete();
  } catch {
    // RDKit throws on some malformed input instead of returning null
  }
  return null;
}

/** Run `fn` with RDKit's log captured; returns its result and the log text. */
function captured<T>(rdkit: RDKitModule, fn: () => T): [T, string] {
  let log: ReturnType<RDKitModule["set_log_capture"]> = null;
  try {
    log = rdkit.set_log_capture("rdApp.*");
  } catch {
    log = null;
  }
  try {
    const out = fn();
    return [out, log ? log.get_buffer() : ""];
  } finally {
    log?.delete();
  }
}

/** Atom indices named in RDKit sanitization messages. */
export function badAtomIndices(log: string): number[] {
  const out = new Set<number>();
  for (const m of log.matchAll(/atom # ?(\d+)/g)) out.add(Number(m[1]));
  for (const m of log.matchAll(/Unkekulized atoms:([\d ]+)/g)) {
    for (const t of m[1].trim().split(/\s+/)) if (t) out.add(Number(t));
  }
  return [...out].sort((a, b) => a - b);
}

/** The last RDKit message without its "[hh:mm:ss] " stamp. */
function lastMessage(log: string): string | null {
  const lines = log.split("\n").map((l) => l.replace(/^\[\d\d:\d\d:\d\d\]\s*/, "").trim()).filter(Boolean);
  return lines.length ? lines[lines.length - 1] : null;
}

function fallbackCounts(m: SketchModel): Map<number, number> {
  return new Map(m.atoms.map((a) => [a.id, fallbackHCount(a.el, a.charge, bondOrderSum(m, a.id), a.radical ?? 0)]));
}

function formulaOf(m: SketchModel, h: ReadonlyMap<number, number>): string {
  const counts = new Map<string, number>();
  for (const a of m.atoms) {
    counts.set(a.el, (counts.get(a.el) ?? 0) + 1);
    const n = h.get(a.id) ?? 0;
    if (n) counts.set("H", (counts.get("H") ?? 0) + n);
  }
  return hillFormula(counts, m.atoms.reduce((s, a) => s + a.charge, 0));
}

/** Implicit H counts from RDKit's JSON (atom order = model order). */
function rdkitHCounts(m: SketchModel, mol: RDKitMol): Map<number, number> {
  const json = JSON.parse(mol.get_json()) as {
    defaults?: { atom?: { impHs?: number } };
    molecules?: { atoms?: { impHs?: number }[] }[];
  };
  const dflt = json.defaults?.atom?.impHs ?? 0;
  const atoms = json.molecules?.[0]?.atoms ?? [];
  return new Map(m.atoms.map((a, i) => [a.id, atoms[i]?.impHs ?? dflt]));
}

export function analyzeSketch(rdkit: RDKitModule, m: SketchModel): SketchAnalysis {
  const base = { forModel: m, smiles: "", inchi: "", descriptors: null, badAtoms: [] as number[], stereo: NO_STEREO };
  if (m.atoms.length === 0) {
    return { ...base, empty: true, valid: false, formula: "", error: null, hCounts: new Map() };
  }
  const mb = toMolblock(m);
  const [mol, log] = captured(rdkit, () => parse(rdkit, mb, NO_REMOVE_HS));
  if (!mol) {
    const bad = badAtomIndices(log).map((i) => m.atoms[i]?.id).filter((id): id is number => id !== undefined);
    const h = fallbackCounts(m);
    const error = lastMessage(log) ?? "RDKit rejected the drawing";
    return { ...base, empty: false, valid: false, formula: formulaOf(m, h), error, badAtoms: bad, hCounts: h };
  }
  try {
    const h = rdkitHCounts(m, mol);
    const [ids] = captured(rdkit, () => identifiers(rdkit, mb));
    const stereo = stereoOf(m, mol);
    return { ...base, ...ids, stereo, empty: false, valid: true, formula: formulaOf(m, h), error: null, hCounts: h };
  } finally {
    mol.delete();
  }
}

/** The CIP labels of a parsed drawing (atom order = model order); none when RDKit cannot. */
function stereoOf(m: SketchModel, mol: RDKitMol): StereoLabels {
  try {
    return parseStereoTags(m, mol.get_stereo_tags());
  } catch {
    return NO_STEREO;
  }
}

/** The CIP labels of a drawing (R/S, r/s, ? on atoms; E/Z on double bonds), by id. */
export function stereoLabels(rdkit: RDKitModule, m: SketchModel): StereoLabels {
  if (m.atoms.length === 0) return NO_STEREO;
  const [out] = captured(rdkit, () => {
    const mol = parse(rdkit, toMolblock(m), NO_REMOVE_HS);
    if (!mol) return NO_STEREO;
    try {
      return stereoOf(m, mol);
    } finally {
      mol.delete();
    }
  });
  return out;
}

function identifiers(rdkit: RDKitModule, mb: string): Pick<SketchAnalysis, "smiles" | "inchi" | "descriptors"> {
  const mol = parse(rdkit, mb, JSON.stringify({ removeHs: true }));
  if (!mol) return { smiles: "", inchi: "", descriptors: null };
  try {
    let inchi = "";
    try {
      inchi = mol.get_inchi();
    } catch {
      inchi = "";
    }
    return { smiles: mol.get_smiles(), inchi, descriptors: JSON.parse(mol.get_descriptors()) as Record<string, number> };
  } finally {
    mol.delete();
  }
}

/** RDKit.js's 2D layout of the drawing (CoordGen), read back by atom order and scaled to our
 *  bond length about the old centroid: the Clean fallback when the server's depiction (POST
 *  /api/depict2d, ring templates) cannot be reached. Wedges are read back too: RDKit re-wedges so that the new
 *  geometry keeps each stereocentre. An unsanitizable drawing is laid out without
 *  sanitization. Null when RDKit cannot parse it at all. */
export function cleanUpSketch(rdkit: RDKitModule, m: SketchModel): SketchModel | null {
  if (m.atoms.length === 0) return m;
  const mb = toMolblock(m);
  const [laid] = captured(rdkit, () => {
    const mol = parse(rdkit, mb, NO_REMOVE_HS) ?? parse(rdkit, mb, JSON.stringify({ removeHs: false, sanitize: false }));
    if (!mol) return null;
    try {
      mol.set_new_coords(true);
      return mol.get_v2Kmolblock();
    } finally {
      mol.delete();
    }
  });
  if (!laid) return null;
  const fresh = fromMolblock(laid);
  if (fresh.atoms.length !== m.atoms.length || fresh.bonds.length !== m.bonds.length) return null;
  const atoms = m.atoms.map((a, i) => ({ ...a, x: fresh.atoms[i].x, y: fresh.atoms[i].y }));
  const stereo = molblockBondStereo(laid);
  const bonds = m.bonds.map((b, k) => {
    const [i, j, st] = stereo[k];
    return { ...b, a: m.atoms[i].id, b: m.atoms[j].id, stereo: b.order === 1 ? st : b.stereo };
  });
  // a new layout is no longer a projection: its depth would order crossings it never had
  return clearDepth(normalizeBondLength({ ...m, atoms, bonds }, centroid(m)));
}

export class ImportError extends Error {}

/** A sketch from pasted or dropped text: a V2000 molfile is read directly (coordinates kept);
 *  V3000, aromatic-bond molfiles and SMILES go through RDKit (kekulized V2000 with 2D
 *  coordinates). Coordinates are rescaled to the standard bond length. */
export function importSketch(rdkit: RDKitModule | null, text: string): SketchModel {
  const input = text.trim();
  if (!input) throw new ImportError("nothing to import");
  if (looksLikeMolfile(text)) {
    try {
      return fromMolblock(text, { normalize: true });
    } catch (err) {
      if (!(err instanceof MolfileError) || !rdkit) throw new ImportError(String((err as Error).message));
    }
  }
  if (!rdkit) throw new ImportError("RDKit.js is not loaded: only V2000 molfiles can be imported");
  const source = looksLikeMolfile(text) ? text.split(/^\$\$\$\$\s*$/m)[0] : input.split(/\s+/)[0];
  const [mb, log] = captured(rdkit, () => {
    const mol = parse(rdkit, source, NO_REMOVE_HS);
    if (!mol) return null;
    try {
      if (!mol.has_coords()) mol.set_new_coords(true);
      return mol.get_v2Kmolblock(JSON.stringify({ kekulize: true }));
    } finally {
      mol.delete();
    }
  });
  if (!mb) throw new ImportError(lastMessage(log) ?? "RDKit cannot parse the input");
  return fromMolblock(mb, { normalize: true });
}

export type ExportFormat = "smiles" | "molfile" | "inchi";

/** The drawing as SMILES, molfile (our V2000 writer, coordinates as drawn) or InChI; "" when
 *  RDKit rejects it (the molfile is always written). */
export function exportSketch(rdkit: RDKitModule | null, m: SketchModel, fmt: ExportFormat): string {
  if (fmt === "molfile") return toMolblock(m);
  if (!rdkit || m.atoms.length === 0) return "";
  const [ids] = captured(rdkit, () => identifiers(rdkit, toMolblock(m)));
  return fmt === "smiles" ? ids.smiles : ids.inchi;
}

/** The canonical isomeric SMILES of a molblock (e.g. the record's 3D one), "" when refused. */
export function molblockSmiles(rdkit: RDKitModule, molblock: string): string {
  const [ids] = captured(rdkit, () => identifiers(rdkit, molblock));
  return ids.smiles;
}
