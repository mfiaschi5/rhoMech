// The numeric per-atom properties of GET /molecule/{id}/properties, as one catalogue shared by
// the atom table, the 3D labels and the colour-by-property scheme. Diagnostics (hybridization,
// aromaticity, functional group) are strings and never model input; they are table-only.
import type { AtomPropsOut, MoleculePropertiesOut } from "../apiTypes";

export interface AtomPropDef {
  key: string;
  label: string;
  /** Hover text for the column header. */
  title: string;
  get(a: AtomPropsOut): number | null;
  /** Signed values get a diverging colour scale centred on 0. */
  signed: boolean;
  digits: number;
}

const BASE: AtomPropDef[] = [
  { key: "mulliken_charge", label: "q Mulliken", title: "Mulliken charge (e)", get: (a) => a.mulliken_charge, signed: true, digits: 3 },
  { key: "formal_charge", label: "formal", title: "formal charge of the drawn Lewis structure", get: (a) => a.formal_charge, signed: true, digits: 0 },
  { key: "f_plus_fd", label: "f+ FD", title: "finite-difference Fukui f+ (N+1 − N)", get: (a) => a.f_plus_fd, signed: true, digits: 3 },
  { key: "f_minus_fd", label: "f− FD", title: "finite-difference Fukui f− (N − N−1)", get: (a) => a.f_minus_fd, signed: true, digits: 3 },
  { key: "f_zero_fd", label: "f0 FD", title: "finite-difference radical Fukui f0", get: (a) => a.f_zero_fd, signed: true, digits: 3 },
  { key: "dual_fd", label: "dual FD", title: "finite-difference dual descriptor f+ − f−", get: (a) => a.dual_fd, signed: true, digits: 3 },
  { key: "f_plus_fmo", label: "f+ FMO", title: "frozen-orbital Fukui f+ (LUMO block weight)", get: (a) => a.f_plus_fmo, signed: false, digits: 3 },
  { key: "f_minus_fmo", label: "f− FMO", title: "frozen-orbital Fukui f− (HOMO block weight)", get: (a) => a.f_minus_fmo, signed: false, digits: 3 },
];

const TAIL: AtomPropDef[] = [
  { key: "vbur", label: "%Vbur", title: "percent buried volume", get: (a) => a.vbur, signed: false, digits: 1 },
  { key: "wiberg_valence", label: "Wiberg val.", title: "sum of the Wiberg bond indices of the atom", get: (a) => a.wiberg_valence, signed: false, digits: 3 },
  { key: "chirality", label: "chirality", title: "chirality sign, set only where the SMILES specifies the stereocentre", get: (a) => a.chirality, signed: true, digits: 0 },
];

/** "homo_minus_1" or "HOMO-1" -> "HOMO−1" for display. */
const levelLabel = (s: string) =>
  s
    .toUpperCase()
    .replace(/_MINUS_/g, "−")
    .replace(/_PLUS_/g, "+")
    .replace(/([A-Z])-(\d)/g, "$1−$2");

/** The catalogue for one properties payload (the frontier weight columns follow fmo_levels). */
export function atomPropDefs(p: MoleculePropertiesOut | null | undefined): AtomPropDef[] {
  const levels = p?.fmo_levels?.length ? p.fmo_levels : ["HOMO-1", "HOMO", "LUMO", "LUMO+1"];
  const weights: AtomPropDef[] = levels.map((lv, i) => ({
    key: `fmo_w${i}`,
    label: `w ${levelLabel(lv)}`,
    title: `frontier weight of the atom in ${levelLabel(lv)} (degenerate blocks within 0.05 eV summed)`,
    get: (a) => a.fmo_weights[i] ?? null,
    signed: false,
    digits: 3,
  }));
  return [...BASE, ...weights, ...TAIL];
}

export function findProp(defs: readonly AtomPropDef[], key: string | null): AtomPropDef | null {
  return defs.find((d) => d.key === key) ?? null;
}

/** [min, max] of a property over the atoms (null when no atom has a value). */
export function propRange(def: AtomPropDef, atoms: readonly AtomPropsOut[]): [number, number] | null {
  const vals = atoms.map(def.get).filter((v): v is number => v !== null && Number.isFinite(v));
  if (!vals.length) return null;
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  if (def.signed) {
    const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    lo = -m;
    hi = m;
  } else if (hi === lo) {
    hi = lo + 1;
  }
  return [lo, hi];
}
