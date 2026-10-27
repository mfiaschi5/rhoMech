// The workbench's hash parameters (selection state survives reloads and deep links):
//   smiles  a SMILES imported into the sketcher on load (once)
//   m       the computed molecule id (m + 20 hex)
//   set     orbital set: canonical (default), ibo, pm_mulliken
//   start   LMO start: cholesky (default), canonical
//   orb     the selected orbital index in that set
//   rep     representation (REPS)
//   layers  pinned surface layers, comma separated "kind:set:start:index" (empty fields allowed)
//   q       cube quality: draft, normal (default), fine
// Pure functions only (no React), so they can be unit-checked in node.
import type { CubeKind, CubeQuality } from "../../api/workbench";
import { DEFAULT_REP, REPS, type Rep } from "@ksmech/viz/viewer/rep";

export { DEFAULT_REP, REPS, type Rep } from "@ksmech/viz/viewer/rep";

export const QUALITIES: readonly CubeQuality[] = ["draft", "normal", "fine"];
export const DEFAULT_QUALITY: CubeQuality = "normal";

export const CANONICAL = "canonical";
export const DEFAULT_SET = CANONICAL;
export const DEFAULT_START = "cholesky";
export const FALLBACK_SETS = ["canonical", "ibo", "pm_mulliken"];
export const FALLBACK_STARTS = ["cholesky", "canonical"];

export const CUBE_KINDS: readonly CubeKind[] = ["orbital", "orbital_density", "density", "fukui_minus", "fukui_plus", "dual", "esp"];
export const ORBITAL_KINDS: ReadonlySet<CubeKind> = new Set<CubeKind>(["orbital", "orbital_density"]);

export const KIND_LABELS: Record<CubeKind, string> = {
  orbital: "orbital ψ",
  orbital_density: "orbital |ψ|²",
  density: "valence density ρ",
  fukui_minus: "Fukui f−",
  fukui_plus: "Fukui f+",
  dual: "dual descriptor",
  esp: "ESP on ρ (approx.)",
};

/** What a surface layer shows: the identity that decides the fetch (never iso or colours). */
export interface LayerSpec {
  kind: CubeKind;
  set: string | null;
  start: string | null;
  index: number | null;
}

export function isOrbitalKind(kind: CubeKind): boolean {
  return ORBITAL_KINDS.has(kind);
}

/** The spec with the fields its kind does not use cleared (canonical has no start). */
export function normalizeSpec(s: LayerSpec): LayerSpec {
  if (!isOrbitalKind(s.kind)) return { kind: s.kind, set: null, start: null, index: null };
  const set = s.set || DEFAULT_SET;
  return { kind: s.kind, set, start: set === CANONICAL ? null : s.start || DEFAULT_START, index: s.index };
}

export function specKey(s: LayerSpec): string {
  const n = normalizeSpec(s);
  return [n.kind, n.set ?? "", n.start ?? "", n.index ?? ""].join(":");
}

export function parseSpec(text: string): LayerSpec | null {
  const [kind, set = "", start = "", index = ""] = text.split(":");
  if (!(CUBE_KINDS as readonly string[]).includes(kind)) return null;
  const k = kind as CubeKind;
  if (!isOrbitalKind(k)) return normalizeSpec({ kind: k, set: null, start: null, index: null });
  const i = Number.parseInt(index, 10);
  if (!Number.isFinite(i) || i < 0) return null;
  return normalizeSpec({ kind: k, set: set || null, start: start || null, index: i });
}

export function parseLayers(value: string | null): LayerSpec[] {
  if (!value) return [];
  return value
    .split(",")
    .map((t) => parseSpec(t.trim()))
    .filter((s): s is LayerSpec => s !== null);
}

export function formatLayers(specs: readonly LayerSpec[]): string {
  return specs.map(specKey).join(",");
}

/** The `rep` param, else `fallback` (the user's default representation). */
export function parseRep(value: string | null, fallback: Rep = DEFAULT_REP): Rep {
  return (REPS as readonly string[]).includes(value ?? "") ? (value as Rep) : fallback;
}

export function parseQuality(value: string | null): CubeQuality {
  return (QUALITIES as readonly string[]).includes(value ?? "") ? (value as CubeQuality) : DEFAULT_QUALITY;
}

export function parseIndex(value: string | null): number | null {
  if (value === null || value === "") return null;
  const n = Number.parseInt(value, 10);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

export function isMoleculeId(value: string | null): value is string {
  return !!value && /^m[0-9a-f]{20}$/.test(value);
}
