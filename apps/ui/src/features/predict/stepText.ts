// Prose of the step viewport (pure; node --test runs it): the diagnostic orbital labels as
// chemists write them (display only, never a model input: "lone_pair(O)" -> "n(O)",
// "sigma_star(C-Cl)" -> "σ*(C–Cl)"), the one-line orbital caption of a reading, R entries in
// words, and an alternative's position.
import type { OrbitalReadingOut } from "./v9types";

const KIND: Record<string, string> = {
  lone_pair: "n",
  sigma: "σ",
  pi: "π",
  delocalized_pi: "π(deloc.)",
  sigma_star: "σ*",
  pi_star: "π*",
  delocalized_pi_star: "π*(deloc.)",
  empty_p: "p",
  empty_s: "s",
  empty_d: "d",
  unknown: "?",
};

/** "sigma_star(C-Cl)" -> "σ*(C–Cl)"; null stays null; an unknown kind is kept as written. */
export function diagSymbol(diag: string | null | undefined): string | null {
  if (!diag) return null;
  const m = /^([a-z_]+)\((.*)\)$/.exec(diag);
  if (!m) return diag;
  const kind = KIND[m[1]] ?? m[1].replace(/_/g, " ");
  const centres = m[2].replace(/-/g, "–");
  return kind.endsWith(")") ? `${kind.slice(0, -1)}, ${centres})` : `${kind}(${centres})`;
}

const minus = (s: string) => s.replace(/^-/, "−");
/** A signed number with a typographic minus. */
export const signedNum = (v: number, digits: number): string => minus(v.toFixed(digits));
const mo = (label: string) => label.replace(/([A-Z])-(\d)/g, "$1−$2");

export interface CaptionParts {
  channel: string;
  g: string;
  canonical: string | null;
}

/** The caption of one reading: "n(O) → σ*(C–Cl)", "G^atom −1.28 eV⁻¹", "HOMO → LUMO, −8.55 / −4.21 eV". */
export function captionParts(r: OrbitalReadingOut): CaptionParts {
  const d = diagSymbol(r.donor_diag) ?? `orbital ${r.donor_token}`;
  const a = diagSymbol(r.acceptor_diag) ?? `orbital ${r.acceptor_token}`;
  const c = r.canonical;
  return {
    channel: `${d} → ${a}`,
    g: `G^atom ${signedNum(r.g_atom, 2)} eV⁻¹`,
    canonical: c ? `${mo(c.k_label)} → ${mo(c.m_label)}, ${signedNum(c.e_k_ev, 2)} / ${signedNum(c.e_m_ev, 2)} eV` : null,
  };
}

export const captionText = (r: OrbitalReadingOut): string => {
  const p = captionParts(r);
  return [p.channel, p.g, p.canonical].filter(Boolean).join(" · ");
};

/** The dominant reading: the largest |term| (the channel that carries most of G^atom). */
export function dominantReading(rs: readonly OrbitalReadingOut[]): number {
  let best = -1;
  rs.forEach((r, k) => {
    if (best < 0 || Math.abs(r.term) > Math.abs(rs[best].term)) best = k;
  });
  return best;
}

/** An R entry in words. */
export function entryWords(kind: "bond" | "lone", delta: number, before: number): string {
  if (kind === "lone") return delta > 0 ? "gains a lone pair" : "gives up a lone pair";
  if (delta > 0) return before === 0 ? "bond forms" : `bond order ${before} → ${before + delta}`;
  return before + delta === 0 ? "bond breaks" : `bond order ${before} → ${before + delta}`;
}

/** The next alternative index (-1 = the edge itself) going `dir` through n alternatives. */
export function stepAlternative(index: number, n: number, dir: 1 | -1): number {
  if (n <= 0) return -1;
  const k = index + 1 + dir;
  return ((k % (n + 1)) + n + 1) % (n + 1) - 1;
}
