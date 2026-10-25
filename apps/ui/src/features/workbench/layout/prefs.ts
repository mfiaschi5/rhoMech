// The workbench's user preferences as pure data: sketch label font size, bond line width, atom
// colouring, label style, the sketch's drawing style (a preset or the custom style, style.ts),
// the default 3D representation, the default orbital isovalue and the theme override. The font
// size, line width and atom colouring set the Screen drawing style. sanitizePrefs() turns anything read from storage into valid preferences
// (out-of-range numbers clamped, unknown values back to the default).
//
// Theme: "auto" follows prefers-color-scheme (no data-theme on <html>); "light" / "dark" set
// <html data-theme>, which styles.css already honours over the media query.
import type { LabelStyle } from "../sketch/drawing";
import type { Rep } from "../params";
import type { DrawingStyle, StyleId } from "../sketch/style";
// @ts-ignore TS5097: node --experimental-strip-types needs the extension (Vite accepts it)
import { DEFAULT_REP, REPS } from "../params.ts";
// @ts-ignore TS5097: see above
import { ACS_1996_STYLE, sanitizeStyle, STYLE_IDS } from "../sketch/style.ts";

export const THEMES = ["auto", "light", "dark"] as const;
export type ThemeChoice = (typeof THEMES)[number];
export const LABEL_STYLE_CHOICES: readonly LabelStyle[] = ["skeletal", "all", "no-h"];
export const LABEL_STYLE_TITLES: Record<LabelStyle, string> = {
  skeletal: "skeletal (carbons as vertices)",
  all: "all atoms labelled",
  "no-h": "no implicit H",
};

export interface Prefs {
  fontPx: number;
  bondWidth: number;
  colourAtoms: boolean;
  labelStyle: LabelStyle;
  /** The sketch's drawing style. */
  drawingStyle: StyleId;
  /** The custom drawing style (the style editor's), kept while another style is chosen. */
  customStyle: DrawingStyle;
  rep: Rep;
  /** Default isovalue of new orbital (psi) surface layers. */
  orbitalIso: number;
  theme: ThemeChoice;
}

export const PREF_BOUNDS = {
  fontPx: { min: 9, max: 22, step: 1 },
  bondWidth: { min: 0.6, max: 4, step: 0.1 },
  orbitalIso: { min: 0.005, max: 0.2, step: 0.005 },
} as const;

export const DEFAULT_PREFS: Prefs = {
  fontPx: 13,
  bondWidth: 1.5,
  colourAtoms: true,
  labelStyle: "skeletal",
  drawingStyle: "screen",
  customStyle: ACS_1996_STYLE,
  rep: DEFAULT_REP,
  orbitalIso: 0.05,
  theme: "auto",
};

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function num(x: unknown, key: keyof typeof PREF_BOUNDS): number {
  const b = PREF_BOUNDS[key];
  return typeof x === "number" && Number.isFinite(x) ? clamp(x, b.min, b.max) : DEFAULT_PREFS[key];
}

const oneOf = <T extends string>(x: unknown, list: readonly T[], dflt: T): T => ((list as readonly unknown[]).includes(x) ? (x as T) : dflt);

/** Valid preferences from anything (storage); missing or bad fields take the default. */
export function sanitizePrefs(x: unknown): Prefs {
  const v = (x && typeof x === "object" ? x : {}) as Partial<Record<keyof Prefs, unknown>>;
  return {
    fontPx: Math.round(num(v.fontPx, "fontPx")),
    bondWidth: Math.round(num(v.bondWidth, "bondWidth") * 10) / 10,
    colourAtoms: typeof v.colourAtoms === "boolean" ? v.colourAtoms : DEFAULT_PREFS.colourAtoms,
    labelStyle: oneOf(v.labelStyle, LABEL_STYLE_CHOICES, DEFAULT_PREFS.labelStyle),
    drawingStyle: oneOf(v.drawingStyle, STYLE_IDS, DEFAULT_PREFS.drawingStyle),
    customStyle: sanitizeStyle(v.customStyle, DEFAULT_PREFS.customStyle),
    rep: oneOf(v.rep, REPS, DEFAULT_PREFS.rep),
    orbitalIso: num(v.orbitalIso, "orbitalIso"),
    theme: oneOf(v.theme, THEMES, DEFAULT_PREFS.theme),
  };
}

/** The preferences with `patch` applied, still valid. */
export const updatePrefs = (p: Prefs, patch: Partial<Prefs>): Prefs => sanitizePrefs({ ...p, ...patch });

/** The data-theme value for <html> (null: none, follow the system). */
export const themeAttribute = (t: ThemeChoice): "light" | "dark" | null => (t === "auto" ? null : t);
