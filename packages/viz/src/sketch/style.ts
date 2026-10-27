// Drawing styles of the sketcher: what a ChemDraw document style controls, as pure data in
// points (1 pt = 1/72 in), plus the presets and the point -> screen-pixel mapping.
//
// The fixed bond length is the length a standard model bond (model.BOND_LENGTH = 1.5 units, what
// the tools draw and what imports and Clean normalise to) has in the style. A style with
// scaleWithZoom maps points to pixels through the zoom: pxPerPt = view.s * BOND_LENGTH /
// bondLengthPt, so every dimension keeps its proportion to the bonds at any zoom (what the
// export prints is what the canvas shows, scaled). "Screen" (the app's own look) keeps its sizes
// in fixed pixels instead (1 pt = 4/3 CSS px), as the sketcher always did: labels stay readable
// on a large molecule fitted in a small panel.
//
// Presets:
//   screen            the app default: 13 px labels, 1.5 px lines, colour (the font size, line
//                     width and atom colour preferences set this style)
//   acs1996           ChemDraw's "ACS Document 1996" style sheet. Values from Revvity Signals
//                     (ChemDraw's vendor) support article 4408234173332, "When I print the
//                     structures, the line thickness for chemical bonds has increased ...":
//                     chain angle 120 deg, bond spacing 18 % of length, fixed length 14.4 pt,
//                     bold width 2 pt, line width 0.6 pt, margin width 1.6 pt, hash spacing
//                     2.5 pt, atom labels Arial/Helvetica 10 pt (also the Middlebury science
//                     writing guide, sites.middlebury.edu/middsciwriting/?p=334). The
//                     subscript and charge sizes, the hash tick width (= line width), the
//                     crossing gap (= margin) and the inner-line shortening are not part of the
//                     published list; ours are noted below.
//   total-synthesis   the bold look of total-synthesis papers and graphical abstracts. There
//                     is no official standard: these are the app's own values (thick 1.1 pt
//                     lines on a 15 pt bond, 3.2 pt tapered wedges, wide hashes, 20 % spacing,
//                     Arial 11 pt, generous margins), black and white.
//   custom            the user's own (the style editor), started from any preset.
// RSC and Wiley styles are not offered: neither publisher's ChemDraw settings could be verified
// from a published source (docs/decisions.md, 2026-10-25).
//
// Pure, no imports beyond types: runs under node --experimental-strip-types.

export const STYLE_IDS = ["screen", "acs1996", "total-synthesis", "custom"] as const;
export type StyleId = (typeof STYLE_IDS)[number];
export type PresetId = Exclude<StyleId, "custom">;
export const PRESET_IDS: readonly PresetId[] = ["screen", "acs1996", "total-synthesis"];

export const FONT_FAMILIES = ["ui", "arial", "helvetica", "times"] as const;
export type FontFamily = (typeof FONT_FAMILIES)[number];
export type WedgeShape = "wedge" | "tapered";
export type ChargePlacement = "superscript" | "circled";
export type TerminalDouble = "centred" | "offset";

export interface DrawingStyle {
  /** Fixed bond length (pt): the length of a standard model bond. */
  bondLengthPt: number;
  lineWidthPt: number;
  /** Width (pt) of the wide end of a wedge (ChemDraw's bold width). */
  boldWidthPt: number;
  /** Centre-to-centre spacing (pt) of the ticks of a hashed wedge. */
  hashSpacingPt: number;
  /** Stroke width (pt) of a hash tick. */
  hashWidthPt: number;
  /** Distance between the lines of a double bond, % of the fixed bond length. */
  bondSpacingPct: number;
  /** How much the inner line of a ring double bond is shortened at each end, % of its length. */
  innerShortenPct: number;
  /** Double bond at a chain end next to a branch point (C=O on a ring): centred or offset. */
  terminalDouble: TerminalDouble;
  /** Margin width (pt): the clearance between a label and the bonds that reach it. */
  marginPt: number;
  fontFamily: FontFamily;
  fontSizePt: number;
  fontWeight: 400 | 700;
  /** H-count subscript size, % of the label font. */
  subscriptPct: number;
  /** Charge sign size, % of the label font. */
  chargePct: number;
  chargePlacement: ChargePlacement;
  /** Element colours on the labels (false: black and white). */
  colour: boolean;
  /** Gap (pt) between the front bond and the ends of the bond broken behind a crossing. */
  breakGapPt: number;
  wedgeShape: WedgeShape;
  /** Chain angle (degrees): the angle between consecutive bonds of a chain the tools draw. */
  chainAngleDeg: number;
  labelCarbons: boolean;
  /** Terminal methyls drawn as "CH3" / "H3C". */
  labelMethyls: boolean;
  /** Sizes follow the zoom (publication styles) or stay in fixed screen pixels (Screen). */
  scaleWithZoom: boolean;
}

/** CSS pixels per point (96 px per inch / 72 pt per inch). */
export const CSS_PX_PER_PT = 4 / 3;
/** The model length of a standard bond (model.BOND_LENGTH; repeated to stay import-free). */
export const MODEL_BOND = 1.5;

export const SCREEN_STYLE: DrawingStyle = {
  bondLengthPt: 28.125, // 37.5 px at 100 % zoom (Sketcher.DEFAULT_SCALE 25 px per unit)
  lineWidthPt: 1.125, // 1.5 px
  boldWidthPt: 5.4, // 7.2 px
  hashSpacingPt: 3.375, // 4.5 px
  hashWidthPt: 1.125,
  bondSpacingPct: 12, // 4.5 px
  innerShortenPct: 14,
  terminalDouble: "centred",
  marginPt: 2,
  fontFamily: "ui",
  fontSizePt: 9.75, // 13 px
  fontWeight: 400,
  subscriptPct: 73,
  chargePct: 77,
  chargePlacement: "superscript",
  colour: true,
  breakGapPt: 2.4, // 3.2 px
  wedgeShape: "wedge",
  chainAngleDeg: 120,
  labelCarbons: false,
  labelMethyls: false,
  scaleWithZoom: false,
};

export const ACS_1996_STYLE: DrawingStyle = {
  bondLengthPt: 14.4, // 0.508 cm (0.2 in)
  lineWidthPt: 0.6,
  boldWidthPt: 2.0,
  hashSpacingPt: 2.5,
  hashWidthPt: 0.6, // ours: the line width
  bondSpacingPct: 18,
  innerShortenPct: 12, // ours
  terminalDouble: "centred",
  marginPt: 1.6,
  fontFamily: "arial",
  fontSizePt: 10,
  fontWeight: 400,
  subscriptPct: 75, // ours
  chargePct: 75, // ours
  chargePlacement: "superscript",
  colour: false,
  breakGapPt: 1.6, // ours: the margin width
  wedgeShape: "wedge",
  chainAngleDeg: 120,
  labelCarbons: false,
  labelMethyls: false,
  scaleWithZoom: true,
};

export const TOTAL_SYNTHESIS_STYLE: DrawingStyle = {
  bondLengthPt: 15,
  lineWidthPt: 1.1,
  boldWidthPt: 3.2,
  hashSpacingPt: 2.6,
  hashWidthPt: 1.0,
  bondSpacingPct: 20,
  innerShortenPct: 12,
  terminalDouble: "centred",
  marginPt: 2.2,
  fontFamily: "arial",
  fontSizePt: 11,
  fontWeight: 400,
  subscriptPct: 75,
  chargePct: 80,
  chargePlacement: "superscript",
  colour: false,
  breakGapPt: 2.2,
  wedgeShape: "tapered",
  chainAngleDeg: 120,
  labelCarbons: false,
  labelMethyls: false,
  scaleWithZoom: true,
};

export const PRESETS: Record<PresetId, DrawingStyle> = {
  screen: SCREEN_STYLE,
  acs1996: ACS_1996_STYLE,
  "total-synthesis": TOTAL_SYNTHESIS_STYLE,
};

export const STYLE_TITLES: Record<StyleId, string> = {
  screen: "Screen",
  acs1996: "ACS Document 1996",
  "total-synthesis": "Total synthesis",
  custom: "Custom",
};

export const STYLE_TIPS: Record<StyleId, string> = {
  screen: "The app's screen style: fixed-pixel labels and lines (font size, line width and atom colours from Layout → Preferences)",
  acs1996: "ChemDraw's ACS Document 1996 style sheet: 14.4 pt (0.508 cm) bonds, 0.6 pt lines, 2.0 pt wedges, 2.5 pt hash spacing, 18 % double-bond spacing, 1.6 pt margins, Arial 10 pt, black and white",
  "total-synthesis": "Total synthesis: the bold look of synthesis papers and graphical abstracts. Not an official standard, the app's own preset: 15 pt bonds, 1.1 pt lines, 3.2 pt tapered wedges, 20 % spacing, Arial 11 pt, black and white",
  custom: "Your own style: edit every field, starting from any preset",
};

type NumKey = { [K in keyof DrawingStyle]: DrawingStyle[K] extends number ? K : never }[keyof DrawingStyle];

/** Bounds of the numeric fields (the editor's inputs and sanitizeStyle). */
export const STYLE_BOUNDS: Record<Exclude<NumKey, "fontWeight">, { min: number; max: number; step: number; label: string; unit: string }> = {
  bondLengthPt: { min: 5, max: 60, step: 0.1, label: "bond length", unit: "pt" },
  lineWidthPt: { min: 0.1, max: 5, step: 0.05, label: "line width", unit: "pt" },
  boldWidthPt: { min: 0.5, max: 12, step: 0.1, label: "bold (wedge) width", unit: "pt" },
  hashSpacingPt: { min: 0.8, max: 10, step: 0.1, label: "hash spacing", unit: "pt" },
  hashWidthPt: { min: 0.1, max: 5, step: 0.05, label: "hash tick width", unit: "pt" },
  bondSpacingPct: { min: 5, max: 40, step: 1, label: "double-bond spacing", unit: "% of bond" },
  innerShortenPct: { min: 0, max: 30, step: 1, label: "inner line shortening", unit: "% each end" },
  marginPt: { min: 0, max: 8, step: 0.1, label: "margin width", unit: "pt" },
  fontSizePt: { min: 4, max: 30, step: 0.5, label: "label font size", unit: "pt" },
  subscriptPct: { min: 40, max: 100, step: 1, label: "subscript size", unit: "% of font" },
  chargePct: { min: 40, max: 120, step: 1, label: "charge size", unit: "% of font" },
  breakGapPt: { min: 0, max: 8, step: 0.1, label: "crossing gap", unit: "pt" },
  chainAngleDeg: { min: 90, max: 150, step: 1, label: "chain angle", unit: "°" },
};

export const NUMERIC_FIELDS = Object.keys(STYLE_BOUNDS) as (keyof typeof STYLE_BOUNDS)[];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const oneOf = <T,>(x: unknown, list: readonly T[], dflt: T): T => ((list as readonly unknown[]).includes(x) ? (x as T) : dflt);
const bool = (x: unknown, dflt: boolean) => (typeof x === "boolean" ? x : dflt);

/** A valid style from anything (storage, the editor); bad fields take `base`'s value. */
export function sanitizeStyle(x: unknown, base: DrawingStyle = ACS_1996_STYLE): DrawingStyle {
  const v = (x && typeof x === "object" ? x : {}) as Partial<Record<keyof DrawingStyle, unknown>>;
  const out: DrawingStyle = { ...base };
  for (const k of NUMERIC_FIELDS) {
    const b = STYLE_BOUNDS[k];
    const n = v[k];
    if (typeof n === "number" && Number.isFinite(n)) out[k] = Math.round(clamp(n, b.min, b.max) * 1000) / 1000;
  }
  out.fontWeight = v.fontWeight === 700 ? 700 : v.fontWeight === 400 ? 400 : base.fontWeight;
  out.fontFamily = oneOf(v.fontFamily, FONT_FAMILIES, base.fontFamily);
  out.wedgeShape = oneOf<WedgeShape>(v.wedgeShape, ["wedge", "tapered"], base.wedgeShape);
  out.chargePlacement = oneOf<ChargePlacement>(v.chargePlacement, ["superscript", "circled"], base.chargePlacement);
  out.terminalDouble = oneOf<TerminalDouble>(v.terminalDouble, ["centred", "offset"], base.terminalDouble);
  out.colour = bool(v.colour, base.colour);
  out.labelCarbons = bool(v.labelCarbons, base.labelCarbons);
  out.labelMethyls = bool(v.labelMethyls, base.labelMethyls);
  out.scaleWithZoom = bool(v.scaleWithZoom, base.scaleWithZoom);
  return out;
}

/** What the screen preferences change in the Screen style. */
export interface ScreenPrefs {
  fontPx?: number;
  bondWidth?: number;
  colourAtoms?: boolean;
}

/** The style to draw with: a preset (Screen with the screen preferences applied) or `custom`. */
export function resolveStyle(id: StyleId, custom: DrawingStyle | null | undefined, screen: ScreenPrefs = {}): DrawingStyle {
  if (id === "custom") return custom ?? ACS_1996_STYLE;
  if (id !== "screen") return PRESETS[id];
  const s = { ...SCREEN_STYLE };
  if (screen.fontPx) s.fontSizePt = screen.fontPx / CSS_PX_PER_PT;
  if (screen.bondWidth) s.lineWidthPt = s.hashWidthPt = screen.bondWidth / CSS_PX_PER_PT;
  if (screen.colourAtoms === false) s.colour = false;
  return s;
}

/** Screen pixels per point at zoom `viewScale` (px per model unit). */
export function pxPerPt(style: DrawingStyle, viewScale: number): number {
  return style.scaleWithZoom ? (viewScale * MODEL_BOND) / style.bondLengthPt : CSS_PX_PER_PT;
}

/** The zoom (px per model unit) at which a standard bond is drawn at its true size on a 96 dpi
 *  CSS screen ("Actual size"). */
export const actualSizeScale = (style: DrawingStyle): number => (style.bondLengthPt * CSS_PX_PER_PT) / MODEL_BOND;

/** A style's dimensions in output units (screen px, or pt for an export): what drawing.ts uses. */
export interface DrawMetrics {
  /** Output units per point. */
  perPt: number;
  bond: number;
  line: number;
  bold: number;
  hashSpacing: number;
  hashWidth: number;
  doubleGap: number;
  innerShorten: number;
  margin: number;
  font: number;
  sub: number;
  charge: number;
  breakGap: number;
  wedgeShape: WedgeShape;
  chargePlacement: ChargePlacement;
  terminalDouble: TerminalDouble;
  fontFamily: FontFamily;
  fontWeight: 400 | 700;
  colour: boolean;
  labelCarbons: boolean;
  labelMethyls: boolean;
  /** Label widths from Helvetica/Arial metrics (publication fonts) or the UI-font estimate. */
  metricText: boolean;
  /** Sizes in fixed screen pixels (Screen), not following the zoom. */
  fixedScreen: boolean;
}

export function drawMetrics(s: DrawingStyle, perPt: number): DrawMetrics {
  const font = s.fontSizePt * perPt;
  return {
    perPt,
    bond: s.bondLengthPt * perPt,
    line: s.lineWidthPt * perPt,
    bold: s.boldWidthPt * perPt,
    hashSpacing: s.hashSpacingPt * perPt,
    hashWidth: s.hashWidthPt * perPt,
    doubleGap: (s.bondSpacingPct / 100) * s.bondLengthPt * perPt,
    innerShorten: s.innerShortenPct / 100,
    margin: s.marginPt * perPt,
    font,
    sub: (font * s.subscriptPct) / 100,
    charge: (font * s.chargePct) / 100,
    breakGap: s.breakGapPt * perPt,
    wedgeShape: s.wedgeShape,
    chargePlacement: s.chargePlacement,
    terminalDouble: s.terminalDouble,
    fontFamily: s.fontFamily,
    fontWeight: s.fontWeight,
    colour: s.colour,
    labelCarbons: s.labelCarbons,
    labelMethyls: s.labelMethyls,
    metricText: s.fontFamily !== "ui",
    fixedScreen: !s.scaleWithZoom,
  };
}

/** The metrics of the Screen style with label font `fontPx` (drawing.ts's default). */
export const screenMetrics = (fontPx = 13): DrawMetrics => drawMetrics({ ...SCREEN_STYLE, fontSizePt: fontPx / CSS_PX_PER_PT }, CSS_PX_PER_PT);

/** CSS font-family of a style font (`ui`: the page font on screen, sans-serif elsewhere). */
export function cssFontFamily(f: FontFamily, onScreen: boolean): string {
  if (f === "ui") return onScreen ? "var(--font)" : "Arial, Helvetica, sans-serif";
  if (f === "arial") return "Arial, Helvetica, 'Liberation Sans', sans-serif";
  if (f === "helvetica") return "Helvetica, Arial, 'Liberation Sans', sans-serif";
  return "'Times New Roman', Times, 'Liberation Serif', serif";
}
