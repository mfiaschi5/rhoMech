// The workbench's customizable layout as pure state (no React, no DOM, runs under node --test):
// which panels sit in which column, in which order, the column widths, the panels' heights,
// which are hidden or collapsed, and whether the orbital info is open beside the diagram.
//
// Areas: 1 to MAX_COLUMNS columns side by side, then a full-width row ("wide") under them.
// Every panel is in exactly one area; a hidden panel keeps its place (showing it puts it back).
// Moves: up / down swap with the next visible panel of the area; down from the last panel of a
// column goes to the top of the wide row, up from the top of the wide row to the end of the
// column with the fewest visible panels; left / right go to the neighbouring column (right of
// the last one opens a new column while there are fewer than MAX_COLUMNS), from the wide row to
// the first or last column. A column left without panels is removed and the widths renormalize.
// Every action but a preset marks the layout "custom". sanitizeLayout() turns anything read
// from storage into a valid state (unknown panels dropped, missing ones added to the wide row,
// widths and heights clamped), so a stale or hand-edited entry never breaks the page.

export const PANELS = ["sketch", "viewer", "orbitals", "surfaces", "properties"] as const;
export type PanelId = (typeof PANELS)[number];

export const PANEL_TITLES: Record<PanelId, string> = {
  sketch: "Sketch",
  viewer: "3D view",
  orbitals: "Orbitals",
  surfaces: "Surfaces",
  properties: "Properties",
};

/** The panels whose body height a row splitter sets (px): bounds and default. */
export const HEIGHTS = {
  sketch: { min: 200, max: 1000, dflt: 340 },
  viewer: { min: 200, max: 1400, dflt: 460 },
  orbitals: { min: 220, max: 1000, dflt: 380 },
  properties: { min: 160, max: 1200, dflt: 380 },
} as const;
export type HeightPanel = keyof typeof HEIGHTS;
export const isHeightPanel = (p: PanelId): p is HeightPanel => p in HEIGHTS;

export const MAX_COLUMNS = 3;
/** The narrowest a column may get, as a fraction of the visible columns' total. */
export const MIN_FRACTION = 0.18;

export const PRESETS = ["default", "sketch", "3d", "orbitals"] as const;
export type PresetName = (typeof PRESETS)[number];
export const PRESET_TITLES: Record<PresetName, string> = {
  default: "Default",
  sketch: "Sketch-focused",
  "3d": "3D-focused",
  orbitals: "Orbitals-focused",
};

export interface LayoutState {
  columns: PanelId[][];
  /** Column widths as fractions (same length as columns, sum 1). */
  widths: number[];
  wide: PanelId[];
  hidden: PanelId[];
  collapsed: PanelId[];
  heights: Record<HeightPanel, number>;
  /** The orbital list and stats open beside the level diagram. */
  orbitalInfo: boolean;
  preset: PresetName | "custom";
}

export type Direction = "left" | "right" | "up" | "down";

export type LayoutAction =
  | { type: "preset"; name: PresetName }
  | { type: "reset" }
  | { type: "move"; panel: PanelId; dir: Direction }
  | { type: "hide"; panel: PanelId }
  | { type: "show"; panel: PanelId }
  | { type: "collapse"; panel: PanelId; collapsed?: boolean }
  | { type: "resizeColumns"; left: number; right: number; from: readonly [number, number]; delta: number }
  | { type: "setHeight"; panel: HeightPanel; px: number }
  | { type: "orbitalInfo"; open?: boolean };

const defaultHeights = (): Record<HeightPanel, number> => ({
  sketch: HEIGHTS.sketch.dflt,
  viewer: HEIGHTS.viewer.dflt,
  orbitals: HEIGHTS.orbitals.dflt,
  properties: HEIGHTS.properties.dflt,
});

/** The layout of a preset (nothing hidden or collapsed). */
export function presetLayout(name: PresetName): LayoutState {
  const base = { wide: ["properties"] as PanelId[], hidden: [], collapsed: [], heights: defaultHeights(), orbitalInfo: true, preset: name };
  switch (name) {
    case "sketch":
      return { ...base, columns: [["sketch"], ["viewer", "orbitals", "surfaces"]], widths: [0.6, 0.4], heights: { ...base.heights, sketch: 560, viewer: 380 }, orbitalInfo: false };
    case "3d":
      return { ...base, columns: [["viewer", "surfaces"], ["sketch", "orbitals"]], widths: [0.64, 0.36], heights: { ...base.heights, viewer: 680, sketch: 300 }, orbitalInfo: false };
    case "orbitals":
      return { ...base, columns: [["orbitals", "sketch"], ["viewer", "surfaces"]], widths: [0.56, 0.44], heights: { ...base.heights, orbitals: 560, sketch: 280 } };
    default:
      // the orbitals full width, so the level diagram and the orbital list sit side by side
      return { ...base, columns: [["sketch"], ["viewer", "surfaces"]], widths: [5 / 12, 7 / 12], wide: ["orbitals", "properties"], heights: { ...base.heights, sketch: 400 } };
  }
}

export const DEFAULT_LAYOUT: LayoutState = presetLayout("default");

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Fractions scaled to sum 1 (equal parts when they cannot be). */
export function normalizeWidths(w: readonly number[]): number[] {
  const ok = w.map((x) => (Number.isFinite(x) && x > 0 ? x : 0));
  const sum = ok.reduce((s, x) => s + x, 0);
  return sum > 0 ? ok.map((x) => x / sum) : w.map(() => 1 / Math.max(1, w.length));
}

/** Drop empty columns (always keeping one) with their widths. */
function prune(s: LayoutState): LayoutState {
  const keep = s.columns.map((c, i) => [c, i] as const).filter(([c]) => c.length > 0);
  if (keep.length === s.columns.length) return s;
  if (keep.length === 0) return { ...s, columns: [[]], widths: [1] };
  return { ...s, columns: keep.map(([c]) => c), widths: normalizeWidths(keep.map(([, i]) => s.widths[i])) };
}

type Place = { area: number | "wide"; index: number };

function placeOf(s: LayoutState, p: PanelId): Place | null {
  const w = s.wide.indexOf(p);
  if (w >= 0) return { area: "wide", index: w };
  for (let c = 0; c < s.columns.length; c++) {
    const i = s.columns[c].indexOf(p);
    if (i >= 0) return { area: c, index: i };
  }
  return null;
}

const areaList = (s: LayoutState, a: number | "wide") => (a === "wide" ? s.wide : s.columns[a]);

function withArea(s: LayoutState, a: number | "wide", list: PanelId[]): LayoutState {
  if (a === "wide") return { ...s, wide: list };
  return { ...s, columns: s.columns.map((c, i) => (i === a ? list : c)) };
}

/** `p` taken out of its area and inserted into `to` at `index` (clamped). */
function relocate(s: LayoutState, p: PanelId, to: number | "wide", index: number): LayoutState {
  const from = placeOf(s, p);
  if (!from) return s;
  let next = withArea(s, from.area, areaList(s, from.area).filter((x) => x !== p));
  if (to !== "wide" && to >= next.columns.length) {
    const n = next.columns.length + 1;
    next = { ...next, columns: [...next.columns, []], widths: [...next.widths.map((w) => (w * (n - 1)) / n), 1 / n] };
  }
  const list = [...areaList(next, to)];
  list.splice(clamp(index, 0, list.length), 0, p);
  return prune(withArea(next, to, list));
}

const visibleCount = (s: LayoutState, list: readonly PanelId[]) => list.filter((p) => !s.hidden.includes(p)).length;

/** Swap `p` with the next visible panel of its area in `step` direction; null at the end. */
function swapInArea(s: LayoutState, at: Place, step: -1 | 1): LayoutState | null {
  const list = areaList(s, at.area);
  let j = at.index + step;
  while (j >= 0 && j < list.length && s.hidden.includes(list[j])) j += step;
  if (j < 0 || j >= list.length) return null;
  const next = [...list];
  [next[at.index], next[j]] = [next[j], next[at.index]];
  return withArea(s, at.area, next);
}

function shortestColumn(s: LayoutState): number {
  let best = 0;
  s.columns.forEach((c, i) => {
    if (visibleCount(s, c) < visibleCount(s, s.columns[best])) best = i;
  });
  return best;
}

/** The state after moving `p` one step in `dir` (the same state when it cannot move). */
export function movePanel(s: LayoutState, p: PanelId, dir: Direction): LayoutState {
  const at = placeOf(s, p);
  if (!at) return s;
  if (dir === "up" || dir === "down") {
    const swapped = swapInArea(s, at, dir === "up" ? -1 : 1);
    if (swapped) return swapped;
    if (at.area !== "wide" && dir === "down") return relocate(s, p, "wide", 0);
    if (at.area === "wide" && dir === "up") {
      const c = shortestColumn(s);
      return relocate(s, p, c, s.columns[c].length);
    }
    return s;
  }
  if (at.area === "wide") {
    const c = dir === "left" ? 0 : s.columns.length - 1;
    return relocate(s, p, c, s.columns[c].length);
  }
  const c = at.area + (dir === "left" ? -1 : 1);
  const alone = s.columns[at.area].length === 1;
  if (c < 0 || c >= MAX_COLUMNS || (c >= s.columns.length && alone)) return s;
  return relocate(s, p, c, at.index);
}

export const canMove = (s: LayoutState, p: PanelId, dir: Direction): boolean => movePanel(s, p, dir) !== s;

function toggleIn(list: readonly PanelId[], p: PanelId, on: boolean): PanelId[] {
  const has = list.includes(p);
  if (has === on) return list as PanelId[];
  return on ? [...list, p] : list.filter((x) => x !== p);
}

/** The two widths of a column splitter drag: `delta` (a fraction of the visible total) moves
 *  from the right column to the left one, neither going under MIN_FRACTION of their pair's
 *  total share. */
function resized(s: LayoutState, a: Extract<LayoutAction, { type: "resizeColumns" }>): LayoutState {
  const { left, right, from, delta } = a;
  if (left === right || !s.widths[left] || !s.widths[right] || !Number.isFinite(delta)) return s;
  const pair = from[0] + from[1];
  const min = Math.min(MIN_FRACTION, pair / 2);
  const l = clamp(from[0] + delta, min, pair - min);
  const widths = s.widths.map((w, i) => (i === left ? l : i === right ? pair - l : w));
  return { ...s, widths: normalizeWidths(widths) };
}

export function layoutReducer(s: LayoutState, a: LayoutAction): LayoutState {
  const custom = (next: LayoutState) => (next === s ? s : { ...next, preset: "custom" as const });
  switch (a.type) {
    case "preset":
      return presetLayout(a.name);
    case "reset":
      return presetLayout("default");
    case "move":
      return custom(movePanel(s, a.panel, a.dir));
    case "hide":
    case "show": {
      const hidden = toggleIn(s.hidden, a.panel, a.type === "hide");
      return hidden === s.hidden ? s : custom({ ...s, hidden });
    }
    case "collapse": {
      const collapsed = toggleIn(s.collapsed, a.panel, a.collapsed ?? !s.collapsed.includes(a.panel));
      return collapsed === s.collapsed ? s : custom({ ...s, collapsed });
    }
    case "resizeColumns":
      return custom(resized(s, a));
    case "setHeight": {
      const b = HEIGHTS[a.panel];
      const px = Math.round(clamp(a.px, b.min, b.max));
      return !b || px === s.heights[a.panel] ? s : custom({ ...s, heights: { ...s.heights, [a.panel]: px } });
    }
    case "orbitalInfo": {
      const open = a.open ?? !s.orbitalInfo;
      return open === s.orbitalInfo ? s : { ...s, orbitalInfo: open };
    }
  }
}

const isPanel = (x: unknown): x is PanelId => typeof x === "string" && (PANELS as readonly string[]).includes(x);

/** Unique known panels of a stored list, skipping those already `seen`. */
function panelList(x: unknown, seen: Set<PanelId>): PanelId[] {
  const out: PanelId[] = [];
  for (const p of Array.isArray(x) ? x : []) {
    if (isPanel(p) && !seen.has(p)) {
      seen.add(p);
      out.push(p);
    }
  }
  return out;
}

function sanitizeHeights(x: unknown): Record<HeightPanel, number> {
  const h = defaultHeights();
  const v = (x && typeof x === "object" ? x : {}) as Record<string, unknown>;
  for (const k of Object.keys(HEIGHTS) as HeightPanel[]) {
    const n = v[k];
    if (typeof n === "number" && Number.isFinite(n)) h[k] = Math.round(clamp(n, HEIGHTS[k].min, HEIGHTS[k].max));
  }
  return h;
}

/** A valid layout from anything (storage); DEFAULT_LAYOUT for nothing usable. */
export function sanitizeLayout(x: unknown): LayoutState {
  if (!x || typeof x !== "object") return DEFAULT_LAYOUT;
  const v = x as Partial<Record<keyof LayoutState, unknown>>;
  const seen = new Set<PanelId>();
  const cols = (Array.isArray(v.columns) ? v.columns : []).slice(0, MAX_COLUMNS).map((c) => panelList(c, seen));
  const wide = panelList(v.wide, seen);
  for (const p of PANELS) if (!seen.has(p)) wide.push(p);
  const rawW = Array.isArray(v.widths) ? v.widths : [];
  const widths = normalizeWidths(cols.map((_, i) => (typeof rawW[i] === "number" ? (rawW[i] as number) : 0)));
  const preset = (PRESETS as readonly unknown[]).includes(v.preset) ? (v.preset as PresetName) : "custom";
  const state: LayoutState = {
    columns: cols.length ? cols : [[]],
    widths: cols.length ? widths : [1],
    wide,
    hidden: panelList(v.hidden, new Set()),
    collapsed: panelList(v.collapsed, new Set()),
    heights: sanitizeHeights(v.heights),
    orbitalInfo: v.orbitalInfo !== false,
    preset,
  };
  return prune(state);
}

export interface VisibleColumn {
  /** Index into state.columns (the resize action's left / right). */
  index: number;
  panels: PanelId[];
  /** Share of the visible columns' width. */
  fraction: number;
}

/** The columns with at least one shown panel, their widths renormalized among them. */
export function visibleColumns(s: LayoutState): VisibleColumn[] {
  const shown = s.columns.map((c, index) => ({ index, panels: c })).filter((c) => visibleCount(s, c.panels) > 0);
  const f = normalizeWidths(shown.map((c) => s.widths[c.index]));
  return shown.map((c, k) => ({ ...c, fraction: f[k] }));
}

/** The CSS grid template of the visible columns with an 8 px splitter track between them. */
export function gridTemplate(cols: readonly VisibleColumn[], splitterPx = 8): string {
  return cols.map((c) => `minmax(0, ${c.fraction.toFixed(4)}fr)`).join(` ${splitterPx}px `) || "minmax(0, 1fr)";
}
