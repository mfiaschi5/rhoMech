// LevelDiagram: the canonical MO energy levels in hand-made SVG, at the panel's full size.
// Degenerate MOs (within the degeneracy threshold, regrouped client-side: levels.ts) sit side by
// side; occupations as half-arrows; the HOMO-1, HOMO, LUMO and LUMO+1 blocks labelled with their
// energies, the labels kept apart with leaders. Click a level to show that MO. When an LMO set
// is shown, a second column holds the LMO energies E_r (click: show that LMO) and the selected
// LMO's spectral weights are drawn as bars on the canonical levels.
//
// Zoom and pan of the energy axis: the mouse wheel (or Ctrl+wheel) zooms about the cursor,
// Shift+wheel and dragging pan, a double-click resets to the window; the buttons zoom in, zoom
// out, fit all levels and show the frontier window (both also set the window select); with the
// drawing focused, + / - zoom, 0 resets and the arrow keys pan. The degeneracy threshold input
// (0.001-0.5 eV) regroups the levels; the page persists it with the layout preferences.
//
// DOM: <figure data-testid="level-diagram" data-mode="canonical|lmo" data-lo data-hi (the
// visible energy range, eV) data-span data-zoom (full span / visible span) data-groups (the
// number of degenerate groups at the threshold) data-threshold data-window>; buttons
// level-zoom-in, level-zoom-out, level-fit, level-frontier; <input data-testid=
// "degeneracy-threshold">; <svg data-testid="level-svg">; canonical level lines <g data-testid=
// "level-<i>" data-frontier="homo|lumo|homo_minus_1|lumo_plus_1|" (at the threshold)
// data-group data-selected data-weight>, frontier labels <text data-testid="level-label-<f>">,
// LMO lines <g data-testid="lmo-level-<i>">.
import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent, type RefObject } from "react";
import type { OrbitalRowOut } from "../apiTypes";
import { exact, fmt, niceTicks } from "../format";
import { DEGENERACY, energyWindow, frontierBlocks, frontierLabel, groupLevels, panRange, sanitizeDegeneracy, sideBySide, spectralWeights, spreadLabels, zoomRange, type LevelGroup, type Range, type Window } from "./levels";
import "./orbitals.css";

export interface LevelDiagramProps {
  canonical: OrbitalRowOut[];
  /** The LMO rows when an LMO set is shown (null: canonical only). */
  lmos: OrbitalRowOut[] | null;
  /** The selected orbital of the shown set. */
  selected: number | null;
  window: Window;
  onPickCanonical(index: number): void;
  onPickLmo(index: number): void;
  /** Height (px) of the diagram with its tool row; 380 by default. */
  height?: number;
  /** The window select follows fit / frontier. */
  onWindow?(w: Window): void;
  /** Degeneracy threshold (eV); without it the diagram keeps its own (default 0.05). */
  degeneracyEv?: number;
  onDegeneracy?(ev: number): void;
}

const DEFAULT_H = 380;
const TOOLS_H = 34;
const TOP = 18;
const BOTTOM = 16;
const AXIS_X = 50;
const LEVEL_W = 46;
const LEVEL_GAP = 10;
const LABEL_W = 132;
/** Below this width the frontier labels drop their energies (in the level's tooltip). */
const COMPACT_W = 380;
const LABEL_W_COMPACT = 74;
const ZOOM_STEP = 1.25;

interface Geometry {
  W: number;
  H: number;
  canX0: number;
  canX1: number;
  labelX: number;
  lmoX0: number;
  lmoX1: number;
}

function geometry(W: number, H: number, lmo: boolean): Geometry {
  const lmoW = lmo ? Math.min(110, Math.max(60, W * 0.2)) : 0;
  const lmoX1 = W - 8;
  const lmoX0 = lmoX1 - lmoW;
  const labelX = (lmo ? lmoX0 - 16 : W - 6) - (W < COMPACT_W ? LABEL_W_COMPACT : LABEL_W);
  return { W, H, canX0: AXIS_X + 14, canX1: Math.max(AXIS_X + 60, labelX - 14), labelX, lmoX0, lmoX1 };
}

interface Scale {
  lo: number;
  hi: number;
  y(e: number): number;
  e(y: number): number;
  plotH: number;
}

function scaleOf(r: Range, H: number): Scale {
  const [lo, hi] = r;
  const plotH = H - TOP - BOTTOM;
  return { lo, hi, plotH, y: (e) => TOP + ((hi - e) / (hi - lo)) * plotH, e: (y) => hi - ((y - TOP) / plotH) * (hi - lo) };
}

/** Tick text with as many decimals as the step needs (2.5-type steps one more). */
export function tickLabel(v: number, step: number): string {
  const e = Math.floor(Math.log10(step) + 1e-9);
  const mantissa = step / 10 ** e;
  const d = Math.max(0, Math.min(4, -e + (Math.abs(mantissa - 2.5) < 1e-6 ? 1 : 0)));
  const t = v.toFixed(d);
  return (Number(t) === 0 ? (0).toFixed(d) : t).replace("-", "−");
}

function Axis({ s, H }: { s: Scale; H: number }) {
  const ticks = niceTicks(s.lo, s.hi, Math.max(3, Math.round(s.plotH / 46)));
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
  return (
    <g className="wb-axis">
      <line x1={AXIS_X} x2={AXIS_X} y1={TOP} y2={H - BOTTOM} />
      {ticks.map((t) => (
        <g key={t}>
          <line x1={AXIS_X - 5} x2={AXIS_X} y1={s.y(t)} y2={s.y(t)} />
          <line className="wb-grid" x1={AXIS_X} x2={AXIS_X + 6} y1={s.y(t)} y2={s.y(t)} />
          <text x={AXIS_X - 8} y={s.y(t) + 4} textAnchor="end">
            {tickLabel(t, step)}
          </text>
        </g>
      ))}
      <text x={6} y={12} className="wb-axis-title">
        E (eV)
      </text>
    </g>
  );
}

/** Half-arrows of the occupation: one up for a SOMO, up and down for a pair. */
function OccArrows({ x, y, occ, h }: { x: number; y: number; occ: number; h: number }) {
  if (occ <= 0.5 || h <= 0) return null;
  const up = (cx: number) => `M${cx},${y + h} V${y - h} L${cx - h * 0.45},${y - h * 0.45}`;
  const down = (cx: number) => `M${cx},${y - h} V${y + h} L${cx + h * 0.45},${y + h * 0.45}`;
  const d = occ > 1.5 ? `${up(x - 3.5)} ${down(x + 3.5)}` : up(x);
  return <path className="wb-occ-arrow" d={d} />;
}

interface LevelsProps {
  groups: LevelGroup[];
  frontier: Map<number, string>;
  s: Scale;
  g: Geometry;
  selected: number | null;
  weights: Map<number, number> | null;
  onPick(i: number): void;
}

/** Half the free height (px) around group `i` (for the arrows), up to 7; 0 (no arrows) when
 *  the neighbours are closer than 8 px: zoom in to see them. */
function roomAt(groups: LevelGroup[], i: number, s: Scale): number {
  const yy = s.y(groups[i].energy);
  const gaps = [i > 0 ? s.y(groups[i - 1].energy) - yy : 99, i + 1 < groups.length ? yy - s.y(groups[i + 1].energy) : 99];
  const room = Math.min(7, Math.min(...gaps) / 2 - 1);
  return room < 3 ? 0 : room;
}

function CanonicalLevels(p: LevelsProps) {
  const pad = (6 / p.s.plotH) * (p.s.hi - p.s.lo);
  const visible = p.groups.map((grp, i) => ({ grp, i })).filter(({ grp }) => grp.energy >= p.s.lo - pad && grp.energy <= p.s.hi + pad);
  return (
    <g>
      {visible.map(({ grp, i }) => {
        const { xs, w } = sideBySide(grp.rows.length, p.g.canX0, p.g.canX1, LEVEL_W, LEVEL_GAP);
        const yy = p.s.y(grp.energy);
        const h = roomAt(p.groups, i, p.s);
        return grp.rows.map((r, k) => {
          const wt = p.weights?.get(r.index) ?? 0;
          const sel = p.weights === null && r.index === p.selected;
          const cls = `wb-level ${r.occupation > 0.5 ? "occ" : "virt"}${sel ? " selected" : ""}`;
          return (
            <g key={r.index} className={cls} data-testid={`level-${r.index}`} data-frontier={p.frontier.get(r.index) ?? ""} data-group={grp.level} data-selected={sel ? "true" : "false"} data-weight={p.weights ? exact(wt) : undefined} onClick={() => p.onPick(r.index)} role="button">
              <title>{`MO ${r.index}${r.label ? ` (${r.label})` : ""}: ${fmt(r.energy_ev, 3)} eV, occupation ${fmt(r.occupation, 2)}${grp.rows.length > 1 ? `, ${grp.rows.length}-fold degenerate` : ""}`}</title>
              <rect className="wb-hit" x={xs[k] - 3} y={yy - 8} width={w + 6} height={16} />
              {sel && <rect className="wb-level-halo" x={xs[k] - 3} y={yy - 5} width={w + 6} height={10} rx={3} />}
              <line x1={xs[k]} x2={xs[k] + w} y1={yy} y2={yy} />
              <OccArrows x={xs[k] + w / 2} y={yy} occ={r.occupation} h={h} />
              {wt > 0 && <rect className="wb-weight" x={xs[k]} y={yy + 2.5} width={Math.max(2, w * Math.min(1, wt / 0.5))} height={4} />}
            </g>
          );
        });
      })}
    </g>
  );
}

const LABEL_ORDER = ["lumo_plus_1", "lumo", "homo", "homo_minus_1"];

/** The frontier block labels at the right of the levels, spread apart, with leaders. */
function FrontierLabels({ groups, frontier, s, g }: Pick<LevelsProps, "groups" | "frontier" | "s" | "g">) {
  // a block (frontierBlocks: within the threshold of its anchor) may straddle two drawn groups
  // at a large threshold: the label gives the block's own size and mean energy
  const items = LABEL_ORDER.flatMap((f) => {
    const members = groups.flatMap((x) => x.rows).filter((r) => frontier.get(r.index) === f);
    if (!members.length) return [];
    const energy = members.reduce((t, r) => t + r.energy_ev, 0) / members.length;
    if (energy < s.lo || energy > s.hi) return [];
    const grp = groups.filter((x) => x.rows.some((r) => frontier.get(r.index) === f)).sort((a, b) => b.rows.length - a.rows.length)[0];
    const right = sideBySide(grp.rows.length, g.canX0, g.canX1, LEVEL_W, LEVEL_GAP);
    return [{ f, n: members.length, energy, y: s.y(grp.energy), x: right.xs[right.xs.length - 1] + right.w }];
  });
  const ys = spreadLabels(items.map((it) => it.y), 15, TOP + 4, g.H - BOTTOM - 2);
  return (
    <g>
      {items.map((it, k) => {
        const main = it.f === "homo" || it.f === "lumo";
        const n = it.n;
        return (
          <g key={it.f} className={`wb-frontier-label${main ? " main" : ""}`}>
            <polyline className="wb-leader" points={`${it.x + 4},${it.y} ${g.labelX - 10},${it.y} ${g.labelX - 4},${ys[k]}`} />
            <text className="wb-frontier" data-testid={`level-label-${it.f}`} x={g.labelX} y={ys[k] + 4}>
              {frontierLabel(it.f)}
              {n > 1 ? (g.W < COMPACT_W ? ` ×${n}` : ` (×${n})`) : ""}
              {g.W >= COMPACT_W && (
                <tspan className="wb-frontier-e" dx={5}>
                  {fmt(it.energy, 2).replace("-", "−")}
                </tspan>
              )}
              <title>{`${frontierLabel(it.f)}: ${n} MO${n > 1 ? "s" : ""}, mean ${fmt(it.energy, 3)} eV`}</title>
            </text>
          </g>
        );
      })}
    </g>
  );
}

function LmoLevels(p: { rows: OrbitalRowOut[]; s: Scale; g: Geometry; selected: number | null; onPick(i: number): void }) {
  const { lmoX0, lmoX1 } = p.g;
  return (
    <g>
      <text className="wb-axis-title" x={(lmoX0 + lmoX1) / 2} y={12} textAnchor="middle">
        LMO E_r
      </text>
      {p.rows
        .filter((r) => r.energy_ev >= p.s.lo && r.energy_ev <= p.s.hi)
        .map((r) => {
          const yy = p.s.y(r.energy_ev);
          const sel = r.index === p.selected;
          return (
            <g key={r.index} className={`wb-level lmo ${r.occupation > 0.5 ? "occ" : "virt"}${sel ? " selected" : ""}`} data-testid={`lmo-level-${r.index}`} data-selected={sel ? "true" : "false"} onClick={() => p.onPick(r.index)} role="button">
              <title>{`LMO ${r.index}: E_r ${fmt(r.energy_ev, 3)} eV, occupation ${fmt(r.occupation, 2)}`}</title>
              <rect className="wb-hit" x={lmoX0 - 2} y={yy - 5} width={lmoX1 - lmoX0 + 4} height={10} />
              {sel && <rect className="wb-level-halo" x={lmoX0 - 3} y={yy - 5} width={lmoX1 - lmoX0 + 6} height={10} rx={3} />}
              <line x1={lmoX0} x2={lmoX1} y1={yy} y2={yy} />
            </g>
          );
        })}
    </g>
  );
}

/** Counts of the canonical levels above and below the visible range, at the plot's edges. */
function OutOfView({ rows, s, g }: { rows: OrbitalRowOut[]; s: Scale; g: Geometry }) {
  const above = rows.filter((r) => r.energy_ev > s.hi).length;
  const below = rows.filter((r) => r.energy_ev < s.lo).length;
  return (
    <g className="wb-outside">
      {above > 0 && (
        <text x={g.canX0} y={TOP - 5} data-testid="level-more-above">
          ▲ {above} above
        </text>
      )}
      {below > 0 && (
        <text x={g.canX0} y={g.H - 3} data-testid="level-more-below">
          ▼ {below} below
        </text>
      )}
    </g>
  );
}

function useWidth(ref: RefObject<HTMLElement | null>): number {
  const [w, setW] = useState(340);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => setW(Math.max(220, Math.floor(el.getBoundingClientRect().width)));
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

interface View {
  range: Range;
  limit: Range;
  mode: Window;
  zoom(factor: number, at?: number): void;
  pan(dE: number): void;
  set(r: Range): void;
  reset(): void;
  show(w: Window): void;
}

/** The visible energy range: the window's until the user zooms or pans; back to it when the
 *  molecule, the set or the window changes. */
function useView(rows: OrbitalRowOut[], all: OrbitalRowOut[], window: Window, onWindow?: (w: Window) => void): View {
  const [mode, setMode] = useState<Window>(window);
  const [range, setRange] = useState<Range | null>(null);
  useEffect(() => setMode(window), [window]);
  useEffect(() => setRange(null), [rows, mode]);
  const limit = useMemo(() => energyWindow(all, "all"), [all]);
  const base = useMemo(() => energyWindow(all, mode), [all, mode]);
  const cur = range ?? base;
  const show = (w: Window) => {
    setMode(w);
    setRange(null);
    onWindow?.(w);
  };
  return {
    range: cur,
    limit,
    mode,
    zoom: (factor, at) => setRange(zoomRange(cur, factor, at ?? (cur[0] + cur[1]) / 2, limit)),
    pan: (dE) => setRange(panRange(cur, dE, limit)),
    set: (r) => setRange(panRange(r, 0, limit)),
    reset: () => setRange(null),
    show,
  };
}

/** Wheel zoom (passive: false, so the page does not scroll or zoom), drag pan, click guard. */
function useGestures(svgRef: RefObject<SVGSVGElement | null>, view: View, s: Scale) {
  const latest = useRef({ view, s });
  latest.current = { view, s };
  const drag = useRef<{ y: number; range: Range; moved: boolean; id: number } | null>(null);
  const swallow = useRef(false);
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const { view: v, s: sc } = latest.current;
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      if (e.shiftKey) return v.pan((-dy / sc.plotH) * (sc.hi - sc.lo));
      const y = e.clientY - el.getBoundingClientRect().top;
      v.zoom(Math.exp(dy * 0.0015), sc.e(y));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [svgRef]);
  const onPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    drag.current = { y: e.clientY, range: view.range, moved: false, id: e.pointerId };
  };
  const onPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    if (!d) return;
    const dy = e.clientY - d.y;
    if (!d.moved && Math.abs(dy) < 4) return;
    if (!d.moved) svgRef.current?.setPointerCapture(d.id);
    d.moved = true;
    view.set([d.range[0] + (dy / s.plotH) * (d.range[1] - d.range[0]), d.range[1] + (dy / s.plotH) * (d.range[1] - d.range[0])]);
  };
  const onPointerUp = () => {
    if (drag.current?.moved) swallow.current = true;
    drag.current = null;
  };
  const onClickCapture = (e: ReactMouseEvent) => {
    if (!swallow.current) return;
    swallow.current = false;
    e.stopPropagation();
  };
  return { onPointerDown, onPointerMove, onPointerUp, onPointerCancel: onPointerUp, onClickCapture };
}

function onKey(e: KeyboardEvent<SVGSVGElement>, view: View): void {
  const span = view.range[1] - view.range[0];
  const act: Record<string, () => void> = {
    "+": () => view.zoom(1 / ZOOM_STEP),
    "=": () => view.zoom(1 / ZOOM_STEP),
    "-": () => view.zoom(ZOOM_STEP),
    _: () => view.zoom(ZOOM_STEP),
    "0": () => view.reset(),
    ArrowUp: () => view.pan(span * 0.1),
    ArrowDown: () => view.pan(-span * 0.1),
    PageUp: () => view.pan(span * 0.5),
    PageDown: () => view.pan(-span * 0.5),
  };
  const f = act[e.key];
  if (!f) return;
  e.preventDefault();
  f();
}

/** The threshold input: a draft while typing, committed when valid (bounds of levels.ts). */
function ThresholdInput({ value, onChange }: { value: number; onChange(v: number): void }) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = (text: string) => {
    setDraft(text);
    const v = Number(text);
    if (text.trim() !== "" && Number.isFinite(v) && v >= DEGENERACY.min && v <= DEGENERACY.max) onChange(sanitizeDegeneracy(v));
  };
  return (
    <label className="wb-inline wb-degeneracy" title="Canonical MOs within this energy of a level's first MO form one degenerate level (drawn side by side; the HOMO / LUMO blocks follow). The server uses 0.05 eV.">
      degenerate ≤
      <input type="number" data-testid="degeneracy-threshold" min={DEGENERACY.min} max={DEGENERACY.max} step={DEGENERACY.step} value={draft ?? String(value)} onChange={(e) => commit(e.target.value)} onBlur={() => setDraft(null)} aria-label="degeneracy threshold (eV)" />
      eV
    </label>
  );
}

function Tools({ view, threshold, onThreshold }: { view: View; threshold: number; onThreshold(v: number): void }) {
  const btn = (id: string, label: string, title: string, f: () => void, active = false) => (
    <button type="button" className={`btn btn-small${active ? " active" : ""}`} data-testid={id} title={title} aria-label={title} onClick={f}>
      {label}
    </button>
  );
  return (
    <div className="wb-level-tools">
      {btn("level-zoom-in", "+", "Zoom in (wheel up, +)", () => view.zoom(1 / ZOOM_STEP))}
      {btn("level-zoom-out", "−", "Zoom out (wheel down, −)", () => view.zoom(ZOOM_STEP))}
      {btn("level-fit", "all", "Fit all levels", () => view.show("all"))}
      {btn("level-frontier", "frontier", "Frontier window: HOMO − 12 eV to LUMO + 8 eV", () => view.show("frontier"))}
      <ThresholdInput value={threshold} onChange={onThreshold} />
    </div>
  );
}

function useThreshold(p: LevelDiagramProps): [number, (v: number) => void] {
  const [own, setOwn] = useState<number>(DEGENERACY.dflt);
  if (p.degeneracyEv !== undefined && p.onDegeneracy) return [sanitizeDegeneracy(p.degeneracyEv), p.onDegeneracy];
  return [p.degeneracyEv !== undefined ? sanitizeDegeneracy(p.degeneracyEv) : own, (v) => (p.onDegeneracy ? p.onDegeneracy(v) : setOwn(v))];
}

export function LevelDiagram(p: LevelDiagramProps) {
  const figRef = useRef<HTMLElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const W = useWidth(figRef);
  const H = Math.max(220, (p.height ?? DEFAULT_H) - TOOLS_H);
  const [threshold, setThreshold] = useThreshold(p);
  const all = useMemo(() => [...p.canonical, ...(p.lmos ?? [])], [p.canonical, p.lmos]);
  const view = useView(p.canonical, all, p.window, p.onWindow);
  const s = scaleOf(view.range, H);
  const g = geometry(W, H, !!p.lmos);
  const groups = useMemo(() => groupLevels(p.canonical, threshold), [p.canonical, threshold]);
  const frontier = useMemo(() => frontierBlocks(p.canonical, threshold), [p.canonical, threshold]);
  const lmo = p.lmos?.find((r) => r.index === p.selected);
  const weights = p.lmos ? spectralWeights(lmo) : null;
  const gestures = useGestures(svgRef, view, s);
  const clip = useId().replace(/:/g, "");
  const span = view.range[1] - view.range[0];
  return (
    <figure ref={figRef} className="wb-levels wb-levels-zoom" data-testid="level-diagram" data-mode={p.lmos ? "lmo" : "canonical"} data-lo={view.range[0].toFixed(3)} data-hi={view.range[1].toFixed(3)} data-span={span.toFixed(3)} data-zoom={((view.limit[1] - view.limit[0]) / span).toFixed(2)} data-groups={groups.length} data-threshold={threshold} data-window={view.mode}>
      <Tools view={view} threshold={threshold} onThreshold={setThreshold} />
      <svg ref={svgRef} data-testid="level-svg" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="orbital energy levels: wheel to zoom, drag to pan, double-click to reset" tabIndex={0} onKeyDown={(e) => onKey(e, view)} onDoubleClick={view.reset} {...gestures}>
        <defs>
          <clipPath id={clip}>
            <rect x={0} y={TOP - 7} width={W} height={H - TOP - BOTTOM + 14} />
          </clipPath>
        </defs>
        <rect className="wb-level-bg" x={0} y={0} width={W} height={H} />
        <Axis s={s} H={H} />
        <g clipPath={`url(#${clip})`}>
          <CanonicalLevels groups={groups} frontier={frontier} s={s} g={g} selected={p.selected} weights={weights} onPick={p.onPickCanonical} />
          <FrontierLabels groups={groups} frontier={frontier} s={s} g={g} />
          {p.lmos && <LmoLevels rows={p.lmos} s={s} g={g} selected={p.selected} onPick={p.onPickLmo} />}
        </g>
        <OutOfView rows={p.canonical} s={s} g={g} />
      </svg>
      <figcaption className="muted wb-small">
        {p.lmos ? "Bars: the selected LMO's spectral weights on the canonical MOs. " : ""}Click a level to show that MO · wheel: zoom · drag: pan · double-click: reset.
      </figcaption>
    </figure>
  );
}
