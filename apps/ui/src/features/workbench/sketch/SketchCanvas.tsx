// SketchCanvas: the SVG drawing of a sketch (drawing.buildScene turned into elements). It draws
// only; SketchPointer handlers come in as props and the hit testing is geometric, so the atom
// and bond groups exist for styling and test hooks (each keeps a transparent hit shape so a
// click at its centre reaches the canvas through it).
//
// Props:
//   model, view          the sketch and the screen transform
//   hCounts              implicit H by atom id (RDKit's), null for the default-valence fallback
//   hover, selection     the hovered object and the selection (tools.Hit / tools.Selection)
//   highlight, invalid   atom ids drawn with the highlight halo / in the error colour
//   rubber               select rubber band (screen px) or null
//   frame                the selection frame (frame.ts) or null: drawn with its handles
//   height               px; the width follows the container
//   pointer              the useSketchPointer handlers
//   svgRef
//   cip                  CIP labels to draw (the R/S toggle; null: none)
//   display              label font size and label style (the page's sketch preferences) and
//                        the drawing style's metrics in screen px (style.ts; Screen without)
//   styleId              the drawing style's id, as data-style
//
// DOM: <svg data-testid="sketch-canvas" data-style data-px-per-pt data-bond-px (a standard
//      bond) data-line-px data-bold-px data-font-px> with
//   <g data-testid="sketch-bond-<id>" data-order="1|2|3" data-stereo data-selected data-hovered
//      data-broken="<n>" (only on a bond drawn broken behind n crossings) data-cip="E|Z" (only
//      while R/S is on and RDKit labels it)>, with <text data-testid="sketch-cip-bond-<id>">
//   <g data-testid="sketch-atom-<id>" data-element data-charge data-h data-radical="1|2" (only
//      with unpaired electrons; drawn as dots circle.sk-radical) data-cip="R|S|r|s|?" (only
//      while R/S is on) data-selected data-highlighted data-invalid data-hovered>, with
//      <text data-testid="sketch-cip-atom-<id>"> and the label <g class="sk-label"
//      data-hside data-hw data-charge-slot data-label-scale (subscript / charge shrunk)
//      data-displaced (moved off its atom, with line.sk-leader)> of one <text data-glyph=
//      "symbol|h|count|charge"> per glyph run at its explicit baseline (labelGlyphs.ts, placed
//      collision-aware by labelLayout.ts; data-label-issues on the svg counts what still
//      overlaps)
//   <g data-testid="sketch-frame" data-atoms="<n>"> with sketch-frame-rotate (circle) and
//      sketch-frame-corner-<0..3> (top-left, top-right, bottom-right, bottom-left)
import type { CSSProperties, RefObject } from "react";
import type { SketchModel } from "./model";
import type { Rect, Vec, View } from "./geometry";
import type { Hit, Selection } from "./tools";
import type { SketchPointer } from "./useSketchPointer";
import type { SelectionFrame } from "./frame";
import type { StereoLabels } from "./cip";
import type { DrawMetrics } from "./style";
import { buildScene, chargeAnchor, hLabelWidth, labelGlyphs, type AtomScene, type BondScene, type Glyph, type LabelStyle, type Mark } from "./drawing";
import { cssFontFamily } from "./style";
import { elementClass } from "./elements";

export interface SketchCanvasProps {
  model: SketchModel;
  view: View;
  hCounts: ReadonlyMap<number, number> | null;
  hover: Hit;
  selection: Selection;
  highlight: ReadonlySet<number>;
  invalid: ReadonlySet<number>;
  rubber: Rect | null;
  frame: SelectionFrame | null;
  height: number;
  pointer: SketchPointer;
  svgRef: RefObject<SVGSVGElement | null>;
  readOnly: boolean;
  cip?: StereoLabels | null;
  display?: { fontPx?: number; labelStyle?: LabelStyle; metrics?: DrawMetrics };
  styleId?: string;
}

/** A band of half-width w around p-q (a bond's hit target; a line has a zero-area bbox when
 *  vertical or horizontal, which test drivers treat as invisible). */
function hitBand(p: Vec, q: Vec, w: number): Vec[] {
  const dx = q[0] - p[0];
  const dy = q[1] - p[1];
  const l = Math.hypot(dx, dy) || 1;
  const nx = (-dy / l) * w;
  const ny = (dx / l) * w;
  return [
    [p[0] + nx, p[1] + ny],
    [q[0] + nx, q[1] + ny],
    [q[0] - nx, q[1] - ny],
    [p[0] - nx, p[1] - ny],
  ];
}

const pts = (vs: readonly Vec[]) => vs.map((v) => `${v[0].toFixed(2)},${v[1].toFixed(2)}`).join(" ");

function CipMark({ m, testId }: { m: Mark; testId: string }) {
  return (
    <text className="sk-cip" data-testid={testId} x={m.at[0]} y={m.at[1]} textAnchor="middle" dominantBaseline="central">
      {m.text}
    </text>
  );
}

const cipCode = (m: Mark | undefined) => (m ? m.text.replace(/[()]/g, "") : undefined);

function BondView({ s, selected, hovered }: { s: BondScene; selected: boolean; hovered: boolean }) {
  const { bond, shape, axis } = s;
  return (
    <g
      data-testid={`sketch-bond-${bond.id}`}
      data-order={bond.order}
      data-stereo={bond.stereo}
      data-broken={s.broken > 0 ? s.broken : undefined}
      data-cip={cipCode(s.cip)}
      data-selected={selected}
      data-hovered={hovered}
      className="sk-bond"
    >
      {(selected || hovered) && (
        <line className={selected ? "sk-halo sk-selected" : "sk-halo"} x1={axis.p[0]} y1={axis.p[1]} x2={axis.q[0]} y2={axis.q[1]} />
      )}
      {shape.lines.map((l, i) => (
        <line key={i} className={bond.stereo === "hash" && bond.order === 1 ? "sk-stroke sk-hash" : "sk-stroke"} x1={l.p[0]} y1={l.p[1]} x2={l.q[0]} y2={l.q[1]} />
      ))}
      {shape.wedges?.map((w, i) => <polygon key={`w${i}`} className="sk-wedge" points={pts(w)} />)}
      <polygon className="sk-hit" points={pts(hitBand(axis.p, axis.q, 5))} />
      {s.cip && <CipMark m={s.cip} testId={`sketch-cip-bond-${bond.id}`} />}
    </g>
  );
}

const fpx = (v: number) => `${Math.round(v * 1000) / 1000}px`;

/** One glyph run of a label at its explicit baseline (labelGlyphs: the export draws the same). */
function GlyphText({ g, at, M, cy }: { g: Glyph; at: Vec; M: DrawMetrics; cy: number }) {
  if (g.kind === "charge" && g.circled) {
    const r = g.w / 2;
    return (
      <g className="sk-charge-circled">
        <circle className="sk-charge-ring" cx={at[0] + g.x + r} cy={at[1] + cy} r={r * 0.95} />
        <text className="sk-charge" x={at[0] + g.x + r} y={at[1] + cy} textAnchor="middle" dominantBaseline="central" style={{ fontSize: fpx(g.size) }}>
          {g.text}
        </text>
      </g>
    );
  }
  const cls = g.kind === "charge" ? "sk-charge" : g.kind === "count" ? "sk-sub" : undefined;
  const size = g.kind === "charge" || g.kind === "count" ? { fontSize: fpx(g.size) } : undefined;
  // the symbol is centred on its box (the browser measures the UI font; the width is exact for
  // the metric fonts), the other runs start at their x
  const sym = g.kind === "symbol";
  return (
    <text className={cls} x={at[0] + g.x + (sym ? g.w / 2 : 0)} y={at[1] + g.y} textAnchor={sym ? "middle" : undefined} style={size} data-glyph={g.kind} data-w={M.metricText ? g.w.toFixed(2) : undefined}>
      {g.text}
    </text>
  );
}

function AtomLabelView({ s, M }: { s: AtomScene; M: DrawMetrics }) {
  const { label, at } = s;
  if (!label) return null;
  const glyphs = labelGlyphs(label, M);
  const cy = label.charge ? chargeAnchor(label, M)[1] : 0;
  return (
    <g
      className={`sk-label ${elementClass(label.el)}`}
      data-hside={label.hCount > 0 ? label.hSide : undefined}
      data-hw={label.hCount > 0 ? hLabelWidth(label.hCount, M, label.scale ?? 1).toFixed(1) : undefined}
      data-charge-slot={label.charge ? (label.chargeSlot ?? "ur") : undefined}
      data-label-scale={label.scale ? label.scale : undefined}
      data-displaced={label.offset ? "true" : undefined}
    >
      {s.leader && <line className="sk-leader" x1={s.leader.p[0]} y1={s.leader.p[1]} x2={s.leader.q[0]} y2={s.leader.q[1]} />}
      {glyphs.map((g, i) => (
        <GlyphText key={i} g={g} at={at} M={M} cy={cy} />
      ))}
    </g>
  );
}

function AtomView(props: { s: AtomScene; M: DrawMetrics; dotR: number; selected: boolean; hovered: boolean; highlighted: boolean; invalid: boolean }) {
  const { s, M, dotR, selected, hovered, highlighted, invalid } = props;
  const { atom, at, label } = s;
  const cls = ["sk-atom", invalid ? "sk-invalid" : "", highlighted ? "sk-highlighted" : ""].join(" ").trim();
  return (
    <g
      data-testid={`sketch-atom-${atom.id}`}
      data-element={atom.el}
      data-charge={atom.charge}
      data-h={label ? label.hCount : undefined}
      data-radical={atom.radical}
      data-cip={cipCode(s.cip)}
      data-selected={selected}
      data-highlighted={highlighted}
      data-invalid={invalid}
      data-hovered={hovered}
      className={cls}
    >
      {highlighted && <circle className="sk-highlight" cx={at[0]} cy={at[1]} r={12} />}
      {invalid && <circle className="sk-invalid-ring" cx={at[0]} cy={at[1]} r={10} />}
      {(selected || hovered) && <circle className={selected ? "sk-dot sk-selected" : "sk-dot"} cx={at[0]} cy={at[1]} r={8} />}
      <AtomLabelView s={s} M={M} />
      {s.dots.map((d, i) => (
        <circle key={`r${i}`} className="sk-radical" cx={d[0]} cy={d[1]} r={dotR} />
      ))}
      {s.cip && <CipMark m={s.cip} testId={`sketch-cip-atom-${atom.id}`} />}
      <circle className="sk-hit" cx={at[0]} cy={at[1]} r={9} />
    </g>
  );
}

/** The style's sizes as CSS variables of the canvas (sketch.css reads them). */
const px = (v: number) => `${Math.round(v * 1000) / 1000}px`;

function canvasVars(M: DrawMetrics): CSSProperties {
  return {
    "--sk-font": px(M.font),
    "--sk-sub-font": px(M.sub),
    "--sk-charge-font": px(M.charge),
    "--sk-font-family": cssFontFamily(M.fontFamily, true),
    "--sk-font-weight": String(M.fontWeight),
    "--sk-bond-width": px(M.line),
    "--sk-hash-width": px(M.hashWidth),
    "--sk-wedge-stroke": M.fixedScreen ? "0.6px" : "0px",
    "--sk-charge-weight": M.fixedScreen ? "600" : String(M.fontWeight),
  } as CSSProperties;
}

const CORNER_CURSORS = ["nwse-resize", "nesw-resize", "nwse-resize", "nesw-resize"];

function FrameView({ f }: { f: SelectionFrame }) {
  const { box, rotate } = f;
  const top: Vec = [(box.x0 + box.x1) / 2, box.y0];
  return (
    <g className="sk-frame" data-testid="sketch-frame" data-atoms={f.ids.length}>
      <rect className="sk-frame-box" x={box.x0} y={box.y0} width={box.x1 - box.x0} height={box.y1 - box.y0} />
      <line className="sk-frame-box" x1={top[0]} y1={top[1]} x2={rotate[0]} y2={rotate[1]} />
      <circle className="sk-frame-handle sk-frame-rotate" data-testid="sketch-frame-rotate" cx={rotate[0]} cy={rotate[1]} r={5.5}>
        <title>Drag to rotate about the centre (Shift: 15° steps)</title>
      </circle>
      {f.corners.map((c, i) => (
        <rect key={i} className="sk-frame-handle" data-testid={`sketch-frame-corner-${i}`} x={c[0] - 4.5} y={c[1] - 4.5} width={9} height={9} style={{ cursor: CORNER_CURSORS[i] }}>
          <title>Drag to scale (Shift: 5 % steps)</title>
        </rect>
      ))}
    </g>
  );
}

export function SketchCanvas(props: SketchCanvasProps) {
  const { model, view, hCounts, hover, selection, highlight, invalid, rubber, frame, height, pointer, svgRef, readOnly } = props;
  const scene = buildScene(model, view, hCounts, { ...props.display, cip: props.cip ?? null });
  const M = scene.metrics;
  const isHover = (kind: "atom" | "bond", id: number) => hover?.kind === kind && hover.id === id;
  return (
    <svg
      ref={svgRef}
      data-testid="sketch-canvas"
      className={`sketch-svg${readOnly ? " read-only" : ""}`}
      width="100%"
      height={height}
      style={canvasVars(M)}
      data-style={props.styleId ?? "screen"}
      data-px-per-pt={M.perPt.toFixed(4)}
      data-bond-px={M.bond.toFixed(3)}
      data-line-px={M.line.toFixed(3)}
      data-bold-px={M.bold.toFixed(3)}
      data-font-px={M.font.toFixed(3)}
      data-label-issues={scene.issues.length}
      tabIndex={0}
      onPointerDown={pointer.onPointerDown}
      onPointerMove={pointer.onPointerMove}
      onPointerUp={pointer.onPointerUp}
      onPointerLeave={pointer.onPointerLeave}
      onContextMenu={(e) => e.preventDefault()}
    >
      <g className="sk-bonds">
        {scene.bonds.map((s) => (
          <BondView key={s.bond.id} s={s} selected={selection.bonds.has(s.bond.id)} hovered={isHover("bond", s.bond.id)} />
        ))}
      </g>
      <g className="sk-atoms">
        {scene.atoms.map((s) => (
          <AtomView
            key={s.atom.id}
            s={s}
            M={M}
            dotR={scene.dotR}
            selected={selection.atoms.has(s.atom.id)}
            hovered={isHover("atom", s.atom.id)}
            highlighted={highlight.has(s.atom.id)}
            invalid={invalid.has(s.atom.id)}
          />
        ))}
      </g>
      {frame && !readOnly && <FrameView f={frame} />}
      {rubber && <rect className="sk-rubber" x={rubber.x0} y={rubber.y0} width={rubber.x1 - rubber.x0} height={rubber.y1 - rubber.y0} />}
    </svg>
  );
}
