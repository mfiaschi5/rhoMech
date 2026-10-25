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
//
// DOM: <svg data-testid="sketch-canvas"> with
//   <g data-testid="sketch-bond-<id>" data-order="1|2|3" data-stereo data-selected data-hovered
//      data-broken="<n>" (only on a bond drawn broken behind n crossings)>
//   <g data-testid="sketch-atom-<id>" data-element data-charge data-h data-selected
//      data-highlighted data-invalid data-hovered>
//   <g data-testid="sketch-frame" data-atoms="<n>"> with sketch-frame-rotate (circle) and
//      sketch-frame-corner-<0..3> (top-left, top-right, bottom-right, bottom-left)
import type { RefObject } from "react";
import type { SketchModel } from "./model";
import type { Rect, Vec, View } from "./geometry";
import type { Hit, Selection } from "./tools";
import type { SketchPointer } from "./useSketchPointer";
import type { SelectionFrame } from "./frame";
import { buildScene, FONT_PX, hLabelWidth, labelHalfWidth, type AtomScene, type BondScene } from "./drawing";
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

function BondView({ s, selected, hovered }: { s: BondScene; selected: boolean; hovered: boolean }) {
  const { bond, shape, axis } = s;
  return (
    <g
      data-testid={`sketch-bond-${bond.id}`}
      data-order={bond.order}
      data-stereo={bond.stereo}
      data-broken={s.broken > 0 ? s.broken : undefined}
      data-selected={selected}
      data-hovered={hovered}
      className="sk-bond"
    >
      {(selected || hovered) && (
        <line className={selected ? "sk-halo sk-selected" : "sk-halo"} x1={axis.p[0]} y1={axis.p[1]} x2={axis.q[0]} y2={axis.q[1]} />
      )}
      {shape.lines.map((l, i) => (
        <line key={i} className="sk-stroke" x1={l.p[0]} y1={l.p[1]} x2={l.q[0]} y2={l.q[1]} />
      ))}
      {shape.wedges?.map((w, i) => <polygon key={`w${i}`} className="sk-wedge" points={pts(w)} />)}
      <polygon className="sk-hit" points={pts(hitBand(axis.p, axis.q, 5))} />
    </g>
  );
}

const subDigits = (n: number) => (n > 1 ? String(n) : "");

function AtomLabelView({ s }: { s: AtomScene }) {
  const { label, at } = s;
  if (!label) return null;
  const half = labelHalfWidth(label.el);
  const hText = label.hCount > 0 ? "H" : "";
  const hWidth = hLabelWidth(label.hCount);
  const right = label.hSide === "right";
  const chargeX = at[0] + half + (right ? hWidth : 0) + 1;
  return (
    <g className={`sk-label ${elementClass(label.el)}`}>
      <text x={at[0]} y={at[1]} textAnchor="middle" dominantBaseline="central">
        {label.el}
      </text>
      {hText && (
        <text x={right ? at[0] + half : at[0] - half} y={at[1]} textAnchor={right ? "start" : "end"} dominantBaseline="central">
          H<tspan className="sk-sub" dy={4}>{subDigits(label.hCount)}</tspan>
        </text>
      )}
      {label.charge && (
        <text className="sk-charge" x={chargeX} y={at[1] - FONT_PX * 0.55} textAnchor="start" dominantBaseline="central">
          {label.charge}
        </text>
      )}
    </g>
  );
}

function AtomView(props: { s: AtomScene; selected: boolean; hovered: boolean; highlighted: boolean; invalid: boolean }) {
  const { s, selected, hovered, highlighted, invalid } = props;
  const { atom, at, label } = s;
  const cls = ["sk-atom", invalid ? "sk-invalid" : "", highlighted ? "sk-highlighted" : ""].join(" ").trim();
  return (
    <g
      data-testid={`sketch-atom-${atom.id}`}
      data-element={atom.el}
      data-charge={atom.charge}
      data-h={label ? label.hCount : undefined}
      data-selected={selected}
      data-highlighted={highlighted}
      data-invalid={invalid}
      data-hovered={hovered}
      className={cls}
    >
      {highlighted && <circle className="sk-highlight" cx={at[0]} cy={at[1]} r={12} />}
      {invalid && <circle className="sk-invalid-ring" cx={at[0]} cy={at[1]} r={10} />}
      {(selected || hovered) && <circle className={selected ? "sk-dot sk-selected" : "sk-dot"} cx={at[0]} cy={at[1]} r={8} />}
      <AtomLabelView s={s} />
      <circle className="sk-hit" cx={at[0]} cy={at[1]} r={9} />
    </g>
  );
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
  const scene = buildScene(model, view, hCounts);
  const isHover = (kind: "atom" | "bond", id: number) => hover?.kind === kind && hover.id === id;
  return (
    <svg
      ref={svgRef}
      data-testid="sketch-canvas"
      className={`sketch-svg${readOnly ? " read-only" : ""}`}
      width="100%"
      height={height}
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
