// PoolCanvas: a pool state drawn by the workbench sketcher's renderer (SketchCanvas, read-only)
// with the reaction matrix, its arrows and the editor's gestures on an SVG layer on top.
//
// Props:
//   structure    PoolStructure            the reactant B (poolStructure(depiction))
//   testId       string                   the container's test id and the overlay's prefix
//   height       number                   px (default 300); the width follows the container
//   matrix       MatrixIO | null          R to overlay
//   arrows       ArrowReadingOut[]        the curved arrows to draw
//   arrowsFaded  boolean                  draw the arrows faded (a stale reading)
//   chargesAfter Map<atom, charge>        formal charges after R (the check's)
//   highlight    Set<atom>                haloed atoms (an issue under the pointer)
//   pending      atom | null              the bond tool's first atom
//   selectedKeys Set<entry key>           R entries drawn as selected
//   interaction  {tool, onAtom, onBond, onArrow, onArrowClick, onNotice} | null   the editor
//
// DOM: <div data-testid={testId} data-atom-xy='{"<atom>": [x, y]}' (px in the canvas)
// data-elements='{"<atom>": "C"}' (every pool atom, hidden H included) data-charges='{"<atom>": -1}'
// (non-zero) data-bonds="i-j:order,.." data-hidden-h="<atoms of undrawn H>"
// data-bond-px data-tool>, the SketchCanvas svg, and svg.pv-layer with per-atom
// circle[data-testid=<testId>-atom-<i>] and per-pair line[data-testid=<testId>-bond-<i>-<j>]
// hit targets, the ROverlay marks and (while an arrow is dragged) line.pv-drag.
import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { SketchCanvas } from "../workbench/sketch/SketchCanvas";
import { NO_SELECTION } from "../workbench/sketch/tools";
import type { SketchPointer } from "../workbench/sketch/useSketchPointer";
import "../workbench/sketch/sketch.css";
import type { Vec } from "./arrowPath";
import type { Tool } from "./editor";
import type { PoolStructure } from "./poolStructure";
import { arrowContext, hitPairs, hitTest, nearEnd, poolView, type PoolHit } from "./poolView";
import { ROverlay } from "./ROverlay";
import type { Endpoint } from "./rmatrix";
import type { ArrowReadingOut, MatrixIO } from "./v9types";

export interface CanvasInteraction {
  tool: Tool;
  onAtom(i: number): void;
  onBond(i: number, j: number): void;
  onArrow(from: Endpoint, to: Endpoint): void;
  onArrowClick(index: number): void;
}

export interface PoolCanvasProps {
  structure: PoolStructure;
  testId: string;
  height?: number;
  matrix?: MatrixIO | null;
  arrows?: readonly ArrowReadingOut[];
  arrowsFaded?: boolean;
  chargesAfter?: ReadonlyMap<number, number> | null;
  highlight?: ReadonlySet<number>;
  pending?: number | null;
  selectedKeys?: ReadonlySet<string>;
  interaction?: CanvasInteraction | null;
}

const NO_POINTER: SketchPointer = {
  onPointerDown: () => undefined,
  onPointerMove: () => undefined,
  onPointerUp: () => undefined,
  onPointerLeave: () => undefined,
  rubber: null,
  inside: { current: false },
  cancel: () => undefined,
};
const EMPTY = new Set<number>();

/** The container's width, followed with a ResizeObserver. */
function useWidth(): [(el: HTMLDivElement | null) => void, number] {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!el) return;
    setW(el.clientWidth);
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, w];
}

interface Drag {
  hit: NonNullable<PoolHit>;
  from: Vec;
  at: Vec;
}

const local = (e: ReactPointerEvent<Element>, svg: SVGSVGElement | null): Vec => {
  const r = svg?.getBoundingClientRect();
  return r ? [e.clientX - r.left, e.clientY - r.top] : [0, 0];
};

function endpointOf(hit: NonNullable<PoolHit>, near?: number): Endpoint {
  return hit.kind === "atom" ? { kind: "atom", i: hit.i } : { kind: "bond", i: hit.i, j: hit.j, near };
}

function usePoolGestures(interaction: CanvasInteraction | null | undefined, pv: ReturnType<typeof poolView>, pairs: [number, number][]) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const down = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!interaction || e.button !== 0) return;
    const p = local(e, svgRef.current);
    const hit = hitTest(pv, pairs, p);
    if (!hit) return;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    setDrag({ hit, from: p, at: p });
  };
  const move = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (drag) setDrag({ ...drag, at: local(e, svgRef.current) });
  };
  const up = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!drag || !interaction) return setDrag(null);
    const p = local(e, svgRef.current);
    const end = hitTest(pv, pairs, p);
    setDrag(null);
    if (interaction.tool === "arrow") {
      if (!end) return;
      const near = drag.hit.kind === "bond" ? nearEnd(pv, drag.hit, p) : undefined;
      return interaction.onArrow(endpointOf(drag.hit, near), endpointOf(end));
    }
    if (drag.hit.kind === "atom") interaction.onAtom(drag.hit.i);
    else interaction.onBond(drag.hit.i, drag.hit.j);
  };
  return { svgRef, drag, handlers: { onPointerDown: down, onPointerMove: move, onPointerUp: up, onPointerCancel: () => setDrag(null) } };
}

export function PoolCanvas(props: PoolCanvasProps) {
  const { structure: s, testId, height = 300, interaction } = props;
  const [ref, width] = useWidth();
  const pv = useMemo(() => poolView(s, Math.max(width, 1), height), [s, width, height]);
  const ctx = useMemo(() => arrowContext(s, pv), [s, pv]);
  const pairs = useMemo(() => hitPairs(s, props.matrix ?? null), [s, props.matrix]);
  const { svgRef, drag, handlers } = usePoolGestures(interaction, pv, pairs);
  const xyJson = useMemo(() => JSON.stringify(Object.fromEntries([...pv.xy].map(([i, p]) => [i, [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]]))), [pv]);
  const facts = useMemo(
    () => ({
      elements: JSON.stringify(Object.fromEntries(s.atoms.map((a) => [a.index, a.el]))),
      charges: JSON.stringify(Object.fromEntries(s.atoms.filter((a) => a.charge !== 0).map((a) => [a.index, a.charge]))),
      bonds: s.bonds.map((b) => `${Math.min(b.i, b.j)}-${Math.max(b.i, b.j)}:${b.order}`).join(","),
      hidden: s.atoms.filter((a) => a.sketchId === null).map((a) => a.index).join(","),
    }),
    [s],
  );
  const arrowDown = interaction?.tool === "eraser" ? (k: number, e: ReactPointerEvent<SVGPathElement>) => (e.stopPropagation(), interaction.onArrowClick(k)) : undefined;
  return (
    <div ref={ref} className="sketch pv-canvas" data-testid={testId} data-atom-xy={width ? xyJson : "{}"} data-elements={facts.elements} data-charges={facts.charges} data-bonds={facts.bonds} data-hidden-h={facts.hidden} data-bond-px={pv.bondPx.toFixed(2)} data-tool={interaction?.tool ?? ""} style={{ height }}>
      {width > 0 && (
        <>
          <SketchCanvas model={s.model} view={pv.view} hCounts={s.hCounts} hover={null} selection={NO_SELECTION} highlight={EMPTY} invalid={EMPTY} rubber={null} frame={null} height={height} pointer={NO_POINTER} svgRef={{ current: null }} readOnly />
          <svg ref={svgRef} className={`pv-layer${interaction ? ` pv-tool-${interaction.tool}` : ""}`} width={width} height={height} {...(interaction ? handlers : {})}>
            {/* the hit targets go under the marks so an arrow's hit stroke (eraser) is on top */}
            {interaction && <HitTargets pv={pv} pairs={pairs} prefix={testId} />}
            <ROverlay structure={s} pv={pv} ctx={ctx} prefix={testId} matrix={props.matrix ?? null} arrows={props.arrows ?? []} arrowsFaded={props.arrowsFaded} chargesAfter={props.chargesAfter} highlight={props.highlight} pending={props.pending} selectedKeys={props.selectedKeys} onArrowPointerDown={arrowDown} />
            {drag && interaction?.tool === "arrow" && <line className="pv-drag" x1={drag.from[0]} y1={drag.from[1]} x2={drag.at[0]} y2={drag.at[1]} />}
          </svg>
        </>
      )}
    </div>
  );
}

/** Invisible per-atom and per-pair targets (styling and test hooks; hit testing is geometric). */
function HitTargets({ pv, pairs, prefix }: { pv: ReturnType<typeof poolView>; pairs: [number, number][]; prefix: string }) {
  return (
    <g className="pv-hits">
      {pairs.map(([i, j]) => {
        const a = pv.xy.get(i);
        const b = pv.xy.get(j);
        if (!a || !b) return null;
        const m: Vec = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        return <circle key={`${i}-${j}`} className="pv-hit-bond" data-testid={`${prefix}-bond-${i}-${j}`} cx={m[0]} cy={m[1]} r={5} />;
      })}
      {[...pv.xy].map(([i, p]) => (
        <circle key={i} className="pv-hit-atom" data-testid={`${prefix}-atom-${i}`} cx={p[0]} cy={p[1]} r={10} />
      ))}
    </g>
  );
}
