// StepScheme: the textbook scheme of a step on screen (stepScheme.ts): reactants, the reaction
// arrow with its caption, the product, lone pairs on the atoms the step touches and curved arrows
// routed clear of the drawing; the R marks (ROverlay) on top, shown or hidden. Fits its box: the
// scale follows the width up to `maxBondPx` and the height follows the content.
//
// Hover: atoms and bonds (R's forming pairs included) report the atoms under the pointer; the
// same atoms are haloed on both sides (atoms keep their identity across the step).
//
// Props: sides, step, prefix ("predict-step"), style, mono (labels in the bond colour), showR,
// hover, onHover, above / below (reaction arrow text), productSmiles, maxBondPx, maxHeight.
// DOM: <div data-testid="<prefix>-canvas" data-variant data-bond-px> with the svg; the product
// <g data-testid="<prefix>-product-drawing" data-smiles>; the ROverlay ids (<prefix>-canvas-r-...,
// <prefix>-canvas-arrow-<k>).
import { useMemo } from "react";
import type { DrawingStyle } from "../workbench/sketch/style";
import type { Vec } from "./arrowPath";
import { arrowFromControls } from "./arrowPath";
import { arrowContext, type PoolView } from "./poolView";
import { ROverlay } from "./ROverlay";
import { PRODUCT_ID, drawScheme, schemeGapFor, schemeGeometry, type SchemeDrawing, type SchemeInput } from "./stepScheme";
import { arrowStyleFor } from "./stepExport";
import { hoverOf, useBoxWidth, type HoverSet, type ShownStep, type StepSides } from "./stepHooks";
import { dotRadius, fitScheme, type FitOptions, DEFAULT_FIT } from "./stepScene";
import { Halos, HoverTargets, LonePairs, ReactionArrow, SceneLayer, sceneVars, type HoverTarget } from "./StepSvg";

export interface StepSchemeProps {
  sides: StepSides;
  step: ShownStep;
  prefix: string;
  style: DrawingStyle;
  mono?: boolean;
  showR: boolean;
  hover: HoverSet;
  onHover: (h: HoverSet | null) => void;
  above?: string;
  below?: string;
  productSmiles: string;
  fit?: Partial<FitOptions>;
  variant: string;
  /** Touched carbons drawn with their symbol (false: vertices, the textbook look). */
  labelCarbons?: boolean;
}

function hoverTargets(d: SchemeDrawing, step: ShownStep): HoverTarget[] {
  const out: HoverTarget[] = [];
  const seen = new Set<string>();
  const bond = (sceneBonds: SchemeDrawing["react"]["bonds"], off: number) => {
    for (const b of sceneBonds) {
      const [i, j] = [b.bond.a - off, b.bond.b - off];
      const key = `${off ? "p" : "r"}b:${Math.min(i, j)}-${Math.max(i, j)}`;
      if (!seen.has(key)) out.push({ key, atoms: [i, j], bond: [i, j], p: b.axis.p, q: b.axis.q }), seen.add(key);
    }
  };
  bond(d.react.bonds, 0);
  for (const [i, j] of step.matrix.bonds) {
    const [p, q] = [d.xyR.get(i), d.xyR.get(j)];
    const key = `rb:${Math.min(i, j)}-${Math.max(i, j)}`;
    if (p && q && !seen.has(key)) out.push({ key, atoms: [i, j], bond: [i, j], p, q }), seen.add(key);
  }
  if (d.prod) bond(d.prod.bonds, PRODUCT_ID);
  for (const a of d.react.atoms) out.push({ key: `ra:${a.atom.id}`, atoms: [a.atom.id], p: a.at });
  for (const a of d.prod?.atoms ?? []) out.push({ key: `pa:${a.atom.id}`, atoms: [a.atom.id - PRODUCT_ID], p: a.at });
  return out;
}

function haloGeometry(d: SchemeDrawing, hover: HoverSet): { atoms: Vec[]; bonds: [Vec, Vec][] } {
  const atoms: Vec[] = [];
  const bonds: [Vec, Vec][] = [];
  const P = (i: number) => d.geomP?.get(i + PRODUCT_ID)?.at;
  for (const i of hover.atoms) {
    const r = d.xyR.get(i);
    if (r) atoms.push(r);
    const p = P(i);
    if (p) atoms.push(p);
  }
  for (const k of hover.bonds) {
    const [i, j] = k.split("-").map(Number);
    const [a, b] = [d.xyR.get(i), d.xyR.get(j)];
    if (a && b) bonds.push([a, b]);
    const [c, e] = [P(i), P(j)];
    if (c && e) bonds.push([c, e]);
  }
  return { atoms, bonds };
}

export function StepScheme(props: StepSchemeProps) {
  const { sides, step, prefix, style } = props;
  const [ref, width] = useBoxWidth();
  const src = sides.source;
  const input: SchemeInput | null = useMemo(() => (src ? { source: src, product: sides.product, aligned: sides.aligned, matrix: step.matrix, arrows: step.arrows } : null), [src, sides.product, sides.aligned, step.matrix, step.arrows]);
  const textBonds = 13 / (props.fit?.maxBondPx ?? DEFAULT_FIT.maxBondPx);
  const geo = useMemo(() => (input ? schemeGeometry(input, schemeGapFor([props.above, props.below], textBonds)) : null), [input, props.above, props.below, textBonds]);
  const fitted = useMemo(() => (geo && width > 0 ? fitScheme(geo.box, width, { ...DEFAULT_FIT, ...props.fit }) : null), [geo, width, props.fit]);
  const d = useMemo(() => (input && geo && fitted ? drawScheme(input, geo, fitted.view, style, false, props.labelCarbons ?? true) : null), [input, geo, fitted, style, props.labelCarbons]);
  const st = useMemo(() => (d ? arrowStyleFor(d.M) : null), [d]);
  const paths = useMemo(() => (d && st ? d.arrows.map((a) => (a ? arrowFromControls(a.tail, a.c1, a.c2, a.head, st) : null)) : []), [d, st]);
  const s = sides.structure;
  const pv: PoolView | null = d ? { view: d.view, xy: d.xyR, bondPx: d.M.bond, centroid: centroidOf(d.xyR) } : null;
  const ctx = useMemo(() => (s && pv ? arrowContext(s, pv) : null), [s, pv?.xy, pv?.bondPx]);
  const targets = useMemo(() => (d ? hoverTargets(d, step) : []), [d, step]);
  const halo = d ? haloGeometry(d, props.hover) : { atoms: [], bonds: [] };
  const height = fitted?.height ?? 180;
  return (
    <div ref={ref} className={`sketch ss-canvas${props.mono ? " sk-mono" : ""}`} data-testid={`${prefix}-canvas`} data-variant={props.variant} data-bond-px={d ? d.M.bond.toFixed(1) : ""} style={{ height }}>
      {d && pv && ctx && s && st && (
        <svg className="ss-svg" width={width} height={height} style={{ ...sceneVars(d.M), ["--ss-arrow-width" as string]: `${st.stroke.toFixed(2)}px` }}>
          <Halos atoms={halo.atoms} bonds={halo.bonds} r={d.M.font * 0.75} />
          <SceneLayer scene={d.react} />
          <LonePairs marks={[...d.pairsR.values()]} r={dotRadius(d.M)} />
          {d.prod && (
            <g data-testid={`${prefix}-product-drawing`} data-smiles={props.productSmiles}>
              <SceneLayer scene={d.prod} />
              <LonePairs marks={[...d.pairsP.values()]} r={dotRadius(d.M)} />
            </g>
          )}
          {d.rxn && <ReactionArrow from={d.rxn.from} to={d.rxn.to} above={props.above} below={props.below} size={Math.max(11, Math.min(15, d.M.font * 0.42))} />}
          <ROverlay structure={s} pv={pv} ctx={ctx} prefix={`${prefix}-canvas`} matrix={step.matrix} arrows={step.arrows} paths={paths} marksOpacity={props.showR ? 1 : 0} chargesAfter={step.chargesAfter} />
          <HoverTargets targets={targets} r={Math.max(8, d.M.font * 0.55)} onHover={(t) => props.onHover(t ? hoverOf(t.atoms, t.bond ? [t.bond] : []) : null)} />
        </svg>
      )}
    </div>
  );
}

export function centroidOf(xy: ReadonlyMap<number, Vec>): Vec {
  const ps = [...xy.values()];
  return ps.length ? [ps.reduce((t, p) => t + p[0], 0) / ps.length, ps.reduce((t, p) => t + p[1], 0) / ps.length] : [0, 0];
}
