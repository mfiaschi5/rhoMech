// StepAnimated (step viewport variant B): the step played as a morph from reactants to product
// (stepMorph.ts): atoms keep their identity (source pool atoms) and move from their reactant to
// their product position, breaking bonds fade out, forming bonds fade in, the curved arrows draw
// on first, the lone pair an arrow takes fades and the new one appears, the charges switch.
// Controls: play / pause (the space bar too, while the pointer or focus is not in a field),
// restart, a scrubber, speed 0.5x / 1x / 2x and loop. Beside the stage a card lists R's entries
// (with the bond orders and charges before and after) and the orbital reading; hovering a row
// haloes its atoms and bond on the stage.
//
// DOM: predict-step-animated > predict-step-canvas (the stage, data-t; the ROverlay ids, its
// marks fading as the step runs), predict-step-play[aria-pressed], predict-step-restart,
// predict-step-scrub, predict-step-speed-<0.5|1|2>, predict-step-loop, predict-step-card >
// predict-step-card-entry-<k>, predict-step-card-reading-<k>.
import { useEffect, useMemo, useRef, useState } from "react";
import type { DrawingStyle } from "../workbench/sketch/style";
import { buildScene } from "../workbench/sketch/drawing";
import { SCREEN_STYLE, drawMetrics, pxPerPt } from "../workbench/sketch/style";
import { arrowFromControls, type Vec } from "./arrowPath";
import { arrowContext, type PoolView } from "./poolView";
import { touchedAtoms } from "./poolStructure";
import { ROverlay } from "./ROverlay";
import { chargeDeltas } from "./rmatrix";
import { arrowStyleFor } from "./stepExport";
import { bounds, pairKey, relaxLayout, type Side, type XY } from "./stepGeometry";
import { BOND_LENGTH } from "./stepScene";
import { hoverOf, useBoxWidth, type HoverSet } from "./stepHooks";
import { PHASES, morphFrame, morphPlan, phase, type MorphPlan } from "./stepMorph";
import { atomGeometry, bondKeys, donorPreferences, dotRadius, fitScheme, lonePairLayout, routeReading, sideModel, DEFAULT_FIT } from "./stepScene";
import { spreadSide } from "./stepScheme";
import { captionParts } from "./stepText";
import { centroidOf } from "./StepScheme";
import { Halos, LonePairs, SceneLayer, sceneVars } from "./StepSvg";
import type { StepVariantProps } from "./StepTextbook";
import type { MatrixIO } from "./v9types";

/** The stage: the screen look (colour labels, UI font) scaled with the drawing, labels 0.4 bond. */
export const STAGE_STYLE: DrawingStyle = { ...SCREEN_STYLE, bondLengthPt: 30, fontSizePt: 12, lineWidthPt: 1.25, chargePlacement: "circled", labelMethyls: true, scaleWithZoom: true };
const STAGE_SPREAD = { charW: 0.24, labelCarbons: true, gapBonds: 1.6 };
const DURATION_MS = 3600;
const SPEEDS = [0.5, 1, 2] as const;
const REST_T = PHASES.arrows[1];

/** The plan of the morph: the reactant spread, the product relaxed from it (stepGeometry). */
function usePlan(source: Side | null, product: Side | null, aligned: XY | null): MorphPlan | null {
  return useMemo(() => {
    if (!source) return null;
    const from = spreadSide(source, new Map([...source.atoms.values()].map((a) => [a.index, [a.x, a.y] as Vec])), STAGE_SPREAD);
    const before: Side = { ...source, atoms: new Map([...source.atoms].map(([i, a]) => [i, { ...a, x: from.get(i)![0], y: from.get(i)![1] }])) };
    const after = product ?? source;
    // the product reached by moving the reactant drawing as little as it takes (atoms only the
    // product draws start from the fitted product depiction)
    const start = new Map([...(aligned ?? new Map()), ...from]);
    const to = product ? spreadSide(product, relaxLayout(start, product, { bond: BOND_LENGTH, iterations: 400 }), STAGE_SPREAD) : from;
    return morphPlan(before, after, to);
  }, [source, product, aligned]);
}

/** Playback: t in [0, 1], playing, speed and loop; the space bar toggles. */
function usePlayback(resetKey: string) {
  const [t, setT] = useState(REST_T);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(1);
  const [loop, setLoop] = useState(false);
  const last = useRef<number | null>(null);
  useEffect(() => {
    setT(REST_T);
    setPlaying(false);
  }, [resetKey]);
  useEffect(() => {
    if (!playing) return void (last.current = null);
    let raf = 0;
    const tick = (now: number) => {
      const dt = last.current === null ? 0 : now - last.current;
      last.current = now;
      setT((x) => {
        const next = x + (dt * speed) / DURATION_MS;
        if (next < 1) return next;
        if (loop) return next - 1;
        setPlaying(false);
        return 1;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, loop]);
  const toggle = () => {
    if (!playing && t >= 1) setT(0);
    setPlaying((p) => !p);
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT", "BUTTON", "A", "SUMMARY"].includes(el.tagName))) return;
      e.preventDefault();
      toggle();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  return { t, setT, playing, setPlaying, toggle, speed, setSpeed, loop, setLoop };
}

interface StageProps {
  plan: MorphPlan;
  t: number;
  width: number;
  props: StepVariantProps;
}

function Stage({ plan, t, width, props }: StageProps) {
  const { step, sides, prefix, hover } = props;
  const fitted = useMemo(() => fitScheme(bounds([plan.from, plan.to])!, Math.max(width, 1), { ...DEFAULT_FIT, maxBondPx: 76, maxHeight: 400, minHeight: 220, padBonds: 1.1 }), [plan, width]);
  const { view } = fitted;
  const M = useMemo(() => drawMetrics(STAGE_STYLE, pxPerPt(STAGE_STYLE, view.s)), [view.s]);
  const frame = morphFrame(plan, t);
  const beforeOrders = useMemo(() => new Map(plan.bonds.map((b) => [b.key, b.before] as const)), [plan]);
  const afterOrders = useMemo(() => new Map(plan.bonds.map((b) => [b.key, b.after] as const)), [plan]);
  const bm = sideModel(plan.before, 0, frame.xy, beforeOrders);
  const am = sideModel(plan.after, 0, frame.xy, afterOrders);
  const sb = buildScene(bm.model, view, bm.hCounts, { metrics: M });
  const sa = buildScene(am.model, view, am.hCounts, { metrics: M });
  const kb = bondKeys(bm.model);
  const ka = bondKeys(am.model);
  // the t = 0 drawing routes the arrows once (they draw on while the atoms are still)
  const rest = useMemo(() => {
    const m = sideModel(plan.before, 0, plan.from, beforeOrders);
    const scene = buildScene(m.model, view, m.hCounts, { metrics: M });
    const geom = atomGeometry(scene, M);
    const touched = touchedAtoms(step.matrix, step.arrows);
    const want = new Map([...touched].filter((i) => plan.before.atoms.get(i)?.lone).map((i) => [i, { n: plan.before.atoms.get(i)!.lone!, prefer: donorPreferences(step.arrows, geom, (x) => x).get(i) ?? null }] as const));
    const pairs = lonePairLayout(geom, want, M, scene.bonds.map((b) => [b.axis.p, b.axis.q] as const));
    const st = arrowStyleFor(M);
    return { routed: routeReading(step.arrows, scene, geom, pairs, M), st, touched };
  }, [plan, view, M, step.matrix, step.arrows, beforeOrders]);
  const late = frame.productLabels;
  const pairs = pairsAt(late ? sa : sb, late ? plan.after : plan.before, rest.touched, step.matrix, M, late);
  const donors = new Set(step.matrix.lone.filter(([, d]) => d < 0).map(([i]) => i));
  const xyPx = new Map([...frame.xy].map(([i, p]) => [i, [view.s * p[0] + view.ox, -view.s * p[1] + view.oy] as Vec] as const));
  const s = sides.structure!;
  const pv: PoolView = { view, xy: xyPx, bondPx: M.bond, centroid: centroidOf(xyPx) };
  const ctx = arrowContext(s, pv);
  const restXy = new Map([...plan.from].map(([i, p]) => [i, [view.s * p[0] + view.ox, -view.s * p[1] + view.oy] as Vec] as const));
  const paths = step.arrows.map((a, k) => {
    const r = rest.routed[k];
    if (!r) return null;
    const dT = drift(a.source, restXy, xyPx);
    const dH = drift(a.target, restXy, xyPx);
    const mv = (p: Vec, d: Vec): Vec => [p[0] + d[0], p[1] + d[1]];
    return arrowFromControls(mv(r.tail, dT), mv(r.c1, dT), mv(r.c2, dH), mv(r.head, dH), rest.st);
  });
  const marks = 1 - phase(t, [REST_T, REST_T + 0.12]);
  const halos = haloAt(xyPx, hover);
  return (
    <svg className="ss-svg" width={width} height={fitted.height} style={{ ...sceneVars(M), ["--ss-arrow-width" as string]: `${rest.st.stroke.toFixed(2)}px` }}>
      <Halos atoms={halos.atoms} bonds={halos.bonds} r={M.font * 1.15} />
      <SceneLayer scene={sb} bondAlpha={(id) => frame.bonds.get(kb.get(id)!)?.before ?? 1} labels={!late} />
      <SceneLayer scene={sa} bondAlpha={(id) => frame.bonds.get(ka.get(id)!)?.after ?? 0} labels={late} />
      <LonePairs marks={[...pairs.values()]} r={dotRadius(M)} alpha={(id, k) => (!late && donors.has(id) && k === 0 ? frame.loneOut : late && step.matrix.lone.some(([i, d]) => i === id && d > 0) && k === 0 ? frame.loneIn : 1)} />
      <ROverlay structure={s} pv={pv} ctx={ctx} prefix={`${prefix}-canvas`} matrix={step.matrix} arrows={step.arrows} paths={paths} marksOpacity={marks} arrowProgress={frame.arrowDraw} arrowOpacity={frame.arrowAlpha} chargesAfter={step.chargesAfter} />
    </svg>
  );
}

/** How far a set of atoms moved, on average, from the rest drawing (an arrow end follows it). */
function drift(atoms: readonly number[], rest: Map<number, Vec>, now: Map<number, Vec>): Vec {
  const ds = atoms.map((i) => [rest.get(i), now.get(i)] as const).filter((x): x is readonly [Vec, Vec] => !!x[0] && !!x[1]);
  if (!ds.length) return [0, 0];
  return [ds.reduce((s, [a, b]) => s + b[0] - a[0], 0) / ds.length, ds.reduce((s, [a, b]) => s + b[1] - a[1], 0) / ds.length];
}

function haloAt(xy: Map<number, Vec>, hover: HoverSet): { atoms: Vec[]; bonds: [Vec, Vec][] } {
  const atoms = [...hover.atoms].map((i) => xy.get(i)).filter((p): p is Vec => !!p);
  const bonds = [...hover.bonds]
    .map((k) => k.split("-").map(Number))
    .map(([i, j]) => [xy.get(i), xy.get(j)] as const)
    .filter((b): b is readonly [Vec, Vec] => !!b[0] && !!b[1])
    .map((b) => [b[0], b[1]] as [Vec, Vec]);
  return { atoms, bonds };
}

/** Lone pairs of the touched atoms on the current scene; the product side's new pair faces away
 *  from the partner it lost (pair 0). */
function pairsAt(scene: ReturnType<typeof buildScene>, side: Side, touched: Set<number>, m: MatrixIO, M: ReturnType<typeof drawMetrics>, late: boolean) {
  const geom = atomGeometry(scene, M);
  const want = new Map<number, { n: number; prefer?: number | null }>();
  for (const i of touched) {
    const a = side.atoms.get(i);
    if (!a?.lone) continue;
    let prefer: number | null = null;
    const broken = late ? m.bonds.find(([x, y, d]) => d < 0 && (x === i || y === i) && m.lone.some(([k, dd]) => k === i && dd > 0)) : undefined;
    if (broken) {
      const j = broken[0] === i ? broken[1] : broken[0];
      const [p, q] = [geom.get(i)?.at, geom.get(j)?.at];
      if (p && q) prefer = Math.atan2(p[1] - q[1], p[0] - q[0]);
    }
    want.set(i, { n: a.lone, prefer });
  }
  return lonePairLayout(geom, want, M, scene.bonds.map((b) => [b.axis.p, b.axis.q] as const));
}

function Controls({ pb, prefix }: { pb: ReturnType<typeof usePlayback>; prefix: string }) {
  return (
    <div className="ss-player">
      <button type="button" className="ss-play" data-testid={`${prefix}-play`} aria-pressed={pb.playing} onClick={pb.toggle} title="play / pause (space)">
        {pb.playing ? "❚❚" : "▶"}
      </button>
      <button type="button" className="btn btn-small" data-testid={`${prefix}-restart`} onClick={() => (pb.setT(0), pb.setPlaying(true))} title="play from the start">
        ↺
      </button>
      <input className="ss-scrub" type="range" min={0} max={1} step={0.001} value={pb.t} data-testid={`${prefix}-scrub`} aria-label="step progress" onChange={(e) => (pb.setPlaying(false), pb.setT(Number(e.target.value)))} />
      <span className="ss-seg ss-speed" role="group" aria-label="speed">
        {SPEEDS.map((v) => (
          <button key={v} type="button" className={`ss-seg-btn${pb.speed === v ? " ss-seg-on" : ""}`} aria-pressed={pb.speed === v} data-testid={`${prefix}-speed-${v}`} onClick={() => pb.setSpeed(v)}>
            {v}×
          </button>
        ))}
      </span>
      <label className="field-inline pv-small">
        <input type="checkbox" data-testid={`${prefix}-loop`} checked={pb.loop} onChange={(e) => pb.setLoop(e.target.checked)} />
        loop
      </label>
    </div>
  );
}

const PHASE_LABELS: [number, string][] = [
  [0, "electrons move (arrows)"],
  [PHASES.move[0], "bonds rearrange"],
  [PHASES.charge, "charges settle"],
  [0.88, "product"],
];

function Card({ props, plan }: { props: StepVariantProps; plan: MorphPlan }) {
  const { step, label, onHover, prefix } = props;
  const q = chargeDeltas(step.matrix);
  const rows = [
    ...step.matrix.bonds.map(([i, j, d]) => ({ key: `b${i}-${j}`, atoms: [i, j], bond: [i, j] as [number, number], what: `${label(i)}–${label(j)}`, delta: d, words: `order ${plan.bonds.find((b) => b.key === pairKey(i, j))?.before ?? 0} → ${plan.bonds.find((b) => b.key === pairKey(i, j))?.after ?? 0}` })),
    ...step.matrix.lone.map(([i, d]) => ({ key: `l${i}`, atoms: [i], bond: null, what: `lone pair ${label(i)}`, delta: d, words: `charge ${fmtQ(plan.before.atoms.get(i)?.charge ?? 0)} → ${fmtQ((plan.before.atoms.get(i)?.charge ?? 0) + (q.get(i) ?? 0))}` })),
  ];
  return (
    <aside className="ss-card" data-testid={`${prefix}-card`}>
      <h5 className="ss-card-h">Reaction matrix R</h5>
      <table className="ss-card-table">
        <tbody>
          {rows.map((r, k) => (
            <tr key={r.key} data-testid={`${prefix}-card-entry-${k}`} onPointerEnter={() => onHover(hoverOf(r.atoms, r.bond ? [r.bond] : []))} onPointerLeave={() => onHover(null)}>
              <td className={`ss-delta ${r.delta > 0 ? "ss-pos" : "ss-neg"}`}>{r.delta > 0 ? `+${r.delta}` : `−${-r.delta}`}</td>
              <td>{r.what}</td>
              <td className="muted">{r.words}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h5 className="ss-card-h">Orbital reading</h5>
      {step.readings.length ? (
        <ul className="ss-card-readings">
          {step.readings.map((r, k) => {
            const p = captionParts(r);
            return (
              <li key={k} data-testid={`${prefix}-card-reading-${k}`} onPointerEnter={() => onHover(hoverOf([r.i, r.j], [[r.i, r.j]]))} onPointerLeave={() => onHover(null)}>
                <div>
                  <b>
                    {label(r.i)}–{label(r.j)}
                  </b>{" "}
                  <span className="ss-cap-channel">{p.channel}</span>
                </div>
                <div className="muted">
                  {p.g}
                  {p.canonical ? ` · ${p.canonical}` : ""}
                </div>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="muted pv-small">{step.note ?? "no forming bond to read"}</p>
      )}
    </aside>
  );
}

const fmtQ = (q: number) => (q === 0 ? "0" : q > 0 ? `+${q}` : `−${-q}`);

export function StepAnimated(props: StepVariantProps) {
  const { sides, step, prefix } = props;
  const plan = usePlan(sides.source, sides.product, sides.aligned);
  const [ref, width] = useBoxWidth();
  const pb = usePlayback(`${JSON.stringify(step.matrix)}|${sides.product ? 1 : 0}`);
  const phaseLabel = [...PHASE_LABELS].reverse().find(([at]) => pb.t >= at)?.[1] ?? "";
  return (
    <div className="ss-animated" data-testid={`${prefix}-animated`}>
      <div className="ss-frame ss-stage-frame">
        <div ref={ref} className="sketch ss-canvas" data-testid={`${prefix}-canvas`} data-variant="animated" data-t={pb.t.toFixed(3)}>
          {plan && sides.structure && width > 0 ? <Stage plan={plan} t={pb.t} width={width} props={props} /> : <div className="pv-canvas-placeholder muted">{sides.loading ? "loading the depiction…" : (sides.error ?? "no depiction")}</div>}
          <div className="ss-phase muted" aria-live="off">
            {phaseLabel}
            {!sides.product && !sides.loading ? " · no product depiction: bonds change in place" : ""}
          </div>
        </div>
        <Controls pb={pb} prefix={prefix} />
      </div>
      {plan && <Card props={props} plan={plan} />}
    </div>
  );
}
