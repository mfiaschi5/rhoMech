// ROverlay: the reaction matrix and its curved arrows drawn over a pool drawing (SVG, screen px).
//   bond +1 on a non-bonded pair   green dashed line (a forming bond)
//   bond +1 on a bond              green dashed line beside it (the order goes up)
//   bond -1                        the bond in red with a cross at its midpoint
//   lone ±2                        a badge beside the atom's lone-pair side
//   formal-charge change           a circled new charge, upper right of the atom by preference
// Badges are placed clear of labels, bonds, arrows and each other (badges.ts).
//   arrows                         cubic Bézier arrows with filled heads (arrowPath.ts)
// Atoms with an issue are haloed (highlight), the bond tool's first atom ringed (pending).
// The step viewport passes its own routed arrows (paths), fades or hides the R marks
// (marksOpacity; 0 keeps them in the DOM) and draws the arrows on (arrowProgress, arrowOpacity).
//
// DOM (prefix = the canvas test id): <g data-testid="<prefix>-r-bond-<i>-<j>" data-delta
// data-forming>, <g data-testid="<prefix>-r-lone-<i>" data-delta>, <g data-testid=
// "<prefix>-charge-<i>" data-delta data-charge>, <g data-testid="<prefix>-arrow-<k>" data-kind
// data-source data-target data-mx data-my (a point on the curve, px)> with path.pv-arrow-line, polygon.pv-arrow-head, path.pv-arrow-hit.
import type { PointerEvent as ReactPointerEvent } from "react";
import { cubicPoint, layoutReading, lonePairPoint, type ArrowGeomContext, type ArrowPath, type Vec } from "./arrowPath";
import { boxAround, placeBadges, segmentBoxes, type BadgeRequest, type Box } from "./badges";
import type { PoolStructure } from "./poolStructure";
import { chargeDeltas } from "./rmatrix";
import type { ArrowReadingOut, MatrixIO } from "./v9types";
import type { PoolView } from "./poolView";

export interface ROverlayProps {
  structure: PoolStructure;
  pv: PoolView;
  ctx: ArrowGeomContext;
  prefix: string;
  matrix: MatrixIO | null;
  arrows: readonly ArrowReadingOut[];
  arrowsFaded?: boolean;
  /** Formal charges after R (the check's), by atom; else q + Δq from R. */
  chargesAfter?: ReadonlyMap<number, number> | null;
  highlight?: ReadonlySet<number>;
  pending?: number | null;
  selectedKeys?: ReadonlySet<string>;
  onArrowPointerDown?: (index: number, e: ReactPointerEvent<SVGPathElement>) => void;
  /** Arrow geometry from a router (the step viewport); default: layoutReading. */
  paths?: readonly (ArrowPath | null)[];
  /** Opacity of the R marks (bonds, lone badges, charges; 0 keeps them in the DOM, hidden). */
  marksOpacity?: number;
  /** Stroke progress (0-1) and opacity of the arrows (the animated step draws them on). */
  arrowProgress?: number;
  arrowOpacity?: number;
}

const sub = (p: Vec, q: Vec): Vec => [p[0] - q[0], p[1] - q[1]];
const unit = (p: Vec): Vec => {
  const l = Math.hypot(p[0], p[1]) || 1;
  return [p[0] / l, p[1] / l];
};
const signed = (v: number) => (v > 0 ? `+${v}` : v < 0 ? `−${-v}` : "0");

function BondMark({ a, b, delta, order, testId, selected }: { a: Vec; b: Vec; delta: number; order: number; testId: string; selected: boolean }) {
  const u = unit(sub(b, a));
  const n: Vec = [-u[1], u[0]];
  const trim = 9;
  const p: Vec = [a[0] + u[0] * trim, a[1] + u[1] * trim];
  const q: Vec = [b[0] - u[0] * trim, b[1] - u[1] * trim];
  const m: Vec = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const forming = delta > 0 && order === 0;
  const off = delta > 0 && !forming ? 5 + 2.5 * order : 0;
  const cls = `pv-rbond ${delta > 0 ? "pv-form" : "pv-break"}${selected ? " pv-selected" : ""}`;
  return (
    <g className={cls} data-testid={testId} data-delta={delta} data-forming={forming ? "true" : "false"}>
      {delta > 0 ? (
        <line x1={p[0] + n[0] * off} y1={p[1] + n[1] * off} x2={q[0] + n[0] * off} y2={q[1] + n[1] * off} className="pv-form-line" />
      ) : (
        <>
          <line x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} className="pv-break-line" />
          <path className="pv-cross" d={`M ${m[0] - 5} ${m[1] - 5} L ${m[0] + 5} ${m[1] + 5} M ${m[0] - 5} ${m[1] + 5} L ${m[0] + 5} ${m[1] - 5}`} />
        </>
      )}
      {Math.abs(delta) !== 1 && (
        <text x={m[0] + n[0] * 12} y={m[1] + n[1] * 12} className="pv-mark-text">
          {signed(delta)}
        </text>
      )}
    </g>
  );
}

function Badge({ at, text, cls, testId, data }: { at: Vec; text: string; cls: string; testId: string; data: Record<string, string> }) {
  const w = 7 + 6.5 * text.length;
  return (
    <g className={cls} data-testid={testId} {...data}>
      <rect x={at[0] - w / 2} y={at[1] - 8} width={w} height={16} rx={8} />
      <text x={at[0]} y={at[1] + 0.5} textAnchor="middle" dominantBaseline="central">
        {text}
      </text>
    </g>
  );
}

const chargeText = (q: number) => (q === 0 ? "0" : q > 0 ? (q === 1 ? "+" : `${q}+`) : q === -1 ? "−" : `${-q}−`);

/** What the badges must not cover: atom labels, bonds (the reactant's and R's), arrow curves. */
function obstacles(s: PoolStructure, pv: PoolView, m: MatrixIO | null, paths: readonly (ArrowPath | null)[]): Box[] {
  const out: Box[] = [];
  const shown = new Map(s.model.atoms.map((a) => [a.id, a] as const));
  for (const a of s.drawn) {
    const p = pv.xy.get(a.index);
    const sk = shown.get(a.sketchId!);
    if (!p || !sk) continue;
    const h = s.hCounts.get(a.sketchId!) ?? 0;
    const labelled = a.el !== "C" || sk.explicitH || a.charge !== 0;
    const chars = a.el.length + (h > 0 ? (h > 1 ? 2 : 1) : 0);
    out.push(boxAround(p, labelled ? 4 + 4.5 * chars : 4, labelled ? 8 : 4));
  }
  const pairs = [...s.bonds.map((b) => [b.i, b.j] as const), ...(m?.bonds ?? []).map(([i, j]) => [i, j] as const)];
  for (const [i, j] of pairs) {
    const p = pv.xy.get(i);
    const q = pv.xy.get(j);
    if (p && q) out.push(...segmentBoxes(p, q));
  }
  for (const c of paths) {
    if (!c) continue;
    for (let k = 0; k < 20; k++) out.push(...segmentBoxes(cubicPoint(c.tail, c.c1, c.c2, c.tip, k / 20), cubicPoint(c.tail, c.c1, c.c2, c.tip, (k + 1) / 20), 4, 2));
  }
  return out;
}

/** Centres of the lone-pair badges ("l:<i>") and the charge marks ("q:<i>"), collision-aware. */
function badgeLayout(s: PoolStructure, pv: PoolView, ctx: ArrowGeomContext, m: MatrixIO, paths: readonly (ArrowPath | null)[]): Map<string, Vec> {
  const reqs: BadgeRequest[] = [];
  for (const [i, d] of m.lone) {
    const p = ctx.pos(i);
    const lp = lonePairPoint(ctx, i);
    if (p && lp) reqs.push({ key: `l:${i}`, at: p, w: 7 + 6.5 * signed(d).length, h: 16, prefer: Math.atan2(lp.dir[1], lp.dir[0]) + (65 * Math.PI) / 180 });
  }
  for (const [i] of chargeDeltas(m)) {
    const p = pv.xy.get(i);
    if (p) reqs.push({ key: `q:${i}`, at: p, w: 18, h: 18, prefer: -Math.PI / 4 });
  }
  return placeBadges(reqs, obstacles(s, pv, m, paths), pv.bondPx);
}

function LoneMarks({ m, at, prefix, selected }: { m: MatrixIO; at: Map<string, Vec>; prefix: string; selected?: ReadonlySet<string> }) {
  return (
    <>
      {m.lone.map(([i, d]) => {
        const p = at.get(`l:${i}`);
        if (!p) return null;
        const cls = `pv-badge ${d > 0 ? "pv-form" : "pv-break"}${selected?.has(`l:${i}`) ? " pv-selected" : ""}`;
        return <Badge key={i} at={p} text={signed(d)} cls={cls} testId={`${prefix}-r-lone-${i}`} data={{ "data-delta": String(d) }} />;
      })}
    </>
  );
}

function ChargeMarks({ s, m, at, after, prefix }: { s: PoolStructure; m: MatrixIO; at: Map<string, Vec>; after?: ReadonlyMap<number, number> | null; prefix: string }) {
  return (
    <>
      {[...chargeDeltas(m)].map(([i, d]) => {
        const p = at.get(`q:${i}`);
        const a = s.atom(i);
        if (!p || !a) return null;
        const q = after?.get(i) ?? a.charge + d;
        return (
          <g key={i} className="pv-charge" data-testid={`${prefix}-charge-${i}`} data-delta={d} data-charge={q}>
            <title>{`formal charge ${signed(a.charge)} → ${signed(q)}`}</title>
            <circle cx={p[0]} cy={p[1]} r={8.5} />
            <text x={p[0]} y={p[1] + 0.5} textAnchor="middle" dominantBaseline="central">
              {chargeText(q)}
            </text>
          </g>
        );
      })}
    </>
  );
}

function Arrows(props: { arrows: readonly ArrowReadingOut[]; paths: readonly (ArrowPath | null)[]; prefix: string; faded?: boolean; onDown?: ROverlayProps["onArrowPointerDown"]; progress?: number; opacity?: number }) {
  const { paths } = props;
  const prog = props.progress ?? 1;
  const draw = prog < 1 ? { pathLength: 1, strokeDasharray: "1 1", strokeDashoffset: 1 - prog } : {};
  return (
    <g className={`pv-arrows${props.faded ? " pv-faded" : ""}`} opacity={props.opacity !== undefined && props.opacity < 1 ? props.opacity : undefined}>
      {paths.map((p, k) => {
        const a = props.arrows[k];
        if (!p) return null;
        return (
          <g key={k} className="pv-arrow" data-testid={`${props.prefix}-arrow-${k}`} data-kind={a.kind} data-source={a.source.join(",")} data-target={a.target.join(",")} data-mx={p.mid[0].toFixed(1)} data-my={p.mid[1].toFixed(1)}>
            <path className="pv-arrow-hit" d={p.d} onPointerDown={props.onDown ? (e) => props.onDown?.(k, e) : undefined} />
            <path className="pv-arrow-line" d={p.d} {...draw} />
            <polygon className="pv-arrow-head" points={p.head.map((v) => `${v[0].toFixed(2)},${v[1].toFixed(2)}`).join(" ")} opacity={prog < 0.97 ? 0 : undefined} />
          </g>
        );
      })}
    </g>
  );
}

export function ROverlay(props: ROverlayProps) {
  const { structure: s, pv, ctx, prefix, matrix } = props;
  const paths = props.paths ?? layoutReading(props.arrows, ctx);
  const at = matrix ? badgeLayout(s, pv, ctx, matrix, paths) : new Map<string, Vec>();
  const mo = props.marksOpacity ?? 1;
  return (
    <g className="pv-overlay">
      <g className={mo <= 0.001 ? "pv-marks pv-marks-hidden" : "pv-marks"} opacity={mo > 0.001 && mo < 1 ? mo : undefined}>
      {[...(props.highlight ?? [])].map((i) => {
        const p = pv.xy.get(i);
        return p ? <circle key={`h${i}`} className="pv-halo" cx={p[0]} cy={p[1]} r={14} data-testid={`${prefix}-halo-${i}`} /> : null;
      })}
      {props.pending != null && pv.xy.get(props.pending) && (
        <circle className="pv-pending" cx={pv.xy.get(props.pending)![0]} cy={pv.xy.get(props.pending)![1]} r={12} data-testid={`${prefix}-pending`} />
      )}
      {matrix?.bonds.map(([i, j, d]) => {
        const a = pv.xy.get(i);
        const b = pv.xy.get(j);
        if (!a || !b) return null;
        return <BondMark key={`${i}-${j}`} a={a} b={b} delta={d} order={s.bondOrder(i, j)} testId={`${prefix}-r-bond-${i}-${j}`} selected={!!props.selectedKeys?.has(`b:${i}-${j}`)} />;
      })}
      {matrix && <LoneMarks m={matrix} at={at} prefix={prefix} selected={props.selectedKeys} />}
      {matrix && <ChargeMarks s={s} m={matrix} at={at} after={props.chargesAfter} prefix={prefix} />}
      </g>
      <Arrows arrows={props.arrows} paths={paths} prefix={prefix} faded={props.arrowsFaded} onDown={props.onArrowPointerDown} progress={props.arrowProgress} opacity={props.arrowOpacity} />
    </g>
  );
}
