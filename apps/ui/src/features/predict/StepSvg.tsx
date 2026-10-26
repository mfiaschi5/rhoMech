// SVG pieces of the step drawings (StepTextbook, StepAnimated, StepOrbital): a workbench scene
// (drawing.buildScene) drawn with per-bond opacity, lone-pair dots, hover halos, invisible
// atom / bond hover targets, routed curved arrows (optionally drawing on) and the reaction arrow.
// Colours come from the theme (sketch.css element tokens, --fg, --arrow), so light and dark both
// read; the export (stepExport.ts) writes the same geometry in black.
import type { CSSProperties } from "react";
import type { Scene } from "../workbench/sketch/drawing";
import { chargeAnchor, labelGlyphs } from "../workbench/sketch/drawing";
import { elementClass } from "../workbench/sketch/elements";
import { cssFontFamily, type DrawMetrics } from "../workbench/sketch/style";
import type { ArrowPath, Vec } from "./arrowPath";
import type { LonePairMark } from "./stepScene";

const px = (v: number) => `${Math.round(v * 1000) / 1000}px`;

/** The style's sizes as the CSS variables sketch.css reads. */
export function sceneVars(M: DrawMetrics): CSSProperties {
  return {
    "--sk-font": px(M.font),
    "--sk-sub-font": px(M.sub),
    "--sk-charge-font": px(M.charge),
    "--sk-font-family": cssFontFamily(M.fontFamily, true),
    "--sk-font-weight": String(M.fontWeight),
    "--sk-bond-width": px(M.line),
    "--sk-hash-width": px(M.hashWidth),
    "--sk-wedge-stroke": "0px",
    "--sk-charge-weight": String(M.fontWeight),
  } as CSSProperties;
}

const pts = (vs: readonly Vec[]) => vs.map((v) => `${v[0].toFixed(2)},${v[1].toFixed(2)}`).join(" ");

export interface SceneLayerProps {
  scene: Scene;
  /** Opacity of a bond by its id (default 1; 0 is not drawn). */
  bondAlpha?: (id: number) => number;
  labels?: boolean;
  labelAlpha?: number;
  className?: string;
}

/** A scene's bonds and labels. */
export function SceneLayer({ scene, bondAlpha, labels = true, labelAlpha = 1, className }: SceneLayerProps) {
  const M = scene.metrics;
  return (
    <g className={`ss-layer${className ? ` ${className}` : ""}`}>
      <g className="sk-bonds">
        {scene.bonds.map((b) => {
          const a = bondAlpha ? bondAlpha(b.bond.id) : 1;
          if (a <= 0.001) return null;
          return (
            <g key={b.bond.id} opacity={a < 1 ? a : undefined}>
              {b.shape.lines.map((l, i) => (
                <line key={i} className="sk-stroke" x1={l.p[0]} y1={l.p[1]} x2={l.q[0]} y2={l.q[1]} />
              ))}
              {b.shape.wedges?.map((w, i) => <polygon key={`w${i}`} className="sk-wedge" points={pts(w)} />)}
            </g>
          );
        })}
      </g>
      {labels && (
        <g className="sk-atoms" opacity={labelAlpha < 1 ? labelAlpha : undefined}>
          {scene.atoms.map((s) => {
            const label = s.label;
            if (!label) return null;
            const cy = label.charge ? chargeAnchor(label, M)[1] : 0;
            return (
              <g key={s.atom.id} className={`sk-label ${elementClass(label.el)}`} data-atom={s.atom.id}>
                {s.leader && <line className="sk-leader" x1={s.leader.p[0]} y1={s.leader.p[1]} x2={s.leader.q[0]} y2={s.leader.q[1]} />}
                {labelGlyphs(label, M).map((g, i) => {
                  if (g.kind === "charge" && g.circled) {
                    const r = g.w / 2;
                    return (
                      <g key={i} className="sk-charge-circled">
                        <circle className="sk-charge-ring" cx={s.at[0] + g.x + r} cy={s.at[1] + cy} r={r * 0.95} />
                        <text className="sk-charge" x={s.at[0] + g.x + r} y={s.at[1] + cy} textAnchor="middle" dominantBaseline="central" style={{ fontSize: px(g.size) }}>
                          {g.text}
                        </text>
                      </g>
                    );
                  }
                  const sym = g.kind === "symbol";
                  const cls = g.kind === "charge" ? "sk-charge" : g.kind === "count" ? "sk-sub" : undefined;
                  const size = g.kind === "charge" || g.kind === "count" ? { fontSize: px(g.size) } : undefined;
                  return (
                    <text key={i} className={cls} x={s.at[0] + g.x + (sym ? g.w / 2 : 0)} y={s.at[1] + g.y} textAnchor={sym ? "middle" : undefined} style={size}>
                      {g.text}
                    </text>
                  );
                })}
              </g>
            );
          })}
        </g>
      )}
    </g>
  );
}

/** Lone-pair dots; `alpha(id, k)` fades pair k of atom id. */
export function LonePairs({ marks, r, alpha, className }: { marks: readonly LonePairMark[]; r: number; alpha?: (id: number, k: number) => number; className?: string }) {
  return (
    <g className={`ss-lone${className ? ` ${className}` : ""}`}>
      {marks.flatMap((m) =>
        m.dots.map((d, k) => {
          const a = alpha ? alpha(m.id, k) : 1;
          if (a <= 0.001) return null;
          return (
            <g key={`${m.id}-${k}`} opacity={a < 1 ? a : undefined} data-atom={m.id}>
              <circle cx={d[0][0]} cy={d[0][1]} r={r} />
              <circle cx={d[1][0]} cy={d[1][1]} r={r} />
            </g>
          );
        }),
      )}
    </g>
  );
}

/** Soft halos under hovered atoms and bonds. */
export function Halos({ atoms, bonds, r }: { atoms: readonly Vec[]; bonds: readonly (readonly [Vec, Vec])[]; r: number }) {
  return (
    <g className="ss-halos">
      {bonds.map(([p, q], k) => (
        <line key={`b${k}`} className="ss-halo-bond" x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} style={{ strokeWidth: r * 1.1 }} />
      ))}
      {atoms.map((p, k) => (
        <circle key={`a${k}`} className="ss-halo-atom" cx={p[0]} cy={p[1]} r={r} />
      ))}
    </g>
  );
}

export interface HoverTarget {
  key: string;
  atoms: number[];
  bond?: [number, number];
  p: Vec;
  q?: Vec;
}

/** Invisible hover targets (atoms as discs, bonds as wide strokes). */
export function HoverTargets({ targets, r, onHover }: { targets: readonly HoverTarget[]; r: number; onHover: (t: HoverTarget | null) => void }) {
  return (
    <g className="ss-hits">
      {targets.map((t) =>
        t.q ? (
          <line key={t.key} className="ss-hit" x1={t.p[0]} y1={t.p[1]} x2={t.q[0]} y2={t.q[1]} style={{ strokeWidth: r * 1.2 }} onPointerEnter={() => onHover(t)} onPointerLeave={() => onHover(null)} data-hit={t.key} />
        ) : (
          <circle key={t.key} className="ss-hit" cx={t.p[0]} cy={t.p[1]} r={r} onPointerEnter={() => onHover(t)} onPointerLeave={() => onHover(null)} data-hit={t.key} />
        ),
      )}
    </g>
  );
}

/** The reaction arrow with text above and below it. */
export function ReactionArrow({ from, to, above, below, size }: { from: Vec; to: Vec; above?: string; below?: string; size: number }) {
  const hl = Math.max(8, size * 0.9);
  const hw = hl * 0.42;
  const tip = to;
  const base: Vec = [to[0] - hl, to[1]];
  return (
    <g className="ss-rxn">
      <line className="ss-rxn-line" x1={from[0]} y1={from[1]} x2={base[0] + hl * 0.2} y2={base[1]} />
      <polygon className="ss-rxn-head" points={pts([tip, [base[0], base[1] - hw], [base[0] + hl * 0.22, base[1]], [base[0], base[1] + hw]])} />
      {above && (
        <text className="ss-rxn-text" x={(from[0] + to[0]) / 2} y={from[1] - size * 0.7} textAnchor="middle" style={{ fontSize: px(size) }}>
          {above}
        </text>
      )}
      {below && (
        <text className="ss-rxn-text ss-rxn-sub" x={(from[0] + to[0]) / 2} y={from[1] + size * 1.45} textAnchor="middle" style={{ fontSize: px(size * 0.9) }}>
          {below}
        </text>
      )}
    </g>
  );
}

/** Head and stroke of one arrow (no test ids: the ROverlay draws the e2e-facing arrows). */
export function ArrowShape({ p, className }: { p: ArrowPath; className?: string }) {
  return (
    <g className={className}>
      <path className="pv-arrow-line" d={p.d} />
      <polygon className="pv-arrow-head" points={pts(p.head)} />
    </g>
  );
}
