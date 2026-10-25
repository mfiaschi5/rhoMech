// LevelDiagram: the canonical MO energy levels in hand-made SVG. Degenerate MOs sit side by
// side; occupations as ↑↓; HOMO and LUMO labelled. Click a level to show that MO. When an LMO
// set is shown, a second column holds the LMO energies E_r (click: show that LMO) and the
// selected LMO's spectral weights are drawn as bars on the canonical levels.
//
// DOM: <figure data-testid="level-diagram" data-mode="canonical|lmo">; canonical level lines
// <g data-testid="level-<i>" data-frontier="homo|lumo|homo_minus_1|...|" data-selected
// data-weight>, LMO lines <g data-testid="lmo-level-<i>">.
import type { OrbitalRowOut } from "../../../api/types.gen";
import { exact, fmt, niceTicks, tickText } from "../../../components/format";
import { energyWindow, frontierLabel, groupLevels, sideBySide, spectralWeights, type Window } from "./levels";

export interface LevelDiagramProps {
  canonical: OrbitalRowOut[];
  /** The LMO rows when an LMO set is shown (null: canonical only). */
  lmos: OrbitalRowOut[] | null;
  /** The selected orbital of the shown set. */
  selected: number | null;
  window: Window;
  onPickCanonical(index: number): void;
  onPickLmo(index: number): void;
  /** Height of the drawing in viewBox units (px at 340 px wide); 380 by default. */
  height?: number;
}

const W = 340;
const DEFAULT_H = 380;
const TOP = 14;
const BOTTOM = 14;
const AXIS_X = 40;
const CAN_X0 = 52;
const CAN_X1 = 196;
const LMO_X0 = 250;
const LMO_X1 = 330;
const MAX_LINE = 40;

function occText(occ: number): string {
  if (occ > 1.5) return "↑↓";
  if (occ > 0.5) return "↑";
  return "";
}

function Axis({ y, lo, hi, H }: { y: (e: number) => number; lo: number; hi: number; H: number }) {
  return (
    <g className="wb-axis">
      <line x1={AXIS_X} x2={AXIS_X} y1={TOP} y2={H - BOTTOM} />
      {niceTicks(lo, hi, Math.max(4, Math.round(H / 64))).map((t) => (
        <g key={t}>
          <line x1={AXIS_X - 4} x2={AXIS_X} y1={y(t)} y2={y(t)} />
          <text x={AXIS_X - 6} y={y(t) + 3} textAnchor="end">
            {tickText(t)}
          </text>
        </g>
      ))}
      <text x={10} y={TOP + 4} className="wb-axis-title">
        eV
      </text>
    </g>
  );
}

function CanonicalLevels(p: { rows: OrbitalRowOut[]; y: (e: number) => number; lo: number; hi: number; selected: number | null; weights: Map<number, number> | null; onPick(i: number): void }) {
  return (
    <g>
      {groupLevels(p.rows)
        .filter((g) => g.energy >= p.lo && g.energy <= p.hi)
        .map((g) => {
          const { xs, w } = sideBySide(g.rows.length, CAN_X0, CAN_X1 - 30, MAX_LINE, 5);
          const yy = p.y(g.energy);
          const label = g.rows.map((r) => frontierLabel(r.frontier)).find((l) => l === "HOMO" || l === "LUMO");
          return (
            <g key={g.level}>
              {g.rows.map((r, k) => {
                const wt = p.weights?.get(r.index) ?? 0;
                const sel = p.weights === null && r.index === p.selected;
                const cls = `wb-level ${r.occupation > 0.5 ? "occ" : "virt"}${sel ? " selected" : ""}`;
                return (
                  <g key={r.index} className={cls} data-testid={`level-${r.index}`} data-frontier={r.frontier ?? ""} data-selected={sel ? "true" : "false"} data-weight={p.weights ? exact(wt) : undefined} onClick={() => p.onPick(r.index)} role="button">
                    <title>{`MO ${r.index}${r.label ? ` (${r.label})` : ""}: ${fmt(r.energy_ev, 3)} eV, occupation ${fmt(r.occupation, 2)}`}</title>
                    <rect className="wb-hit" x={xs[k] - 2} y={yy - 6} width={w + 4} height={12} />
                    <line x1={xs[k]} x2={xs[k] + w} y1={yy} y2={yy} />
                    <text className="wb-occ" x={xs[k] + w / 2} y={yy - 2} textAnchor="middle">
                      {occText(r.occupation)}
                    </text>
                    {wt > 0 && <rect className="wb-weight" x={xs[k]} y={yy + 1.5} width={Math.max(1.5, w * Math.min(1, wt / 0.5))} height={4} />}
                  </g>
                );
              })}
              {label && (
                <text className="wb-frontier" x={CAN_X1 - 26} y={yy + 3}>
                  {label}
                </text>
              )}
            </g>
          );
        })}
    </g>
  );
}

function LmoLevels(p: { rows: OrbitalRowOut[]; y: (e: number) => number; lo: number; hi: number; selected: number | null; onPick(i: number): void }) {
  return (
    <g>
      <text className="wb-axis-title" x={(LMO_X0 + LMO_X1) / 2} y={TOP + 2} textAnchor="middle">
        LMO E_r
      </text>
      {p.rows
        .filter((r) => r.energy_ev >= p.lo && r.energy_ev <= p.hi)
        .map((r) => {
          const yy = p.y(r.energy_ev);
          const sel = r.index === p.selected;
          return (
            <g key={r.index} className={`wb-level lmo ${r.occupation > 0.5 ? "occ" : "virt"}${sel ? " selected" : ""}`} data-testid={`lmo-level-${r.index}`} data-selected={sel ? "true" : "false"} onClick={() => p.onPick(r.index)} role="button">
              <title>{`LMO ${r.index}: E_r ${fmt(r.energy_ev, 3)} eV, occupation ${fmt(r.occupation, 2)}`}</title>
              <rect className="wb-hit" x={LMO_X0 - 2} y={yy - 4} width={LMO_X1 - LMO_X0 + 4} height={8} />
              <line x1={LMO_X0} x2={LMO_X1} y1={yy} y2={yy} />
            </g>
          );
        })}
    </g>
  );
}

export function LevelDiagram(p: LevelDiagramProps) {
  const H = p.height ?? DEFAULT_H;
  const [lo, hi] = energyWindow([...p.canonical, ...(p.lmos ?? [])], p.window);
  const y = (e: number) => TOP + ((hi - e) / (hi - lo)) * (H - TOP - BOTTOM);
  const lmo = p.lmos?.find((r) => r.index === p.selected);
  const weights = p.lmos ? spectralWeights(lmo) : null;
  const hidden = p.canonical.filter((r) => r.energy_ev < lo || r.energy_ev > hi).length;
  return (
    <figure className="wb-levels" data-testid="level-diagram" data-mode={p.lmos ? "lmo" : "canonical"}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="orbital energy levels">
        <Axis y={y} lo={lo} hi={hi} H={H} />
        <CanonicalLevels rows={p.canonical} y={y} lo={lo} hi={hi} selected={p.selected} weights={weights} onPick={p.onPickCanonical} />
        {p.lmos && <LmoLevels rows={p.lmos} y={y} lo={lo} hi={hi} selected={p.selected} onPick={p.onPickLmo} />}
      </svg>
      <figcaption className="muted wb-small">
        {p.lmos ? "Bars: the selected LMO's spectral weights on the canonical MOs. Click a canonical level to show that MO." : "Click a level to show that MO."}
        {hidden > 0 ? ` ${hidden} level${hidden === 1 ? "" : "s"} outside the window.` : ""}
      </figcaption>
    </figure>
  );
}
