// EnergyLevels: canonical MO levels behind the gap kernel G, hand-made SVG (design §7.5, §11.7).
// The donor species' occupied levels on the left and the acceptor species' virtual levels on the
// right, on one eV axis. Line length is proportional to the level's weight in the LMO; the
// dominant pair k and m is bold and labelled ("HOMO−1 −9.21 eV"); a dashed double arrow shows
// E_m − E_k; dotted lines mark the LMO energies E_r and E_s.
//
// Props:
//   levels     EnergyLevelsOut
//   dominant   DominantPairOut
//
// DOM: <figure data-testid="energy-levels"> > svg; level lines carry data-canonical, data-side and
// data-dominant.
import { useId } from "react";
import type { DominantPairOut, EnergyLevelsOut, LevelOut } from "../../api/types.gen";
import { exact, fmt, moLabel, niceTicks, tickText } from "../../components/format";

export interface EnergyLevelsProps {
  levels: EnergyLevelsOut;
  dominant: DominantPairOut;
}

const W = 480;
const H = 330;
const TOP = 34;
const BOTTOM = 16;
const AXIS_X = 44;
const DONOR_RIGHT = 214;
const ACCEPTOR_LEFT = 274;
const MAX_LEN = 104;
const MIN_LEN = 16;

function scale(levels: EnergyLevelsOut, dominant: DominantPairOut) {
  const es = [
    ...levels.donor_levels.map((l) => l.energy),
    ...levels.acceptor_levels.map((l) => l.energy),
    levels.donor_lmo_energy,
    levels.acceptor_lmo_energy,
    dominant.E_k,
    dominant.E_m,
  ].filter((e) => Number.isFinite(e));
  let lo = es.length ? Math.min(...es) : -1;
  let hi = es.length ? Math.max(...es) : 1;
  if (!(hi > lo)) {
    lo -= 1;
    hi += 1;
  }
  const pad = 0.06 * (hi - lo);
  lo -= pad;
  hi += pad;
  return { lo, hi, ys: (e: number) => TOP + ((hi - e) / (hi - lo)) * (H - TOP - BOTTOM) };
}

interface ColumnProps {
  side: "donor" | "acceptor";
  levels: LevelOut[];
  dominantCanonical: number;
  ys: (e: number) => number;
  maxW: number;
}

function Column({ side, levels, dominantCanonical, ys, maxW }: ColumnProps) {
  return (
    <g className={`levels-${side}`}>
      {levels.map((l) => {
        const dom = l.canonical === dominantCanonical;
        const len = MIN_LEN + (MAX_LEN - MIN_LEN) * (l.weight !== null && maxW > 0 ? l.weight / maxW : 0);
        const [x1, x2] = side === "donor" ? [DONOR_RIGHT - len, DONOR_RIGHT] : [ACCEPTOR_LEFT, ACCEPTOR_LEFT + len];
        const y = ys(l.energy);
        const text = `${moLabel(l.label)} (MO ${l.canonical}): ${exact(l.energy)} eV, weight ${l.weight === null ? "n/a" : exact(l.weight)}`;
        return (
          <g key={`${side}-${l.canonical}`} className={`level${dom ? " dominant" : ""}${l.weight === null ? " unweighted" : ""}`} data-canonical={l.canonical} data-side={side} data-dominant={dom ? "true" : "false"}>
            <title>{text}</title>
            <line x1={x1} x2={x2} y1={y} y2={y} />
            {dom && (
              <text x={side === "donor" ? DONOR_RIGHT : ACCEPTOR_LEFT} y={y - 5} textAnchor={side === "donor" ? "end" : "start"} className="level-label">
                {`${moLabel(l.label)} ${fmt(l.energy, 2).replace("-", "\u2212")} eV`}
              </text>
            )}
          </g>
        );
      })}
    </g>
  );
}

function GapArrow({ uid, yk, ym, gapText }: { uid: string; yk: number; ym: number; gapText: string }) {
  const mx = (DONOR_RIGHT + ACCEPTOR_LEFT) / 2;
  const my = (yk + ym) / 2;
  return (
    <g className="gap-arrow" data-testid="energy-gap">
      <defs>
        <marker id={`${uid}-gap-head`} viewBox="0 0 8 8" refX={7} refY={4} markerWidth={7} markerHeight={7} orient="auto-start-reverse">
          <path d="M0 0 L8 4 L0 8 z" />
        </marker>
      </defs>
      <line x1={DONOR_RIGHT + 3} y1={yk} x2={ACCEPTOR_LEFT - 3} y2={ym} markerStart={`url(#${uid}-gap-head)`} markerEnd={`url(#${uid}-gap-head)`} />
      <text x={mx} y={my - 6} textAnchor="middle" className="gap-label">
        {gapText}
      </text>
    </g>
  );
}

function LmoLine({ side, energy, ys }: { side: "donor" | "acceptor"; energy: number; ys: (e: number) => number }) {
  const y = ys(energy);
  const [x1, x2] = side === "donor" ? [DONOR_RIGHT - MAX_LEN - 4, DONOR_RIGHT] : [ACCEPTOR_LEFT, ACCEPTOR_LEFT + MAX_LEN + 4];
  const label = side === "donor" ? "E_r" : "E_s";
  return (
    <g className="lmo-line" data-lmo={label}>
      <title>{`${label} (LMO energy) = ${exact(energy)} eV`}</title>
      <line x1={x1} x2={x2} y1={y} y2={y} />
      <text x={side === "donor" ? x1 - 3 : x2 + 3} y={y + 4} textAnchor={side === "donor" ? "end" : "start"}>
        {label}
      </text>
    </g>
  );
}

export function EnergyLevels({ levels, dominant }: EnergyLevelsProps) {
  const uid = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const { lo, hi, ys } = scale(levels, dominant);
  const maxW = Math.max(1e-12, ...[...levels.donor_levels, ...levels.acceptor_levels].map((l) => l.weight ?? 0));
  const gap = dominant.E_m - dominant.E_k;
  return (
    <figure className="energy-levels" data-testid="energy-levels" data-k={dominant.k_canonical} data-m={dominant.m_canonical}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="canonical MO levels of the donor and acceptor species">
        <text x={(AXIS_X + DONOR_RIGHT) / 2 + 20} y={14} textAnchor="middle" className="col-title">
          {levels.donor_species} occupied
        </text>
        <text x={ACCEPTOR_LEFT + MAX_LEN / 2} y={14} textAnchor="middle" className="col-title">
          {levels.acceptor_species} virtual
        </text>
        <line className="chart-axis" x1={AXIS_X} x2={AXIS_X} y1={TOP} y2={H - BOTTOM} />
        {niceTicks(lo, hi, 6).map((t) => (
          <g key={t} className="chart-tick">
            <line x1={AXIS_X - 4} x2={AXIS_X} y1={ys(t)} y2={ys(t)} />
            <text x={AXIS_X - 6} y={ys(t) + 4} textAnchor="end">
              {tickText(t)}
            </text>
          </g>
        ))}
        <text x={12} y={TOP - 10} className="axis-unit">
          eV
        </text>
        <LmoLine side="donor" energy={levels.donor_lmo_energy} ys={ys} />
        <LmoLine side="acceptor" energy={levels.acceptor_lmo_energy} ys={ys} />
        <Column side="donor" levels={levels.donor_levels} dominantCanonical={dominant.k_canonical} ys={ys} maxW={maxW} />
        <Column side="acceptor" levels={levels.acceptor_levels} dominantCanonical={dominant.m_canonical} ys={ys} maxW={maxW} />
        <GapArrow uid={uid} yk={ys(dominant.E_k)} ym={ys(dominant.E_m)} gapText={`E_m − E_k = ${fmt(gap, 2)} eV`} />
      </svg>
      <figcaption className="muted">
        line length ∝ weight in the LMO; HOMO-LUMO gap {fmt(levels.homo_lumo_gap, 2)} eV
        {!dominant.same_molecule && "; k and m index two different molecules' spectra (separate QM runs)"}
      </figcaption>
    </figure>
  );
}
