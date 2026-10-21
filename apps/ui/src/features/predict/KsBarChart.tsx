// KsBarChart: the Klopman-Salem terms of one (donor, acceptor) pair as hand-made SVG
// (design §11.7). Three panels, each on its own labelled axis because the units differ:
//   1. KS1, KS2 and KS3 (eV), signed bars around zero, with the floor marker -(KS3 + KS2);
//   2. G (1/eV);
//   3. beta^2 at 2.0, 2.3 and 3.0 Angstrom (eV^2), the scored 2.3 Angstrom bar emphasised.
// Hovering (or focusing) a bar shows its exact value under the chart.
//
// Props:
//   physics   PhysicsOut
//
// DOM: <div data-testid="ks-bar-chart"> with one <svg data-panel="ks|G|beta2"> per panel;
// each bar <g data-bar data-value>.
import { useState } from "react";
import type { PhysicsOut } from "../../api/types.gen";
import { exact, fmtSig, niceTicks, tickText } from "./format";

export interface KsBarChartProps {
  physics: PhysicsOut;
}

interface Bar {
  key: string;
  label: string;
  value: number | null;
  note?: string;
  emphasis?: boolean;
}

interface Marker {
  value: number;
  label: string;
}

const W = 400;
const X0 = 70;
const X1 = W - 64;
const ROW = 22;
const TOP = 6;
const AXIS = 22;

function domain(bars: Bar[], marker?: Marker): [number, number] {
  const vals = bars.map((b) => b.value).filter((v): v is number => v !== null && Number.isFinite(v));
  if (marker && Number.isFinite(marker.value)) vals.push(marker.value);
  let lo = Math.min(0, ...vals);
  let hi = Math.max(0, ...vals);
  if (hi - lo < 1e-12) {
    const m = Math.max(Math.abs(hi), 1e-6);
    lo -= m;
    hi += m;
  }
  const pad = 0.06 * (hi - lo);
  return [lo < 0 ? lo - pad : lo, hi > 0 ? hi + pad : hi];
}

interface PanelProps {
  id: string;
  title: string;
  unit: string;
  bars: Bar[];
  marker?: Marker;
  onHover: (text: string | null) => void;
}

interface RowProps {
  bar: Bar;
  y: number;
  zero: number;
  xs: (v: number) => number;
  unit: string;
  onHover: (text: string | null) => void;
}

function hoverText(b: Bar, unit: string): string {
  const note = b.note ? ` (${b.note})` : "";
  return b.value === null || !Number.isFinite(b.value) ? `${b.label} undefined${note}` : `${b.label} = ${exact(b.value)} ${unit}${note}`;
}

function BarRow({ bar: b, y, zero, xs, unit, onHover }: RowProps) {
  const text = hoverText(b, unit);
  const v = b.value;
  const finite = v !== null && Number.isFinite(v);
  return (
    <g
      data-bar={b.key}
      data-value={exact(v)}
      tabIndex={0}
      className={`chart-bar${b.emphasis ? " emphasis" : ""}`}
      onMouseEnter={() => onHover(text)}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(text)}
      onBlur={() => onHover(null)}
    >
      <title>{text}</title>
      <rect className="chart-row-hit" x={0} y={y} width={W} height={ROW} />
      <text className="chart-label" x={X0 - 6} y={y + ROW / 2 + 4} textAnchor="end">
        {b.label}
      </text>
      {finite ? (
        <rect className={v >= 0 ? "bar-pos" : "bar-neg"} x={Math.min(zero, xs(v))} y={y + 4} width={Math.max(1, Math.abs(xs(v) - zero))} height={ROW - 8} />
      ) : (
        <text className="chart-undefined" x={zero + 4} y={y + ROW / 2 + 4}>
          undefined
        </text>
      )}
      <text className="chart-value" x={W - 2} y={y + ROW / 2 + 4} textAnchor="end">
        {finite ? fmtSig(v, 4) : "–"}
      </text>
    </g>
  );
}

function MarkerLine({ marker, xs, top, bottom, unit }: { marker: Marker; xs: (v: number) => number; top: number; bottom: number; unit: string }) {
  if (!Number.isFinite(marker.value)) return null;
  return (
    <g className="chart-marker" data-marker={marker.label}>
      <title>{`${marker.label} = ${exact(marker.value)} ${unit}`}</title>
      <line x1={xs(marker.value)} x2={xs(marker.value)} y1={top - 4} y2={bottom + 2} />
      <text x={xs(marker.value)} y={top + 2} textAnchor="middle" dy={-1}>
        {marker.label}
      </text>
    </g>
  );
}

function BarPanel({ id, title, unit, bars, marker, onHover }: PanelProps) {
  const [lo, hi] = domain(bars, marker);
  const xs = (v: number) => X0 + ((v - lo) / (hi - lo)) * (X1 - X0);
  const height = TOP + bars.length * ROW + AXIS;
  const zero = xs(0);
  const yAxis = TOP + bars.length * ROW;
  return (
    <figure className="ks-panel">
      <figcaption>
        {title} <span className="muted">({unit})</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${height}`} width="100%" height={height} data-panel={id} role="img" aria-label={`${title} in ${unit}`}>
        {niceTicks(lo, hi, 4).map((t) => (
          <g key={t} className="chart-tick">
            <line x1={xs(t)} x2={xs(t)} y1={TOP} y2={yAxis + 4} />
            <text x={xs(t)} y={yAxis + 15} textAnchor="middle">
              {tickText(t)}
            </text>
          </g>
        ))}
        <line className="chart-zero" x1={zero} x2={zero} y1={TOP - 2} y2={yAxis + 2} />
        {bars.map((b, k) => (
          <BarRow key={b.key} bar={b} y={TOP + k * ROW} zero={zero} xs={xs} unit={unit} onHover={onHover} />
        ))}
        {marker && <MarkerLine marker={marker} xs={xs} top={TOP} bottom={yAxis} unit={unit} />}
      </svg>
    </figure>
  );
}

export function KsBarChart({ physics: p }: KsBarChartProps) {
  const [hover, setHover] = useState<string | null>(null);
  const ks: Bar[] = [
    { key: "ks1", label: "KS1", value: p.ks1_defined ? p.ks1 : null, note: "closed-shell repulsion" },
    { key: "ks2", label: "KS2", value: p.ks2_defined ? p.ks2 : null, note: `Coulomb, ${p.ks2_detail.form} form` },
    { key: "ks3", label: "KS3", value: p.ks3, note: "filled-empty, beta^2 G", emphasis: true },
  ];
  const g: Bar[] = [{ key: "G", label: "G", value: p.G, note: `exact ${exact(p.G_exact)}` }];
  const primary = p.contact_distance_ang.toFixed(1);
  const beta2: Bar[] = Object.entries(p.beta2_by_distance)
    .sort(([a], [b]) => Number(a) - Number(b))
    .map(([d, v]) => ({ key: `beta2-${d}`, label: `β² ${d} Å`, value: v, emphasis: Number(d).toFixed(1) === primary }));
  return (
    <div className="ks-bar-chart" data-testid="ks-bar-chart">
      <BarPanel id="ks" title="Klopman-Salem terms" unit="eV" bars={ks} marker={{ value: p.floor, label: "floor" }} onHover={setHover} />
      <BarPanel id="G" title="gap kernel G" unit="1/eV" bars={g} onHover={setHover} />
      <BarPanel id="beta2" title="β² by contact distance" unit="eV²" bars={beta2} onHover={setHover} />
      <div className="chart-readout muted" data-testid="ks-bar-readout" aria-live="polite">
        {hover ?? "hover a bar for its exact value; floor = -(KS3 + KS2), higher is better"}
      </div>
    </div>
  );
}
