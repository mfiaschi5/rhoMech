// InfoBar: the RDKit verdict under the canvas: canonical SMILES (or the sanitization error) and
// formula, MW, logP, TPSA, HBD/HBA.
//
// Props: info (useSketchInfo), report (the last projection / fix, useLayoutActions; null hides).
// DOM: <div data-testid="sketch-info" data-status data-valid> with sketch-info-smiles,
// sketch-info-error, <span data-testid="sketch-desc-<key>"> per descriptor and
// <div data-testid="sketch-fix-report" data-kind data-label-overlaps data-close-contacts
// data-atom-on-bond data-crossings data-bond-rms data-stereo> (values after the fix).
import type { SketchInfo } from "./useSketchInfo";
import type { LayoutReport } from "./useLayoutActions";

function FixReport({ r }: { r: LayoutReport }) {
  const { before: b, after: a } = r;
  const pair = (x: number, y: number) => (x === y ? String(y) : `${x}→${y}`);
  const tilt = r.tilt && (r.tilt[0] || r.tilt[1]) ? ` (view tilted ${r.tilt[0]}°, ${r.tilt[1]}°)` : "";
  const stereo = { none: "", kept: " · stereo kept", repaired: " · stereo repaired", mismatch: " · stereo differs!" }[r.stereo];
  return (
    <div
      className={`sk-fix-report${r.stereo === "mismatch" ? " sk-error" : ""}`}
      data-testid="sketch-fix-report"
      data-kind={r.kind}
      data-label-overlaps={a.labelOverlaps}
      data-close-contacts={a.closeContacts}
      data-atom-on-bond={a.atomOnBond}
      data-crossings={a.crossings}
      data-bond-rms={a.bondLengthRms.toFixed(3)}
      data-stereo={r.stereo}
    >
      <span className="muted">{r.kind === "project" ? `3D→2D${tilt}:` : "Fix layout:"}</span> label overlaps {pair(b.labelOverlaps, a.labelOverlaps)} · close
      contacts {pair(b.closeContacts, a.closeContacts)} · atom on bond {pair(b.atomOnBond, a.atomOnBond)} · crossings {pair(b.crossings, a.crossings)} · bond
      RMS {b.bondLengthRms.toFixed(2)}→{a.bondLengthRms.toFixed(2)}
      {stereo}
    </div>
  );
}

function fmt(v: number | undefined, digits: number): string {
  if (v === undefined || !Number.isFinite(v)) return "–";
  const s = v.toFixed(digits);
  return Number(s) === 0 ? (0).toFixed(digits) : s; // never "-0.00"
}

export function InfoBar({ info, report = null }: { info: SketchInfo; report?: LayoutReport | null }) {
  const d = info.descriptors;
  const items: [string, string, string][] = d
    ? [
        ["mw", "MW", fmt(d.amw, 2)],
        ["logp", "logP", fmt(d.CrippenClogP, 2)],
        ["tpsa", "TPSA", fmt(d.tpsa, 1)],
        ["hbd", "HBD", fmt(d.NumHBD, 0)],
        ["hba", "HBA", fmt(d.NumHBA, 0)],
      ]
    : [];
  return (
    <div
      className={`sk-info${info.stale ? " stale" : ""}`}
      data-testid="sketch-info"
      data-status={info.status}
      data-valid={info.valid}
    >
      {info.status === "empty" && <span className="muted">Draw a molecule, or paste a SMILES or molfile (Ctrl+V).</span>}
      {info.status === "loading" && <span className="muted">Loading RDKit…</span>}
      {info.error && (
        <span className="sk-error" data-testid="sketch-info-error">
          {info.error}
        </span>
      )}
      {info.smiles && (
        <code className="sk-smiles" data-testid="sketch-info-smiles" title="Canonical SMILES (RDKit)">
          {info.smiles}
        </code>
      )}
      {info.formula && (
        <span className="sk-desc" data-testid="sketch-desc-formula">
          {info.formula}
        </span>
      )}
      {items.map(([key, label, value]) => (
        <span key={key} className="sk-desc" data-testid={`sketch-desc-${key}`}>
          <span className="muted">{label}</span> {value}
        </span>
      ))}
      {report && <FixReport r={report} />}
    </div>
  );
}
