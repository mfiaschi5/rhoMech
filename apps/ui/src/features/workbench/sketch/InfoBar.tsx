// InfoBar: the RDKit verdict under the canvas: canonical SMILES (or the sanitization error) and
// formula, MW, logP, TPSA, HBD/HBA.
//
// Props: info (useSketchInfo).
// DOM: <div data-testid="sketch-info" data-status data-valid> with sketch-info-smiles,
// sketch-info-error and <span data-testid="sketch-desc-<key>"> per descriptor.
import type { SketchInfo } from "./useSketchInfo";

function fmt(v: number | undefined, digits: number): string {
  if (v === undefined || !Number.isFinite(v)) return "–";
  const s = v.toFixed(digits);
  return Number(s) === 0 ? (0).toFixed(digits) : s; // never "-0.00"
}

export function InfoBar({ info }: { info: SketchInfo }) {
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
    </div>
  );
}
