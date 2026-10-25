// InfoBar: the RDKit verdict under the canvas: canonical SMILES (or the sanitization error) and
// formula, MW, logP, TPSA, HBD/HBA, and the report of the last Clean / 3D->2D.
//
// Props: info (useSketchInfo), report (the last layout action, useLayoutActions; null hides).
// DOM: <div data-testid="sketch-info" data-status data-valid> with sketch-info-smiles,
// sketch-info-error, <span data-testid="sketch-desc-<key>"> per descriptor and
// <div data-testid="sketch-depict-report" data-kind (clean | project) data-method data-template
// data-stereo (none | kept | mismatch) data-score data-label-overlaps data-close-contacts
// data-crossings data-view-residual data-fallback (true when the browser laid it out)
// data-error>.
import type { SketchInfo } from "./useSketchInfo";
import type { LayoutReport } from "./useLayoutActions";

function describe(r: LayoutReport): string {
  const what = r.kind === "project" ? "3D→2D" : "Clean";
  if (r.error) return `${what} failed: ${r.error}`;
  if (!r.result) return `${what}: ${r.fallback ?? ""}`;
  const o = r.result;
  const d = o.defects;
  const how = o.template ? `${o.style} template ${o.template}` : o.method;
  const fit = o.view_residual === null ? "" : ` · view fit ${o.view_residual.toFixed(2)}`;
  const stereo = { none: "", kept: " · stereo kept", mismatch: " · stereo differs!" }[o.stereo];
  const notes = o.notes.length ? ` · ${o.notes.join("; ")}` : "";
  return (
    `${what} (${how}): score ${d.score.toFixed(1)} · label overlaps ${d.label_overlaps} · close contacts ` +
    `${d.close_contacts} · atom on bond ${d.atom_on_bond} · crossings ${d.crossings}${fit}${stereo}${notes}`
  );
}

function DepictReport({ r }: { r: LayoutReport }) {
  const o = r.result;
  const bad = !!r.error || o?.stereo === "mismatch";
  return (
    <div
      className={`sk-fix-report${bad ? " sk-error" : ""}`}
      data-testid="sketch-depict-report"
      data-kind={r.kind}
      data-method={o?.method ?? "browser-coordgen"}
      data-template={o?.template ?? ""}
      data-stereo={o?.stereo ?? "none"}
      data-score={o ? o.defects.score.toFixed(3) : ""}
      data-label-overlaps={o?.defects.label_overlaps ?? ""}
      data-close-contacts={o?.defects.close_contacts ?? ""}
      data-crossings={o?.defects.crossings ?? ""}
      data-view-residual={o?.view_residual ?? ""}
      data-fallback={r.fallback ? "true" : "false"}
      data-error={r.error ?? ""}
    >
      {describe(r)}
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
      {report && <DepictReport r={report} />}
    </div>
  );
}
