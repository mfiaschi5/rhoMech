// Import and export popovers of the sketcher.
//
// ImportPanel props: onImport(text) -> error message or null; onClose. A textarea for a SMILES
// or a molfile (also a dropped .mol/.sdf file) and a Load button.
// DOM: <div data-testid="sketch-import-panel"> with sketch-import-text, sketch-import-load,
// sketch-import-error.
//
// ExportPanel props: smiles, molfile, inchi (strings; "" = unavailable), onClose.
// DOM: <div data-testid="sketch-export-panel"> with sketch-export-smiles, -molfile, -inchi
// (read-only textareas), copy buttons and sketch-export-download (the .mol file).
import { useState, type DragEvent } from "react";

export async function readDroppedText(e: DragEvent): Promise<string | null> {
  const file = e.dataTransfer.files?.[0];
  if (file) return file.text();
  const text = e.dataTransfer.getData("text/plain");
  return text || null;
}

export function ImportPanel({ onImport, onClose }: { onImport(text: string): string | null; onClose(): void }) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const load = (t: string) => {
    const err = onImport(t);
    setError(err);
    if (!err) onClose();
  };
  const onDrop = async (e: DragEvent<HTMLTextAreaElement>) => {
    e.preventDefault();
    const t = await readDroppedText(e);
    if (t) {
      setText(t);
      load(t);
    }
  };
  return (
    <div className="sk-popover sk-panel" data-testid="sketch-import-panel">
      <div className="sk-panel-head">
        <strong>Import</strong>
        <span className="muted">SMILES, molfile or SDF text; or drop a .mol / .sdf file</span>
        <button type="button" className="btn btn-small" onClick={onClose}>
          Close
        </button>
      </div>
      <textarea
        data-testid="sketch-import-text"
        rows={6}
        spellCheck={false}
        value={text}
        placeholder="CC(=O)C"
        onChange={(e) => setText(e.target.value)}
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) load(text);
        }}
      />
      <div className="row">
        <button type="button" className="btn btn-primary btn-small" data-testid="sketch-import-load" onClick={() => load(text)}>
          Load
        </button>
        {error && (
          <span className="sk-error" data-testid="sketch-import-error">
            {error}
          </span>
        )}
      </div>
    </div>
  );
}

function Field({ id, label, value, rows }: { id: string; label: string; value: string; rows: number }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(value).then(
      () => setCopied(true),
      () => setCopied(false),
    );
  };
  return (
    <label className="sk-field">
      <span className="sk-field-head">
        {label}
        <button type="button" className="btn btn-small" disabled={!value} onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </span>
      <textarea data-testid={id} readOnly rows={rows} spellCheck={false} value={value || "(unavailable)"} />
    </label>
  );
}

export function ExportPanel(p: { smiles: string; molfile: string; inchi: string; onClose(): void }) {
  const href = `data:chemical/x-mdl-molfile;charset=utf-8,${encodeURIComponent(p.molfile)}`;
  return (
    <div className="sk-popover sk-panel" data-testid="sketch-export-panel">
      <div className="sk-panel-head">
        <strong>Export</strong>
        <a className="btn btn-small" data-testid="sketch-export-download" href={href} download="sketch.mol">
          Download .mol
        </a>
        <button type="button" className="btn btn-small" onClick={p.onClose}>
          Close
        </button>
      </div>
      <Field id="sketch-export-smiles" label="SMILES" value={p.smiles} rows={2} />
      <Field id="sketch-export-inchi" label="InChI" value={p.inchi} rows={2} />
      <Field id="sketch-export-molfile" label="Molfile (V2000)" value={p.molfile} rows={8} />
    </div>
  );
}
