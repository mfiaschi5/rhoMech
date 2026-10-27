// Import and export popovers of the sketcher.
//
// ImportPanel props: onImport(text) -> error message or null; onClose; onImage(file) (OCR,
// useImageImport.tsx) with imageOff (why image import is off, null when on). A textarea for a
// SMILES or a molfile (also a dropped .mol/.sdf file; a dropped or pasted image goes to OCR), a
// Load button and the "Image (OCR)" file picker.
// DOM: <div data-testid="sketch-import-panel"> with sketch-import-text, sketch-import-load,
// sketch-import-error, <input type=file data-testid="sketch-import-image" disabled while image
// import is off> and sketch-import-image-off (the reason).
//
// ExportPanel props: smiles, molfile, inchi (strings; "" = unavailable), onClose, children (the
// image export section, ImageExport.tsx, shown first).
// DOM: <div data-testid="sketch-export-panel"> with sketch-export-smiles, -molfile, -inchi
// (read-only textareas), copy buttons and sketch-export-download (the .mol file).
import { useState, type ClipboardEvent, type DragEvent, type ReactNode } from "react";
import { isImageFile, OCR_IMAGE_TYPES, pickImage, pickPaste } from "./imageImport";

/** The text of a drop (a .mol / .sdf file or dragged text); null for an image or nothing. */
export async function readDroppedText(e: DragEvent): Promise<string | null> {
  const file = e.dataTransfer.files?.[0];
  if (file) return isImageFile(file) ? null : file.text();
  const text = e.dataTransfer.getData("text/plain");
  return text || null;
}

export interface ImportPanelProps {
  onImport(text: string): string | null;
  onClose(): void;
  onImage(file: File): void;
  imageOff: string | null;
}

export function ImportPanel({ onImport, onClose, onImage, imageOff }: ImportPanelProps) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const load = (t: string) => {
    const err = onImport(t);
    setError(err);
    if (!err) onClose();
  };
  const image = (file: File) => {
    onImage(file);
    onClose();
  };
  const onPaste = (e: ClipboardEvent<HTMLTextAreaElement>) => {
    const pick = pickPaste(e.clipboardData);
    if (pick?.kind !== "image") return;
    e.preventDefault();
    image(pick.file);
  };
  const onDrop = async (e: DragEvent<HTMLTextAreaElement>) => {
    e.preventDefault();
    const img = pickImage(e.dataTransfer);
    if (img) return image(img);
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
        <span className="muted">SMILES, molfile or SDF text; or drop a .mol / .sdf file, or an image of a structure (OCR)</span>
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
        onPaste={onPaste}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) load(text);
        }}
      />
      <div className="row">
        <button type="button" className="btn btn-primary btn-small" data-testid="sketch-import-load" onClick={() => load(text)}>
          Load
        </button>
        <label className={`btn btn-small sk-file${imageOff ? " disabled" : ""}`} title={imageOff ?? "Read a structure from a picture (scan, screenshot, figure): PNG, JPEG, WebP, GIF, BMP or TIFF, up to 10 MB. Also: drop an image on the canvas, or paste one (Ctrl+V)"}>
          Image (OCR)…
          <input
            type="file"
            data-testid="sketch-import-image"
            accept={OCR_IMAGE_TYPES.join(",")}
            disabled={!!imageOff}
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) image(file);
            }}
          />
        </label>
        {error && (
          <span className="sk-error" data-testid="sketch-import-error">
            {error}
          </span>
        )}
      </div>
      {imageOff && (
        <div className="muted" data-testid="sketch-import-image-off">
          {imageOff}
        </div>
      )}
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

export function ExportPanel(p: { smiles: string; molfile: string; inchi: string; onClose(): void; children?: ReactNode }) {
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
      {p.children}
      <Field id="sketch-export-smiles" label="SMILES" value={p.smiles} rows={2} />
      <Field id="sketch-export-inchi" label="InChI" value={p.inchi} rows={2} />
      <Field id="sketch-export-molfile" label="Molfile (V2000)" value={p.molfile} rows={8} />
    </div>
  );
}
