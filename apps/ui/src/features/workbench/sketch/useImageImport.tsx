// The sketcher's image import (OCR): one image (dropped, pasted or picked in the import panel)
// sent to the page's OcrSource; its molfile replaces the drawing as one undoable step through
// the usual import path (importSketch: the image's 2D layout kept, scaled to the standard bond
// length), else its SMILES. The notice under the canvas tells what happened: busy, the result
// with confidence and warnings, the error, or why image import is off.
//
// DOM: <div data-testid="sketch-ocr" data-state="busy|done|error|off" data-engine
// data-confidence data-warnings> with sketch-ocr-text, sketch-ocr-warning (one per warning) and
// sketch-ocr-dismiss.
import { useEffect, useRef, useState } from "react";
import { imageProblem, LOW_CONFIDENCE, ocrSummary, type OcrResult, type OcrSource } from "./imageImport";

export type OcrState =
  | { state: "idle" }
  | { state: "busy"; name: string }
  | { state: "done"; result: OcrResult }
  | { state: "error"; message: string; code: string }
  | { state: "off"; message: string };

export interface ImageImport {
  ocr: OcrState;
  /** Send one image to OCR (or explain why not). */
  run(file: File): void;
  dismiss(): void;
  /** False without an OcrSource or while the capability is off (the reason is in `offReason`). */
  available: boolean;
  offReason: string | null;
}

/** The error of a failed recognition as code and message (an ApiError or any Error). */
function describeError(err: unknown): { code: string; message: string } {
  const e = err as { code?: unknown; message?: unknown };
  const code = typeof e?.code === "string" ? e.code : "CLIENT_ERROR";
  return { code, message: typeof e?.message === "string" ? e.message : String(err) };
}

/** Import the molfile (the image's layout), else the SMILES; the error, or null. */
function importResult(r: OcrResult, importText: (text: string) => string | null): string | null {
  const err = r.molfile.trim() ? importText(r.molfile) : "no molfile";
  if (!err) return null;
  return r.smiles.trim() ? importText(r.smiles) && err : err;
}

export function useImageImport(source: OcrSource | undefined, importText: (text: string) => string | null): ImageImport {
  const [ocr, setOcr] = useState<OcrState>({ state: "idle" });
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);
  const offReason = !source ? "image import (OCR) is not available here" : source.enabled ? null : `image import (OCR) is off: ${source.reason ?? "no reason given"}`;
  const run = (file: File) => {
    if (!source || !source.enabled) return setOcr({ state: "off", message: offReason ?? "" });
    const problem = imageProblem(file);
    if (problem) return setOcr({ state: "error", code: "BAD_IMAGE", message: problem });
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    setOcr({ state: "busy", name: file.name || "image" });
    source.recognize(file, ctl.signal).then(
      (result) => {
        if (ctl.signal.aborted) return;
        const err = importResult(result, importText);
        setOcr(err ? { state: "error", code: "OCR_IMPORT", message: `the recognized structure cannot be drawn: ${err}` } : { state: "done", result });
      },
      (err: unknown) => {
        if (ctl.signal.aborted) return;
        setOcr({ state: "error", ...describeError(err) });
      },
    );
  };
  return { ocr, run, dismiss: () => setOcr({ state: "idle" }), available: !offReason, offReason };
}

export function OcrNotice({ ocr, onDismiss }: { ocr: OcrState; onDismiss(): void }) {
  if (ocr.state === "idle") return null;
  const r = ocr.state === "done" ? ocr.result : null;
  const low = r?.confidence !== null && r?.confidence !== undefined && r.confidence < LOW_CONFIDENCE;
  const bad = ocr.state === "error" || ocr.state === "off" || low;
  const text =
    ocr.state === "busy" ? `Reading ${ocr.name}…` : ocr.state === "done" ? ocrSummary(ocr.result) : ocr.state === "error" ? `Image import failed (${ocr.code}): ${ocr.message}` : ocr.message;
  return (
    <div
      className={`sk-ocr${bad ? " sk-ocr-warn" : ""}`}
      data-testid="sketch-ocr"
      data-state={ocr.state}
      data-engine={r?.engine ?? ""}
      data-confidence={r?.confidence ?? ""}
      data-warnings={r?.warnings.length ?? 0}
      role={ocr.state === "error" ? "alert" : "status"}
    >
      {ocr.state === "busy" && <span className="spinner" aria-hidden="true" />}
      <span data-testid="sketch-ocr-text">{text}</span>
      {low && <span className="badge badge-warn">low confidence: check the drawing</span>}
      {r?.warnings.map((w, k) => (
        <span key={k} className="sk-ocr-warning" data-testid="sketch-ocr-warning">
          {w}
        </span>
      ))}
      {ocr.state !== "busy" && (
        <button type="button" className="btn btn-small" data-testid="sketch-ocr-dismiss" aria-label="dismiss" onClick={onDismiss}>
          ×
        </button>
      )}
    </div>
  );
}
