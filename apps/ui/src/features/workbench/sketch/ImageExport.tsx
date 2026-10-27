// Image export of the sketch in its drawing style, at true physical size (exportImage.ts):
// downloads (SVG, PNG at a chosen DPI with the resolution written into the file, vector PDF),
// clipboard copies (the image as PNG plus SVG where the browser allows it, the SVG code, a LaTeX
// figure snippet, chemfig code) and a drag handle that carries the files.
//
// useSchemeFiles() renders the files in the background whenever the drawing, the style or the
// options change (debounced), so a dragstart or a copy can attach them synchronously, inside the
// user gesture, as the browser requires.
//
// Drag (sketch-drag): the DataTransfer gets scheme.png, scheme.pdf and scheme.svg as File items,
// text/html with an <img> (PNG data URL, sized to the physical size) and Chrome's DownloadURL
// (scheme.png). What a target takes depends on the target: see docs/decisions.md (2026-10-25).
//
// DOM: in the export panel, <div data-testid="sketch-image-export" data-width-pt data-height-pt
// data-bond-pt data-line-pt> (the size and the style of the files rendered last)
// with sketch-export-dpi, sketch-export-transparent, sketch-export-cip, sketch-export-svg,
// sketch-export-png, sketch-export-pdf, sketch-copy-image, sketch-copy-svg, sketch-copy-latex,
// sketch-copy-chemfig, and <span data-testid="sketch-copy-status">; on the canvas, the drag
// handle <div data-testid="sketch-drag" draggable data-ready data-line-pt>.
import { useEffect, useRef, useState, type DragEvent } from "react";
import type { SketchModel } from "@ksmech/viz/sketch/model";
import type { StereoLabels } from "@ksmech/viz/sketch/cip";
import type { LabelStyle } from "@ksmech/viz/sketch/drawing";
import { exportLayout, pixelSize, pngWithDpi, sizeCm, toSvg, type ExportLayout } from "@ksmech/viz/sketch/exportImage";
import { pdfBytes, toPdf } from "./pdf";
import { latexFigure, toChemfig } from "./latex";
import type { DrawingStyle } from "@ksmech/viz/sketch/style";

export interface ImageOptions {
  dpi: number;
  transparent: boolean;
  cip: boolean;
}

export const DPI_CHOICES = [150, 300, 600, 1200] as const;
export const DEFAULT_IMAGE_OPTIONS: ImageOptions = { dpi: 300, transparent: false, cip: false };

export interface SchemeFiles {
  layout: ExportLayout;
  svg: string;
  pdf: Uint8Array;
  png: Uint8Array;
  pngUrl: string;
  pngDataUrl: string;
  dpi: number;
  latex: string;
  chemfig: string;
}

async function renderPng(svg: string, l: ExportLayout, dpi: number, transparent: boolean): Promise<Uint8Array> {
  const { w, h } = pixelSize(l, dpi);
  const img = new Image();
  img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  await img.decode();
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2D canvas");
  if (!transparent) {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);
  }
  ctx.drawImage(img, 0, 0, w, h);
  const blob = await new Promise<Blob | null>((res) => canvas.toBlob(res, "image/png"));
  if (!blob) throw new Error("PNG encoding failed");
  return pngWithDpi(new Uint8Array(await blob.arrayBuffer()), dpi);
}

function dataUrl(bytes: Uint8Array, type: string): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return `data:${type};base64,${btoa(bin)}`;
}

export interface SchemeInput {
  model: SketchModel;
  hCounts: ReadonlyMap<number, number> | null;
  style: DrawingStyle;
  labelStyle?: LabelStyle;
  stereo: StereoLabels | null;
  options: ImageOptions;
  /** false: render nothing (a read-only sketch has no export). */
  enabled?: boolean;
}

/** The files of the drawing, rendered `delay` ms after the last change (null while empty or
 *  rendering the first time). */
export function useSchemeFiles(input: SchemeInput, delay = 250): SchemeFiles | null {
  const [files, setFiles] = useState<SchemeFiles | null>(null);
  const { model, hCounts, style, labelStyle, stereo, options, enabled = true } = input;
  useEffect(() => {
    if (!enabled || model.atoms.length === 0) {
      setFiles(null);
      return;
    }
    let live = true;
    const t = window.setTimeout(() => {
      const layout = exportLayout(model, hCounts, style, { cip: options.cip ? stereo : null, labelStyle });
      const bg = options.transparent ? null : "#ffffff";
      const svg = toSvg(layout, { background: bg, title: "ksmech scheme" });
      const pdf = pdfBytes(toPdf(layout, { background: bg, title: "ksmech scheme" }));
      renderPng(svg, layout, options.dpi, options.transparent).then(
        (png) => {
          if (!live) return;
          const pngUrl = URL.createObjectURL(new Blob([png as BlobPart], { type: "image/png" }));
          const latex = latexFigure(layout, "scheme.pdf");
          setFiles({ layout, svg, pdf, png, pngUrl, pngDataUrl: dataUrl(png, "image/png"), dpi: options.dpi, latex, chemfig: toChemfig(model, hCounts, style) });
        },
        (err) => console.warn("sketch image:", err),
      );
    }, delay);
    return () => {
      live = false;
      window.clearTimeout(t);
    };
  }, [model, hCounts, style, labelStyle, stereo, options.dpi, options.transparent, options.cip, delay, enabled]);
  useEffect(() => () => void (files && URL.revokeObjectURL(files.pngUrl)), [files]);
  return files;
}

function download(data: BlobPart, type: string, name: string): void {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

const supports = (type: string): boolean => {
  const ci = (globalThis as { ClipboardItem?: { supports?: (t: string) => boolean } }).ClipboardItem;
  try {
    return !!ci?.supports?.(type);
  } catch {
    return false;
  }
};

/** Copy the image: PNG, plus the SVG as image/svg+xml (or Chrome's "web image/svg+xml") when
 *  the browser can write it. Returns what was written. */
export async function copyImage(f: SchemeFiles): Promise<string[]> {
  const items: Record<string, Blob> = { "image/png": new Blob([f.png as BlobPart], { type: "image/png" }) };
  const svgType = supports("image/svg+xml") ? "image/svg+xml" : supports("web image/svg+xml") ? "web image/svg+xml" : null;
  if (svgType) items[svgType] = new Blob([f.svg], { type: svgType });
  try {
    await navigator.clipboard.write([new ClipboardItem(items)]);
    return Object.keys(items);
  } catch {
    await navigator.clipboard.write([new ClipboardItem({ "image/png": items["image/png"] })]);
    return ["image/png"];
  }
}

/** Marks a drag that starts at the drag handle (the sketcher's own drop ignores it). */
export const SCHEME_DRAG_TYPE = "application/x-ksmech-scheme";

/** Fill a DataTransfer with the scheme: File items (PNG, PDF, SVG), text/html, DownloadURL. */
export function fillDataTransfer(dt: DataTransfer, f: SchemeFiles): void {
  dt.effectAllowed = "copy";
  dt.setData(SCHEME_DRAG_TYPE, "1");
  const add = (data: BlobPart, name: string, type: string) => {
    try {
      dt.items.add(new File([data], name, { type }));
    } catch {
      // a browser that refuses File items in dragstart still gets the HTML and DownloadURL
    }
  };
  add(f.png as BlobPart, "scheme.png", "image/png");
  add(f.pdf as BlobPart, "scheme.pdf", "application/pdf");
  add(f.svg, "scheme.svg", "image/svg+xml");
  const w = Math.round(f.layout.widthPt * (4 / 3));
  const h = Math.round(f.layout.heightPt * (4 / 3));
  dt.setData("text/html", `<img src="${f.pngDataUrl}" width="${w}" height="${h}" alt="chemical structure">`);
  dt.setData("DownloadURL", `image/png:scheme.png:${f.pngUrl}`);
}

export function DragHandle({ files }: { files: SchemeFiles | null }) {
  const imgRef = useRef<HTMLImageElement>(null);
  const onDragStart = (e: DragEvent<HTMLDivElement>) => {
    if (!files) {
      e.preventDefault();
      return;
    }
    fillDataTransfer(e.dataTransfer, files);
    if (imgRef.current) e.dataTransfer.setDragImage(imgRef.current, 8, 8);
  };
  const title = files
    ? "Drag the drawing (PNG, PDF, SVG files at true size) into Word, Overleaf's file tree, a file manager or another app"
    : "Preparing the image…";
  return (
    <div className="sk-drag" data-testid="sketch-drag" draggable={!!files} data-ready={files ? "true" : "false"} data-line-pt={files?.layout.lineWidthPt} title={title} onDragStart={onDragStart}>
      {files ? <img ref={imgRef} src={files.pngDataUrl} alt="" draggable={false} /> : <span aria-hidden="true">⠿</span>}
    </div>
  );
}

/** The image options: PNG resolution, transparent background, CIP labels. */
function OptionsRow({ options, onOptions }: { options: ImageOptions; onOptions(o: ImageOptions): void }) {
  return (
    <div className="row sk-image-options">
      <label title="PNG resolution (written into the file, so Word places it at its true size)">
        PNG dpi{" "}
        <select data-testid="sketch-export-dpi" value={options.dpi} onChange={(e) => onOptions({ ...options, dpi: Number(e.target.value) })}>
          {DPI_CHOICES.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </label>
      <label>
        <input type="checkbox" data-testid="sketch-export-transparent" checked={options.transparent} onChange={(e) => onOptions({ ...options, transparent: e.target.checked })} /> transparent
      </label>
      <label>
        <input type="checkbox" data-testid="sketch-export-cip" checked={options.cip} onChange={(e) => onOptions({ ...options, cip: e.target.checked })} /> R/S labels
      </label>
    </div>
  );
}

export function ImageExportSection(p: { files: SchemeFiles | null; options: ImageOptions; onOptions(o: ImageOptions): void; fileStem?: string }) {
  const { files, options } = p;
  const [status, setStatus] = useState("");
  const stem = p.fileStem ?? "scheme";
  const run = (label: string, fn: () => Promise<unknown> | void) => async () => {
    try {
      const r = await fn();
      setStatus(typeof r === "string" ? r : label);
    } catch (err) {
      setStatus(`failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };
  const cm = files ? sizeCm(files.layout) : null;
  const btn = (id: string, label: string, title: string, fn: () => Promise<unknown> | void, done: string) => (
    <button type="button" className="btn btn-small" data-testid={id} disabled={!files} title={title} onClick={run(done, fn)}>
      {label}
    </button>
  );
  return (
    <div
      className="sk-image-export"
      data-testid="sketch-image-export"
      data-width-pt={files?.layout.widthPt.toFixed(2)}
      data-height-pt={files?.layout.heightPt.toFixed(2)}
      data-bond-pt={files?.layout.bondLengthPt}
      data-line-pt={files?.layout.lineWidthPt}
    >
      <div className="sk-field-head">
        <span>Image, in the drawing style at true size{cm ? ` (${cm.w.toFixed(2)} × ${cm.h.toFixed(2)} cm)` : ""}</span>
      </div>
      <OptionsRow options={options} onOptions={p.onOptions} />
      <div className="row sk-image-buttons">
        {files && btn("sketch-export-svg", "SVG", "Download the SVG (size in pt)", () => download(files.svg, "image/svg+xml", `${stem}.svg`), "SVG saved")}
        {files && btn("sketch-export-png", "PNG", `Download the PNG at ${options.dpi} dpi`, () => download(files.png as BlobPart, "image/png", `${stem}.png`), "PNG saved")}
        {files && btn("sketch-export-pdf", "PDF", "Download a vector PDF (Helvetica, nothing embedded): for LaTeX / Overleaf", () => download(files.pdf as BlobPart, "application/pdf", `${stem}.pdf`), "PDF saved")}
        {files && btn("sketch-copy-image", "Copy image", "Copy the image (PNG, and SVG where the browser allows) to paste into Word, PowerPoint…", () => copyImage(files).then((t) => `copied ${t.join(" + ")}`), "")}
        {files && btn("sketch-copy-svg", "Copy SVG", "Copy the SVG code", () => navigator.clipboard.writeText(files.svg), "SVG code copied")}
        {files && btn("sketch-copy-latex", "Copy LaTeX", "Copy a figure snippet that includes scheme.pdf at its physical width (upload the PDF to Overleaf)", () => navigator.clipboard.writeText(files.latex), "LaTeX snippet copied")}
        {files && btn("sketch-copy-chemfig", "Copy chemfig", "Copy chemfig code that redraws the molecule in LaTeX (\\usepackage{chemfig}); no radical dots", () => navigator.clipboard.writeText(files.chemfig), "chemfig code copied")}
        {!files && <span className="muted">preparing…</span>}
      </div>
      <span className="muted sk-copy-status" data-testid="sketch-copy-status" aria-live="polite">
        {status}
      </span>
    </div>
  );
}
