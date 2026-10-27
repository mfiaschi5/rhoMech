// StepTextbook (step viewport variant A): the step as a publication scheme, reactants -> product
// in one row in the ACS 1996 style with circled charges (StepScheme), lone pairs where the step
// touches, routed curved arrows, the predictor and score over the reaction arrow, and the orbital
// reading as a figure caption under it (hover a caption line: its bond is haloed). R marks can be
// shown on the reactants. Export: SVG and PNG files (300 dpi, the true ACS size) and clipboard
// copies, from stepExport.ts.
//
// DOM: predict-step-textbook > the StepScheme ids, predict-step-show-r, predict-step-export-svg,
// predict-step-export-png, predict-step-copy-image, predict-step-copy-svg,
// predict-step-export-status, predict-step-caption > li[data-testid=predict-step-caption-<k>].
import { useState } from "react";
import { ACS_1996_STYLE, type DrawingStyle } from "../sketch/style";
import { pngWithDpi } from "../sketch/exportImage";
import { schemeSvg } from "./stepExport";
import { StepScheme } from "./StepScheme";
import { ProductView } from "./ProductView";
import { hoverOf, type HoverSet, type ShownStep, type StepSides } from "./stepSides";
import { captionParts, dominantReading } from "./stepText";
import type { OrbitalReadingOut } from "../v9types";

/** ACS 1996 lines, fonts and spacing with circled charges, written-out methyls and a 17 pt bond
 *  (the 14.4 pt bond leaves only stubs between two labels). */
export const TEXTBOOK_STYLE: DrawingStyle = { ...ACS_1996_STYLE, bondLengthPt: 17, chargePlacement: "circled", labelMethyls: true };

export interface StepVariantProps {
  sides: StepSides;
  step: ShownStep;
  prefix: string;
  productSmiles: readonly string[];
  sigmaKeys: readonly string[];
  label: (i: number) => string;
  above: string;
  below: string;
  hover: HoverSet;
  onHover: (h: HoverSet | null) => void;
}

/** The reading's caption lines, dominant channel first. */
export function StepCaption({ readings, label, onHover, prefix }: { readings: readonly OrbitalReadingOut[]; label: (i: number) => string; onHover: (h: HoverSet | null) => void; prefix: string }) {
  if (!readings.length) return null;
  const top = dominantReading(readings);
  const order = [top, ...readings.map((_, k) => k).filter((k) => k !== top)];
  return (
    <ol className="ss-caption" data-testid={`${prefix}-caption`}>
      {order.map((k) => {
        const r = readings[k];
        const p = captionParts(r);
        return (
          <li key={k} data-testid={`${prefix}-caption-${k}`} onPointerEnter={() => onHover(hoverOf([r.i, r.j], [[r.i, r.j]]))} onPointerLeave={() => onHover(null)}>
            <span className="ss-cap-bond">
              {label(r.i)}–{label(r.j)}
            </span>
            <span className="ss-cap-channel">{p.channel}</span>
            <span className="ss-cap-sep">·</span>
            <span className="ss-cap-g">
              <i>G</i>
              <sup>atom</sup> {p.g.replace(/^G\^atom /, "")}
            </span>
            {p.canonical && (
              <>
                <span className="ss-cap-sep">·</span>
                <span className="ss-cap-mo">{p.canonical}</span>
              </>
            )}
          </li>
        );
      })}
    </ol>
  );
}

async function svgToPng(svg: string, wPt: number, hPt: number, dpi: number): Promise<Blob> {
  const [w, h] = [Math.max(1, Math.round((wPt / 72) * dpi)), Math.max(1, Math.round((hPt / 72) * dpi))];
  const img = new Image();
  img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  await img.decode();
  const canvas = document.createElement("canvas");
  [canvas.width, canvas.height] = [w, h];
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2D canvas");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
  const blob = await new Promise<Blob | null>((res) => canvas.toBlob(res, "image/png"));
  if (!blob) throw new Error("PNG encoding failed");
  return new Blob([pngWithDpi(new Uint8Array(await blob.arrayBuffer()), dpi) as BlobPart], { type: "image/png" });
}

function download(data: Blob, name: string): void {
  const url = URL.createObjectURL(data);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

function ExportBar({ props, prefix }: { props: StepVariantProps; prefix: string }) {
  const [status, setStatus] = useState("");
  const file = () => {
    const { source, product, aligned } = props.sides;
    if (!source) return null;
    const input = { source, product, aligned, matrix: props.step.matrix, arrows: props.step.arrows };
    return schemeSvg(input, TEXTBOOK_STYLE, { above: props.above, below: props.below, background: "#ffffff", labelCarbons: false });
  };
  const run = (what: string, job: () => Promise<void>) => job().then(() => setStatus(what), (err) => setStatus(`failed: ${String(err)}`));
  const svg = () => run("SVG saved", async () => void download(new Blob([file()?.svg ?? ""], { type: "image/svg+xml" }), "step.svg"));
  const png = () => run("PNG saved (300 dpi)", async () => {
    const f = file();
    if (f) download(await svgToPng(f.svg, f.widthPt, f.heightPt, 300), "step.png");
  });
  const copyImage = () => run("image copied", async () => {
    const f = file();
    if (!f) return;
    await navigator.clipboard.write([new ClipboardItem({ "image/png": svgToPng(f.svg, f.widthPt, f.heightPt, 300) })]);
  });
  const copySvg = () => run("SVG code copied", async () => navigator.clipboard.writeText(file()?.svg ?? ""));
  return (
    <span className="ss-export" role="group" aria-label="export the scheme">
      <button type="button" className="btn btn-small" data-testid={`${prefix}-export-svg`} onClick={svg} title="download the scheme as SVG (ACS 1996 lines and fonts, true size)">
        SVG
      </button>
      <button type="button" className="btn btn-small" data-testid={`${prefix}-export-png`} onClick={png} title="download a 300 dpi PNG">
        PNG
      </button>
      <button type="button" className="btn btn-small" data-testid={`${prefix}-copy-image`} onClick={copyImage} title="copy the scheme as an image">
        Copy
      </button>
      <button type="button" className="btn btn-small" data-testid={`${prefix}-copy-svg`} onClick={copySvg} title="copy the SVG code">
        Copy SVG
      </button>
      <span className="muted ss-export-status" data-testid={`${prefix}-export-status`} aria-live="polite">
        {status}
      </span>
    </span>
  );
}

export function StepTextbook(props: StepVariantProps) {
  const { sides, step, prefix } = props;
  const [showR, setShowR] = useState(false);
  return (
    <div className="ss-textbook" data-testid={`${prefix}-textbook`}>
      <div className="ss-frame">
        <div className="ss-frame-tools">
          <label className="field-inline pv-small" title="the reaction matrix on the reactants: forming bonds dashed, breaking bonds crossed, lone-pair and charge changes">
            <input type="checkbox" data-testid={`${prefix}-show-r`} checked={showR} onChange={(e) => setShowR(e.target.checked)} />
            R marks
          </label>
          <ExportBar props={props} prefix={prefix} />
        </div>
        {sides.source ? (
          <StepScheme sides={sides} step={step} prefix={prefix} style={TEXTBOOK_STYLE} mono showR={showR || !step.arrows.length} hover={props.hover} onHover={props.onHover} above={props.above} below={props.below} productSmiles={props.productSmiles.join(".")} variant="textbook" labelCarbons={false} fit={{ maxBondPx: 52, maxHeight: 380 }} />
        ) : (
          <div className="pv-canvas-placeholder muted">{sides.loading ? "loading the depiction…" : (sides.error ?? "no depiction")}</div>
        )}
        {sides.source && !sides.product && !sides.loading && <ProductView smiles={props.productSmiles} prefix={`${prefix}-fallback`} />}
        <StepCaption readings={step.readings} label={props.label} onHover={props.onHover} prefix={prefix} />
      </div>
    </div>
  );
}
