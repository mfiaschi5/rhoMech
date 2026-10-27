// @ts-nocheck -- run by `node --experimental-strip-types --test` (see model.test.ts).
// Publication export (exportImage.ts, pdf.ts, latex.ts): true physical size of the SVG and the
// PDF, a well-formed deterministic PDF, the PNG resolution chunk, the LaTeX figure snippet and
// chemfig code for a few molecules (RDKit's 2D layout via importSketch).
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { addAtom, addBond, emptySketch, setCharge } from "@ksmech/viz/sketch/model";
import { crc32, exportLayout, pixelSize, pngWithDpi, sizeCm, toSvg } from "@ksmech/viz/sketch/exportImage";
import { pdfBytes, toPdf } from "./pdf.ts";
import { latexFigure, toChemfig } from "./latex.ts";
import { importSketch } from "./rdkitSketch.ts";
import { ACS_1996_STYLE, TOTAL_SYNTHESIS_STYLE } from "@ksmech/viz/sketch/style";

const wasmJs = fileURLToPath(new URL("../../../../node_modules/@rdkit/rdkit/dist/RDKit_minimal.js", import.meta.url));
const rdOpts = { skip: existsSync(wasmJs) ? false : "RDKit.js not installed (npm --prefix apps/ui install)" };
let rdkitPromise = null;
const rdkit = () => (rdkitPromise ??= createRequire(import.meta.url)(wasmJs)());

/** Ethane drawn horizontally: one standard bond, two bare vertices. */
function ethane() {
  let m = emptySketch();
  let a, b;
  [m, a] = addAtom(m, "C", 0, 0);
  [m, b] = addAtom(m, "C", 1.5, 0);
  [m] = addBond(m, a, b, 1);
  return m;
}

const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

test("the export is at true size: an ACS bond is 14.4 pt (0.508 cm) in the SVG", () => {
  const l = exportLayout(ethane(), null, ACS_1996_STYLE, { padPt: 2 });
  // 14.4 pt of bond + half a line width of round cap at each end + 2 pt margin each side
  assert.ok(near(l.widthPt, 14.4 + 0.6 + 4), `width ${l.widthPt}`);
  assert.ok(near(l.heightPt, 0.6 + 4), `height ${l.heightPt}`);
  const svg = toSvg(l);
  assert.match(svg, /width="19pt" height="4.6pt" viewBox="0 0 19 4.6"/);
  const line = l.prims.find((p) => p.k === "line");
  assert.ok(near(Math.hypot(line.q[0] - line.p[0], line.q[1] - line.p[1]), 14.4));
  assert.ok(near(line.w, 0.6));
  assert.ok(near(sizeCm(l).w, (19 / 72) * 2.54));
  assert.deepEqual(pixelSize(l, 600), { w: Math.round((19 / 72) * 600), h: Math.round((4.6 / 72) * 600) });
  const ts = exportLayout(ethane(), null, TOTAL_SYNTHESIS_STYLE, { padPt: 0 });
  assert.ok(near(ts.widthPt, 15 + 1.1));
});

test("labels, charges and the background in the SVG", () => {
  let m = emptySketch();
  let a, b;
  [m, a] = addAtom(m, "C", 0, 0);
  [m, b] = addAtom(m, "N", 1.5, 0);
  [m] = addBond(m, a, b, 1);
  m = setCharge(m, b, 1);
  const svg = toSvg(exportLayout(m, new Map([[a, 3], [b, 3]]), ACS_1996_STYLE), { background: "#ffffff" });
  assert.match(svg, /<rect width="[\d.]+" height="[\d.]+" fill="#ffffff"\/>/);
  assert.match(svg, />N<\/text>/);
  assert.match(svg, />H<\/text>/);
  assert.match(svg, />3<\/text>/);
  assert.match(svg, />\+<\/text>/);
  assert.match(svg, /font-family="Arial, Helvetica/);
  assert.doesNotMatch(svg, /dominant-baseline|sk-|data-testid/, "no UI classes, explicit baselines");
});

test("the PDF is well formed, deterministic and its page is the physical size", () => {
  const l = exportLayout(ethane(), null, ACS_1996_STYLE, { padPt: 2 });
  const pdf = toPdf(l);
  assert.equal(pdf, toPdf(l), "deterministic");
  assert.ok(pdf.startsWith("%PDF-1.4\n"));
  assert.ok(pdf.endsWith("%%EOF\n"));
  assert.match(pdf, /\/MediaBox \[0 0 19 4\.6\]/);
  const startxref = Number(/startxref\n(\d+)\n%%EOF\n$/.exec(pdf)[1]);
  assert.ok(pdf.slice(startxref).startsWith("xref\n0 9\n"));
  const offsets = [...pdf.slice(startxref).matchAll(/^(\d{10}) 00000 n $/gm)].map((x) => Number(x[1]));
  assert.equal(offsets.length, 8);
  offsets.forEach((o, i) => assert.ok(pdf.startsWith(`${i + 1} 0 obj\n`, o), `object ${i + 1} at its xref offset`));
  const stream = /<< \/Length (\d+) >>\nstream\n([\s\S]*?)endstream/.exec(pdf);
  assert.equal(Number(stream[1]), stream[2].length, "stream length");
  assert.match(stream[2], /0\.6 w 2\.3 2\.3 m 16\.7 2\.3 l S/, "the bond as a stroked line, y flipped");
  assert.match(pdf, /\/BaseFont \/Helvetica /);
  const bytes = pdfBytes(pdf);
  assert.equal(bytes.length, pdf.length);
  assert.ok([...pdf].every((c) => c.charCodeAt(0) < 256));
});

test("PDF text: Helvetica, escaped strings, the minus sign as an en dash", () => {
  let m = emptySketch();
  let a;
  [m, a] = addAtom(m, "O", 0, 0);
  m = setCharge(m, a, -1);
  const pdf = toPdf(exportLayout(m, new Map([[a, 1]]), ACS_1996_STYLE));
  assert.match(pdf, /\/F1 10 Tf 1 0 0 1 [\d.]+ [\d.]+ Tm \(O\) Tj/);
  assert.match(pdf, /\(\\226\) Tj/);
});

test("pngWithDpi writes a pHYs chunk after IHDR and keeps the rest", () => {
  const chunk = (type, data) => {
    const body = new Uint8Array([...type].map((c) => c.charCodeAt(0)).concat([...data]));
    const out = new Uint8Array(12 + data.length);
    const v = new DataView(out.buffer);
    v.setUint32(0, data.length);
    out.set(body, 4);
    v.setUint32(8 + data.length, crc32(body));
    return out;
  };
  assert.equal(crc32(new TextEncoder().encode("IEND")), 0xae426082);
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  const png = new Uint8Array([...sig, ...chunk("IHDR", new Uint8Array(13)), ...chunk("pHYs", new Uint8Array(9)), ...chunk("IEND", new Uint8Array(0))]);
  const out = pngWithDpi(png, 300);
  const v = new DataView(out.buffer);
  const at = 8 + 25; // after IHDR
  assert.equal(String.fromCharCode(...out.subarray(at + 4, at + 8)), "pHYs");
  assert.equal(v.getUint32(at + 8), Math.round(300 / 0.0254));
  assert.equal(out[at + 16], 1);
  assert.equal(v.getUint32(at + 17), crc32(out.subarray(at + 4, at + 17)));
  assert.equal(out.length, png.length, "the old pHYs replaced, not added");
  assert.equal(String.fromCharCode(...out.subarray(out.length - 8, out.length - 4)), "IEND");
});

test("the LaTeX snippet includes the PDF at its physical width", () => {
  const l = exportLayout(ethane(), null, ACS_1996_STYLE, { padPt: 2 });
  const tex = latexFigure(l, "scheme.pdf");
  assert.match(tex, /\\begin\{figure\}\[htbp\]/);
  assert.match(tex, /\\includegraphics\[width=0\.67cm\]\{scheme\.pdf\}/, "19 pt = 0.67 cm");
  assert.match(tex, /\\end\{figure\}$/);
});

test("chemfig: angles, branches, ring hooks, offset doubles, wedges, labels and charges", rdOpts, async () => {
  const rd = await rdkit();
  const body = (smi) => /\\chemfig\{(.*)\}/.exec(toChemfig(importSketch(rd, smi), null, ACS_1996_STYLE))[1];
  assert.match(body("CCO"), /^-\[:-?\d+\]-\[:-?\d+\]\{(HO|OH)\}$/, "two bare carbons and the OH label");
  const benzene = body("c1ccccc1");
  assert.equal((benzene.match(/\?\[a/g) ?? []).length, 2, "one ring closure hook pair");
  assert.equal((benzene.match(/=[\^_]/g) ?? []).length, 3, "three doubles offset into the ring");
  const ala = body("C[C@@H](C(=O)[O-])[NH3+]");
  assert.match(ala, /[<>]:?\[/, "the wedge or hash of the stereocentre");
  assert.match(ala, /\{?H_3N\^\{\+\}\}?|\{?NH_3\^\{\+\}\}?/);
  assert.match(ala, /O\^\{-\}/);
  const full = toChemfig(importSketch(rd, "CC.O"), null, ACS_1996_STYLE);
  assert.equal((full.match(/\\chemfig\{/g) ?? []).length, 2, "one \\chemfig per fragment");
  assert.match(full, /\\setchemfig\{atom sep=14\.4bp,bond offset=1\.6bp,double bond sep=2\.59bp,cram width=2bp/);
});
