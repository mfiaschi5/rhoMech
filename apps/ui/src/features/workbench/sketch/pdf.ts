// A small dependency-free PDF writer for the export primitives (exportImage.ts): one page the
// size of the drawing (MediaBox in pt, so the file is the drawing at its true size), stroked
// lines with round caps, filled polygons (wedges), circles (radical dots, circled charges) as
// four Bezier arcs, and text in the PDF base-14 fonts (Helvetica, Helvetica-Bold, -Oblique,
// Times-Roman for the Times family), WinAnsiEncoding, so nothing is embedded. The minus sign is
// written as the en dash. Deterministic: no dates or ids, the same layout gives the same bytes.
// The output is a string whose every char code is < 256 (one byte each); pdfBytes() makes the
// Uint8Array for a Blob.
import type { ExportLayout, Prim } from "@ksmech/viz/sketch/exportImage";

const n = (x: number) => {
  const r = Math.round(x * 1000) / 1000;
  return Object.is(r, -0) ? "0" : String(r);
};

function rgb(hex: string): string {
  const h = hex.replace("#", "");
  const c = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  return c.map((v) => n(Number.isFinite(v) ? v : 0)).join(" ");
}

/** A PDF string literal in WinAnsi: \, ( and ) escaped; U+2212 and U+2013 as the en dash. */
function pdfString(s: string): string {
  let out = "";
  for (const ch of s) {
    const c = ch.codePointAt(0) ?? 63;
    if (ch === "\\" || ch === "(" || ch === ")") out += `\\${ch}`;
    else if (c === 0x2212 || c === 0x2013) out += "\\226";
    else if (c >= 32 && c <= 126) out += ch;
    else out += "?";
  }
  return `(${out})`;
}

const K = 0.5523; // Bezier circle constant

function circlePath(cx: number, cy: number, r: number): string {
  const k = r * K;
  return [
    `${n(cx + r)} ${n(cy)} m`,
    `${n(cx + r)} ${n(cy + k)} ${n(cx + k)} ${n(cy + r)} ${n(cx)} ${n(cy + r)} c`,
    `${n(cx - k)} ${n(cy + r)} ${n(cx - r)} ${n(cy + k)} ${n(cx - r)} ${n(cy)} c`,
    `${n(cx - r)} ${n(cy - k)} ${n(cx - k)} ${n(cy - r)} ${n(cx)} ${n(cy - r)} c`,
    `${n(cx + k)} ${n(cy - r)} ${n(cx + r)} ${n(cy - k)} ${n(cx + r)} ${n(cy)} c`,
  ].join("\n");
}

/** The font resource of a text primitive. */
function fontOf(p: Extract<Prim, { k: "text" }>): string {
  return p.italic ? "/F3" : p.weight === 700 ? "/F2" : "/F1";
}

/** The page content: y flipped (PDF y up). */
function content(l: ExportLayout, background: string | null): string {
  const H = l.heightPt;
  const Y = (y: number) => H - y;
  const ops: string[] = ["1 J 1 j"];
  if (background) ops.push(`${rgb(background)} rg 0 0 ${n(l.widthPt)} ${n(H)} re f`);
  for (const p of l.prims) {
    if (p.k === "line") ops.push(`${rgb(p.colour)} RG ${n(p.w)} w ${n(p.p[0])} ${n(Y(p.p[1]))} m ${n(p.q[0])} ${n(Y(p.q[1]))} l S`);
    else if (p.k === "poly") {
      const [first, ...rest] = p.pts;
      ops.push(`${rgb(p.colour)} rg ${n(first[0])} ${n(Y(first[1]))} m ${rest.map((v) => `${n(v[0])} ${n(Y(v[1]))} l`).join(" ")} h f`);
    } else if (p.k === "circle") {
      const path = circlePath(p.c[0], Y(p.c[1]), p.r);
      ops.push(p.fill ? `${rgb(p.colour)} rg\n${path}\nh f` : `${rgb(p.colour)} RG ${n(p.w)} w\n${path}\nh S`);
    } else {
      ops.push(`BT ${rgb(p.colour)} rg ${fontOf(p)} ${n(p.size)} Tf 1 0 0 1 ${n(p.x)} ${n(Y(p.y))} Tm ${pdfString(p.text)} Tj ET`);
    }
  }
  return ops.join("\n") + "\n";
}

/** Base-14 font names for the layout's family (Times for a serif family, else Helvetica). */
function baseFonts(l: ExportLayout): [string, string, string] {
  return /times/i.test(l.fontFamily) && !/^arial|^helvetica/i.test(l.fontFamily) ? ["Times-Roman", "Times-Bold", "Times-Italic"] : ["Helvetica", "Helvetica-Bold", "Helvetica-Oblique"];
}

/** A one-page PDF of the layout (a byte string). `background` null: transparent. */
export function toPdf(l: ExportLayout, opts: { background?: string | null; title?: string } = {}): string {
  const W = n(l.widthPt);
  const H = n(l.heightPt);
  const stream = content(l, opts.background ?? null);
  const [f1, f2, f3] = baseFonts(l);
  const font = (name: string) => `<< /Type /Font /Subtype /Type1 /BaseFont /${name} /Encoding /WinAnsiEncoding >>`;
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${W} ${H}] /Resources << /Font << /F1 5 0 R /F2 6 0 R /F3 7 0 R >> >> /Contents 4 0 R >>`,
    `<< /Length ${stream.length} >>\nstream\n${stream}endstream`,
    font(f1),
    font(f2),
    font(f3),
    `<< /Producer (ksmech workbench) /Title ${pdfString(opts.title ?? "scheme")} >>`,
  ];
  let out = "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n";
  const offsets: number[] = [];
  objects.forEach((body, i) => {
    offsets.push(out.length);
    out += `${i + 1} 0 obj\n${body}\nendobj\n`;
  });
  const xref = out.length;
  out += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (const o of offsets) out += `${String(o).padStart(10, "0")} 00000 n \n`;
  out += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R /Info ${objects.length} 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return out;
}

/** The bytes of a PDF byte string. */
export function pdfBytes(s: string): Uint8Array {
  const b = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i) & 0xff;
  return b;
}
