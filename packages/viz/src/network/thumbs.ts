// Structure thumbnails of pool states, cached per state and size as an SVG string and a data URI
// (cards put them on --depiction-bg, white in both themes). cardThumb draws every species at one bond length (BOND_PX, RDKit's flexicanvas), so structures
// read at the same scale across cards; a composite wider or taller than its box is scaled down.
import type { SpeciesCountOut } from "../apiTypes";
import { smilesToSvg, svgDataUri, type RDKitModule } from "../chem/rdkit";

export interface Thumb {
  svg: string;
  uri: string;
}


/** The inner markup of an RDKit SVG as a nested <svg> placed at x (header and size dropped). */
function nested(svg: string, x: number, w: number, h: number): string {
  const body = svg.replace(/<\?xml[^>]*>\s*/, "").replace(/<svg\b[^>]*>/, (tag) => {
    const vb = /viewBox=['"]([^'"]+)['"]/.exec(tag)?.[1] ?? `0 0 ${w} ${h}`;
    return `<svg x="${x}" y="0" width="${w}" height="${h}" viewBox="${vb}" preserveAspectRatio="xMidYMid meet">`;
  });
  return body;
}

const ONE_ATOM = /^\[([A-Z][a-z]?)(H\d?)?([+-]\d?)?\]$/;

/** A one-atom species as text ("H⁻", "Cl⁻", "O²⁻", "CH₃⁺", "H₂"): RDKit draws a lone H- or H+ as
 *  nothing. */
function atomText(smi: string, x: number, w: number, h: number): string | null {
  const m = ONE_ATOM.exec(smi);
  const parts = smi === "[H][H]" ? { el: "H", sub: "2", charge: "" } : m ? { el: m[1], sub: m[2] ? `H${m[2].slice(1)}` : "", charge: m[3] ?? "" } : null;
  if (!parts) return null;
  const sup = parts.charge.replace(/^([+-])(\d)$/, "$2$1").replace("-", "−");
  const colours: Record<string, string> = { O: "#d0261c", N: "#2850d8", Cl: "#1f8a3c", Br: "#9a2a2a", F: "#1f8a3c", S: "#a08000" };
  const size = Math.min(h * 0.32, 22);
  const small = size * 0.65;
  // "CH3": the H at full size, its count as a subscript; "H2": the 2 as a subscript
  const hText = parts.sub.startsWith("H") ? `H${parts.sub.length > 1 ? `<tspan font-size="${small}" dy="${size * 0.25}">${parts.sub.slice(1)}</tspan><tspan dy="${-size * 0.25}"></tspan>` : ""}` : parts.sub ? `<tspan font-size="${small}" dy="${size * 0.25}">${parts.sub}</tspan><tspan dy="${-size * 0.25}"></tspan>` : "";
  return `<text x="${x + w / 2}" y="${h / 2 + size / 3}" text-anchor="middle" font-family="sans-serif" font-size="${size}" fill="${colours[parts.el] ?? "#1c2330"}">${parts.el}${hText}${sup ? `<tspan font-size="${small}" dy="${-size * 0.45}">${sup}</tspan>` : ""}</text>`;
}

export const BOND_PX = 24;
const GAP = 18;

interface Natural {
  svg: string;
  w: number;
  h: number;
}

const naturals = new Map<string, Natural | null>();

/** One species at BOND_PX per bond (one-atom ions as text), with its natural size. */
function natural(rdkit: RDKitModule, smi: string): Natural | null {
  if (naturals.has(smi)) return naturals.get(smi) ?? null;
  let out: Natural | null = null;
  const text = atomText(smi, 0, 34, 30);
  if (text) out = { svg: `<svg xmlns="http://www.w3.org/2000/svg" width="34" height="30" viewBox="0 0 34 30">${text}</svg>`, w: 34, h: 30 };
  else {
    const svg = smilesToSvg(rdkit, smi, { width: -1, height: -1, draw: { clearBackground: false, padding: 0.04, fixedBondLength: BOND_PX, bondLineWidth: 1.5, minFontSize: 11 } });
    const w = Number(/width=['"]([\d.]+)/.exec(svg ?? "")?.[1]);
    const h = Number(/height=['"]([\d.]+)/.exec(svg ?? "")?.[1]);
    if (svg && w > 0 && h > 0) out = { svg, w, h };
  }
  naturals.set(smi, out);
  return out;
}

export interface CardThumb extends Thumb {
  w: number;
  h: number;
}

const cards = new Map<string, CardThumb | null>();

/** The species of a state at one bond length, "+" between them, within maxW x maxH. */
export function cardThumb(rdkit: RDKitModule, species: readonly SpeciesCountOut[], fallback: string, maxW: number, maxH: number): CardThumb | null {
  const list = species.length ? species.map((s) => ({ smi: s.display_smiles, n: s.count })) : fallback.split(".").map((smi) => ({ smi, n: 1 }));
  const key = `${maxW}x${maxH}|${list.map((s) => `${s.n}*${s.smi}`).join("+")}`;
  if (cards.has(key)) return cards.get(key) ?? null;
  const parts = list.map((s) => ({ ...s, nat: natural(rdkit, s.smi) })).filter((s) => s.nat);
  if (!parts.length) {
    cards.set(key, null);
    return null;
  }
  const count = (n: number) => (n > 1 ? 18 : 0);
  const natW = parts.reduce((t, p) => t + p.nat!.w + count(p.n), 0) + GAP * (parts.length - 1);
  const natH = Math.max(...parts.map((p) => p.nat!.h));
  const k = Math.min(1, maxW / natW, maxH / natH);
  const [W, H] = [Math.ceil(natW * k), Math.ceil(natH * k)];
  let x = 0;
  const body: string[] = [];
  parts.forEach((p, i) => {
    const nat = p.nat!;
    if (p.n > 1) {
      body.push(`<text x="${x}" y="${natH / 2 + 4}" font-family="sans-serif" font-size="12" fill="#5d6878">${p.n}×</text>`);
      x += count(p.n);
    }
    body.push(nested(nat.svg, x, nat.w, nat.h).replace(/<svg x="([\d.]+)" y="0"/, `<svg x="$1" y="${((natH - nat.h) / 2).toFixed(1)}"`));
    x += nat.w;
    if (i < parts.length - 1) body.push(`<text x="${x + GAP / 2}" y="${natH / 2 + 5}" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#8a94a3">+</text>`);
    x += GAP;
  });
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${natW} ${natH}">${body.join("")}</svg>`;
  const t = { svg, uri: svgDataUri(svg), w: W, h: H };
  cards.set(key, t);
  return t;
}
