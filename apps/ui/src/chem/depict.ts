// Pool and step depictions with exact atom positions (design §11.4).
//
// The backend's DepictionOut carries a 2D molblock (maps cleared, H atoms outside keep_maps
// removed) and an atom table depiction_idx -> {global_site, map, element}. renderDepiction draws
// it with RDKit.js and finds every atom's drawing position, in this order of coordinate tiers:
//   1. "metadata": the <metadata> rdkit:atom elements of the SVG (idx 1-based, drawing-x/y).
//                  RDKit.js 2026.9.1 writes no metadata (checked), so this tier is dormant;
//   2. "fit":      a least-squares fit X = s x + tx, Y = -s y + ty of the molblock coordinates to
//                  the single-stroke bond ends at unlabelled atoms, accepted when the RMS is
//                  <= 1.5 px and every labelled atom falls inside its label glyphs (svgFit.ts;
//                  within 0.16 px of RDKit's GetDrawCoords on the fixture molecules);
//   3. "server":   the server's own SVG and its exact GetDrawCoords (GET .../depiction?server_svg=1,
//                  or DepictionOut.server_svg when already present), shown instead. Needed when
//                  fewer than two unlabelled atoms end a single bond (CCl, [OH-] + CCl, ...).
// Positions are in the SVG's viewBox units; the overlay uses the same viewBox.
import { getDepiction } from "../api/client";
import type { DepictionOut } from "../api/types.gen";
import { loadRDKit, type RDKitModule } from "./rdkit";
import { fitAtomPositions, molblockCoords, type Point, type SvgPrimitive } from "./svgFit";

export type { Point } from "./svgFit";
export { FIT_MAX_RMS_PX, molblockCoords } from "./svgFit";
export type ViewBox = [number, number, number, number];
export type CoordsMethod = "metadata" | "fit" | "server";

const RDKIT_NS = "http://www.rdkit.org/xml";
const STATE_REF = /^s[0-9a-f]{20}$/;

export interface DepictionResult {
  /** The SVG markup, its root sized to the requested width x height. */
  svg: string;
  viewBox: ViewBox;
  /** depiction_idx -> drawing position. */
  atomXY: Map<number, Point>;
  /** map number -> drawing position (what overlays and data-atom-xy use). */
  byMap: Map<number, Point>;
  /** map number -> element symbol. */
  elements: Map<number, string>;
  method: CoordsMethod;
  fitRms: number | null;
  /** The atom table the positions refer to (the server's, for the server tier). */
  atoms: DepictionOut["atoms"];
}

export interface RenderOptions {
  width: number;
  height: number;
  /** depiction_idx values to highlight. */
  highlight?: number[];
  /** Tier-3 fetcher. Default: GET /pool/{ref}/depiction?server_svg=1 when ref is a state id. */
  fetchServer?: () => Promise<DepictionOut>;
  /** Skip the client tiers (diagnostics, tests). */
  forceServer?: boolean;
}

export async function renderDepiction(dep: DepictionOut, opts: RenderOptions): Promise<DepictionResult> {
  const rdkit = await loadRDKit();
  if (!opts.forceServer && !(dep.server_svg && dep.server_coords)) {
    const local = drawLocally(rdkit, dep, opts);
    if (local) return local;
  }
  return serverTier(dep, opts);
}

function drawLocally(rdkit: RDKitModule, dep: DepictionOut, opts: RenderOptions): DepictionResult | null {
  let mol = null;
  try {
    mol = rdkit.get_mol(dep.molblock, JSON.stringify({ removeHs: false }));
  } catch {
    mol = null;
  }
  if (!mol) return null;
  try {
    if (!mol.is_valid() || mol.get_num_atoms() !== dep.atoms.length) return null;
    const raw = mol.get_svg_with_highlights(
      JSON.stringify({
        width: opts.width,
        height: opts.height,
        includeMetadata: true,
        includeAtomTags: true,
        addChiralHs: false,
        atoms: opts.highlight ?? [],
        bondLineWidth: 1.5,
      }),
    );
    const doc = parseSvg(raw);
    if (!doc) return null;
    const meta = metadataPositions(doc, dep.atoms.length);
    if (meta) return result(doc, opts, dep, meta, "metadata", null);
    const fit = fitAtomPositions(svgPrimitives(doc), molblockCoords(dep.molblock), dep.atoms.length);
    if (fit) return result(doc, opts, dep, fit.positions, "fit", fit.rms);
    return null;
  } finally {
    mol.delete();
  }
}

async function serverTier(dep: DepictionOut, opts: RenderOptions): Promise<DepictionResult> {
  let server = dep;
  if (!(dep.server_svg && dep.server_coords)) {
    const fetcher = opts.fetchServer ?? defaultServerFetcher(dep);
    if (!fetcher) throw new Error(`no depiction coordinates for ${dep.ref} and no server fallback`);
    server = await fetcher();
  }
  const doc = server.server_svg ? parseSvg(server.server_svg) : null;
  if (!doc || !server.server_coords) throw new Error(`server depiction of ${dep.ref} has no SVG or coordinates`);
  const positions = new Map<number, Point>();
  server.server_coords.forEach(([x, y], i) => {
    if (Number.isFinite(x) && Number.isFinite(y)) positions.set(i, [x, y]);
  });
  return result(doc, opts, server, positions, "server", null);
}

function defaultServerFetcher(dep: DepictionOut): (() => Promise<DepictionOut>) | null {
  if (!STATE_REF.test(dep.ref)) return null;
  const keepMaps = dep.atoms.filter((a) => a.element === "H").map((a) => a.map);
  return () => getDepiction(dep.ref, { keepMaps, serverSvg: true });
}

function result(
  doc: Document,
  opts: RenderOptions,
  dep: DepictionOut,
  atomXY: Map<number, Point>,
  method: CoordsMethod,
  fitRms: number | null,
): DepictionResult {
  const viewBox = normalizeRoot(doc, opts.width, opts.height);
  const byMap = new Map<number, Point>();
  const elements = new Map<number, string>();
  for (const a of dep.atoms) {
    elements.set(a.map, a.element);
    const p = atomXY.get(a.depiction_idx);
    if (p) byMap.set(a.map, p);
  }
  const svg = new XMLSerializer().serializeToString(doc.documentElement);
  return { svg, viewBox, atomXY, byMap, elements, method, fitRms, atoms: dep.atoms };
}

function parseSvg(text: string): Document | null {
  const doc = new DOMParser().parseFromString(text, "image/svg+xml");
  if (doc.getElementsByTagName("parsererror").length > 0) return null;
  return doc.documentElement?.localName === "svg" ? doc : null;
}

/** Size the root <svg> to width x height, keeping (or adding) its viewBox. */
function normalizeRoot(doc: Document, width: number, height: number): ViewBox {
  const root = doc.documentElement;
  let vb = (root.getAttribute("viewBox") ?? "").trim().split(/[\s,]+/).map(Number);
  if (vb.length !== 4 || vb.some((v) => !Number.isFinite(v))) {
    const w = Number.parseFloat(root.getAttribute("width") ?? "") || width;
    const h = Number.parseFloat(root.getAttribute("height") ?? "") || height;
    vb = [0, 0, w, h];
    root.setAttribute("viewBox", vb.join(" "));
  }
  root.setAttribute("width", String(width));
  root.setAttribute("height", String(height));
  root.setAttribute("preserveAspectRatio", "xMidYMid meet");
  return [vb[0], vb[1], vb[2], vb[3]];
}

// ---- tier 1: SVG metadata

function metadataPositions(doc: Document, nAtoms: number): Map<number, Point> | null {
  let nodes = Array.from(doc.getElementsByTagNameNS(RDKIT_NS, "atom"));
  if (nodes.length === 0) nodes = Array.from(doc.getElementsByTagName("rdkit:atom"));
  const out = new Map<number, Point>();
  for (const el of nodes) {
    const idx = Number.parseInt(el.getAttribute("idx") ?? "", 10) - 1;
    const x = Number.parseFloat(el.getAttribute("drawing-x") ?? "");
    const y = Number.parseFloat(el.getAttribute("drawing-y") ?? "");
    if (idx >= 0 && idx < nAtoms && Number.isFinite(x) && Number.isFinite(y)) out.set(idx, [x, y]);
  }
  return out.size === nAtoms ? out : null;
}

// ---- tier 2: least-squares fit (svgFit.ts)

/** The drawn elements that carry RDKit atom / bond classes. */
function svgPrimitives(doc: Document): SvgPrimitive[] {
  return Array.from(doc.querySelectorAll("[class*='atom-'], [class*='bond-']"), (el) => {
    const x = Number.parseFloat(el.getAttribute("x") ?? "");
    const y = Number.parseFloat(el.getAttribute("y") ?? "");
    return {
      tag: el.localName,
      cls: el.getAttribute("class") ?? "",
      d: el.getAttribute("d") ?? undefined,
      x: Number.isFinite(x) ? x : undefined,
      y: Number.isFinite(y) ? y : undefined,
    };
  });
}
