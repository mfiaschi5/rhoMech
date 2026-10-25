// Curved-arrow geometry (design §11.5), in depiction viewBox units.
//
// Main (token) arrow: tail at the donor site, or at the a-c bond midpoint for a bond source
// (source_partner_map); head at the acceptor site. A quadratic Bezier with control point
// mid + n * 0.35 * |PQ|, n the unit normal pointing away from the depiction centroid,
// alternating side for further arrows on the same anchor pair. The tail is shortened by 8 px
// and the head by 11 px (proportionally less on very short arrows).
// Implied sink arrow: from the b-d midpoint to d (sink_partner_map), thinner, not clickable.
// Dataset arrows (map mode): tail at the source atom or bond midpoint, head at the sink atom or
// bond midpoint; a radical source gets the half arrowhead.
import type { DatasetArrowOut } from "../api/types.gen";
import type { Point } from "./depict";

export type { Point } from "./depict";

export type ArrowRole = "main" | "implied" | "dataset";

/** What a token arrow needs to be drawn (the v3 token arrows; dataset arrows are drawn by map). */
export interface TokenArrowLike {
  donor: number;
  donor_site: number;
  acceptor: number;
  acceptor_site: number;
  donor_site_map: number;
  acceptor_site_map: number;
  donor_element: string;
  acceptor_element: string;
  source_partner_map?: number | null;
  sink_partner_map?: number | null;
  electrons?: 1 | 2;
}

export const BOW = 0.35;
export const SHORTEN_TAIL_PX = 8;
export const SHORTEN_HEAD_PX = 11;

export interface Curve {
  /** Unshortened anchors and control point of the full curve. */
  p0: Point;
  c: Point;
  p1: Point;
  /** The visible (shortened) curve as an SVG path. */
  d: string;
  /** The full curve at t = 0.5 (where the click handle sits). */
  mid: Point;
}

export interface OverlayItem {
  role: ArrowRole;
  /** Index of the arrow in the list it came from (an implied arrow shares its main arrow's). */
  index: number;
  key: string;
  curve: Curve;
  electrons: 1 | 2;
  /** data-* attributes of the arrow's <g> (without the data- prefix), strings. */
  data: Record<string, string>;
}

export interface MissingAtoms {
  role: ArrowRole;
  index: number;
  maps: number[];
}

export interface OverlayLayout {
  items: OverlayItem[];
  missing: MissingAtoms[];
}

export type PositionOf = (map: number) => Point | undefined;

export const midpoint = (a: Point, b: Point): Point => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

export function centroidOf(points: Iterable<Point>): Point {
  let n = 0, x = 0, y = 0;
  for (const p of points) {
    n += 1;
    x += p[0];
    y += p[1];
  }
  return n ? [x / n, y / n] : [0, 0];
}

export function quadPoint(p0: Point, c: Point, p1: Point, t: number): Point {
  const u = 1 - t;
  return [u * u * p0[0] + 2 * u * t * c[0] + t * t * p1[0], u * u * p0[1] + 2 * u * t * c[1] + t * t * p1[1]];
}

/** The control point of the sub-curve [t0, t1] of a quadratic Bezier. */
function subControl(p0: Point, c: Point, p1: Point, t0: number, t1: number): Point {
  const b0 = quadPoint(p0, c, p1, t0);
  const k = t1 - t0;
  return [
    b0[0] + k * ((1 - t0) * (c[0] - p0[0]) + t0 * (p1[0] - c[0])),
    b0[1] + k * ((1 - t0) * (c[1] - p0[1]) + t0 * (p1[1] - c[1])),
  ];
}

/** The t at which the curve is `dist` (chord length) away from its start (fromEnd: its end). */
function tAtDistance(p0: Point, c: Point, p1: Point, dist: number, fromEnd: boolean): number {
  const anchor = fromEnd ? p1 : p0;
  let lo = 0, hi = 0.5;
  for (let i = 0; i < 40; i++) {
    const m = (lo + hi) / 2;
    const q = quadPoint(p0, c, p1, fromEnd ? 1 - m : m);
    if (Math.hypot(q[0] - anchor[0], q[1] - anchor[1]) < dist) lo = m;
    else hi = m;
  }
  const t = (lo + hi) / 2;
  return fromEnd ? 1 - t : t;
}

const fmt = (v: number) => (Math.round(v * 100) / 100).toString();

export interface CurveOptions {
  centroid: Point;
  /** +1: away from the centroid; -1: towards it. */
  side?: 1 | -1;
  bow?: number;
  shortenTail?: number;
  shortenHead?: number;
}

export function curvedArrow(tail: Point, head: Point, opts: CurveOptions): Curve {
  const dx = head[0] - tail[0];
  const dy = head[1] - tail[1];
  const len = Math.hypot(dx, dy);
  const mid = midpoint(tail, head);
  let n: Point = len > 1e-9 ? [-dy / len, dx / len] : [0, -1];
  const away = (mid[0] - opts.centroid[0]) * n[0] + (mid[1] - opts.centroid[1]) * n[1];
  if (away < 0) n = [-n[0], -n[1]];
  const sign = opts.side ?? 1;
  const bow = (opts.bow ?? BOW) * Math.max(len, 12);
  const c: Point = [mid[0] + sign * n[0] * bow, mid[1] + sign * n[1] * bow];
  const wantTail = opts.shortenTail ?? SHORTEN_TAIL_PX;
  const wantHead = opts.shortenHead ?? SHORTEN_HEAD_PX;
  const scale = len > 0 ? Math.min(1, (0.6 * len) / Math.max(wantTail + wantHead, 1e-9)) : 0;
  const t0 = wantTail * scale > 0 ? tAtDistance(tail, c, head, wantTail * scale, false) : 0;
  const t1 = wantHead * scale > 0 ? tAtDistance(tail, c, head, wantHead * scale, true) : 1;
  const a = quadPoint(tail, c, head, t0);
  const b = quadPoint(tail, c, head, t1);
  const q = subControl(tail, c, head, t0, t1);
  const d = `M ${fmt(a[0])} ${fmt(a[1])} Q ${fmt(q[0])} ${fmt(q[1])} ${fmt(b[0])} ${fmt(b[1])}`;
  return { p0: tail, c, p1: head, d, mid: quadPoint(tail, c, head, 0.5) };
}

/** Side and bow for the k-th arrow (0-based) on one anchor pair: +, -, +wider, -wider, ... */
function sideFor(seen: Map<string, number>, key: string): { side: 1 | -1; bow: number } {
  const k = seen.get(key) ?? 0;
  seen.set(key, k + 1);
  return { side: k % 2 === 0 ? 1 : -1, bow: BOW * (1 + 0.5 * Math.floor(k / 2)) };
}

function anchorData(curve: Curve): Record<string, string> {
  return {
    x1: fmt(curve.p0[0]),
    y1: fmt(curve.p0[1]),
    x2: fmt(curve.p1[0]),
    y2: fmt(curve.p1[1]),
    mx: fmt(curve.mid[0]),
    my: fmt(curve.mid[1]),
  };
}

function tokenData(a: TokenArrowLike): Record<string, string> {
  return {
    "donor-site": String(a.donor_site),
    "acceptor-site": String(a.acceptor_site),
    "donor-site-map": String(a.donor_site_map),
    "acceptor-site-map": String(a.acceptor_site_map),
    "donor-element": a.donor_element,
    "acceptor-element": a.acceptor_element,
    donor: String(a.donor),
    acceptor: String(a.acceptor),
  };
}

/** Main and implied arrows of a list of token arrows. Arrows with an atom that has no position
 *  are not drawn and are reported in `missing` (the overlay sets data-overlay-error). */
export function layoutTokenArrows(
  arrows: readonly TokenArrowLike[],
  pos: PositionOf,
  centroid: Point,
): OverlayLayout {
  const items: OverlayItem[] = [];
  const missing: MissingAtoms[] = [];
  const seen = new Map<string, number>();
  arrows.forEach((a, index) => {
    const partner = a.source_partner_map ?? null;
    const need = [a.donor_site_map, a.acceptor_site_map, ...(partner !== null ? [partner] : [])];
    const lost = need.filter((m) => !pos(m));
    if (lost.length) return void missing.push({ role: "main", index, maps: lost });
    const site = pos(a.donor_site_map)!;
    const tail = partner !== null ? midpoint(site, pos(partner)!) : site;
    const head = pos(a.acceptor_site_map)!;
    const key = `${partner !== null ? [a.donor_site_map, partner].sort().join("-") : a.donor_site_map}>${a.acceptor_site_map}`;
    const curve = curvedArrow(tail, head, { centroid, ...sideFor(seen, key) });
    const data = { ...tokenData(a), ...anchorData(curve) };
    items.push({ role: "main", index, key: `main-${index}`, curve, electrons: a.electrons ?? 2, data });
    const sink = a.sink_partner_map ?? null;
    if (sink === null) return;
    const d = pos(sink);
    if (!d) return void missing.push({ role: "implied", index, maps: [sink] });
    const implied = curvedArrow(midpoint(head, d), d, { centroid, shortenTail: 2, ...sideFor(seen, `${a.acceptor_site_map}-${sink}`) });
    const idata = { ...tokenData(a), ...anchorData(implied), "sink-partner-map": String(sink) };
    items.push({ role: "implied", index, key: `implied-${index}`, curve: implied, electrons: a.electrons ?? 2, data: idata });
  });
  return { items, missing };
}

/** Dataset arrows (map numbers), for Browse and curated steps. */
export function layoutDatasetArrows(
  arrows: readonly DatasetArrowOut[],
  pos: PositionOf,
  centroid: Point,
  elementOf: (map: number) => string | undefined = () => undefined,
): OverlayLayout {
  const items: OverlayItem[] = [];
  const missing: MissingAtoms[] = [];
  const seen = new Map<string, number>();
  arrows.forEach((a, index) => {
    const lost = [...a.source_atoms, ...a.sink_atoms].filter((m) => !pos(m));
    if (lost.length || !a.source_atoms.length || !a.sink_atoms.length) {
      return void missing.push({ role: "dataset", index, maps: lost });
    }
    const at = (maps: number[]): Point => (maps.length === 1 ? pos(maps[0])! : midpoint(pos(maps[0])!, pos(maps[1])!));
    const tail = at(a.source_atoms);
    const head = at(a.sink_atoms);
    const key = `${a.source_atoms.join("-")}>${a.sink_atoms.join("-")}`;
    const curve = curvedArrow(tail, head, { centroid, ...sideFor(seen, key) });
    const data = {
      "source-kind": a.source_kind,
      "sink-kind": a.sink_kind,
      "donor-site-map": a.source_atoms.join(","),
      "acceptor-site-map": a.sink_atoms.join(","),
      "donor-element": a.source_atoms.map((m) => elementOf(m) ?? "?").join(","),
      "acceptor-element": a.sink_atoms.map((m) => elementOf(m) ?? "?").join(","),
      ...anchorData(curve),
    };
    const electrons: 1 | 2 = a.source_kind === "radical" ? 1 : 2;
    items.push({ role: "dataset", index, key: `dataset-${index}`, curve, electrons, data });
  });
  return { items, missing };
}
