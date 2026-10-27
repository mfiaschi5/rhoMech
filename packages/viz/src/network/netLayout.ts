// Pure layout and reading helpers of the network viewports (node --test runs them).
//
//   layeredLayout   left -> right by depth (a reaction scheme): one column per depth, nodes in a
//                   column ordered by the barycentre of their parents (two sweeps), centred; a
//                   column of more than maxPerColumn states becomes a brick of sub-columns (the
//                   k-th state in sub-column k mod s, rows dy / s apart), so a wide layer does
//                   not make the scheme tall.
//   backEdges       steps that go back to a shallower or equal depth (cycles, recurrences).
//   edgeStrength    a 0..1 strength per step: exp(weight) (the log beam weight) over the largest.
//   stepType        a coarse name of a step from its arrow reading (substitution, addition, ...).
//   readingLabel    the reading as text with element symbols ("lp O2 → C6 · C6–Cl7 → Cl7").
//   neighbour       keyboard navigation over a layered layout (arrows move through neighbours).
//   schemeLayout    the scheme of variant A from card sizes: columns by depth in barycentre
//                   order, uniform width per column, cards stacked with fixed gaps (sub-columns
//                   past maxPerColumn), columns centred vertically: no two cards ever overlap.
//   pathwayTree     the ranked pathways as a trie: shared prefixes merged, a weight per branch.
//   treeRows        that trie in columns: each column lists its cards in depth-first order
//                   (planar: ribbons never cross), best rank first, packed from the top, so the
//                   best pathway is one straight top lane and no column has gaps.
import type { ArrowReadingOut, EdgeOut, NetworkOut, NodeOut } from "../v9types";

export type XY = { x: number; y: number };

export interface LayeredOptions {
  dx?: number;
  dy?: number;
  /** States per column before it splits into sub-columns (default 6). */
  maxPerColumn?: number;
  /** Distance between sub-columns (default 0.72 dx). */
  subDx?: number;
}

export interface Column {
  layer: number;
  /** x of the first and the last sub-column. */
  x0: number;
  x1: number;
  count: number;
}

export interface Layered {
  pos: Map<string, XY & { layer: number }>;
  layers: string[][];
  columns: Column[];
}

const byLogp = (nodes: Map<string, NodeOut>) => (a: string, b: string) => (nodes.get(b)?.best_logp ?? 0) - (nodes.get(a)?.best_logp ?? 0) || a.localeCompare(b);

/** Order a layer by the mean index of each node's neighbours in the reference layer. */
function sweep(layer: string[], ref: Map<string, number>, adj: Map<string, string[]>, tie: (a: string, b: string) => number): string[] {
  const bary = new Map(
    layer.map((id) => {
      const ks = (adj.get(id) ?? []).map((n) => ref.get(n)).filter((k): k is number => k !== undefined);
      return [id, ks.length ? ks.reduce((s, k) => s + k, 0) / ks.length : Number.POSITIVE_INFINITY] as const;
    }),
  );
  return [...layer].sort((a, b) => bary.get(a)! - bary.get(b)! || tie(a, b));
}

export function layeredLayout(net: NetworkOut, opts: LayeredOptions = {}): Layered {
  const dx = opts.dx ?? 240;
  const dy = opts.dy ?? 150;
  const nodes = new Map(net.nodes.map((n) => [n.id, n] as const));
  const depth = (id: string) => (id === net.root ? 0 : (nodes.get(id)?.depth ?? 0));
  const maxDepth = Math.max(0, ...net.nodes.map((n) => depth(n.id)));
  const tie = byLogp(nodes);
  let layers: string[][] = Array.from({ length: maxDepth + 1 }, (_, d) => net.nodes.filter((n) => depth(n.id) === d).map((n) => n.id).sort(tie));
  const parents = new Map<string, string[]>();
  const children = new Map<string, string[]>();
  for (const e of net.edges) {
    if (depth(e.target) <= depth(e.source)) continue;
    parents.set(e.target, [...(parents.get(e.target) ?? []), e.source]);
    children.set(e.source, [...(children.get(e.source) ?? []), e.target]);
  }
  const index = (layer: string[]) => new Map(layer.map((id, k) => [id, k] as const));
  for (let pass = 0; pass < 2; pass++) {
    for (let d = 1; d < layers.length; d++) layers[d] = sweep(layers[d], index(layers[d - 1]), parents, tie);
    for (let d = layers.length - 2; d >= 1; d--) layers[d] = sweep(layers[d], index(layers[d + 1]), children, tie);
  }
  layers = layers.filter((l) => l.length);
  const per = opts.maxPerColumn ?? 6;
  const subDx = opts.subDx ?? 0.72 * dx;
  const pos = new Map<string, XY & { layer: number }>();
  const columns: Column[] = [];
  let x = 0;
  for (const layer of layers) {
    const subs = Math.ceil(layer.length / per);
    const d = depth(layer[0]);
    layer.forEach((id, k) => pos.set(id, { x: x + (k % subs) * subDx, y: ((k - (layer.length - 1) / 2) * dy) / subs, layer: d }));
    columns.push({ layer: d, x0: x, x1: x + (subs - 1) * subDx, count: layer.length });
    x += (subs - 1) * subDx + dx;
  }
  return { pos, layers, columns };
}

export interface Size {
  w: number;
  h: number;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface SchemeColumn {
  layer: number;
  /** Left edge of the first sub-column and right edge of the last. */
  x0: number;
  x1: number;
  /** Card width of the column (every card in it has it). */
  width: number;
  count: number;
}

export interface Scheme {
  /** Card rectangles by node id (top-left corner, model px). */
  rects: Map<string, Rect>;
  /** Card centres with their layer (for keyboard navigation). */
  pos: Map<string, XY & { layer: number }>;
  columns: SchemeColumn[];
  bounds: Rect;
}

export interface SchemeOptions {
  /** Gap between columns (room for the step curves). */
  colGap?: number;
  /** Gap between sub-columns of one column. */
  subGap?: number;
  /** Vertical gap between cards. */
  rowGap?: number;
  maxPerColumn?: number;
  minWidth?: number;
  maxWidth?: number;
}

/** Cards of a column split over sub-columns: card k in sub-column k mod s, each stacked. */
function stack(ids: string[], subs: number, size: (id: string) => Size, gap: number): { sub: number; y: number }[][] {
  const cols: { id: string; h: number }[][] = Array.from({ length: subs }, () => []);
  ids.forEach((id, k) => cols[k % subs].push({ id, h: size(id).h }));
  return cols.map((c) => {
    const total = c.reduce((t, x) => t + x.h, 0) + gap * Math.max(0, c.length - 1);
    let y = -total / 2;
    return c.map((x, k) => {
      const out = { sub: k, y };
      y += x.h + gap;
      return out;
    });
  });
}

export function schemeLayout(net: NetworkOut, size: (id: string) => Size, opts: SchemeOptions = {}): Scheme {
  const colGap = opts.colGap ?? 96;
  const subGap = opts.subGap ?? 28;
  const rowGap = opts.rowGap ?? 18;
  const per = opts.maxPerColumn ?? 7;
  const { layers } = layeredLayout(net);
  const nodes = new Map(net.nodes.map((n) => [n.id, n] as const));
  const rects = new Map<string, Rect>();
  const pos = new Map<string, XY & { layer: number }>();
  const columns: SchemeColumn[] = [];
  let x = 0;
  for (const ids of layers) {
    const layer = ids[0] === net.root ? 0 : (nodes.get(ids[0])?.depth ?? 0);
    const width = Math.min(opts.maxWidth ?? 320, Math.max(opts.minWidth ?? 150, ...ids.map((id) => size(id).w)));
    const subs = Math.ceil(ids.length / per);
    const placed = stack(ids, subs, size, rowGap);
    ids.forEach((id, k) => {
      const sub = k % subs;
      const y = placed[sub][Math.floor(k / subs)].y;
      const r = { x: x + sub * (width + subGap), y, w: width, h: size(id).h };
      rects.set(id, r);
      pos.set(id, { x: r.x + r.w / 2, y: r.y + r.h / 2, layer });
    });
    const x1 = x + subs * width + (subs - 1) * subGap;
    columns.push({ layer, x0: x, x1, width, count: ids.length });
    x = x1 + colGap;
  }
  const all = [...rects.values()];
  const b = all.length ? { x: Math.min(...all.map((r) => r.x)), y: Math.min(...all.map((r) => r.y)), x1: Math.max(...all.map((r) => r.x + r.w)), y1: Math.max(...all.map((r) => r.y + r.h)) } : { x: 0, y: 0, x1: 0, y1: 0 };
  return { rects, pos, columns, bounds: { x: b.x, y: b.y, w: b.x1 - b.x, h: b.y1 - b.y } };
}

/** Do two rectangles intersect (touching edges do not)? */
export const intersects = (a: Rect, b: Rect) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/** Steps that do not go deeper (cycle closures, recurrences): drawn as curved back-edges. */
export function backEdges(net: NetworkOut): Set<string> {
  const depth = new Map(net.nodes.map((n) => [n.id, n.id === net.root ? 0 : n.depth] as const));
  return new Set(net.edges.filter((e) => (depth.get(e.target) ?? 0) <= (depth.get(e.source) ?? 0) || e.recurrence).map((e) => e.id));
}

/** exp(weight) over the network's largest (weight: the step's log beam weight); 1 for user steps. */
export function edgeStrength(net: NetworkOut): Map<string, number> {
  const ws = net.edges.map((e) => (Number.isFinite(e.weight) ? e.weight : -10));
  const top = Math.max(...ws, -10);
  return new Map(net.edges.map((e, k) => [e.id, e.kind === "corrected" ? 1 : Math.max(0.04, Math.exp(ws[k] - top))] as const));
}

export type StepType = "substitution" | "addition" | "elimination" | "proton transfer" | "heterolysis" | "relay" | "other";

export const STEP_TYPES: readonly StepType[] = ["substitution", "addition", "elimination", "proton transfer", "heterolysis", "relay", "other"];

/** Element symbols of a mapped SMILES by atom index (every atom is a bracket atom there). */
export function mappedSymbols(smiles: string): string[] {
  return [...smiles.matchAll(/\[([A-Za-z][a-z]?)/g)].map((m) => {
    const s = m[1];
    if (/^(cl|br)$/i.test(s)) return s[0].toUpperCase() + s[1].toLowerCase();
    return s.length === 2 && s[1] === s[1].toLowerCase() && /^[A-Z]/.test(s) ? s : s[0].toUpperCase();
  });
}

/** A coarse name of a step from its arrow kinds and the atoms they touch. */
export function stepType(arrows: readonly ArrowReadingOut[], symbols: readonly string[] = []): StepType {
  const kinds = arrows.map((a) => a.kind);
  const forms = arrows.filter((a) => a.kind !== "bond_to_lone_pair").map((a) => a.target);
  const hForm = forms.some((t) => t.some((i) => symbols[i] === "H"));
  const leaves = arrows.filter((a) => a.kind === "bond_to_lone_pair");
  if (hForm && leaves.length) return "proton transfer";
  if (kinds.includes("bond_to_bond") && arrows.length >= 2) return kinds.filter((k) => k === "bond_to_bond").length > 1 ? "relay" : "elimination";
  if (kinds.length === 1 && kinds[0] === "bond_to_lone_pair") return "heterolysis";
  if (kinds.length === 1 && kinds[0] === "lone_pair_to_bond") return "addition";
  if (kinds.includes("lone_pair_to_bond") && leaves.length) {
    const formedAt = new Set(forms.flatMap((t) => t));
    return leaves.some((a) => a.source.some((i) => formedAt.has(i))) ? "substitution" : "addition";
  }
  return "other";
}

/** "lp O2 → C6 · C6–Cl7 → Cl7" with element symbols (indices alone without them). */
export function readingLabel(arrows: readonly ArrowReadingOut[], symbols: readonly string[] = []): string {
  const at = (i: number) => `${symbols[i] ?? ""}${i}`;
  const end = (xs: number[], target: boolean) => (xs.length === 2 ? (target ? at(xs[1]) : `${at(xs[0])}–${at(xs[1])}`) : `lp ${at(xs[0])}`);
  return arrows.map((a) => `${end(a.source, false)} → ${end(a.target, true)}`).join(" · ");
}

export type Direction = "left" | "right" | "up" | "down";

/** The node an arrow key moves to: right/left along out/in steps (nearest in y), up/down within
 *  the same layer; null when there is none. */
export function neighbour(net: NetworkOut, pos: ReadonlyMap<string, XY & { layer: number }>, from: string, dir: Direction): string | null {
  const p = pos.get(from);
  if (!p) return null;
  const nearest = (ids: string[]) => ids.filter((id) => id !== from && pos.has(id)).sort((a, b) => Math.abs(pos.get(a)!.y - p.y) - Math.abs(pos.get(b)!.y - p.y))[0] ?? null;
  if (dir === "right") return nearest(net.edges.filter((e) => e.source === from && pos.get(e.target)!.layer > p.layer).map((e) => e.target));
  if (dir === "left") return nearest(net.edges.filter((e) => e.target === from && pos.get(e.source)!.layer < p.layer).map((e) => e.source));
  const same = [...pos].filter(([id, q]) => q.layer === p.layer && id !== from && (dir === "up" ? q.y < p.y : q.y > p.y));
  same.sort((a, b) => Math.abs(a[1].y - p.y) - Math.abs(b[1].y - p.y));
  return same[0]?.[0] ?? null;
}

// ---- pathway trie (variant C)

export interface TreeNode {
  /** Path key: the edge ids from the root, joined ("" for the root). */
  key: string;
  nodeId: string;
  edgeId: string | null;
  depth: number;
  /** Sum of the weights of the pathways through this branch (0..1 over all shown). */
  weight: number;
  /** Ranks of the pathways through it, ascending. */
  ranks: number[];
  children: TreeNode[];
}

/** Pathway weights: their probability when given, else exp(logp) normalized to sum 1. */
export function pathwayWeights(net: NetworkOut): Map<string, number> {
  const ps = net.pathways;
  if (ps.every((p) => p.probability !== null)) {
    const s = ps.reduce((t, p) => t + (p.probability ?? 0), 0) || 1;
    return new Map(ps.map((p) => [p.id, (p.probability ?? 0) / s] as const));
  }
  const top = Math.max(...ps.map((p) => p.logp));
  const raw = ps.map((p) => Math.exp(p.logp - top));
  const s = raw.reduce((t, v) => t + v, 0) || 1;
  return new Map(ps.map((p, k) => [p.id, raw[k] / s] as const));
}

/** The ranked pathways as a trie on their edge sequences (shared prefixes merged). */
export function pathwayTree(net: NetworkOut): TreeNode {
  const weights = pathwayWeights(net);
  const edges = new Map(net.edges.map((e) => [e.id, e] as const));
  const root: TreeNode = { key: "", nodeId: net.root, edgeId: null, depth: 0, weight: 0, ranks: [], children: [] };
  for (const p of [...net.pathways].sort((a, b) => a.rank - b.rank)) {
    const w = weights.get(p.id) ?? 0;
    let cur = root;
    cur.weight += w;
    cur.ranks.push(p.rank);
    p.edge_ids.forEach((id, k) => {
      let next = cur.children.find((c) => c.edgeId === id);
      if (!next) {
        const e: EdgeOut | undefined = edges.get(id);
        next = { key: p.edge_ids.slice(0, k + 1).join(","), nodeId: e?.target ?? p.node_ids[k + 1] ?? id, edgeId: id, depth: k + 1, weight: 0, ranks: [], children: [] };
        cur.children.push(next);
      }
      next.weight += w;
      next.ranks.push(p.rank);
      cur = next;
    });
  }
  return root;
}

export interface TreeRow {
  node: TreeNode;
  row: number;
  /** The parent's key ("" has none). */
  parent: string | null;
  collapsed: boolean;
  /** Descendants hidden by a collapse. */
  hidden: number;
}

const countBelow = (t: TreeNode): number => t.children.reduce((s, c) => s + 1 + countBelow(c), 0);

/** The trie in columns: depth-first, best rank first; each card takes the next free slot of its
 *  column (row), so columns are packed from the top and ribbons never cross. A collapsed branch
 *  keeps its card and hides what follows it. */
export function treeRows(root: TreeNode, collapsed: ReadonlySet<string> = new Set()): TreeRow[] {
  const out: TreeRow[] = [];
  const next = new Map<number, number>();
  const visit = (t: TreeNode, parent: string | null) => {
    const isCollapsed = collapsed.has(t.key) && t.children.length > 0;
    const row = next.get(t.depth) ?? 0;
    next.set(t.depth, row + 1);
    out.push({ node: t, row, parent, collapsed: isCollapsed, hidden: isCollapsed ? countBelow(t) : 0 });
    if (isCollapsed) return;
    for (const c of [...t.children].sort((a, b) => a.ranks[0] - b.ranks[0])) visit(c, t.key);
  };
  visit(root, null);
  return out;
}
