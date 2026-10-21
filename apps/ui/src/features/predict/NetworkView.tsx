// NetworkView: the reaction network drawn with cytoscape (design §11.6).
//
// - Nodes: background-image = data-URI SVG of display_smiles (RDKit.js, 110 x 80); the label
//   gives the species counts; the border shows the status (failed: --error, TERMINAL(TURNOVER):
//   double --cycle border, pruned: dashed).
// - Edges: predicted width mapData(probability, 0, 1, 1, 9); .corrected orange, solid, width 5,
//   labelled "corrected"; .superseded dotted at opacity 0.35; .recurrence a looping bezier;
//   turnover virtual edges from cycles[].virtual_edge (teal, dashed, unbundled-bezier, "−P +S");
//   .cycle members of the shown cycles in purple; edges of the shown pathway thickened. Edge labels
//   are --fg on a --panel box, so they follow the light / dark theme.
// - Layout: built-in breadthfirst, directed, rooted at the root node; it reruns only when the
//   set of elements changes, never on a selection change.
// cytoscape draws on a canvas: NetworkLegend is the DOM mirror tests and keyboards use.
//
// Props:
//   net             NetworkOut
//   selectedNode    string | null       node id (pool id)
//   selectedEdge    string | null
//   pathway         PathwayOut | null   the pathway whose edges are thickened
//   cycleEdges      Set<string>         edge ids drawn in the cycle colour
//   onSelectNode    (id) => void        a tap on a node
//   onSelectEdge    (id) => void        a tap on a real edge (virtual edges are not selectable)
//   height          number              default 420
//
// DOM: <div data-testid="network-view" data-node-count data-edge-count> (real edges only).
import cytoscape from "cytoscape";
import { useEffect, useMemo, useRef } from "react";
import type { NetworkOut, PathwayOut } from "../../api/types.gen";
import { smilesToSvg, svgDataUri, useRDKit, type RDKitModule } from "../../chem/rdkit";
import { fmtProb } from "./format";
import { useLatest } from "./hooks";
import { speciesLabel, virtualEdgeId } from "./network";

export interface NetworkViewProps {
  net: NetworkOut;
  selectedNode: string | null;
  selectedEdge: string | null;
  pathway: PathwayOut | null;
  cycleEdges: ReadonlySet<string>;
  onSelectNode: (id: string) => void;
  onSelectEdge: (id: string) => void;
  height?: number;
}

const NODE_W = 110;
const NODE_H = 80;
const imageCache = new Map<string, string | null>();

function nodeImage(rdkit: RDKitModule, smiles: string): string | null {
  if (!imageCache.has(smiles)) {
    const svg = smiles ? smilesToSvg(rdkit, smiles, { width: NODE_W, height: NODE_H, draw: { padding: 0.08 } }) : null;
    imageCache.set(smiles, svg ? svgDataUri(svg) : null);
  }
  return imageCache.get(smiles) ?? null;
}

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

type Palette = Record<
  "fg" | "panel" | "border" | "accent" | "error" | "cycle" | "turnover" | "corrected" | "muted" | "selected",
  string
>;

function palette(): Palette {
  return {
    fg: cssVar("--fg", "#1c2330"),
    panel: cssVar("--panel", "#ffffff"),
    border: cssVar("--border", "#d5dbe3"),
    accent: cssVar("--accent", "#2463c7"),
    error: cssVar("--error", "#c0362c"),
    cycle: cssVar("--cycle", "#7b3fb8"),
    turnover: cssVar("--turnover", "#128a8a"),
    corrected: cssVar("--corrected", "#e07b00"),
    muted: cssVar("--muted", "#5d6878"),
    selected: cssVar("--arrow-selected", "#7b3fb8"),
  };
}

function nodeStyles(c: Palette) {
  return [
    {
      selector: "node",
      style: {
        shape: "round-rectangle",
        width: NODE_W,
        height: NODE_H,
        "background-color": "#ffffff",
        "border-width": 2,
        "border-color": c.border,
        label: "data(label)",
        "font-size": 9,
        color: c.fg,
        "text-valign": "bottom",
        "text-halign": "center",
        "text-margin-y": 4,
        "text-wrap": "ellipsis",
        "text-max-width": "130px",
      },
    },
    { selector: "node[image]", style: { "background-image": "data(image)", "background-fit": "contain", "background-clip": "node" } },
    { selector: "node.root", style: { "border-width": 3, "border-color": c.accent } },
    { selector: "node.status-failed", style: { "border-color": c.error, "border-width": 3 } },
    { selector: "node.status-pruned", style: { "border-style": "dashed" } },
    { selector: "node.terminal-turnover", style: { "border-style": "double", "border-width": 6, "border-color": c.cycle } },
    { selector: "node.in-pathway", style: { "border-color": c.accent, "border-width": 3 } },
    { selector: "node.selected", style: { "border-color": c.selected, "border-width": 5 } },
  ];
}

function edgeStyles(c: Palette) {
  const edge = {
    "curve-style": "bezier",
    width: "mapData(probability, 0, 1, 1, 9)",
    "line-color": c.muted,
    "target-arrow-shape": "triangle",
    "target-arrow-color": c.muted,
    label: "data(label)",
    "font-size": 8,
    // themed like the text: --fg on --panel (a fixed white box made the labels unreadable in dark mode)
    color: c.fg,
    "text-background-color": c.panel,
    "text-background-opacity": 0.8,
    "text-background-padding": "1px",
  };
  const virtual = {
    "curve-style": "unbundled-bezier",
    "control-point-distances": [-60],
    "control-point-weights": [0.5],
    "line-style": "dashed",
    "line-color": c.turnover,
    "target-arrow-color": c.turnover,
    width: 2,
    color: c.turnover,
  };
  return [
    { selector: "edge", style: edge },
    { selector: "edge.in-pathway", style: { width: "mapData(probability, 0, 1, 5, 13)", "line-color": c.accent, "target-arrow-color": c.accent } },
    { selector: "edge.recurrence", style: { "curve-style": "unbundled-bezier", "control-point-distances": [70], "control-point-weights": [0.5], "line-style": "dashed" } },
    { selector: "edge.cycle", style: { "line-color": c.cycle, "target-arrow-color": c.cycle } },
    { selector: "edge.superseded", style: { "line-style": "dotted", opacity: 0.35 } },
    { selector: "edge.corrected", style: { "line-color": c.corrected, "target-arrow-color": c.corrected, "line-style": "solid", width: 5 } },
    { selector: "edge.corrected.in-pathway", style: { width: 8 } },
    { selector: "edge.turnover-virtual", style: virtual },
    { selector: "edge.selected", style: { "line-color": c.selected, "target-arrow-color": c.selected, "z-index": 10 } },
  ];
}

/** The cytoscape stylesheet from the current theme's CSS variables (exported for tests). */
export function stylesheet(): cytoscape.StylesheetJson {
  const c = palette();
  return [...nodeStyles(c), ...edgeStyles(c)] as unknown as cytoscape.StylesheetJson;
}

interface Built {
  defs: cytoscape.ElementDefinition[];
  nodeCount: number;
  edgeCount: number;
}

function buildElements(p: NetworkViewProps, images: Map<string, string | null>): Built {
  const pathwayEdges = new Set(p.pathway?.edge_ids ?? []);
  const pathwayNodes = new Set(p.pathway?.node_ids ?? []);
  const defs: cytoscape.ElementDefinition[] = p.net.nodes.map((n) => {
    const cls = [`status-${n.status}`];
    if (n.is_root) cls.push("root");
    if (n.status === "terminal" && n.terminal_reason === "TURNOVER") cls.push("terminal-turnover");
    if (pathwayNodes.has(n.id)) cls.push("in-pathway");
    if (n.id === p.selectedNode) cls.push("selected");
    const image = images.get(n.id) ?? undefined;
    return { group: "nodes", data: { id: n.id, label: speciesLabel(n), ...(image ? { image } : {}) }, classes: cls.join(" ") };
  });
  for (const e of p.net.edges) {
    const cls: string[] = [e.kind];
    if (e.status === "superseded") cls.push("superseded");
    if (e.recurrence) cls.push("recurrence");
    if (p.cycleEdges.has(e.id)) cls.push("cycle");
    if (pathwayEdges.has(e.id)) cls.push("in-pathway");
    if (e.id === p.selectedEdge) cls.push("selected");
    const label = e.kind === "corrected" ? "corrected" : fmtProb(e.probability);
    defs.push({ group: "edges", data: { id: e.id, source: e.source, target: e.target, probability: e.probability ?? 0, label }, classes: cls.join(" ") });
  }
  const ids = new Set(p.net.nodes.map((n) => n.id));
  for (const c of p.net.cycles) {
    const v = c.virtual_edge;
    if (!v || !ids.has(v.source) || !ids.has(v.target)) continue;
    const label = v.label ?? "";
    defs.push({ group: "edges", data: { id: virtualEdgeId(c.id), source: v.source, target: v.target, label, probability: 0, virtual: 1 }, classes: "turnover-virtual" });
  }
  return { defs, nodeCount: p.net.nodes.length, edgeCount: p.net.edges.length };
}

const IMMUTABLE = new Set(["id", "source", "target"]);

/** Change only what differs (ids and endpoints are immutable in cytoscape). */
function update(el: cytoscape.CollectionReturnValue, d: cytoscape.ElementDefinition): void {
  const data = d.data as Record<string, unknown>;
  for (const [k, v] of Object.entries(data)) if (!IMMUTABLE.has(k) && el.data(k) !== v) el.data(k, v);
  if (!("image" in data) && el.data("image") !== undefined) el.removeData("image");
  const cls = String(d.classes ?? "");
  if (el.classes().join(" ") !== cls) el.classes(cls);
}

/** Add, update and remove elements in place; true when the element set changed. */
function sync(cy: cytoscape.Core, defs: cytoscape.ElementDefinition[]): boolean {
  let changed = false;
  const want = new Set(defs.map((d) => String(d.data.id)));
  cy.batch(() => {
    cy.elements().forEach((el) => {
      if (!want.has(el.id())) {
        el.remove();
        changed = true;
      }
    });
    for (const d of defs.filter((x) => x.group === "nodes").concat(defs.filter((x) => x.group === "edges"))) {
      const el = cy.getElementById(String(d.data.id));
      if (el.empty()) {
        cy.add(d);
        changed = true;
      } else {
        update(el, d);
      }
    }
  });
  return changed;
}

export function NetworkView(props: NetworkViewProps) {
  const { net, height = 420 } = props;
  const container = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const handlers = useLatest({ node: props.onSelectNode, edge: props.onSelectEdge });
  const { rdkit } = useRDKit();
  const images = useMemo(() => {
    const out = new Map<string, string | null>();
    if (rdkit) for (const n of net.nodes) out.set(n.id, nodeImage(rdkit, n.display_smiles));
    return out;
  }, [rdkit, net.nodes]);
  const built = buildElements(props, images);

  useEffect(() => {
    if (!container.current) return;
    const cy = cytoscape({ container: container.current, style: stylesheet(), minZoom: 0.2, maxZoom: 3 });
    cy.on("tap", "node", (evt) => handlers.current.node(evt.target.id()));
    cy.on("tap", "edge", (evt) => {
      if (!evt.target.data("virtual")) handlers.current.edge(evt.target.id());
    });
    cyRef.current = cy;
    const ro = new ResizeObserver(() => cy.resize());
    ro.observe(container.current);
    return () => {
      ro.disconnect();
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    if (sync(cy, built.defs)) {
      const roots = net.nodes.some((n) => n.id === net.root) ? [net.root] : undefined;
      cy.layout({ name: "breadthfirst", directed: true, roots, spacingFactor: 1.15, padding: 24, fit: true, animate: false }).run();
    }
  });

  return (
    <div
      ref={container}
      className="network-view"
      data-testid="network-view"
      data-node-count={built.nodeCount}
      data-edge-count={built.edgeCount}
      data-revision={net.revision}
      style={{ height }}
      aria-label={`reaction network: ${built.nodeCount} nodes, ${built.edgeCount} edges (the list beside it mirrors it)`}
    />
  );
}
