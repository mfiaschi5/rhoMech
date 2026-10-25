// NetworkLegend: the visible DOM mirror of the cytoscape network (design §11.6). cytoscape draws
// on a canvas, so every node and edge is also a real, focusable <button> row here; a click
// selects it exactly like a tap on the graph.
//
// Props:
//   net            NetworkOut
//   selectedNode   string | null
//   selectedEdge   string | null
//   pathway        PathwayOut | null          its members are marked
//   onSelectNode   (id) => void
//   onSelectEdge   (id) => void
//   probabilities  boolean                    scores are probabilities (the edge's number)
//
// DOM: <details data-testid="predict-network-list" open> "Network list" > <ul>
//   <li data-testid="predict-network-node-{id}" data-root data-status data-terminal-reason data-depth><button/></li>
//   <li data-testid="predict-network-edge-{id}" data-kind data-status data-rank data-closes-cycle
//       data-recurrence data-source data-target><button/></li>
// Root edges are the edge items whose data-source equals the root node's id. Cycles are listed
// after the edges (data-testid="predict-network-cycle-{id}"); they are not edges.
import { shortId } from "../../components/format";
import { matrixText, rankLabel, rankOf, scoreText, speciesLabel } from "./network";
import type { EdgeOut, NetworkOut, NodeOut, PathwayOut } from "./v9types";

export interface NetworkLegendProps {
  net: NetworkOut;
  selectedNode: string | null;
  selectedEdge: string | null;
  pathway: PathwayOut | null;
  onSelectNode: (id: string) => void;
  onSelectEdge: (id: string) => void;
  probabilities: boolean;
}

function NodeItem({ n, selected, inPathway, onSelect }: { n: NodeOut; selected: boolean; inPathway: boolean; onSelect: () => void }) {
  return (
    <li
      data-testid={`predict-network-node-${n.id}`}
      className={`legend-item legend-node status-${n.status}${selected ? " selected" : ""}${inPathway ? " in-pathway" : ""}`}
      data-root={n.is_root ? "true" : "false"}
      data-status={n.status}
      data-terminal-reason={n.terminal_reason ?? ""}
      data-depth={n.depth}
    >
      <button type="button" className="legend-button" aria-pressed={selected} onClick={onSelect} title={n.pool_key}>
        <span className="legend-kind">{n.is_root ? "root" : `d${n.depth}`}</span>
        <span className="legend-main">{speciesLabel(n)}</span>
        <span className={`badge legend-status status-${n.status}`}>
          {n.status}
          {n.terminal_reason ? `: ${n.terminal_reason}` : ""}
          {n.failure ? `: ${n.failure.code}` : ""}
        </span>
      </button>
    </li>
  );
}

function EdgeItem({ e, nodes, selected, inPathway, onSelect, probabilities }: { e: EdgeOut; nodes: Map<string, NodeOut>; selected: boolean; inPathway: boolean; onSelect: () => void; probabilities: boolean }) {
  const src = nodes.get(e.source);
  const tgt = nodes.get(e.target);
  return (
    <li
      data-testid={`predict-network-edge-${e.id}`}
      data-rank={rankOf(e) ?? ""}
      className={`legend-item legend-edge kind-${e.kind} edge-${e.status}${selected ? " selected" : ""}${inPathway ? " in-pathway" : ""}`}
      data-kind={e.kind}
      data-status={e.status}
      data-closes-cycle={e.closes_cycle ? "true" : "false"}
      data-recurrence={e.recurrence ? "true" : "false"}
      data-source={e.source}
      data-target={e.target}
    >
      <button type="button" className="legend-button" aria-pressed={selected} onClick={onSelect} title={`${src?.pool_key ?? e.source} → ${e.target_pool_key}`}>
        <span className="legend-kind">{rankLabel(e)}</span>
        <span className="legend-main">
          {src ? speciesLabel(src) : shortId(e.source)} → {tgt ? speciesLabel(tgt) : shortId(e.target)}
          <span className="muted"> ({matrixText(e.matrix)})</span>
        </span>
        <span className="legend-prob num">{scoreText(e, probabilities)}</span>
        {e.status === "superseded" && <span className="badge">superseded</span>}
        {e.recurrence && <span className="badge">recurrence</span>}
        {e.closes_cycle && <span className="badge legend-cycle">closes cycle</span>}
      </button>
    </li>
  );
}

export function NetworkLegend(props: NetworkLegendProps) {
  const { net, pathway } = props;
  const nodes = new Map(net.nodes.map((n) => [n.id, n] as const));
  const pNodes = new Set(pathway?.node_ids ?? []);
  const pEdges = new Set(pathway?.edge_ids ?? []);
  return (
    <details className="network-legend-panel" data-testid="predict-network-list" open>
      <summary>
        Network list <span className="muted">({net.nodes.length} nodes, {net.edges.length} edges, {net.cycles.length} cycles)</span>
      </summary>
      <ul className="network-legend">
        {net.nodes.map((n) => (
          <NodeItem key={n.id} n={n} selected={n.id === props.selectedNode} inPathway={pNodes.has(n.id)} onSelect={() => props.onSelectNode(n.id)} />
        ))}
        {net.edges.map((e) => (
          <EdgeItem key={e.id} e={e} nodes={nodes} selected={e.id === props.selectedEdge} inPathway={pEdges.has(e.id)} onSelect={() => props.onSelectEdge(e.id)} probabilities={props.probabilities} />
        ))}
        {net.cycles.map((c) => (
          <li key={c.id} className="legend-item legend-cycle-item" data-testid={`predict-network-cycle-${c.id}`} data-cycle-kind={c.kind} data-anchor={c.anchor}>
            <span className="legend-kind">{c.kind}</span>
            <span className="legend-main">
              anchor {shortId(nodes.get(c.anchor)?.pool_key ?? c.anchor, 24)}, {c.edge_ids.length} edges
              {c.virtual_edge?.label ? `, turnover ${c.virtual_edge.label}` : ""}
              {Object.keys(c.catalyst).length ? `, catalyst ${Object.keys(c.catalyst).join(", ")}` : ""}
            </span>
          </li>
        ))}
      </ul>
    </details>
  );
}
