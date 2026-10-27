// NodePanel: the selected pool state of the Predict job view: its species, status, the pool
// drawing, its outgoing steps, and "Correct from here" (an empty R editor on this state).
//
// Props: jobId, node, net, probabilities, onEdge(edgeId).
// DOM: <section data-testid="predict-node-panel" data-node data-status>, predict-node-canvas,
// predict-node-out-<k>, predict-correct-node.
import { ErrorBanner } from "../../components/ErrorBanner";
import { correctHash } from "../../router";
import { usePool } from "./hooks";
import { matrixText, rankLabel, rankOf, scoreText, speciesLabel } from "@ksmech/viz/network/network";
import { Panel } from "../../components/Panel";
import { PoolCanvas } from "./PoolCanvas";
import type { NetworkOut, NodeOut } from "@ksmech/viz/v9types";

export interface NodePanelProps {
  jobId: string;
  node: NodeOut;
  net: NetworkOut;
  probabilities: boolean;
  onEdge: (edgeId: string) => void;
}

export function NodePanel({ jobId, node, net, probabilities, onEdge }: NodePanelProps) {
  const pool = usePool(node.state_id);
  const out = net.edges.filter((e) => e.source === node.id).sort((a, b) => (rankOf(a) ?? 99) - (rankOf(b) ?? 99));
  const tools = (
    <a className="btn btn-small btn-primary" data-testid="predict-correct-node" href={correctHash(jobId, { node: node.id })}>
      Correct from here
    </a>
  );
  return (
    <section data-testid="predict-node-panel" data-node={node.id} data-status={node.status}>
      <Panel title="State" note={`${speciesLabel(node)} · ${node.status}${node.terminal_reason ? ` (${node.terminal_reason})` : ""}`} tools={tools}>
        <div className="stack">
          {pool.structure && <PoolCanvas structure={pool.structure} testId="predict-node-canvas" height={260} />}
          <ErrorBanner error={pool.error} title="Cannot draw the pool" />
          {node.failure && <div className="banner banner-error">{node.failure.code}: {node.failure.message}</div>}
          {out.length > 0 && (
            <ol className="pv-steps">
              {out.map((e, k) => (
                <li key={e.id}>
                  <button type="button" className="legend-button" data-testid={`predict-node-out-${k}`} onClick={() => onEdge(e.id)}>
                    <span className="legend-kind">{rankLabel(e)}</span>
                    <span className="legend-main">{matrixText(e.matrix, (i) => pool.structure?.label(i) ?? String(i))}</span>
                    <span className="num">{scoreText(e, probabilities)}</span>
                  </button>
                </li>
              ))}
            </ol>
          )}
        </div>
      </Panel>
    </section>
  );
}
