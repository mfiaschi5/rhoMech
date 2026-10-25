// NodePanel: the selected network node (design §11.8): its species and status, the failure or
// terminal reason, the expansion counters, its pool depiction, a "correct" link, the tokens, and
// the candidates toggle. The candidates panel lists the node's tuples (CandidateTable); "preview"
// draws one on the node depiction (CandidatePreview), and clicking that arrow opens the physics
// drawer.
//
// Props:
//   jobId        string
//   node         NodeOut
//   sel          JobSelection            hash selection (cands, cand, cdonor, cexcl, phys)
//   select       (updates) => void       writes the hash
//   predictor    string                  the network's predictor (candidate ranking)
//   temperature  number | null
//   scope        "all" | "inter"
//
// DOM: node-panel[data-node], node-candidates-toggle, node-correct-link, node-tokens-toggle,
// then candidates-panel and candidate-preview.
import { useEffect, useRef, useState } from "react";
import type { CandidateOut, NodeOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import { correctHash } from "../../router";
import { CandidatePreview } from "./CandidatePreview";
import { CandidateTable } from "./CandidateTable";
import { fmt, shortId } from "../../components/format";
import { bodyError } from "./JobProgress";
import { speciesLabel } from "./network";
import { candParam, type JobSelection, type SelectionUpdate } from "./selection";
import { StateDepiction } from "./StateDepiction";
import { TokenTable } from "./TokenTable";

export interface NodePanelProps {
  jobId: string;
  node: NodeOut;
  sel: JobSelection;
  select: (updates: SelectionUpdate) => void;
  predictor: string;
  temperature: number | null;
  scope: "all" | "inter";
}

function NodeFacts({ node }: { node: NodeOut }) {
  const failures = Object.entries(node.apply_failures);
  return (
    <dl className="kv">
      <dt>pool key</dt>
      <dd>
        <code className="code">{node.pool_key}</code>
      </dd>
      <dt>state</dt>
      <dd>
        <code className="code">{node.state_id}</code>
        {node.alias_state_ids.length > 0 && <span className="muted"> (+{node.alias_state_ids.length} alias states)</span>}
      </dd>
      <dt>depth / best log p</dt>
      <dd>
        {node.depth} / {fmt(node.best_logp, 3)}
      </dd>
      <dt>candidates</dt>
      <dd>
        {node.n_valid ?? "–"} valid pairs; scanned mass {fmt(node.scanned_mass, 3)}, unscanned {fmt(node.unscanned_mass, 3)}
      </dd>
      {failures.length > 0 && (
        <>
          <dt>apply failures</dt>
          <dd>{failures.map(([k, n]) => `${k} ${n}`).join(", ")}</dd>
        </>
      )}
      {node.warnings.length > 0 && (
        <>
          <dt>warnings</dt>
          <dd>{node.warnings.join("; ")}</dd>
        </>
      )}
    </dl>
  );
}

function CandidatesSection({ node, sel, select, predictor, temperature, scope }: Omit<NodePanelProps, "jobId">) {
  const [preview, setPreview] = useState<CandidateOut | null>(null);
  const onPreview = (c: CandidateOut) => {
    setPreview(c);
    select({ cand: candParam(c), phys: null, threed: null, arrow: null });
  };
  return (
    <div className="candidates-layout">
      <CandidateTable
        stateId={node.state_id}
        predictor={predictor}
        temperature={temperature}
        scope={scope}
        donor={sel.candDonor}
        onDonorChange={(d) => select({ cdonor: d })}
        showExcluded={sel.candExcluded}
        onShowExcludedChange={(on) => select({ cexcl: on ? 1 : null })}
        activeTuple={sel.cand}
        onPreview={onPreview}
      />
      {sel.cand && (
        <div className="candidates-preview-col">
          <CandidatePreview
            stateId={node.state_id}
            tuple={sel.cand}
            candidate={preview}
            selected={sel.candPhysics}
            onArrowClick={() => select({ phys: "cand", arrow: null })}
            onClose={() => select({ cand: null, phys: null, threed: null })}
          />
        </div>
      )}
    </div>
  );
}

export function NodePanel(props: NodePanelProps) {
  const { jobId, node, sel, select } = props;
  const [tokensOpen, setTokensOpen] = useState(false);
  const ref = useRef<HTMLElement | null>(null);
  useEffect(() => {
    ref.current?.scrollIntoView?.({ block: "nearest" });
  }, [node.id]);
  return (
    <section ref={ref} className="panel node-panel" data-testid="node-panel" data-node={node.id} data-status={node.status}>
      <div className="row">
        <h3>
          {node.is_root ? "Root" : "Node"}: {speciesLabel(node)}
        </h3>
        <span className={`badge status-${node.status}`}>
          {node.status}
          {node.terminal_reason ? `: ${node.terminal_reason}` : ""}
        </span>
        {node.fermi_smeared && <span className="badge badge-warn">Fermi-smeared</span>}
        <a className="btn btn-small" data-testid="node-correct-link" href={correctHash(jobId, { node: node.id })}>
          correct from this node
        </a>
      </div>
      {node.failure && <ErrorBanner error={bodyError(node.failure)} title={`node failed (${shortId(node.id)})`} testId="node-failure" />}
      <div className="row">
        <StateDepiction stateId={node.state_id} width={320} height={220} />
        <NodeFacts node={node} />
      </div>
      <div className="row">
        <button type="button" className="btn" data-testid="node-candidates-toggle" aria-pressed={sel.cands} onClick={() => select({ cands: sel.cands ? null : 1, cand: null, phys: null, threed: null })}>
          {sel.cands ? "hide candidates" : "candidates"}
        </button>
        <button type="button" className="btn" data-testid="node-tokens-toggle" aria-pressed={tokensOpen} onClick={() => setTokensOpen((o) => !o)}>
          {tokensOpen ? "hide tokens" : "tokens"}
        </button>
      </div>
      {tokensOpen && <TokenTable stateId={node.state_id} />}
      {sel.cands && <CandidatesSection {...props} />}
    </section>
  );
}
