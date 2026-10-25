// PathwayStrip: the selected pathway as a horizontal strip of pool depictions (design §11.8).
// Panel i is the source state of step i with that step's curved arrows; the step probability
// sits between panels; the last panel is the final product. Clicking an arrow selects the edge
// and opens the physics drawer for that arrow.
//
// Props:
//   net             NetworkOut
//   pathway         PathwayOut | null
//   selectedEdge    string | null
//   selectedArrow   number | null
//   onArrow         (edgeId, arrowIndex) => void
//   onEdge          (edgeId) => void
//
// DOM: <div data-testid="pathway-strip" data-pathway> > strip-step-{i}[data-edge] (each holding a
// mol-depiction with its arrow-overlay groups), then strip-final.
import type { EdgeOut, NetworkOut, PathwayOut } from "../../api/types.gen";
import { fmtProb, shortId } from "../../components/format";
import { edgeArrowsText, speciesLabel } from "./network";
import { StateDepiction } from "./StateDepiction";

export interface PathwayStripProps {
  net: NetworkOut;
  pathway: PathwayOut | null;
  selectedEdge: string | null;
  selectedArrow: number | null;
  onArrow: (edgeId: string, index: number) => void;
  onEdge: (edgeId: string) => void;
}

function StepSeparator({ e, onEdge }: { e: EdgeOut; onEdge: () => void }) {
  return (
    <div className={`strip-sep kind-${e.kind}`}>
      <button type="button" className="btn btn-small" onClick={onEdge} title={edgeArrowsText(e)}>
        {e.kind === "corrected" ? <span className="badge badge-corrected">corrected</span> : `p ${fmtProb(e.probability)}`}
      </button>
      <span className="strip-arrow" aria-hidden="true">
        ⟶
      </span>
      {e.status === "superseded" && <span className="badge">superseded</span>}
    </div>
  );
}

export function PathwayStrip({ net, pathway, selectedEdge, selectedArrow, onArrow, onEdge }: PathwayStripProps) {
  if (!pathway) return null;
  const edges = new Map(net.edges.map((e) => [e.id, e] as const));
  const nodes = new Map(net.nodes.map((n) => [n.id, n] as const));
  const steps = pathway.edge_ids.map((id) => edges.get(id)).filter((e): e is EdgeOut => !!e);
  const last = steps[steps.length - 1];
  return (
    <section className="panel strip-panel">
      <h3>
        Pathway #{pathway.rank} <span className="muted">({pathway.n_steps} steps, p {fmtProb(pathway.probability)}, ends {pathway.ends_with})</span>
      </h3>
      <div className="pathway-strip" data-testid="pathway-strip" data-pathway={pathway.id}>
        {steps.map((e, i) => {
          const src = nodes.get(e.source);
          return (
            <div key={e.id} className="strip-cell">
              <div className={`strip-step kind-${e.kind}${e.id === selectedEdge ? " selected" : ""}`} data-testid={`strip-step-${i}`} data-edge={e.id}>
                <StateDepiction
                  stateId={e.source_state_id}
                  arrows={e.arrows}
                  edgeId={e.id}
                  onArrowClick={(k) => onArrow(e.id, k)}
                  selectedArrow={e.id === selectedEdge ? selectedArrow : null}
                  width={260}
                  height={180}
                  caption={src ? speciesLabel(src) : shortId(e.source)}
                />
              </div>
              <StepSeparator e={e} onEdge={() => onEdge(e.id)} />
            </div>
          );
        })}
        {last && (
          <div className="strip-cell">
            <div className="strip-step strip-final" data-testid="strip-final" data-state-id={last.product_state_id}>
              <StateDepiction stateId={last.product_state_id} width={260} height={180} caption={last.product_smiles} />
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
