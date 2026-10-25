// StepPanel: the selected edge of the Predict job view: its kind, rank, predictor and score (a
// log-probability when the predictor's scores are probabilities, else the raw score), the anchor,
// the StepView (pool with R and the arrows, the product), the R entries, the orbital reading
// table and "Correct this step" (the Correct page on the parent state with this R preloaded).
//
// Props: jobId, edge, net, probabilities.
// Also the beam weight, the reading as a PMechDB arrow code, and the alternatives (other R with
// the same product).
// DOM: <section data-testid="predict-step-panel" data-edge data-kind> with predict-step-score
// [data-unit="logp|score"], predict-step-anchor, the StepView "predict-step", predict-step-entries,
// predict-step-pmechdb, predict-step-alternatives > li[data-testid=predict-step-alternative-<k>],
// the ReadingTable "predict-step", predict-correct-step.
import { exact, fmt, fmtSig } from "../../components/format";
import { correctHash } from "../../router";
import { matrixText, rankOf, speciesLabel } from "./network";
import { Panel } from "./Panel";
import { ReadingTable } from "./ReadingTable";
import { StepView } from "./StepView";
import { usePool } from "./hooks";
import type { EdgeOut, NetworkOut } from "./v9types";

export interface StepPanelProps {
  jobId: string;
  edge: EdgeOut;
  net: NetworkOut;
  probabilities: boolean;
}

function Score({ edge, probabilities }: { edge: EdgeOut; probabilities: boolean }) {
  if (edge.kind === "corrected") return <span className="badge badge-corrected" data-testid="predict-step-score" data-unit="user">user step</span>;
  const useLogp = probabilities && edge.logp !== null;
  return (
    <span className="badge" data-testid="predict-step-score" data-unit={useLogp ? "logp" : "score"} data-value={exact(useLogp ? edge.logp : edge.score)} title={useLogp ? "log-probability of the step" : "the predictor's score (not a probability)"}>
      {useLogp ? `log p ${fmt(edge.logp, 3)} (p ${fmt(Math.exp(edge.logp ?? 0), 3)})` : `score ${fmtSig(edge.score, 4)}`}
    </span>
  );
}

function StepFacts({ edge, label, probabilities }: { edge: EdgeOut; label: (i: number) => string; probabilities: boolean }) {
  return (
    <div className="stack pv-facts">
      <div className="muted pv-wrap">
        beam weight <code className="code">{fmt(edge.weight, 3)}</code>
        {edge.pmechdb_code && (
          <>
            {" "}
            · PMechDB code <code className="code" data-testid="predict-step-pmechdb">{edge.pmechdb_code}</code>
          </>
        )}
      </div>
      {edge.alternatives.length > 0 && (
        <details data-testid="predict-step-alternatives">
          <summary>{edge.alternatives.length} other reaction matri{edge.alternatives.length === 1 ? "x" : "ces"} with the same product</summary>
          <ol className="pv-steps">
            {edge.alternatives.map((a, k) => (
              <li key={k} data-testid={`predict-step-alternative-${k}`}>
                <code className="code">{matrixText(a.matrix, label)}</code> <span className="muted">{probabilities && a.logp !== null ? `log p ${fmt(a.logp, 3)}` : `score ${fmtSig(a.score, 3)}`}{a.rank > 0 ? `, rank ${a.rank}` : ""}</span>
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}

export function StepPanel({ jobId, edge, net, probabilities }: StepPanelProps) {
  const pool = usePool(edge.source_state_id);
  const s = pool.structure;
  const label = (i: number) => s?.label(i) ?? String(i);
  const src = net.nodes.find((n) => n.id === edge.source);
  const tools = (
    <a className="btn btn-small btn-primary" data-testid="predict-correct-step" href={correctHash(jobId, { edge: edge.id })} title="edit this step's reaction matrix on its parent state">
      Correct this step
    </a>
  );
  const note = (
    <>
      <span className="badge">{edge.kind}</span> {rankOf(edge) !== null && <span className="badge">rank {rankOf(edge)}</span>} <Score edge={edge} probabilities={probabilities} /> {edge.predictor && <span className="muted">{edge.predictor}</span>}
      {edge.status === "superseded" && <span className="badge">superseded</span>}
    </>
  );
  return (
    <section data-testid="predict-step-panel" data-edge={edge.id} data-kind={edge.kind}>
      <Panel title="Step" note={note} tools={tools}>
        <div className="stack">
          <div className="muted pv-wrap">
            from {src ? speciesLabel(src) : edge.source}
            {edge.anchor && (
              <>
                {" "}
                · anchor{" "}
                <span data-testid="predict-step-anchor" data-i={edge.anchor.i} data-j={edge.anchor.j} data-sign={edge.anchor.sign}>
                  {edge.anchor.sign === "form" ? "forms" : "breaks"} {label(edge.anchor.i)}–{label(edge.anchor.j)}
                </span>
              </>
            )}
          </div>
          <StepView stateId={edge.source_state_id} matrix={edge.matrix} arrows={edge.arrows} productSmiles={edge.product_smiles} sigmaKeys={edge.sigma_keys} chargesAfter={edge.formal_charges ? new Map(edge.formal_charges) : null} prefix="predict-step" />
          <div className="pv-wrap" data-testid="predict-step-entries">
            R: <code className="code">{matrixText(edge.matrix, label)}</code>
          </div>
          <StepFacts edge={edge} label={label} probabilities={probabilities} />
          <h4 className="pv-subtitle">Orbital reading</h4>
          <ReadingTable readings={edge.readings} structure={s} stateId={edge.source_state_id} prefix="predict-step" />
        </div>
      </Panel>
    </section>
  );
}
