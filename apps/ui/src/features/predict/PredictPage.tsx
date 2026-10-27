// PredictPage: #/predict (the input form) and #/predict/job/<jobId> (the job view, spec v9).
// The job view polls the job and shows, as soon as a (partial) network exists: the reaction
// network (cytoscape; nodes are pool states, edges are steps) with its DOM list, the ranked
// pathways, and the selected step (StepPanel: R on the pool, the arrows, the product, the
// orbital reading) or state (NodePanel). Every selection lives in the hash (selection.ts).
//
// Props: PageProps {route}.
// DOM: predict-job[data-job-id], predict-job-header, predict-job-checkpoint (network jobs:
// the checkpoint the job ran, data-run, data-file, data-sha; a smoke run badged), the JobProgress ids (predict-job-status ...),
// predict-network, predict-network-list, predict-pathways, predict-step-panel, predict-node-panel.
import { useJobPoll } from "../../api/poll";
import type { JobOut, Provenance } from "../../api/types.gen";
import { CapabilityNotice } from "../../components/CapabilityNotice";
import { ProvenanceBadge } from "../../components/ProvenanceBadge";
import type { PageProps } from "../../registry";
import { predictPrefillHash, type Route } from "../../router";
import { useCapability } from "../../state/store";
import { recordedLabel } from "./checkpoints";
import { usePredictors } from "./hooks";
import { JobProgress } from "../../components/JobProgress";
import { currentPathway } from "@ksmech/viz/network/network";
import { NetworkPanel } from "./netview/NetworkPanel";
import { useNetHeight } from "./netview/prefs";
import { NodePanel } from "./NodePanel";
import { PathwayList } from "./PathwayList";
import { PredictForm } from "./PredictForm";
import { scoreBadge } from "./PredictorSelect";
import { parseSelection, useSelect, type JobSelection, type SelectionUpdate } from "./selection";
import { StepPanel } from "./step/StepPanel";
import { networkOfJob } from "../../api/predict";
import type { NetworkOut } from "@ksmech/viz/v9types";

export function PredictPage({ route }: PageProps) {
  const cap = useCapability("predict");
  if (cap.known && !cap.enabled) return <CapabilityNotice name="predict" title="Predict is disabled" />;
  // ?rev= changes after an expansion: the job view remounts and reads the grown network
  if (route.parts[0] === "job" && route.parts[1]) return <JobView key={`${route.parts[1]}|${route.params.get("rev") ?? ""}`} jobId={route.parts[1]} route={route} />;
  const prefillKey = route.params.getAll("r").join("|") + "#" + route.params.getAll("n").join("|");
  return <PredictForm key={prefillKey} route={route} />;
}

function JobHeader({ job, net, probabilities, badge }: { job: JobOut | null; net: NetworkOut | null; probabilities: boolean; badge: string | null }) {
  const req = net?.request;
  return (
    <div className="row pv-inline-row pv-job-header" data-testid="predict-job-header" data-probabilities={probabilities ? "true" : "false"}>
      <h2>Prediction</h2>
      {req && <span className="pv-wrap">{req.reactants_smiles.map((smi, i) => `${(req.stoichiometry?.[i] ?? 1) > 1 ? `${req.stoichiometry?.[i]} ` : ""}${smi}`).join(" + ")}</span>}
      {net && (
        <span className="muted">
          {net.provenance.predictor}, beam {net.params.beam}, max steps {net.params.max_steps}
        </span>
      )}
      {badge && <span className="badge badge-warn" data-testid="predict-score-badge">{badge}</span>}
      <CheckpointLine provenance={net?.provenance ?? job?.provenance ?? null} />
      <ProvenanceBadge provenance={net?.provenance ?? job?.provenance} compact />
      {req && (
        <a className="btn btn-small" href={predictPrefillHash(req.reactants_smiles.map((smiles, i) => ({ smiles, count: req.stoichiometry?.[i] ?? 1 })))}>
          new prediction from these reactants
        </a>
      )}
    </div>
  );
}

function CheckpointLine({ provenance }: { provenance: Provenance | null }) {
  const ck = provenance?.checkpoint;
  if (!ck) return null;
  return (
    <span className="muted pv-wrap" data-testid="predict-job-checkpoint" data-run={ck.run} data-file={ck.file} data-sha={ck.sha256} title={`${ck.source}/${ck.run}/${ck.file}.pt, sha256 ${ck.sha256}`}>
      checkpoint {recordedLabel(ck)}
      {ck.smoke && <span className="badge badge-warn"> smoke</span>}
    </span>
  );
}

function NetworkSection({ jobId, net, sel, select, probabilities, onChanged }: { jobId: string; net: NetworkOut; sel: JobSelection; select: (u: SelectionUpdate) => void; probabilities: boolean; onChanged: () => void }) {
  const pathway = currentPathway(net, sel.pathway);
  const edge = sel.edge ? net.edges.find((e) => e.id === sel.edge) : undefined;
  const node = sel.node ? net.nodes.find((n) => n.id === sel.node) : undefined;
  const onNode = (id: string) => select({ node: id, edge: null });
  const onEdge = (id: string) => select({ edge: id, node: null });
  const onPathway = (id: string) => select({ pathway: id, edge: net.pathways.find((p) => p.id === id)?.edge_ids[0] ?? null, node: null });
  const [height, setHeight] = useNetHeight();
  return (
    <>
      <div className="pv-job-grid">
        <NetworkPanel height={height} onHeight={setHeight} jobId={jobId} net={net} selectedNode={sel.node} selectedEdge={sel.edge} pathway={pathway} pathwayFocus={!!sel.pathway} probabilities={probabilities} onSelectNode={onNode} onSelectEdge={onEdge} onSelectPathway={onPathway} onNetworkChanged={onChanged} />
        <div className="pv-side-fill">
          <PathwayList jobId={jobId} net={net} current={pathway} onSelect={onPathway} probabilities={probabilities} selectedEdge={sel.edge} onEdge={onEdge} />
        </div>
      </div>
      {sel.edge && !edge && <div className="banner banner-error">step {sel.edge} is not in this network</div>}
      {sel.node && !node && <div className="banner banner-error">state {sel.node} is not in this network</div>}
      {edge && <StepPanel jobId={jobId} edge={edge} net={net} probabilities={probabilities} />}
      {node && <NodePanel jobId={jobId} node={node} net={net} probabilities={probabilities} onEdge={onEdge} />}
    </>
  );
}

function JobView({ jobId, route }: { jobId: string; route: Route }) {
  const { job, error } = useJobPoll(jobId);
  const sel = parseSelection(route);
  const select = useSelect(route);
  const net = networkOfJob(job);
  const predictors = usePredictors();
  const info = predictors.data?.find((p) => p.name === net?.provenance.predictor);
  const probabilities = net?.scores_are_probabilities ?? info?.scores_are_probabilities ?? !!net?.edges.some((e) => e.logp !== null);
  return (
    <div className="pv-page stack" data-testid="predict-job" data-job-id={jobId}>
      <JobHeader job={job} net={net} probabilities={probabilities} badge={scoreBadge(info)} />
      <JobProgress job={job} pollError={error} title="Prediction job" />
      {net ? <NetworkSection jobId={jobId} net={net} sel={sel} select={select} probabilities={probabilities} onChanged={() => select({ rev: Date.now() })} /> : job && <div className="muted">no network yet</div>}
    </div>
  );
}
