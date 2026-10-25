// PredictPage: #/predict (the input form) and #/predict/job/<jobId> (the job view), design §11.8.
// The job view polls the job and shows, as soon as a (partial) network exists: NetworkView with
// its NetworkLegend, the cycles toggle, PathwayList, PathwayStrip, the EdgePanel / NodePanel of
// the selection, the network's skipped counts and failed proposals, and the physics drawer.
// Every selection lives in the hash (selection.ts).
//
// Props: PageProps {route}.
import { getJob } from "../../api/client";
import { useCached } from "../../api/cache";
import { useJobPoll } from "../../api/poll";
import type { JobOut, NetworkOut } from "../../api/types.gen";
import { CapabilityNotice } from "../../components/CapabilityNotice";
import { ErrorBanner } from "../../components/ErrorBanner";
import { ProvenanceBadge } from "../../components/ProvenanceBadge";
import type { PageProps } from "../../registry";
import { useCapability } from "../../state/store";
import { predictPrefillHash, type Route } from "../../router";
import { EdgePanel } from "./EdgePanel";
import { fmtProb, shortId } from "./format";
import { bodyError, JobProgress } from "./JobProgress";
import { currentPathway, cycleEdgeIds, networkOf, overrideOutcome, speciesLabel } from "./network";
import { NetworkLegend } from "./NetworkLegend";
import { NetworkView } from "./NetworkView";
import { NodePanel } from "./NodePanel";
import { PathwayList } from "./PathwayList";
import { PathwayStrip } from "./PathwayStrip";
import { PhysicsDrawer, drawerTarget } from "./PhysicsDrawer";
import { PredictForm } from "./PredictForm";
import { CLEAR_TARGET, parseSelection, useSelect, type JobSelection, type SelectionUpdate } from "./selection";

export function PredictPage({ route }: PageProps) {
  const cap = useCapability("predict");
  if (cap.known && !cap.enabled) return <CapabilityNotice name="predict" title="Predict is disabled" />;
  if (route.parts[0] === "job" && route.parts[1]) {
    const jobId = route.parts[1];
    // a new override result reloads the network (the override job rewrote it)
    return <JobView key={`${jobId}|${route.params.get("override") ?? ""}`} jobId={jobId} route={route} />;
  }
  const prefillKey = route.params.getAll("r").join("|") + "#" + route.params.getAll("n").join("|");
  return <PredictForm key={prefillKey} route={route} />;
}

function JobHeader({ job, net }: { job: JobOut | null; net: NetworkOut | null }) {
  const req = net?.request;
  return (
    <div className="row job-header">
      <h2>Prediction</h2>
      {req && (
        <span>
          {req.reactants_smiles.map((smi, i) => `${(req.stoichiometry?.[i] ?? 1) > 1 ? `${req.stoichiometry?.[i]} ` : ""}${smi}`).join(" + ")}
        </span>
      )}
      {net && (
        <span className="muted">
          {net.provenance.predictor}, beam {net.params.beam}, max steps {net.params.max_steps}, T {net.params.temperature}, scope {net.params.candidate_scope}
        </span>
      )}
      {req?.conditions && <span className="muted">conditions (recorded only): {req.conditions}</span>}
      <ProvenanceBadge provenance={net?.provenance ?? job?.provenance} compact />
      {req && (
        <a className="btn btn-small" href={predictPrefillHash(req.reactants_smiles.map((smiles, i) => ({ smiles, count: req.stoichiometry?.[i] ?? 1 })))}>
          new prediction from these reactants
        </a>
      )}
    </div>
  );
}

function OverrideResultBanner({ overrideId, net, select }: { overrideId: string; net: NetworkOut | null; select: (u: SelectionUpdate) => void }) {
  const res = useCached(`/jobs/${overrideId}#override`, () => getJob(overrideId));
  const job = res.data;
  const out = job ? overrideOutcome(job) : null;
  const edge = out ? net?.edges.find((e) => e.id === out.corrected_edge_id) : undefined;
  return (
    <div className="banner banner-ok override-result" data-testid="override-result" data-status={job?.status ?? "loading"} data-corrected-edge={out?.corrected_edge_id ?? ""}>
      <strong>{!job ? "Override" : job.status === "done" ? "Correction applied" : `Override job ${job.status}`}</strong>{" "}
      {out ? (
        <>
          corrected step <code className="code">{shortId(out.corrected_edge_id, 12)}</code> into node <code className="code">{shortId(out.target_node, 12)}</code>
          {out.continued_from ? `, prediction continued from ${shortId(out.continued_from, 12)}` : ", not continued"}
          {out.issues.length > 0 && `; issues: ${out.issues.map((i) => i.code).join(", ")}`}
          {edge && (
            <button type="button" className="btn btn-small" onClick={() => select({ edge: edge.id, node: null, ...CLEAR_TARGET })}>
              select it
            </button>
          )}
        </>
      ) : (
        <span className="muted">{res.error ? `cannot read override job ${overrideId}: ${res.error.code}` : "loading…"}</span>
      )}
      <ErrorBanner error={bodyError(job?.error)} title="override job error" testId="override-error" />
      <button type="button" className="btn btn-small banner-dismiss" onClick={() => select({ override: null })}>
        dismiss
      </button>
    </div>
  );
}

function NetworkDiagnostics({ net }: { net: NetworkOut }) {
  const skipped = Object.entries(net.skipped).filter(([, n]) => n > 0);
  if (!skipped.length && !net.failed_proposals.length) return null;
  const nodes = new Map(net.nodes.map((n) => [n.id, n] as const));
  return (
    <details className="panel network-diagnostics" data-testid="network-diagnostics">
      <summary>
        {skipped.reduce((s, [, n]) => s + n, 0)} skipped, {net.failed_proposals.length} failed proposals
      </summary>
      {skipped.length > 0 && <div>skipped: {skipped.map(([k, n]) => `${k} ${n}`).join(", ")}</div>}
      <ul>
        {net.failed_proposals.map((f, i) => (
          <li key={i}>
            {nodes.get(f.node) ? speciesLabel(nodes.get(f.node)!) : shortId(f.node)}, proposal {f.rank} (p {fmtProb(f.probability)}): <code className="code">{f.code}</code> {f.detail}
          </li>
        ))}
      </ul>
    </details>
  );
}

function NetworkSection({ jobId, net, sel, select }: { jobId: string; net: NetworkOut; sel: JobSelection; select: (u: SelectionUpdate) => void }) {
  const pathway = currentPathway(net, sel.pathway);
  const edge = sel.edge ? net.edges.find((e) => e.id === sel.edge) : undefined;
  const node = sel.node ? net.nodes.find((n) => n.id === sel.node) : undefined;
  const { edgeIds } = cycleEdgeIds(net, edge, pathway, sel.cycles);
  // re-selecting the current node or edge keeps its panels (candidates, drawer) as they are
  const onNode = (id: string) => id !== sel.node && select({ node: id, edge: null, ...CLEAR_TARGET });
  const onEdge = (id: string) => id !== sel.edge && select({ edge: id, node: null, ...CLEAR_TARGET });
  const scope = net.params.candidate_scope === "inter" ? "inter" : "all";
  const predictor = net.request.predictor ?? net.provenance.predictor;
  return (
    <>
      <section className="panel network-panel">
        <div className="row">
          <h3>Network</h3>
          <span className="muted">
            revision {net.revision}
            {net.complete ? "" : ", incomplete"}
          </span>
          <label className="field-inline">
            <input type="checkbox" data-testid="cycles-toggle" checked={sel.cycles} onChange={(e) => select({ cycles: e.target.checked ? 1 : null })} />
            highlight every cycle ({net.cycles.length})
          </label>
          <span className="legend-key muted">
            <span className="key key-corrected" /> corrected <span className="key key-superseded" /> superseded <span className="key key-turnover" /> turnover <span className="key key-cycle" /> cycle
          </span>
        </div>
        <div className="network-grid">
          <NetworkView net={net} selectedNode={sel.node} selectedEdge={sel.edge} pathway={pathway} cycleEdges={edgeIds} onSelectNode={onNode} onSelectEdge={onEdge} />
          <NetworkLegend net={net} selectedNode={sel.node} selectedEdge={sel.edge} pathway={pathway} onSelectNode={onNode} onSelectEdge={onEdge} />
        </div>
      </section>
      {sel.edge && !edge && <div className="banner banner-error">edge {sel.edge} is not in this network</div>}
      {sel.node && !node && <div className="banner banner-error">node {sel.node} is not in this network</div>}
      {edge && <EdgePanel jobId={jobId} net={net} edge={edge} sel={sel} select={select} />}
      {node && <NodePanel jobId={jobId} node={node} sel={sel} select={select} predictor={predictor} temperature={net.params.temperature} scope={scope} />}
      <PathwayList jobId={jobId} net={net} current={pathway} onSelect={(id) => select({ pathway: id })} />
      <PathwayStrip net={net} pathway={pathway} selectedEdge={sel.edge} selectedArrow={sel.arrow} onArrow={(id, k) => select({ edge: id, node: null, ...CLEAR_TARGET, arrow: k })} onEdge={onEdge} />
      <NetworkDiagnostics net={net} />
    </>
  );
}

function JobView({ jobId, route }: { jobId: string; route: Route }) {
  const { job, error } = useJobPoll(jobId);
  const sel = parseSelection(route);
  const select = useSelect(route);
  const net = networkOf(job);
  const drawer = net ? drawerTarget(net, sel) : null;
  return (
    <div className={`predict-job${drawer ? " with-drawer" : ""}`} data-testid="predict-job" data-job-id={jobId}>
      <div className="predict-main stack">
        <JobHeader job={job} net={net} />
        <JobProgress job={job} pollError={error} title="Prediction job" />
        {sel.override && <OverrideResultBanner overrideId={sel.override} net={net} select={select} />}
        {net ? <NetworkSection jobId={jobId} net={net} sel={sel} select={select} /> : job && <div className="muted">no network yet</div>}
      </div>
      {net && drawer && <PhysicsDrawer net={net} sel={sel} select={select} />}
    </div>
  );
}
