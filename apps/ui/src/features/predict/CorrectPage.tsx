// CorrectPage: the R editor (spec v9), on a state of a job's network:
//   #/correct/<jobId>/edge/<edgeId>   the edge's parent state, its R preloaded (from Predict's
//                                     "Correct this step")
//   #/correct/<jobId>/node/<nodeId>   that state, R empty
//   #/correct/<jobId>                 the network's root
//   #/correct?r=<smiles>&n=<count>..  bare reactants: a job without prediction (predictor
//                                     "user") is created, then its root opens
// Apply adds the step to the network and opens its product state, so the mechanism is built
// step by step; ?path=<edge ids> keeps the steps applied so far (else the best-parent path from
// the root) and the breadcrumb shows it. Save to curation stores that path (POST /curate,
// action save) and opens the mechanism. "Predict from here" runs the network's predictor from
// the current state (the POST /jobs/{id}/expand job, polled to its end) and opens the network
// in Predict.
//
// DOM: correct-page[data-job-id][data-node], correct-breadcrumb > a[data-testid=correct-crumb-<k>],
// correct-save, correct-save-error, correct-expand, correct-expand-error, correct-start (the bare-reactant form), the CorrectEditor ids.
import { useCallback, useState } from "react";
import { curate, toApiError, type ApiError } from "../../api/client";
import { pollJob, useJobPoll } from "../../api/poll";
import { CapabilityNotice } from "../../components/CapabilityNotice";
import { ErrorBanner } from "../../components/ErrorBanner";
import type { PageProps } from "../../registry";
import { formatRoute, jobHash, navigate, parsePredictPrefill, type Route } from "../../router";
import { useCapability } from "../../state/store";
import { CorrectEditor } from "./CorrectEditor";
import { JobProgress } from "./JobProgress";
import { pathTo, rankLabel, speciesLabel } from "./network";
import { Panel } from "./Panel";
import { ReactantRows, usePreflight, type ReactantRow } from "./ReactantRows";
import { expandNode, networkOfJob, predictV9 } from "./v9api";
import type { EdgeOut, NetworkOut, NodeOut } from "./v9types";

/** Steps applied in this tab, by job: the network a done job carries is a snapshot, so the
 *  states and edges added since are kept here until a reload reads them from the server. */
const applied = new Map<string, { edges: EdgeOut[]; nodes: NodeOut[] }>();

function withApplied(jobId: string, net: NetworkOut | null): NetworkOut | null {
  const extra = applied.get(jobId);
  if (!net || !extra) return net;
  const ids = new Set(net.nodes.map((n) => n.id));
  const eids = new Set(net.edges.map((e) => e.id));
  return { ...net, nodes: [...net.nodes, ...extra.nodes.filter((n) => !ids.has(n.id))], edges: [...net.edges, ...extra.edges.filter((e) => !eids.has(e.id))] };
}

export const correctNodeHash = (jobId: string, nodeId: string, path: readonly string[] = []) =>
  formatRoute({ page: "correct", parts: [jobId, "node", nodeId], params: { path: path.length ? path.join(",") : null } });

/** The steps from the root to `nodeId`: the ?path= edges when they lead there, else best parents. */
function chainTo(net: NetworkOut, nodeId: string, pathParam: string | null): EdgeOut[] {
  const edges = new Map(net.edges.map((e) => [e.id, e] as const));
  const given = (pathParam ?? "").split(",").filter(Boolean).map((id) => edges.get(id));
  const ok = given.length > 0 && given.every((e, k): e is EdgeOut => !!e && (k === 0 || given[k - 1]?.target === e.source));
  if (ok && given[given.length - 1]?.target === nodeId) return given as EdgeOut[];
  return pathTo(net, nodeId);
}

export function CorrectPage({ route }: PageProps) {
  const cap = useCapability("correct");
  if (cap.known && !cap.enabled) return <CapabilityNotice name="correct" title="Correct is disabled" />;
  const jobId = route.parts[0];
  if (!jobId) return <StartForm route={route} />;
  return <JobCorrect key={jobId} jobId={jobId} route={route} />;
}

function StartForm({ route }: { route: Route }) {
  const prefill = parsePredictPrefill({ ...route, page: "predict" });
  const [rows, setRows] = useState<ReactantRow[]>(prefill.length ? prefill : [{ smiles: "", count: 1 }]);
  const preflight = usePreflight(rows);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const filled = rows.filter((r) => r.smiles.trim());
  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const created = await predictV9({ reactants_smiles: filled.map((r) => r.smiles.trim()), stoichiometry: filled.map((r) => r.count), predictor: "user", beam: 1, max_steps: 1 });
      navigate(formatRoute({ page: "correct", parts: [created.job_id] }));
    } catch (err) {
      setError(toApiError(err));
      setBusy(false);
    }
  };
  return (
    <div className="pv-page stack" data-testid="correct-start">
      <Panel title="Correct: draw a mechanism step by step" note="start from bare reactants; no prediction is run">
        <div className="stack">
          <ReactantRows rows={rows} onChange={setRows} preflight={preflight} prefix="correct" />
          <div className="row pv-inline-row">
            <button type="button" className="btn btn-primary" data-testid="correct-start-submit" disabled={busy || !filled.length} onClick={() => void start()}>
              {busy ? "starting…" : "Start"}
            </button>
          </div>
          <ErrorBanner error={error} title="Cannot start" testId="correct-start-error" />
        </div>
      </Panel>
    </div>
  );
}

function Breadcrumb({ jobId, net, nodeId, chain, editing }: { jobId: string; net: NetworkOut; nodeId: string; chain: EdgeOut[]; editing: EdgeOut | null }) {
  const nodes = new Map(net.nodes.map((n) => [n.id, n] as const));
  const ids = [chain[0]?.source ?? nodeId, ...chain.map((e) => e.target)];
  return (
    <nav className="pv-breadcrumb" data-testid="correct-breadcrumb" data-steps={chain.length} aria-label="pathway">
      {ids.map((id, k) => {
        const n = nodes.get(id);
        const text = n ? speciesLabel(n) : id;
        return (
          <span key={`${id}-${k}`} className="pv-crumb">
            {k > 0 && <span className="pv-crumb-sep">→</span>}
            <a href={correctNodeHash(jobId, id, chain.slice(0, k).map((e) => e.id))} data-testid={`correct-crumb-${k}`} data-node={id} aria-current={id === nodeId && !editing ? "page" : undefined} title={text}>
              {k === 0 ? "reactants" : `step ${k}`}: <span className="pv-crumb-text">{text}</span>
            </a>
          </span>
        );
      })}
      {editing && <span className="pv-crumb pv-crumb-edit">→ editing step {rankLabel(editing)}</span>}
    </nav>
  );
}

function SaveButton({ jobId, edgeIds }: { jobId: string; edgeIds: string[] }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      const m = await curate({ job_id: jobId, edge_ids: edgeIds, action: "save", notes: "built on the Correct page" });
      navigate(`#/curate/m/${m.id}`);
    } catch (err) {
      setError(toApiError(err));
      setBusy(false);
    }
  };
  return (
    <>
      <button type="button" className="btn" data-testid="correct-save" disabled={busy || !edgeIds.length} title={edgeIds.length ? `save the ${edgeIds.length}-step pathway to the curation database` : "apply a step first"} onClick={() => void save()}>
        {busy ? "saving…" : "Save to curation"}
      </button>
      <ErrorBanner error={error} title="Cannot save to curation" testId="correct-save-error" />
    </>
  );
}

function ExpandButton({ jobId, nodeId }: { jobId: string; nodeId: string }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const expand = async () => {
    setBusy(true);
    setError(null);
    try {
      const created = await expandNode(jobId, { node_id: nodeId, steps: 3 });
      const done = await pollJob(created.job_id);
      if (done.status !== "done") throw new Error(done.error ? `${done.error.code}: ${done.error.message}` : `the expansion ${done.status}`);
      navigate(jobHash(jobId, { node: nodeId }));
    } catch (err) {
      setError(toApiError(err));
      setBusy(false);
    }
  };
  return (
    <>
      <button type="button" className="btn" data-testid="correct-expand" disabled={busy} title="run the predictor from this state (3 steps) and open the network" onClick={() => void expand()}>
        {busy ? "predicting…" : "Predict from here"}
      </button>
      <ErrorBanner error={error} title="Cannot predict from this state" testId="correct-expand-error" />
    </>
  );
}

function JobCorrect({ jobId, route }: { jobId: string; route: Route }) {
  const { job, error } = useJobPoll(jobId);
  const net = withApplied(jobId, networkOfJob(job));
  const [kind, target] = [route.parts[1], route.parts[2]];
  const edge = kind === "edge" ? (net?.edges.find((e) => e.id === target) ?? null) : null;
  const nodeId = kind === "edge" ? (edge?.source ?? null) : kind === "node" ? (target ?? null) : (net?.root ?? null);
  const node = nodeId ? net?.nodes.find((n) => n.id === nodeId) : undefined;
  const chain = net && node ? chainTo(net, node.id, route.params.get("path")) : [];
  const chainKey = chain.map((e) => e.id).join(",");
  const onApplied = useCallback(
    (e: EdgeOut, n: NodeOut) => {
      const a = applied.get(jobId) ?? { edges: [], nodes: [] };
      applied.set(jobId, { edges: [...a.edges, e], nodes: [...a.nodes, n] });
      navigate(correctNodeHash(jobId, n.id, [...chainKey.split(",").filter(Boolean), e.id]));
    },
    [jobId, chainKey],
  );
  return (
    <div className="pv-page stack" data-testid="correct-page" data-job-id={jobId} data-node={node?.id ?? ""}>
      <div className="row pv-inline-row">
        <h2>Correct</h2>
        <a className="btn btn-small" href={jobHash(jobId, edge ? { edge: edge.id } : node ? { node: node.id } : {})} data-testid="correct-open-predict">
          open in Predict
        </a>
      </div>
      {(!job || job.status !== "done") && <JobProgress job={job} pollError={error} title="Job" />}
      {net && node && <Breadcrumb jobId={jobId} net={net} nodeId={node.id} chain={chain} editing={edge} />}
      {net && !node && <div className="banner banner-error">{kind === "edge" ? `step ${target} is not in this network` : `state ${nodeId ?? "?"} is not in this network`}</div>}
      {node && (
        <CorrectEditor
          key={`${node.state_id}|${edge?.id ?? ""}`}
          jobId={jobId}
          stateId={node.state_id}
          initial={edge ? { matrix: edge.matrix, arrows: edge.arrows } : null}
          onApplied={onApplied}
          extra={
            <>
              <SaveButton jobId={jobId} edgeIds={chain.map((e) => e.id)} />
              <ExpandButton jobId={jobId} nodeId={node.id} />
            </>
          }
        />
      )}
    </div>
  );
}
