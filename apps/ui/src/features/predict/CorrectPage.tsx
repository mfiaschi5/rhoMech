// CorrectPage: #/correct/<jobId>/edge/<edgeId> and #/correct/<jobId>/node/<nodeId> (design §4.6,
// §11.8). It shows the original step (its arrows faded on the source depiction), the source
// state's CandidateTable (a row's "use" adds the tuple to the override), SitePicker for the
// selected override arrow, the ordered OverrideBuilder and a live preview of the user arrows.
// Submit posts /jobs/{id}/override, polls the override job and, once it is done, returns to
// #/predict/job/<parent>?edge=<corrected edge>&override=<override job>, where the corrected edge is
// selected and drawn as corrected. #/correct alone lists recent prediction jobs.
//
// Props: PageProps {route}.
//
// DOM: correct-panel, correct-original, candidate-row (with candidate-select-btn),
// site-option-{donor|acceptor}-{site}, override-arrow-{i}, override-remove-{i},
// override-allow-invalid, override-continue (continue steps), override-continue-prediction,
// override-submit, override-result.
import { useEffect, useRef, useState } from "react";
import { ApiError, isAbort, listJobs, overrideJob, toApiError } from "../../api/client";
import { useCached } from "../../api/cache";
import { pollJob, useJobPoll } from "../../api/poll";
import type { CandidateOut, EdgeOut, JobCreatedOut, JobOut, NetworkOut, NodeOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import type { PageProps } from "../../registry";
import { formatRoute, jobHash, navigate } from "../../router";
import { CandidateTable } from "./CandidateTable";
import { fmtProb, fmtSig, shortId } from "./format";
import { bodyError, JobProgress } from "./JobProgress";
import { arrowText, edgeArrowsText, networkOf, overrideOutcome, speciesLabel } from "./network";
import { builderArrowFrom, DEFAULT_OPTIONS, OverrideBuilder, toArrowIn, type BuilderArrow, type OverrideOptions } from "./OverrideBuilder";
import { SitePicker } from "./SitePicker";
import { StateDepiction } from "./StateDepiction";

export const MAX_OVERRIDE_ARROWS = 6;

export function CorrectPage({ route }: PageProps) {
  const [jobId, kind, target] = route.parts;
  if (!jobId) return <CorrectIndex />;
  if ((kind !== "edge" && kind !== "node") || !target) {
    return (
      <section className="panel" data-testid="correct-panel">
        <ErrorBanner error="use #/correct/<jobId>/edge/<edgeId> or #/correct/<jobId>/node/<nodeId>" title="Unknown correction target" />
      </section>
    );
  }
  return <CorrectView key={`${jobId}/${kind}/${target}`} jobId={jobId} kind={kind} targetId={target} />;
}

function CorrectIndex() {
  const [mounted] = useState(() => Date.now());
  const jobs = useCached(`/jobs?kind=predict&limit=20#${mounted}`, () => listJobs({ kind: "predict", limit: 20 }));
  return (
    <section className="panel" data-testid="correct-panel">
      <h2>Correct a predicted step</h2>
      <p className="muted">Pick a prediction, select a step or node there, and use its "correct" link.</p>
      <ErrorBanner error={jobs.error} title="Cannot list jobs" />
      <ul>
        {(jobs.data ?? []).map((j) => (
          <li key={j.job_id}>
            <a href={jobHash(j.job_id)}>{j.label || j.job_id}</a> <span className="badge">{j.status}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

interface Resolved {
  net: NetworkOut | null;
  edge: EdgeOut | undefined;
  node: NodeOut | undefined;
  stateId: string | null;
  missing: string | null;
}

function resolve(job: JobOut | null, kind: "edge" | "node", targetId: string): Resolved {
  const net = networkOf(job);
  if (!net) return { net, edge: undefined, node: undefined, stateId: null, missing: null };
  const edge = kind === "edge" ? net.edges.find((e) => e.id === targetId) : undefined;
  const nodeId = kind === "node" ? targetId : edge?.source;
  const node = net.nodes.find((n) => n.id === nodeId);
  const stateId = edge?.source_state_id ?? node?.state_id ?? null;
  const missing = kind === "edge" && !edge ? `edge ${targetId} is not in this network` : !node && kind === "node" ? `node ${targetId} is not in this network` : null;
  return { net, edge, node, stateId, missing };
}

function Original({ r }: { r: Resolved }) {
  const { edge, node } = r;
  return (
    <div className="correct-original panel" data-testid="correct-original" data-edge={edge?.id ?? ""} data-node={node?.id ?? ""}>
      {edge ? (
        <>
          <strong>Original step</strong> <span className={`badge${edge.kind === "corrected" ? " badge-corrected" : ""}`}>{edge.kind}</span> <span className="badge">{edge.status}</span>
          <div>
            <code className="code">{edgeArrowsText(edge)}</code> → <code className="code">{edge.product_smiles}</code>
          </div>
          <div className="muted">
            {edge.kind === "corrected" ? "a user correction" : `rank ${edge.rank ?? "–"}, p ${fmtProb(edge.probability)}, score ${fmtSig(edge.score, 4)} (${edge.predictor ?? "?"})`}
            {edge.status === "superseded" && "; already superseded by another correction"}
          </div>
          <StateDepiction stateId={edge.source_state_id} arrows={edge.arrows} edgeId={edge.id} fadedArrows width={320} height={220} caption="the original arrows, faded" />
        </>
      ) : node ? (
        <>
          <strong>From node</strong> {speciesLabel(node)} <span className="badge">{node.status}</span>
          <div className="muted">a node correction adds a new corrected step; no prediction is superseded</div>
          <StateDepiction stateId={node.state_id} width={320} height={220} />
        </>
      ) : null}
    </div>
  );
}

interface RunState {
  created: JobCreatedOut | null;
  job: JobOut | null;
  error: ApiError | null;
  busy: boolean;
}

/** Create the override job and poll it; on success go back to Predict with the corrected edge. */
function useOverrideRun(jobId: string, r: Resolved, kind: "edge" | "node", targetId: string) {
  const [run, setRun] = useState<RunState>({ created: null, job: null, error: null, busy: false });
  const ctl = useRef<AbortController | null>(null);
  useEffect(() => () => ctl.current?.abort(), []);
  const submit = async (arrows: BuilderArrow[], o: OverrideOptions) => {
    ctl.current?.abort();
    const c = new AbortController();
    ctl.current = c;
    setRun({ created: null, job: null, error: null, busy: true });
    try {
      const target = kind === "edge" ? { edge_id: targetId } : { node_id: targetId };
      const body = { ...target, arrows: arrows.map(toArrowIn), allow_invalid: o.allowInvalid, continue_prediction: o.continuePrediction, continue_steps: o.continueSteps };
      const created = await overrideJob(jobId, body, { signal: c.signal });
      setRun((s) => ({ ...s, created }));
      const job = await pollJob(created.job_id, { signal: c.signal, onUpdate: (j) => setRun((s) => ({ ...s, job: j })) });
      setRun({ created, job, error: null, busy: false });
      const out = job.status === "done" ? overrideOutcome(job, kind === "edge" ? { edgeId: targetId } : { nodeId: r.node?.id ?? targetId }) : null;
      if (out) navigate(formatRoute({ page: "predict", parts: ["job", out.parent_job_id || jobId], params: { edge: out.corrected_edge_id, override: created.job_id } }));
    } catch (err) {
      if (!isAbort(err)) setRun((s) => ({ ...s, error: toApiError(err), busy: false }));
    }
  };
  return { run, submit };
}

function OverrideResult({ run, jobId }: { run: RunState; jobId: string }) {
  if (!run.created && !run.error) return null;
  const job = run.job;
  const status = run.error ? "error" : (job?.status ?? "queued");
  const out = job && job.status === "done" ? overrideOutcome(job) : null;
  return (
    <div className={`override-result panel status-${status}`} data-testid="override-result" data-status={status} data-corrected-edge={out?.corrected_edge_id ?? ""}>
      {run.created && <JobProgress job={job} title="Override job" />}
      <ErrorBanner error={run.error} title="Cannot apply the override" testId="override-error" />
      {job?.error && <ErrorBanner error={bodyError(job.error)} title="The override failed" testId="override-job-error" />}
      {out && (
        <div>
          corrected step <code className="code">{shortId(out.corrected_edge_id, 12)}</code>; returning to the network…
        </div>
      )}
      {job?.status === "done" && !out && (
        <div className="banner banner-warn">
          the override job finished without naming its corrected edge; <a href={jobHash(jobId)}>open the network</a>
        </div>
      )}
    </div>
  );
}

interface BuilderState {
  arrows: BuilderArrow[];
  selected: number | null;
}

/** The override column: builder, site picker, preview of the user arrows and the job result. */
function OverrideColumn(p: { jobId: string; stateId: string; b: BuilderState; setB: (b: BuilderState) => void; run: RunState; ready: boolean; onSubmit: (o: OverrideOptions) => void }) {
  const { arrows, selected } = p.b;
  const [options, setOptions] = useState<OverrideOptions>(DEFAULT_OPTIONS);
  const change = (next: BuilderArrow[]) => p.setB({ arrows: next, selected: selected === null || next.length === 0 ? null : Math.min(selected, next.length - 1) });
  const sel = selected !== null ? (arrows[selected] ?? null) : null;
  return (
    <div className="correct-side stack">
      <OverrideBuilder arrows={arrows} selected={selected} onSelect={(i) => p.setB({ arrows, selected: i })} onChange={change} options={options} onOptions={setOptions} onSubmit={() => p.onSubmit(options)} busy={p.run.busy || !p.ready} maxSteps={12} />
      <SitePicker arrow={sel} index={selected ?? 0} onChange={(a) => selected !== null && p.setB({ arrows: arrows.map((x, k) => (k === selected ? a : x)), selected })} />
      {arrows.length > 0 && <StateDepiction stateId={p.stateId} arrows={arrows} edgeId="override" width={340} height={230} testId="override-preview" caption={`your arrows: ${arrows.map(arrowText).join("; ")}`} />}
      <OverrideResult run={p.run} jobId={p.jobId} />
    </div>
  );
}

function CorrectView({ jobId, kind, targetId }: { jobId: string; kind: "edge" | "node"; targetId: string }) {
  const { job, error } = useJobPoll(jobId);
  const r = resolve(job, kind, targetId);
  const [b, setB] = useState<BuilderState>({ arrows: [], selected: null });
  const [donor, setDonor] = useState<number | null>(null);
  const [excluded, setExcluded] = useState(false);
  const { run, submit } = useOverrideRun(jobId, r, kind, targetId);
  const add = (c: CandidateOut) => {
    if (b.arrows.length < MAX_OVERRIDE_ARROWS) setB({ arrows: [...b.arrows, builderArrowFrom(c)], selected: b.arrows.length });
  };
  const net = r.net;
  const last = b.arrows[b.arrows.length - 1];
  return (
    <section className="correct-panel stack" data-testid="correct-panel" data-job-id={jobId} data-target-kind={kind} data-target={targetId}>
      <div className="row">
        <h2>Correct {kind === "edge" ? "a step" : "from a node"}</h2>
        <a className="btn btn-small" href={jobHash(jobId, kind === "edge" ? { edge: targetId } : { node: targetId })}>
          back to the network
        </a>
      </div>
      {(!job || job.status !== "done") && <JobProgress job={job} pollError={error} title="Prediction job" />}
      {r.missing && <ErrorBanner error={new ApiError({ status: 404, code: kind === "edge" ? "EDGE_NOT_FOUND" : "NODE_NOT_FOUND", message: r.missing })} title="Nothing to correct" />}
      {net && !r.missing && r.stateId && (
        <>
          <Original r={r} />
          <div className="correct-grid">
            <CandidateTable
              stateId={r.stateId}
              predictor={net.request.predictor ?? net.provenance.predictor}
              temperature={net.params.temperature}
              scope={net.params.candidate_scope === "inter" ? "inter" : "all"}
              donor={donor}
              onDonorChange={setDonor}
              showExcluded={excluded}
              onShowExcludedChange={setExcluded}
              activeTuple={last ? [last.donor, last.donor_site, last.acceptor, last.acceptor_site] : null}
              onSelect={add}
              selectLabel={b.arrows.length >= MAX_OVERRIDE_ARROWS ? "full" : "use"}
            />
            <OverrideColumn jobId={jobId} stateId={r.stateId} b={b} setB={setB} run={run} ready={job?.status === "done"} onSubmit={(o) => void submit(b.arrows, o)} />
          </div>
        </>
      )}
    </section>
  );
}
