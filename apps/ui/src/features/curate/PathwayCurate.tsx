// Curate a predicted pathway (#/curate/<jobId>?pathway=<pid>&ind=<toggled steps>): its steps with
// their arrows, an "independent of previous step" toggle per step after the first, notes, and
// Validate / Reject / Save through POST /curate (design §9.2, §9.3, §11.8), which receives the
// pathway's edge chain (edge_ids). Validate stores "validated", or "corrected" when a step of the
// pathway was corrected: the server decides.
// Without ?pathway= the job's ranked pathways are listed to choose from.
import { useState } from "react";
import { toApiError, type ApiError } from "../../api/http";
import { getPathways } from "../../api/predict";
import { curate } from "../../api/curate";
import { useJobPoll } from "../../api/poll";
import type { CurateIn, JobOut, NetworkOut, PathwayOut } from "../../api/types.gen";
import { CapabilityNotice } from "../../components/CapabilityNotice";
import { ErrorBanner } from "../../components/ErrorBanner";
import { useCapability } from "../../state/store";
import { ProvenanceBadge } from "../../components/ProvenanceBadge";
import { curateHash, jobHash, navigate, withParams, type Route } from "../../router";
import { PathwayStepList } from "./CurateStepList";
import {
  formatToggleParam,
  groupsFromToggles,
  groupsProblem,
  independenceHints,
  parseToggleParam,
  type IndependenceHint,
} from "./orderGroups";
import { bumpRevision, rememberCreated } from "./session";
import { validateTarget } from "./status";
import { fmtProb } from "./stepdoc";
import { useLoad } from "./useLoad";

function isNetwork(r: JobOut["result"] | undefined): r is NetworkOut {
  return !!r && typeof r === "object" && "nodes" in r && "edges" in r && "pathways" in r;
}

/** The network a job shows: its own result, or (an expand job) its parent's network. */
function useJobNetwork(jobId: string) {
  const primary = useJobPoll(jobId);
  const job = primary.job;
  const own = job && isNetwork(job.result) ? job.result : null;
  const parentId = !own && job ? job.parent_job_id : null;
  const parent = useJobPoll(parentId);
  const network = own ?? (parent.job && isNetwork(parent.job.result) ? parent.job.result : null);
  return { job, network, error: primary.error ?? parent.error, parentId };
}

function PathwayChooser({ jobId, network }: { jobId: string; network: NetworkOut }) {
  if (!network.pathways.length) {
    return <p className="cu-hint-warn">This network has no pathway yet.</p>;
  }
  return (
    <div className="cu-section">
      <h3>Choose a pathway</h3>
      <ol className="cu-choices" data-testid="curate-pathway-choices">
        {network.pathways.map((p) => (
          <li key={p.id}>
            <a href={curateHash(jobId, p.id)} data-testid={`curate-pathway-choice-${p.rank}`}>
              #{p.rank}: {p.n_steps} step{p.n_steps === 1 ? "" : "s"}, {p.probability !== null ? `p ${fmtProb(p.probability)}` : `log score ${fmtProb(p.logp)}`}, ends with {p.ends_with}
              {p.n_corrected > 0 ? `, ${p.n_corrected} corrected` : ""}
            </a>
          </li>
        ))}
      </ol>
    </div>
  );
}

interface FormProps {
  jobId: string;
  pathway: PathwayOut;
  groups: number[];
  hints: readonly IndependenceHint[];
  nCorrected: number;
  blockedReason: string | null;
}

type Action = "save" | "validate" | "reject";

/** The POST /curate body. It names the pathway by its edge chain, never by pathway_id: usePathway
 *  may have found the pathway in the fallback list (GET /jobs/{id}/pathways with pruned ones, up
 *  to 50), and the server resolves a pathway_id only among NetworkOut.pathways (the top
 *  n_pathways, none pruned), so it would answer 404 PATHWAY_NOT_FOUND. The server checks the chain
 *  (existing, not superseded, contiguous) either way. */
export function curateRequest(
  jobId: string,
  edgeIds: readonly string[],
  action: Action,
  notes: string,
  groups: number[],
  conditions: string,
): CurateIn {
  return { job_id: jobId, edge_ids: [...edgeIds], action, notes, order_groups: groups, conditions: conditions.trim() || null };
}

function useCurateSubmit(jobId: string, edgeIds: readonly string[]) {
  const [busy, setBusy] = useState<Action | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const submit = async (action: Action, notes: string, groups: number[], conditions: string) => {
    setBusy(action);
    setError(null);
    try {
      const m = await curate(curateRequest(jobId, edgeIds, action, notes, groups, conditions));
      rememberCreated(m);
      bumpRevision();
      navigate(`#/curate/m/${m.id}`);
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(null);
    }
  };
  return { busy, error, submit };
}

function FormButtons(props: { target: string; busy: Action | null; disabled: boolean; onSubmit: (a: Action) => void }) {
  const { target, busy, disabled, onSubmit } = props;
  return (
    <div className="row cu-controls">
      <button
        type="button"
        className="btn btn-primary"
        data-testid="curate-validate"
        data-target-status={target}
        disabled={disabled}
        onClick={() => onSubmit("validate")}
      >
        {busy === "validate" ? "saving…" : `Validate (stores ${target})`}
      </button>
      <button type="button" className="btn" data-testid="curate-reject" disabled={disabled} onClick={() => onSubmit("reject")}>
        {busy === "reject" ? "saving…" : "Reject"}
      </button>
      <button type="button" className="btn" data-testid="curate-save" disabled={disabled} onClick={() => onSubmit("save")}>
        {busy === "save" ? "saving…" : "Save (no verdict)"}
      </button>
    </div>
  );
}

function CurateForm({ jobId, pathway, groups, hints, nCorrected, blockedReason }: FormProps) {
  const [notes, setNotes] = useState("");
  const [conditions, setConditions] = useState("");
  const { busy, error, submit } = useCurateSubmit(jobId, pathway.edge_ids);
  const edit = useCapability("curate_edit");
  const target = validateTarget(nCorrected);
  const disabled = busy !== null || blockedReason !== null || groupsProblem(groups) !== null;
  const refused = hints.some((h) => h.code === "NOT_INDEPENDENT");
  return (
    <div className="cu-section cu-form">
      <label className="cu-notes-label">
        notes
        <textarea data-testid="curate-notes" value={notes} rows={4} onChange={(e) => setNotes(e.target.value)} />
      </label>
      <label className="cu-notes-label">
        conditions (optional; overrides the ones recorded with the job)
        <input data-testid="curate-conditions" value={conditions} onChange={(e) => setConditions(e.target.value)} />
      </label>
      <CapabilityNotice name="curate_edit" title="Storing a curated pathway is disabled" compact />
      <FormButtons
        target={target}
        busy={busy}
        disabled={disabled || !edit.enabled}
        onSubmit={(a) => submit(a, notes, groups, conditions)}
      />
      <p className="muted">
        {nCorrected > 0
          ? `${nCorrected} step(s) of this pathway were corrected, so Validate stores "corrected".`
          : 'No step was corrected, so Validate stores "validated".'}{" "}
        Group ids sent: [{groups.join(", ")}].
      </p>
      {blockedReason && <div className="cu-hint-warn">{blockedReason}</div>}
      {refused && <div className="cu-hint-error">The server will refuse this grouping (NOT_INDEPENDENT): see the marked steps.</div>}
      <ErrorBanner error={error} title="Not curated" testId="curate-error" />
    </div>
  );
}

function usePathway(jobId: string, network: NetworkOut | null, pid: string | null) {
  const fromNet = network?.pathways.find((p) => p.id === pid) ?? null;
  const extra = useLoad(network && pid && !fromNet ? `/jobs/${jobId}/pathways?all` : null, (o) =>
    getPathways(jobId, { limit: 50, includePruned: true }, o),
  );
  const pathway = fromNet ?? extra.data?.find((p) => p.id === pid) ?? null;
  const missing = !!network && !!pid && !pathway && !extra.loading;
  return { pathway, missing, error: extra.error };
}

interface BodyProps {
  route: Route;
  jobId: string;
  network: NetworkOut;
  pathway: PathwayOut;
  blockedReason: string | null;
}

/** The steps of the chosen pathway, their order groups (from the hash) and the verdict form. */
function PathwayBody({ route, jobId, network, pathway, blockedReason }: BodyProps) {
  const n = pathway.edge_ids.length;
  const independent = parseToggleParam(route.params.get("ind"), n);
  const groups = groupsFromToggles(independent, n);
  const edges = new Map(network.edges.map((e) => [e.id, e]));
  const steps = pathway.edge_ids.map((id) => edges.get(id));
  const hints = independenceHints(
    groups,
    steps.map((e) => ({ reactantStateId: e?.source_state_id ?? null, productStateId: e?.product_state_id ?? null, touchedMaps: e?.touched_maps ?? [] })),
  );
  const onToggle = (i: number, v: boolean) => {
    const next = independent.map((t, k) => (k === i ? v : t));
    navigate(withParams(route, { ind: formatToggleParam(next) }), { replace: true });
  };
  return (
    <>
      <p className="muted" data-testid="curate-pathway-summary">
        Pathway #{pathway.rank}: {pathway.n_steps} step(s), {pathway.probability !== null ? `probability ${fmtProb(pathway.probability)}` : `log score ${fmtProb(pathway.logp)} (the predictor gives scores, not probabilities)`}, ends with{" "}
        {pathway.ends_with}
        {pathway.closes_cycle ? ", closes a cycle" : ""}. <a href={curateHash(jobId)}>other pathways</a>
      </p>
      <PathwayStepList
        jobId={jobId}
        network={network}
        pathway={pathway}
        groups={groups}
        independent={independent}
        hints={hints}
        disabled={false}
        onToggle={onToggle}
      />
      <CurateForm
        jobId={jobId}
        pathway={pathway}
        groups={groups}
        hints={hints}
        nCorrected={steps.filter((e) => e?.kind === "corrected").length}
        blockedReason={blockedReason}
      />
    </>
  );
}

export function PathwayCurate({ route, jobId }: { route: Route; jobId: string }) {
  const { job, network, error, parentId } = useJobNetwork(jobId);
  const pid = route.params.get("pathway");
  const { pathway, missing, error: pathwayError } = usePathway(jobId, network, pid);
  const status = job?.status ?? "loading";
  // queued / running: the server answers 409 JOB_NOT_DONE; a failed or cancelled job with a partial
  // network is left to the server to accept or refuse
  const running = status === "queued" || status === "running" || status === "loading";
  const blocked = running ? `The job is ${status}: curate it once it is done (409 JOB_NOT_DONE otherwise).` : null;
  return (
    <section className="panel cu-pathway" data-testid="curate-pathway" data-job={jobId} data-pathway={pid ?? ""} data-job-status={status}>
      <h2 className="cu-title">
        Curate job <code>{jobId}</code>
      </h2>
      <p className="row cu-controls">
        <a href={jobHash(jobId, pid ? { pathway: pid } : {})}>← back to the prediction</a>
        {parentId && <span className="muted">network of parent job {parentId}</span>}
        {network && <ProvenanceBadge provenance={network.provenance} compact />}
        {network && !network.complete && <span className="badge badge-warn">network incomplete</span>}
        {(status === "failed" || status === "cancelled") && <span className="badge badge-warn">job {status}: partial network</span>}
      </p>
      <ErrorBanner error={error} title={`Job ${jobId}`} testId="curate-error" />
      {!error && !network && <p className="muted">{job ? `job ${job.status}; no network yet` : "loading the job…"}</p>}
      {network && !pid && <PathwayChooser jobId={jobId} network={network} />}
      <ErrorBanner error={pathwayError} title="Cannot list the pathways" />
      {missing && (
        <div className="banner banner-error" data-testid="curate-pathway-missing" data-code="PATHWAY_NOT_FOUND">
          <code>PATHWAY_NOT_FOUND</code> pathway {pid} is not in this network. <a href={curateHash(jobId)}>Choose another.</a>
        </div>
      )}
      {network && pathway && (
        <PathwayBody key={pathway.id} route={route} jobId={jobId} network={network} pathway={pathway} blockedReason={blocked} />
      )}
    </section>
  );
}
