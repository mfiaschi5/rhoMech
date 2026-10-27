// The predict router (serve/predict): the predictors, prediction jobs, the user step R
// (override, step check), expansion of a state, pathways and the local checkpoints.
import { request, type RequestOptions } from "./http";
import type * as T from "./types.gen";

const job = (id: string, rest = "") => `/jobs/${encodeURIComponent(id)}${rest}`;

export const getPredictors = (o?: RequestOptions) =>
  request<T.PredictorInfoOut[]>({ path: "/predictors" }, o);

export const predict = (body: T.PredictIn, o?: RequestOptions) =>
  request<T.JobCreatedOut>({ method: "POST", path: "/predict", body }, o);

export interface JobListQuery {
  kind?: string;
  status?: string;
  limit?: number;
}

export const listJobs = (q: JobListQuery = {}, o?: RequestOptions) =>
  request<T.JobSummaryOut[]>({ path: "/jobs", query: { ...q } }, o);

export const getJob = (jobId: string, q: { sinceRevision?: number } = {}, o?: RequestOptions) =>
  request<T.JobOut>({ path: job(jobId), query: { since_revision: q.sinceRevision } }, o);

export const cancelJob = (jobId: string, q: { force?: boolean } = {}, o?: RequestOptions) =>
  request<T.JobOut>({ method: "POST", path: job(jobId, "/cancel"), query: { force: q.force } }, o);

/** POST /jobs/{id}/override: apply R on a state of the job's network; the new edge and node
 *  (synchronous; 202 pending while the parent's analysis is computed). */
export const overrideJob = (jobId: string, body: T.OverrideIn, o?: RequestOptions) =>
  request<T.OverrideResultOut>({ method: "POST", path: job(jobId, "/override"), body }, o);

/** POST /jobs/{id}/expand: a job (kind "expand", result ExpandResultOut) that runs the
 *  predictor from one state of the job's network. */
export const expandNode = (jobId: string, body: T.ExpandIn, o?: RequestOptions) =>
  request<T.JobCreatedOut>({ method: "POST", path: job(jobId, "/expand"), body }, o);

/** POST /step/check: the product, issues, formal charges and reading of R on a pool state. */
export const checkStep = (body: T.StepCheckIn, o?: RequestOptions) =>
  request<T.StepCheckOut>({ method: "POST", path: "/step/check", body }, o);

export const getPathways = (
  jobId: string,
  q: { limit?: number; includePruned?: boolean } = {},
  o?: RequestOptions,
) =>
  request<T.PathwayOut[]>(
    { path: job(jobId, "/pathways"), query: { limit: q.limit, include_pruned: q.includePruned } },
    o,
  );

/** GET /models: the local checkpoints and the embedding capability they give. */
export const getModels = (o?: RequestOptions) => request<T.ModelsOut>({ path: "/models" }, o);

/** The network of a predict job (also partial ones while it runs). */
export function networkOfJob(j: T.JobOut | null | undefined): T.NetworkOut | null {
  const r = j?.result as Partial<T.NetworkOut> | null | undefined;
  if (!r || typeof r !== "object") return null;
  if (j?.result_kind === "network" || (Array.isArray(r.nodes) && Array.isArray(r.edges) && "root" in r)) return r as T.NetworkOut;
  return null;
}
