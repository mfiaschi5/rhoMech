// The spec-v9 endpoints of the Predict and Correct pages over api/client.ts's request() (error
// envelope, 202 pending retries). Shapes: v9types.ts.
import { request, type RequestOptions } from "../../api/client";
import type { JobCreatedOut, JobOut } from "../../api/types.gen";
import type { ExpandIn, NetworkOut, OverrideV9In, OverrideV9Out, PredictIn, PredictorOut, StepCheckIn, StepCheckOut } from "./v9types";

const job = (id: string, rest = "") => `/jobs/${encodeURIComponent(id)}${rest}`;

export const getPredictorsV9 = (o?: RequestOptions) => request<PredictorOut[]>({ path: "/predictors" }, o);

export const predictV9 = (body: PredictIn, o?: RequestOptions) =>
  request<JobCreatedOut>({ method: "POST", path: "/predict", body }, o);

/** POST /step/check: the product, issues, formal charges and reading of R on a pool state. */
export const checkStep = (body: StepCheckIn, o?: RequestOptions) =>
  request<StepCheckOut>({ method: "POST", path: "/step/check", body }, o);

/** POST /jobs/{id}/override: apply R on a state of the job's network; the new edge and node. */
export const overrideStep = (jobId: string, body: OverrideV9In, o?: RequestOptions) =>
  request<OverrideV9Out>({ method: "POST", path: job(jobId, "/override"), body }, o);

/** POST /jobs/{id}/expand: a job (kind "expand", result ExpandResultOut) that runs the
 *  predictor from one state of the job's network. */
export const expandNode = (jobId: string, body: ExpandIn, o?: RequestOptions) =>
  request<JobCreatedOut>({ method: "POST", path: job(jobId, "/expand"), body }, o);

/** The network of a predict job (also partial ones while it runs). */
export function networkOfJob(j: JobOut | null | undefined): NetworkOut | null {
  const r = j?.result as Partial<NetworkOut> | null | undefined;
  if (!r || typeof r !== "object") return null;
  if (j?.result_kind === "network" || (Array.isArray(r.nodes) && Array.isArray(r.edges) && "root" in r)) return r as NetworkOut;
  return null;
}
