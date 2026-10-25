// Typed client of the ksmech-serve HTTP API (design §6.1, §11.3): one function per endpoint.
//
// - Every non-2xx answer throws ApiError {status, code, message, upstream, detail, requestId},
//   parsed from the server's ErrorOut envelope (code "HTTP_<status>" when the body is not one;
//   "NETWORK_ERROR" when the server cannot be reached). An aborted request rethrows the
//   AbortError unchanged (check with isAbort).
// - A 202 PendingOut (compute still running in the worker) is retried on the same URL after
//   retry_after_s, the wait doubling from 1 s to 5 s, for up to 180 s (then ApiError
//   PENDING_TIMEOUT). Each retry calls onPending(label, worker) so the UI can show
//   "queued behind: <label>" (components/PendingNote). A 202 that is not a PendingOut
//   (JobCreatedOut of /predict, /override, /curate/export, ...) is returned as the result.
// - null is tolerated wherever the schema allows it (types.gen.ts is generated from it).
import type * as T from "./types.gen";

export const API_BASE = "/api";
export const PENDING_TIMEOUT_MS = 180_000;
const PENDING_FIRST_MS = 1_000;
const PENDING_MAX_MS = 5_000;

export interface ApiErrorInit {
  status: number;
  code: string;
  message: string;
  upstream?: string | null;
  detail?: Record<string, unknown> | null;
  requestId?: string;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly upstream: string | null;
  readonly detail: Record<string, unknown> | null;
  readonly requestId: string;

  constructor(init: ApiErrorInit) {
    super(init.message);
    this.name = "ApiError";
    this.status = init.status;
    this.code = init.code;
    this.upstream = init.upstream ?? null;
    this.detail = init.detail ?? null;
    this.requestId = init.requestId ?? "";
  }

  static fromBody(body: T.ErrorBody, status: number): ApiError {
    return new ApiError({
      status,
      code: body.code,
      message: body.message,
      upstream: body.upstream,
      detail: body.detail,
      requestId: body.request_id,
    });
  }
}

export function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

/** Any thrown value as an ApiError (status 0 / code CLIENT_ERROR for non-HTTP failures). */
export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  if (err instanceof Error) return new ApiError({ status: 0, code: "CLIENT_ERROR", message: err.message });
  return new ApiError({ status: 0, code: "CLIENT_ERROR", message: String(err) });
}

export type PendingCallback = (label: string, worker: T.WorkerStatusOut, pending: T.PendingOut) => void;

export interface RequestOptions {
  signal?: AbortSignal;
  /** Called on every 202 PendingOut before the retry. */
  onPending?: PendingCallback;
  /** Give up on a pending computation after this long (default 180 s). */
  pendingTimeoutMs?: number;
}

export type QueryValue = string | number | boolean | null | undefined | readonly (string | number)[];
export type Query = Record<string, QueryValue>;

/** "/api<path>?<query>"; null / undefined values are dropped, arrays repeat the key. */
export function apiUrl(path: string, query: Query = {}): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    const values = Array.isArray(value) ? value : [value];
    for (const v of values) {
      if (v === null || v === undefined) continue;
      params.append(key, typeof v === "boolean" ? (v ? "true" : "false") : String(v));
    }
  }
  const qs = params.toString();
  return `${API_BASE}${path}${qs ? `?${qs}` : ""}`;
}

export interface ApiRequest {
  method?: "GET" | "POST" | "PATCH";
  path: string;
  query?: Query;
  body?: unknown;
  /** "json" (default), "text" (cube files) or "textWithHeaders" (a TextWithHeaders). */
  as?: "json" | "text" | "textWithHeaders";
}

/** A text body with the response headers (the workbench cube carries its metadata there). */
export interface TextWithHeaders {
  text: string;
  headers: Headers;
}

function isPendingOut(body: unknown): body is T.PendingOut {
  const b = body as Partial<T.PendingOut> | null;
  return !!b && b.status === "pending" && typeof b.task_id === "string";
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("aborted", "AbortError"));
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

async function errorFrom(res: Response): Promise<ApiError> {
  let text = "";
  try {
    text = await res.text();
  } catch {
    // body unreadable: fall through to the status line
  }
  try {
    const body = JSON.parse(text) as Partial<T.ErrorOut>;
    if (body?.error && typeof body.error.code === "string") return ApiError.fromBody(body.error, res.status);
  } catch {
    // not JSON: a proxy or static-server error page
  }
  return new ApiError({
    status: res.status,
    code: `HTTP_${res.status}`,
    message: text.slice(0, 500) || res.statusText || `HTTP ${res.status}`,
  });
}

async function fetchOnce(req: ApiRequest, signal?: AbortSignal): Promise<Response> {
  const init: RequestInit = { method: req.method ?? "GET", signal, headers: { Accept: "application/json" } };
  if (req.body !== undefined) {
    init.headers = { ...init.headers, "Content-Type": "application/json" };
    init.body = JSON.stringify(req.body);
  }
  try {
    return await fetch(apiUrl(req.path, req.query), init);
  } catch (err) {
    if (isAbort(err)) throw err;
    throw new ApiError({ status: 0, code: "NETWORK_ERROR", message: `cannot reach the server: ${String(err)}` });
  }
}

async function readBody<R>(res: Response, as: "json" | "text" | "textWithHeaders"): Promise<R> {
  if (res.status === 204) return undefined as R;
  if (as === "text") return (await res.text()) as R;
  if (as === "textWithHeaders") return { text: await res.text(), headers: res.headers } as R;
  const text = await res.text();
  return (text ? JSON.parse(text) : null) as R;
}

/** One API call with the error envelope and the 202 PendingOut retry loop. */
export async function request<R>(req: ApiRequest, opts: RequestOptions = {}): Promise<R> {
  const start = Date.now();
  const timeout = opts.pendingTimeoutMs ?? PENDING_TIMEOUT_MS;
  let backoff = PENDING_FIRST_MS;
  for (;;) {
    const res = await fetchOnce(req, opts.signal);
    if (!res.ok) throw await errorFrom(res);
    if (res.status !== 202) return readBody<R>(res, req.as ?? "json");
    const body: unknown = await readBody<unknown>(res, "json");
    if (!isPendingOut(body)) return body as R;
    opts.onPending?.(body.label, body.worker, body);
    const wait = Math.min(PENDING_MAX_MS, Math.max(backoff, (body.retry_after_s ?? 1) * 1000));
    if (Date.now() - start + wait > timeout) {
      throw new ApiError({
        status: 202,
        code: "PENDING_TIMEOUT",
        message: `still computing after ${Math.round(timeout / 1000)} s: ${body.label}`,
        detail: { task_id: body.task_id, label: body.label },
      });
    }
    await sleep(wait, opts.signal);
    backoff = Math.min(PENDING_MAX_MS, backoff * 2);
  }
}

// ---- meta

export const getHealth = (o?: RequestOptions) => request<T.HealthOut>({ path: "/health" }, o);
export const getMeta = (o?: RequestOptions) => request<T.MetaOut>({ path: "/meta" }, o);
export const getPredictors = (o?: RequestOptions) =>
  request<T.PredictorInfoOut[]>({ path: "/predictors" }, o);

// ---- input, prediction and jobs

export const normalize = (body: T.NormalizeIn, o?: RequestOptions) =>
  request<T.NormalizeOut>({ method: "POST", path: "/normalize", body }, o);

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
  request<T.JobOut>(
    { path: `/jobs/${encodeURIComponent(jobId)}`, query: { since_revision: q.sinceRevision } },
    o,
  );

export const cancelJob = (jobId: string, q: { force?: boolean } = {}, o?: RequestOptions) =>
  request<T.JobOut>(
    { method: "POST", path: `/jobs/${encodeURIComponent(jobId)}/cancel`, query: { force: q.force } },
    o,
  );

export const overrideJob = (jobId: string, body: T.OverrideIn, o?: RequestOptions) =>
  request<T.JobCreatedOut>(
    { method: "POST", path: `/jobs/${encodeURIComponent(jobId)}/override`, body },
    o,
  );

export const getPathways = (
  jobId: string,
  q: { limit?: number; includePruned?: boolean } = {},
  o?: RequestOptions,
) =>
  request<T.PathwayOut[]>(
    {
      path: `/jobs/${encodeURIComponent(jobId)}/pathways`,
      query: { limit: q.limit, include_pruned: q.includePruned },
    },
    o,
  );

// ---- pools, tokens, candidates, physics

const pool = (stateId: string, rest = "") => `/pool/${encodeURIComponent(stateId)}${rest}`;

/** POST /pools: the pool of these reactants, once its analysis is ready (pending meanwhile). */
export const createPool = (body: T.NormalizeIn, o?: RequestOptions) =>
  request<T.PoolOut>({ method: "POST", path: "/pools", body }, o);

export const getPool = (stateId: string, o?: RequestOptions) =>
  request<T.PoolOut>({ path: pool(stateId) }, o);

export const getTokens = (stateId: string, o?: RequestOptions) =>
  request<T.TokensOut>({ path: pool(stateId, "/tokens") }, o);

export interface CandidatesQuery {
  predictor?: string;
  donor?: number | null;
  includeExcluded?: boolean;
  offset?: number;
  limit?: number;
  temperature?: number | null;
  scope?: "all" | "inter";
}

export const getCandidates = (stateId: string, q: CandidatesQuery = {}, o?: RequestOptions) =>
  request<T.CandidatesOut>(
    {
      path: pool(stateId, "/candidates"),
      query: {
        predictor: q.predictor,
        donor: q.donor,
        include_excluded: q.includeExcluded,
        offset: q.offset,
        limit: q.limit,
        temperature: q.temperature,
        scope: q.scope,
      },
    },
    o,
  );

export interface PhysicsQuery {
  donor: number;
  acceptor: number;
  donorSite?: number | null;
  acceptorSite?: number | null;
  predictor?: string;
  temperature?: number | null;
}

export const getPhysics = (stateId: string, q: PhysicsQuery, o?: RequestOptions) =>
  request<T.PhysicsOut>(
    {
      path: pool(stateId, "/physics"),
      query: {
        donor: q.donor,
        acceptor: q.acceptor,
        donor_site: q.donorSite,
        acceptor_site: q.acceptorSite,
        predictor: q.predictor,
        temperature: q.temperature,
      },
    },
    o,
  );

export const getSitePairs = (stateId: string, q: { donor: number; acceptor: number }, o?: RequestOptions) =>
  request<T.SitePairsOut>({ path: pool(stateId, "/site_pairs"), query: { ...q } }, o);

// ---- geometry, depiction, 3D

export const getGeometry = (stateId: string, o?: RequestOptions) =>
  request<T.GeometryOut>({ path: pool(stateId, "/geometry") }, o);

export interface CubeQuery {
  spacing?: number;
  padding?: number;
}

export const orbitalCubePath = (stateId: string, tokenId: number) => pool(stateId, `/orbital/${tokenId}`);

/** The Gaussian cube text of one localized orbital (pending while the grid is computed). */
export const getOrbitalCube = (stateId: string, tokenId: number, q: CubeQuery = {}, o?: RequestOptions) =>
  request<string>({ path: orbitalCubePath(stateId, tokenId), query: { ...q }, as: "text" }, o);

export interface DepictionQuery {
  keepMaps?: readonly number[];
  serverSvg?: boolean;
}

export const getDepiction = (stateId: string, q: DepictionQuery = {}, o?: RequestOptions) =>
  request<T.DepictionOut>(
    {
      path: pool(stateId, "/depiction"),
      query: {
        keep_maps: q.keepMaps && q.keepMaps.length ? q.keepMaps.join(",") : undefined,
        server_svg: q.serverSvg ? 1 : undefined,
      },
    },
    o,
  );

export interface ContactRef {
  stateId: string;
  donor: number;
  donorSite: number;
  acceptor: number;
  acceptorSite: number;
}

export const contactPath = (c: ContactRef, rest = "") =>
  `/contact/${encodeURIComponent(c.stateId)}/${c.donor}/${c.donorSite}/${c.acceptor}/${c.acceptorSite}${rest}`;

export const getContact = (c: ContactRef, q: { cubes?: "urls" | "inline" } = {}, o?: RequestOptions) =>
  request<T.ContactOut>({ path: contactPath(c), query: { cubes: q.cubes } }, o);

export const getContactCube = (c: ContactRef, which: "donor" | "acceptor", o?: RequestOptions) =>
  request<string>({ path: contactPath(c, `/cube/${which}`), as: "text" }, o);

/** A server-given URL ("/api/..." as in ContactOut.cube_urls) fetched as text, pending-aware. */
export function getTextUrl(url: string, o?: RequestOptions): Promise<string> {
  const u = new URL(url, window.location.origin);
  const path = u.pathname.startsWith(API_BASE) ? u.pathname.slice(API_BASE.length) : u.pathname;
  const query: Query = {};
  u.searchParams.forEach((v, k) => {
    query[k] = v;
  });
  return request<string>({ path, query, as: "text" }, o);
}

// ---- curation, export, recheck

export const curate = (body: T.CurateIn, o?: RequestOptions) =>
  request<T.MechanismOut>({ method: "POST", path: "/curate", body }, o);

export interface MechanismListQuery {
  status?: string;
  predictor?: string;
  stale?: boolean;
  limit?: number;
  offset?: number;
}

export const listMechanisms = (q: MechanismListQuery = {}, o?: RequestOptions) =>
  request<T.MechanismSummaryOut[]>({ path: "/curate", query: { ...q } }, o);

export const exportCurated = (body: T.ExportIn = {}, o?: RequestOptions) =>
  request<T.JobCreatedOut>({ method: "POST", path: "/curate/export", body }, o);

export const getLatestExport = (o?: RequestOptions) =>
  request<T.ExportReportOut>({ path: "/curate/export/latest" }, o);

export const recheckMechanisms = (body: T.RecheckIn = {}, o?: RequestOptions) =>
  request<T.JobCreatedOut>({ method: "POST", path: "/curate/recheck", body }, o);

export const getMechanism = (mid: number, o?: RequestOptions) =>
  request<T.MechanismOut>({ path: `/curate/${mid}` }, o);

export const patchMechanism = (mid: number, body: T.MechanismPatchIn, o?: RequestOptions) =>
  request<T.MechanismOut>({ method: "PATCH", path: `/curate/${mid}`, body }, o);

export const recheckMechanism = (mid: number, body: T.RecheckOneIn = {}, o?: RequestOptions) =>
  request<T.JobCreatedOut>({ method: "POST", path: `/curate/${mid}/recheck`, body }, o);

// ---- browse

export const getBrowseStatus = (o?: RequestOptions) =>
  request<T.IndexStatusOut>({ path: "/browse/status" }, o);

/** POST /browse/index: start (or rejoin) the PMechDB index job. */
export const buildBrowseIndex = (o?: RequestOptions) =>
  request<T.JobCreatedOut>({ method: "POST", path: "/browse/index" }, o);

export type BrowseSource = "pmechdb" | "user_curated";

export interface SearchQuery {
  q: string;
  mode?: "exact" | "substructure";
  side?: "reactants" | "products" | "either";
  sources?: readonly BrowseSource[];
  status?: string;
  resonance?: boolean;
  limit?: number;
  offset?: number;
}

export const searchSteps = (s: SearchQuery, o?: RequestOptions) =>
  request<T.SearchOut>(
    {
      path: "/browse/search",
      query: {
        q: s.q,
        mode: s.mode,
        side: s.side,
        source: s.sources,
        status: s.status,
        resonance: s.resonance,
        limit: s.limit,
        offset: s.offset,
      },
    },
    o,
  );

/** GET /browse/step/{ref}; ref is "pmechdb:<step_id>" or "curated:<mid>:<step_index>". */
export const getBrowseStep = (ref: string, o?: RequestOptions) =>
  request<T.StepViewOut>({ path: `/browse/step/${encodeURIComponent(ref)}` }, o);

// ---- molecule workbench (capability "workbench"; display only)

const molecule = (mid: string, rest = "") => `/molecule/${encodeURIComponent(mid)}${rest}`;

/** POST /molecule: register a drawn molecule (molblock) or a SMILES and featurize it. Answers
 *  within the server's interactive wait with status ready, or running (poll getMolecule). Only
 *  this answer carries the sketch mapping. */
export const createMolecule = (body: T.MoleculeIn, o?: RequestOptions) =>
  request<T.MoleculeOut>({ method: "POST", path: "/molecule", body }, o);

/** GET /molecule/{id}: the status (never computes; mapping is always null here). */
export const getMolecule = (mid: string, o?: RequestOptions) => request<T.MoleculeOut>({ path: molecule(mid) }, o);

export const getMoleculeGeometry = (mid: string, o?: RequestOptions) =>
  request<T.MoleculeGeometryOut>({ path: molecule(mid, "/geometry") }, o);

export const getMoleculeProperties = (mid: string, o?: RequestOptions) =>
  request<T.MoleculePropertiesOut>({ path: molecule(mid, "/properties") }, o);

export interface OrbitalSetQuery {
  set?: string | null;
  start?: string | null;
}

export const getOrbitalSet = (mid: string, q: OrbitalSetQuery = {}, o?: RequestOptions) =>
  request<T.OrbitalSetOut>({ path: molecule(mid, "/orbitals"), query: { set: q.set, start: q.start } }, o);

export type CubeKind = "orbital" | "orbital_density" | "density" | "fukui_minus" | "fukui_plus" | "dual" | "esp";
export type CubeQuality = "draft" | "normal" | "fine";

export interface MoleculeCubeQuery {
  kind: CubeKind;
  set?: string | null;
  start?: string | null;
  index?: number | null;
  quality?: CubeQuality | null;
}

/** The metadata of a workbench cube (X-Cube-* headers); null where the header says "none". */
export interface CubeMeta {
  kind: string;
  units: string;
  integral: number | null;
  expected: number | null;
  spacing: number | null;
}

function headerNumber(h: Headers, name: string): number | null {
  const v = h.get(name);
  if (v === null || v === "none" || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export function cubeMeta(h: Headers): CubeMeta {
  return {
    kind: h.get("X-Cube-Kind") ?? "",
    units: h.get("X-Cube-Units") ?? "",
    integral: headerNumber(h, "X-Cube-Integral"),
    expected: headerNumber(h, "X-Cube-Expected"),
    spacing: headerNumber(h, "X-Cube-Spacing"),
  };
}

function cubeQuery(q: MoleculeCubeQuery, download = false): Query {
  const orbital = q.kind === "orbital" || q.kind === "orbital_density";
  return {
    kind: q.kind,
    set: orbital ? q.set : undefined,
    start: orbital ? q.start : undefined,
    index: orbital ? q.index : undefined,
    quality: q.quality,
    download: download ? true : undefined,
  };
}

/** GET /molecule/{id}/cube: the cube text and its X-Cube-* metadata (pending while computed). */
export async function getMoleculeCube(
  mid: string,
  q: MoleculeCubeQuery,
  o?: RequestOptions,
): Promise<{ text: string; headers: Headers; meta: CubeMeta }> {
  const res = await request<TextWithHeaders>({ path: molecule(mid, "/cube"), query: cubeQuery(q), as: "textWithHeaders" }, o);
  return { ...res, meta: cubeMeta(res.headers) };
}

/** The URL of a workbench cube (download: as an attachment), for links. */
export function moleculeCubeUrl(mid: string, q: MoleculeCubeQuery, download = true): string {
  return apiUrl(molecule(mid, "/cube"), cubeQuery(q, download));
}
