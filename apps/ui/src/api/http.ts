// The core of the typed ksmech-serve HTTP client (design §6.1, §11.3): request() and the error
// layer. The endpoints live in one module per serve router (meta, predict, pool, workbench,
// embedding, curate, browse), one function per endpoint.
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
