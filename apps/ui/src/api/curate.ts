// The curate router (serve/curate): curated mechanisms, the PMechDB export and the rechecks.
import { apiUrl, request, type RequestOptions, type TextWithHeaders } from "./http";
import type * as T from "./types.gen";

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

/** The report of a PMechDB export (the X-Ksmech-Export-Report header). */
export interface PmechdbExportReport {
  statuses: string[];
  n_mechanisms: number;
  n_steps_considered: number;
  n_steps_written: number;
  skipped: Record<string, number>;
  warnings?: string[];
}

/** GET /curate/export/pmechdb?status=..: the curated steps as PMechDB raw records (text/csv)
 *  and the report of its header. A download, never training data. */
export async function exportPmechdb(
  statuses: readonly string[],
  o?: RequestOptions,
): Promise<{ csv: string; report: PmechdbExportReport | null }> {
  const res = await request<TextWithHeaders>({ path: "/curate/export/pmechdb", query: { status: statuses }, as: "textWithHeaders" }, o);
  const raw = res.headers.get("X-Ksmech-Export-Report");
  let report: PmechdbExportReport | null = null;
  try {
    report = raw ? (JSON.parse(raw) as PmechdbExportReport) : null;
  } catch {
    report = null;
  }
  return { csv: res.text, report };
}

export const exportPmechdbUrl = (statuses: readonly string[]) => apiUrl("/curate/export/pmechdb", { status: statuses });

export const recheckMechanisms = (body: T.RecheckIn = {}, o?: RequestOptions) =>
  request<T.JobCreatedOut>({ method: "POST", path: "/curate/recheck", body }, o);

export const getMechanism = (mid: number, o?: RequestOptions) =>
  request<T.MechanismOut>({ path: `/curate/${mid}` }, o);

export const patchMechanism = (mid: number, body: T.MechanismPatchIn, o?: RequestOptions) =>
  request<T.MechanismOut>({ method: "PATCH", path: `/curate/${mid}`, body }, o);

export const recheckMechanism = (mid: number, body: T.RecheckOneIn = {}, o?: RequestOptions) =>
  request<T.JobCreatedOut>({ method: "POST", path: `/curate/${mid}/recheck`, body }, o);
