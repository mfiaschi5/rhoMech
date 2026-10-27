// Re-check results (design §9.4): the RecheckReportOut of a finished re-check job and the
// RecheckRowOut history of a mechanism. A re-check never changes a status; the disagreements
// (rank > 1, product not reproduced, a reason code) are highlighted for the curator.
import { useEffect } from "react";
import type { JobOut, RecheckReportOut, RecheckRowOut } from "../../api/types.gen";
import { ProvenanceBadge } from "../../components/ProvenanceBadge";
import { loadPredictors, predictorsStore, useStore } from "../../state/store";
import { fmtDate } from "./stepdoc";

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "–" : `${(100 * v).toFixed(1)} %`);
const yesNo = (v: boolean | null | undefined) => (v === null || v === undefined ? "–" : v ? "yes" : "no");

export function CountList({ counts, empty }: { counts: Record<string, number>; empty: string }) {
  const entries = Object.entries(counts);
  if (!entries.length) return <span className="muted">{empty}</span>;
  return (
    <span className="cu-counts">
      {entries.map(([code, n]) => (
        <span key={code} className="badge" data-code={code}>
          {code}: {n}
        </span>
      ))}
    </span>
  );
}

export function RecheckRowsTable({ rows, testId }: { rows: readonly RecheckRowOut[]; testId: string }) {
  if (!rows.length) return <p className="muted" data-testid={testId} data-rows="0">No re-check yet.</p>;
  return (
    <div className="cu-table-wrap">
      <table className="table" data-testid={testId} data-rows={rows.length}>
        <thead>
          <tr>
            <th>mech.</th>
            <th>step</th>
            <th>when</th>
            <th>predictor</th>
            <th>aligned</th>
            <th className="num">rank</th>
            <th>top1</th>
            <th>top5</th>
            <th>product</th>
            <th>top-1 gives product</th>
            <th>reason</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, k) => {
            const disagrees = r.reason !== null || r.top1 === false || r.top1_reproduces_product === false;
            return (
              <tr key={k} className={disagrees ? "cu-row-warn" : undefined} data-step={r.step_index} data-top1={String(r.top1)}>
                <td>{r.mechanism_id}</td>
                <td>{r.step_index + 1}</td>
                <td>{fmtDate(r.created_at)}</td>
                <td title={`${r.predictor_version}; ${r.spec_version}; physics ${r.physics_fingerprint}`}>{r.predictor}</td>
                <td>{yesNo(r.aligned)}</td>
                <td className="num">
                  {r.first_arrow_rank ?? "–"}
                  {r.n_valid !== null ? ` / ${r.n_valid}` : ""}
                </td>
                <td>{yesNo(r.top1)}</td>
                <td>{yesNo(r.top5)}</td>
                <td>{r.product_match}</td>
                <td>{yesNo(r.top1_reproduces_product)}</td>
                <td>{r.reason ? <code>{r.reason}</code> : ""}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function RecheckReportView({ report, testId = "recheck-report" }: { report: RecheckReportOut; testId?: string }) {
  return (
    <div className="cu-report" data-testid={testId} data-n-checked={report.n_checked} data-n-steps={report.n_steps}>
      <dl className="kv">
        <dt>steps</dt>
        <dd>
          {report.n_checked} checked of {report.n_steps}
        </dd>
        <dt>top-1 / top-5</dt>
        <dd>
          {pct(report.top1)} / {pct(report.top5)}
        </dd>
        <dt>product match</dt>
        <dd>
          <CountList counts={report.product_match} empty="–" />
        </dd>
        <dt>skipped</dt>
        <dd>
          <CountList counts={report.skipped} empty="none" />
        </dd>
        <dt>predictor</dt>
        <dd>
          <ProvenanceBadge provenance={report.provenance} compact />
        </dd>
      </dl>
      <RecheckRowsTable rows={report.rows} testId={`${testId}-rows`} />
    </div>
  );
}


/** The predictor a re-check uses: "" means each mechanism's own predictor (the server default). */
export function RecheckPredictorSelect(props: { value: string; onChange: (v: string) => void; testId: string }) {
  const predictors = useStore(predictorsStore);
  useEffect(() => {
    void loadPredictors();
  }, []);
  return (
    <select
      data-testid={props.testId}
      value={props.value}
      onChange={(e) => props.onChange(e.target.value)}
      title="predictor of the re-check"
    >
      <option value="">own predictor</option>
      {(predictors.data ?? []).map((p) => (
        <option key={p.name} value={p.name} disabled={!p.available} title={p.reason ?? p.description}>
          {p.label || p.name}
          {p.available ? "" : " (unavailable)"}
        </option>
      ))}
    </select>
  );
}

export function isRecheckReport(r: JobOut["result"] | undefined): r is RecheckReportOut {
  return !!r && typeof r === "object" && "n_checked" in r && "rows" in r;
}
