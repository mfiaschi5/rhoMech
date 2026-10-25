// Export (design §9.6): POST /curate/export starts a batch job; its ExportReportOut says what was
// written to data/processed/user_curated/ and why every other step was not: skipped reason codes
// (ARROW_ENCODING_FAILED, DUPLICATE_OF_PMECHDB, DUPLICATE_IN_EXPORT, ...), ingest rejections,
// duplicates, withdrawn and changed ids, the paths and the sha256 of steps.parquet, and the
// export's warnings (PMECHDB_UNAVAILABLE, COMPACT_FALLBACK). Every export rebuilds steps.parquet
// from the database with the chosen statuses only (export.py), so the panel warns when a status
// is unchecked: its previously exported steps are withdrawn.
// DOM: button[data-testid=export-button]; export-withdraw-warning while a status is unchecked;
// ul[data-testid=export-warnings] li[data-code]; div[data-testid=export-job][data-status] while a job is
// followed; div[data-testid=export-report][data-export-id][data-written] for the job's report, or
// for GET /curate/export/latest when no job is in the hash ("export-report-none" when none).
import { useState } from "react";
import { exportCurated, getLatestExport, toApiError, type ApiError } from "../../api/client";
import type { ExportIn, ExportReportOut, JobOut } from "../../api/types.gen";
import { CapabilityNotice } from "../../components/CapabilityNotice";
import { ErrorBanner } from "../../components/ErrorBanner";
import { navigate, withParams, type Route } from "../../router";
import { useCapability, useStore } from "../../state/store";
import { JobLine } from "./JobLine";
import { CountList } from "./RecheckReport";
import { bumpRevision, curationRevision } from "./session";
import { fmtDate } from "./stepdoc";
import { useLoad } from "./useLoad";

export type ExportStatus = "validated" | "corrected";
const EXPORT_STATUSES: ExportStatus[] = ["validated", "corrected"];

export function isExportReport(r: JobOut["result"] | undefined): r is ExportReportOut {
  return !!r && typeof r === "object" && "export_id" in r && "sha256" in r;
}

function StepRows({ steps }: { steps: readonly Record<string, unknown>[] }) {
  if (!steps.length) return null;
  const cell = (v: unknown) => (v === null || v === undefined ? "" : typeof v === "object" ? JSON.stringify(v) : String(v));
  return (
    <details className="cu-details" open={steps.length <= 12}>
      <summary>{steps.length} step rows</summary>
      <div className="cu-table-wrap">
        <table className="table" data-testid="export-report-steps">
          <thead>
            <tr>
              <th>mech.</th>
              <th>step</th>
              <th>written as</th>
              <th>reason</th>
              <th>detail</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((s, k) => (
              <tr key={k} className={s.code ? "cu-row-warn" : undefined} data-code={cell(s.code)}>
                <td>{cell(s.mechanism_id)}</td>
                <td>{typeof s.step_index === "number" ? s.step_index + 1 : cell(s.step_index)}</td>
                <td>{s.step_id ? <code>{cell(s.step_id)}</code> : "–"}</td>
                <td>{s.code ? <code>{cell(s.code)}</code> : ""}</td>
                <td>{cell(s.detail)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

/** "CODE: message" warnings of an export. `warnings` is read through a widened type, so this
 *  compiles whether or not types.gen.ts has the field yet. */
function ExportWarnings({ report }: { report: ExportReportOut }) {
  const warnings = (report as ExportReportOut & { warnings?: readonly string[] }).warnings ?? [];
  if (!warnings.length) return null;
  return (
    <ul className="cu-hints" data-testid="export-warnings">
      {warnings.map((w) => (
        <li key={w} className="cu-hint-warn" data-code={w.split(":", 1)[0].trim()}>
          <code>{w}</code>
        </li>
      ))}
    </ul>
  );
}

export function ExportReportView({ report }: { report: ExportReportOut }) {
  const r = report;
  return (
    <div className="cu-report" data-testid="export-report" data-export-id={r.export_id} data-written={r.n_steps_written}>
      <ExportWarnings report={r} />
      <dl className="kv">
        <dt>export</dt>
        <dd>
          <code>{r.export_id}</code> at {fmtDate(r.created_at)}
        </dd>
        <dt>spec</dt>
        <dd>
          <code>{r.spec_version}</code>
        </dd>
        <dt>mechanisms</dt>
        <dd data-testid="export-n-mechanisms">{r.n_mechanisms}</dd>
        <dt>steps</dt>
        <dd>
          <span data-testid="export-n-written">{r.n_steps_written}</span> written of {r.n_steps_considered} considered
        </dd>
        <dt>skipped</dt>
        <dd>
          <CountList counts={r.skipped} empty="none" />
        </dd>
        <dt>ingest rejections</dt>
        <dd>
          <CountList counts={r.ingest_rejections} empty="none" />
        </dd>
        <dt>duplicates</dt>
        <dd>
          <span data-testid="export-duplicates-pmechdb">{r.duplicates_pmechdb}</span> of PMechDB,{" "}
          <span data-testid="export-duplicates-in-export">{r.duplicates_in_export}</span> within this export
        </dd>
        <dt>ledger</dt>
        <dd>
          <span data-testid="export-withdrawn">{r.withdrawn}</span> withdrawn since earlier exports,{" "}
          <span data-testid="export-changed-ids">{r.changed_ids}</span> changed ids
        </dd>
        <dt>files</dt>
        <dd>
          <ul className="cu-paths">
            {Object.entries(r.paths).map(([name, path]) => (
              <li key={name}>
                {name}: <code>{path}</code>
              </li>
            ))}
          </ul>
        </dd>
        <dt>sha256</dt>
        <dd>
          <code data-testid="export-sha256">{r.sha256}</code>
        </dd>
      </dl>
      <StepRows steps={r.steps} />
    </div>
  );
}

function LatestExport() {
  const revision = useStore(curationRevision);
  const latest = useLoad(`/curate/export/latest#${revision}`, (o) => getLatestExport(o));
  if (latest.error?.code === "EXPORT_NOT_FOUND") {
    return (
      <p className="muted" data-testid="export-report-none">
        No export yet.
      </p>
    );
  }
  if (latest.error) return <ErrorBanner error={latest.error} title="Cannot read the latest export" />;
  if (!latest.data) return <p className="muted">loading the latest export…</p>;
  return (
    <>
      <h4 className="cu-subhead">Latest export</h4>
      <ExportReportView report={latest.data} />
    </>
  );
}

function useStartExport(route: Route) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const start = async (statuses: ExportStatus[]) => {
    setBusy(true);
    setError(null);
    try {
      const body: ExportIn = { statuses };
      const created = await exportCurated(body);
      navigate(withParams(route, { export: created.job_id }), { replace: true });
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(false);
    }
  };
  return { busy, error, start };
}

function JobReport({ jobId }: { jobId: string }) {
  return (
    <JobLine jobId={jobId} testId="export-job" title="Export" onFinished={() => bumpRevision()}>
      {(job) =>
        isExportReport(job.result) ? (
          <ExportReportView report={job.result} />
        ) : (
          <p className="cu-hint-error">The export job finished without an ExportReport (result kind {job.result_kind ?? "none"}).</p>
        )
      }
    </JobLine>
  );
}

/** Shown while a status is unchecked: the export rebuilds steps.parquet from the chosen statuses
 *  only, so the steps an earlier export wrote for the unchecked ones are withdrawn. */
export function ExportWithdrawWarning({ chosen }: { chosen: readonly ExportStatus[] }) {
  const unchecked = EXPORT_STATUSES.filter((s) => !chosen.includes(s));
  if (!unchecked.length) return null;
  return (
    <p className="cu-hint-warn" data-testid="export-withdraw-warning" data-unchecked={unchecked.join(",")}>
      Previously exported {unchecked.join(" and ")} steps will be withdrawn from steps.parquet (only the checked
      statuses are exported).
    </p>
  );
}

export function ExportPanel({ route }: { route: Route }) {
  const jobId = route.params.get("export");
  const [statuses, setStatuses] = useState<Record<ExportStatus, boolean>>({ validated: true, corrected: true });
  const { busy, error, start } = useStartExport(route);
  const exp = useCapability("export");
  const chosen = EXPORT_STATUSES.filter((s) => statuses[s]);
  return (
    <section className="cu-section" data-testid="export-panel">
      <h3>Export</h3>
      <p className="muted" data-testid="export-description">
        Rebuilds <code>data/processed/user_curated/steps.parquet</code> (canonical schema, source user_curated) and{" "}
        <code>pmechdb_arrow_codes.csv</code> from the database on every export (mechanisms of the chosen statuses only),
        de-duplicated against PMechDB and within the export by (reactant pool key, product pool key); steps no longer
        exported are counted as withdrawn. Rejected mechanisms are never exported.
      </p>
      <div className="row cu-controls">
        {EXPORT_STATUSES.map((s) => (
          <label key={s}>
            <input
              type="checkbox"
              data-testid={`export-status-${s}`}
              checked={statuses[s]}
              onChange={(e) => setStatuses({ ...statuses, [s]: e.target.checked })}
            />{" "}
            {s}
          </label>
        ))}
        <button type="button" className="btn btn-primary" data-testid="export-button" disabled={busy || !chosen.length || !exp.enabled} title={exp.enabled ? undefined : (exp.reason ?? undefined)} onClick={() => start(chosen)}>
          {busy ? "starting…" : "Export"}
        </button>
        {jobId && (
          <button type="button" className="btn btn-small" onClick={() => navigate(withParams(route, { export: null }), { replace: true })}>
            hide job
          </button>
        )}
      </div>
      <ExportWithdrawWarning chosen={chosen} />
      <CapabilityNotice name="export" title="Export is disabled" compact />
      <ErrorBanner error={error} title="Export not started" />
      {jobId ? <JobReport jobId={jobId} /> : <LatestExport />}
    </section>
  );
}
