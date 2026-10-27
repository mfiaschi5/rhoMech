// Export (spec v9): GET /curate/export/pmechdb?status=validated&status=corrected downloads the
// curated steps of the chosen statuses as PMechDB raw records (text/csv, one quoted record per
// line: a file for the user, never training data). Its report comes in the X-Ksmech-Export-Report
// header: the statuses, mechanisms, steps considered and written, and the steps skipped by reason
// code. Each download is built from the database anew; nothing is written on the server.
// DOM: <section data-testid="export-panel">, export-status-{validated|corrected}, button
// [data-testid=export-button], export-withdraw-warning while a status is unchecked, a[data-testid=
// export-link] (the plain download URL), div[data-testid=export-report][data-written] after a
// download, ul[data-testid=export-warnings] li[data-code] when the report has warnings.
import { useState } from "react";
import { toApiError, type ApiError } from "../../api/http";
import { exportPmechdb, exportPmechdbUrl, type PmechdbExportReport } from "../../api/curate";
import { CapabilityNotice } from "../../components/CapabilityNotice";
import { ErrorBanner } from "../../components/ErrorBanner";
import type { Route } from "../../router";
import { useCapability } from "../../state/store";
import { CountList } from "./RecheckReport";

export type ExportStatus = "validated" | "corrected";
const EXPORT_STATUSES: ExportStatus[] = ["validated", "corrected"];
export const EXPORT_FILENAME = "ksmech_curated_pmechdb.csv";

/** "CODE: message" warnings of a report, when it has any. */
function ExportWarnings({ warnings }: { warnings: readonly string[] }) {
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

export function ExportReportView({ report }: { report: PmechdbExportReport }) {
  const r = report;
  return (
    <div className="cu-report" data-testid="export-report" data-written={r.n_steps_written} data-considered={r.n_steps_considered}>
      <ExportWarnings warnings={r.warnings ?? []} />
      <dl className="kv">
        <dt>statuses</dt>
        <dd>{(r.statuses ?? []).join(", ") || "–"}</dd>
        <dt>mechanisms</dt>
        <dd data-testid="export-n-mechanisms">{r.n_mechanisms}</dd>
        <dt>steps</dt>
        <dd>
          <span data-testid="export-n-written">{r.n_steps_written}</span> written of {r.n_steps_considered} considered
        </dd>
        <dt>skipped</dt>
        <dd>
          <CountList counts={r.skipped ?? {}} empty="none" />
        </dd>
      </dl>
    </div>
  );
}

/** Shown while a status is unchecked: its mechanisms are left out of the download. */
export function ExportWithdrawWarning({ chosen }: { chosen: readonly ExportStatus[] }) {
  const unchecked = EXPORT_STATUSES.filter((s) => !chosen.includes(s));
  if (!unchecked.length) return null;
  return (
    <p className="cu-hint-warn" data-testid="export-withdraw-warning" data-unchecked={unchecked.join(",")}>
      {unchecked.join(" and ")} mechanisms are left out of this download (only the checked statuses are exported).
    </p>
  );
}

/** Hand a text to the browser as a file download. */
function saveText(text: string, filename: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function useDownload() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [report, setReport] = useState<PmechdbExportReport | null>(null);
  const download = async (statuses: ExportStatus[]) => {
    setBusy(true);
    setError(null);
    try {
      const out = await exportPmechdb(statuses);
      setReport(out.report);
      saveText(out.csv, EXPORT_FILENAME);
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(false);
    }
  };
  return { busy, error, report, download };
}

export function ExportPanel(_props: { route: Route }) {
  const [statuses, setStatuses] = useState<Record<ExportStatus, boolean>>({ validated: true, corrected: true });
  const { busy, error, report, download } = useDownload();
  const exp = useCapability("export");
  const chosen = EXPORT_STATUSES.filter((s) => statuses[s]);
  return (
    <section className="cu-section" data-testid="export-panel">
      <h3>Export</h3>
      <p className="muted" data-testid="export-description">
        Downloads the curated steps of the chosen statuses as PMechDB raw records (<code>{EXPORT_FILENAME}</code>, one
        quoted record per line), built from the database on every download; steps that cannot be written are counted by
        reason in the report. Rejected mechanisms are never exported. A download for the user, not training data.
      </p>
      <div className="row cu-controls">
        {EXPORT_STATUSES.map((s) => (
          <label key={s}>
            <input type="checkbox" data-testid={`export-status-${s}`} checked={statuses[s]} onChange={(e) => setStatuses({ ...statuses, [s]: e.target.checked })} /> {s}
          </label>
        ))}
        <button type="button" className="btn btn-primary" data-testid="export-button" disabled={busy || !chosen.length || !exp.enabled} title={exp.enabled ? undefined : (exp.reason ?? undefined)} onClick={() => void download(chosen)}>
          {busy ? "exporting…" : "Download CSV"}
        </button>
        {chosen.length > 0 && (
          <a className="btn btn-small" data-testid="export-link" href={exportPmechdbUrl(chosen)} download={EXPORT_FILENAME}>
            direct link
          </a>
        )}
      </div>
      <ExportWithdrawWarning chosen={chosen} />
      <CapabilityNotice name="export" title="Export is disabled" compact />
      <ErrorBanner error={error} title="Export failed" testId="export-error" />
      {report && <ExportReportView report={report} />}
    </section>
  );
}
