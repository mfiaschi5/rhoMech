// The Curate page (design §11.8). Routes (§11.2; all state in the hash):
//   #/curate                          recent prediction jobs to curate from, and the mechanisms
//   #/curate/<jobId>?pathway=<pid>    curate one pathway of a job (POST /curate)
//   #/curate/m/<mechId>               one stored mechanism (GET / PATCH /curate/{mid}, re-check)
// Every view also shows the mechanism list (status / stale / predictor filters), the batch
// re-check and the export with its report. DOM: div[data-testid=curate-panel][data-view].
import { listJobs } from "../../api/client";
import { ErrorBanner } from "../../components/ErrorBanner";
import type { PageProps } from "../../registry";
import { curateHash } from "../../router";
import { MechanismsPanel } from "./MechanismList";
import { MechanismView } from "./MechanismView";
import { PathwayCurate } from "./PathwayCurate";
import { fmtDate } from "./stepdoc";
import { useLoad } from "./useLoad";

function RecentJobs() {
  const jobs = useLoad("/jobs?kind=predict", (o) => listJobs({ kind: "predict", limit: 20 }, o));
  return (
    <section className="panel" data-testid="curate-jobs">
      <h2 className="cu-title">Curate</h2>
      <p className="muted">
        Pick a prediction, then one of its pathways: validate it, reject it, or save it without a verdict. Steps you mark as
        independent of the previous one share an order group. Correct a wrong step on the Correct page first; the corrected
        status is then set automatically.
      </p>
      <ErrorBanner error={jobs.error} title="Cannot list jobs" />
      {jobs.data && jobs.data.length === 0 && (
        <p className="muted">
          No prediction yet. <a href="#/predict">Start one.</a>
        </p>
      )}
      {jobs.data && jobs.data.length > 0 && (
        <ul className="cu-choices">
          {jobs.data.map((j) => (
            <li key={j.job_id}>
              <a href={curateHash(j.job_id)} data-testid={`curate-job-${j.job_id}`} data-status={j.status}>
                <code>{j.job_id}</code> {j.label || j.kind}
              </a>{" "}
              <span className={j.status === "done" ? "badge badge-ok" : "badge"}>{j.status}</span>{" "}
              <span className="muted">{fmtDate(j.created_at)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function CuratePage({ route }: PageProps) {
  const [first, second] = route.parts;
  const mid = first === "m" ? Number.parseInt(second ?? "", 10) : null;
  const view = !first ? "landing" : first === "m" ? "mechanism" : "pathway";
  return (
    <div className="cu-page stack" data-testid="curate-panel" data-view={view}>
      {view === "landing" && <RecentJobs />}
      {view === "pathway" && <PathwayCurate key={first} route={route} jobId={first} />}
      {view === "mechanism" && <MechanismView key={second} route={route} mid={mid ?? Number.NaN} />}
      <MechanismsPanel
        route={route}
        currentId={mid !== null && Number.isInteger(mid) ? mid : null}
        recheckTestId={view === "mechanism" ? "recheck-selected-button" : "recheck-button"}
      />
    </div>
  );
}
