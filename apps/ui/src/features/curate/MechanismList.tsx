// The stored mechanisms (GET /curate) with status, stale and predictor filters, a batch re-check
// (POST /curate/recheck) and the Export panel (design §11.8, §9.4, §9.6). Filters and the job ids
// live in the hash: #/curate?status=&stale=&predictor=&offset=&recheckall=<job>&export=<job>.
// DOM: table[data-testid=mechanism-list] with tr[data-testid=mechanism-row-{id}][data-status][data-stale].
import { useState } from "react";
import { listMechanisms, recheckMechanisms, toApiError, type ApiError } from "../../api/client";
import type { MechanismSummaryOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import { PendingNote } from "../../components/PendingNote";
import { formatRoute, navigate, withParams, type Route } from "../../router";
import { useStore } from "../../state/store";
import { ExportPanel } from "./ExportReport";
import { JobLine } from "./JobLine";
import { isRecheckReport, RecheckPredictorSelect, RecheckReportView } from "./RecheckReport";
import { bumpRevision, curationRevision } from "./session";
import { MECHANISM_STATUSES, statusBadgeClass } from "./status";
import { fmtDate } from "./stepdoc";
import { useLoad } from "./useLoad";

const PAGE = 50;
const PREDICTORS = ["floor", "gbdt", "transformer", "stub"];

export function reactantsText(m: Pick<MechanismSummaryOut, "reactants" | "stoichiometry">): string {
  return m.reactants.map((s, i) => `${(m.stoichiometry[i] ?? 1) > 1 ? `${m.stoichiometry[i]} × ` : ""}${s}`).join(" + ");
}

const LIST_PARAMS = ["status", "stale", "predictor", "offset"];

/** "#/curate/m/<id>" keeping the list filters of the current route. */
export function mechanismHash(route: Route, id: number): string {
  const params: Record<string, string> = {};
  for (const k of LIST_PARAMS) {
    const v = route.params.get(k);
    if (v) params[k] = v;
  }
  return formatRoute({ page: "curate", parts: ["m", id], params });
}

function Filters({ route }: { route: Route }) {
  const set = (key: string, value: string) => navigate(withParams(route, { [key]: value || null, offset: null }), { replace: true });
  const p = route.params;
  return (
    <div className="row cu-controls">
      <label>
        status{" "}
        <select data-testid="mechanism-filter-status" value={p.get("status") ?? ""} onChange={(e) => set("status", e.target.value)}>
          <option value="">any</option>
          {MECHANISM_STATUSES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>
      <label>
        stale{" "}
        <select data-testid="mechanism-filter-stale" value={p.get("stale") ?? ""} onChange={(e) => set("stale", e.target.value)}>
          <option value="">any</option>
          <option value="true">stale only</option>
          <option value="false">fresh only</option>
        </select>
      </label>
      <label>
        predictor{" "}
        <select data-testid="mechanism-filter-predictor" value={p.get("predictor") ?? ""} onChange={(e) => set("predictor", e.target.value)}>
          <option value="">any</option>
          {PREDICTORS.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

interface RowsProps {
  route: Route;
  rows: readonly MechanismSummaryOut[];
  currentId: number | null;
  selected: ReadonlySet<number>;
  onSelect: (id: number, on: boolean) => void;
}

function Rows({ route, rows, currentId, selected, onSelect }: RowsProps) {
  return (
    <tbody>
      {rows.map((m) => (
        <tr
          key={m.id}
          data-testid={`mechanism-row-${m.id}`}
          data-status={m.status}
          data-stale={m.stale ? "true" : "false"}
          className={`cu-click-row${m.id === currentId ? " cu-row-current" : ""}`}
          onClick={() => navigate(mechanismHash(route, m.id))}
        >
          <td onClick={(e) => e.stopPropagation()}>
            <input
              type="checkbox"
              aria-label={`select mechanism ${m.id}`}
              data-testid={`mechanism-select-${m.id}`}
              checked={selected.has(m.id)}
              onChange={(e) => onSelect(m.id, e.target.checked)}
            />
          </td>
          <td>
            <a href={mechanismHash(route, m.id)} onClick={(e) => e.stopPropagation()}>
              #{m.id}
            </a>
          </td>
          <td>{fmtDate(m.created_at)}</td>
          <td className="cu-smiles-cell" title={reactantsText(m)}>
            {reactantsText(m)}
          </td>
          <td>
            <span className={statusBadgeClass(m.status)}>{m.status}</span>
          </td>
          <td title={m.checkpoint_hash}>{m.predictor}</td>
          <td className="num">{m.n_steps}</td>
          <td>{m.stale ? <span className="badge badge-warn">stale</span> : <span className="muted">fresh</span>}</td>
          <td>{fmtDate(m.last_recheck)}</td>
          <td className="cu-notes-cell" title={m.notes}>
            {m.notes.split("\n")[0]}
          </td>
        </tr>
      ))}
    </tbody>
  );
}

function MechanismTable(props: RowsProps & { loading: boolean }) {
  return (
    <div className="cu-table-wrap">
      <table
        className="table"
        data-testid="mechanism-list"
        data-count={props.rows.length}
        data-loading={props.loading ? "true" : "false"}
      >
        <thead>
          <tr>
            <th />
            <th>id</th>
            <th>created</th>
            <th>reactants</th>
            <th>status</th>
            <th>predictor</th>
            <th className="num">steps</th>
            <th>stale</th>
            <th>last re-check</th>
            <th>notes</th>
          </tr>
        </thead>
        <Rows {...props} />
      </table>
    </div>
  );
}

function Pager({ offset, count, onPage }: { offset: number; count: number; onPage: (offset: number) => void }) {
  return (
    <div className="row cu-pager">
      <button type="button" className="btn btn-small" disabled={offset === 0} onClick={() => onPage(offset - PAGE)}>
        previous
      </button>
      <span className="muted">{count ? `${offset + 1}–${offset + count}` : "–"}</span>
      <button type="button" className="btn btn-small" disabled={count < PAGE} onClick={() => onPage(offset + PAGE)}>
        next
      </button>
    </div>
  );
}

function BatchRecheck({ route, selected, testId }: { route: Route; selected: readonly number[]; testId: string }) {
  const [staleOnly, setStaleOnly] = useState(true);
  const [predictor, setPredictor] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const body = { mechanism_ids: selected.length ? [...selected] : null, predictor: predictor || null, stale_only: staleOnly };
      const created = await recheckMechanisms(body);
      navigate(withParams(route, { recheckall: created.job_id }), { replace: true });
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(false);
    }
  };
  const jobId = route.params.get("recheckall");
  const what = selected.length ? `the ${selected.length} selected` : "all";
  return (
    <div className="cu-section">
      <div className="row cu-controls">
        <label>
          <input type="checkbox" data-testid="recheck-stale-only" checked={staleOnly} onChange={(e) => setStaleOnly(e.target.checked)} /> stale
          only
        </label>
        <RecheckPredictorSelect value={predictor} onChange={setPredictor} testId="recheck-all-predictor" />
        <button type="button" className="btn" data-testid={testId} disabled={busy} onClick={start}>
          {busy ? "starting…" : `Re-check ${what}${staleOnly ? " (stale)" : ""}`}
        </button>
      </div>
      <ErrorBanner error={error} title="Re-check not started" />
      <JobLine jobId={jobId} testId="recheck-all-job" title="Re-check" onFinished={() => bumpRevision()}>
        {(job) => (isRecheckReport(job.result) ? <RecheckReportView report={job.result} testId="recheck-all-report" /> : null)}
      </JobLine>
    </div>
  );
}

export interface MechanismsPanelProps {
  route: Route;
  currentId: number | null;
  /** "recheck-button" unless the page shows a single mechanism, whose own re-check takes that id. */
  recheckTestId: string;
}

export function MechanismsPanel({ route, currentId, recheckTestId }: MechanismsPanelProps) {
  const revision = useStore(curationRevision);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const p = route.params;
  const status = p.get("status") || undefined;
  const staleParam = p.get("stale");
  const stale = staleParam === "true" ? true : staleParam === "false" ? false : undefined;
  const predictor = p.get("predictor") || undefined;
  const offset = Math.max(0, Number.parseInt(p.get("offset") ?? "0", 10) || 0);
  const key = `/curate?status=${status ?? ""}&stale=${staleParam ?? ""}&predictor=${predictor ?? ""}&offset=${offset}#${revision}`;
  const list = useLoad(key, (o) => listMechanisms({ status, stale, predictor, limit: PAGE, offset }, o));
  const rows = list.data ?? [];
  const onSelect = (id: number, on: boolean) =>
    setSelected((s) => {
      const next = new Set(s);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });
  const page = (o: number) => navigate(withParams(route, { offset: o > 0 ? o : null }), { replace: true });
  return (
    <section className="panel cu-mechanisms" data-testid="mechanisms-panel">
      <h3>Mechanisms</h3>
      <Filters route={route} />
      <ErrorBanner error={list.error} title="Cannot list mechanisms" />
      {list.pending && <PendingNote label={list.pending.label} worker={list.pending.worker} />}
      <MechanismTable route={route} rows={rows} loading={list.loading} currentId={currentId} selected={selected} onSelect={onSelect} />
      {!list.loading && !list.error && rows.length === 0 && <p className="muted">No mechanism matches these filters.</p>}
      <Pager offset={offset} count={rows.length} onPage={page} />
      <BatchRecheck route={route} selected={[...selected].sort((a, b) => a - b)} testId={recheckTestId} />
      <ExportPanel route={route} />
    </section>
  );
}
