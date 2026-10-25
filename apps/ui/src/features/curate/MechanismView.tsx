// One stored mechanism (#/curate/m/<mid>; GET /curate/{mid}): status, staleness reasons,
// warnings such as INDEPENDENCE_UNCHECKED, the steps with was_corrected marks and their original
// arrows, the re-check history, and the edits a PATCH allows (design §9.3): only the status moves
// of the transition table, notes, and order groups (not on a rejected mechanism). Steps
// themselves are immutable; a different correction means a new override and a new mechanism.
// DOM: section[data-testid=mechanism-view][data-id][data-status][data-stale];
// mechanism-stale-reasons[data-rechecked]; mechanism-warnings li[data-code] (the bare code of a
// "CODE: message" warning); curate-notes,
// curate-validate / curate-reject / curate-reopen (only the allowed moves), curate-save,
// recheck-button (POST /curate/{mid}/recheck; the job is followed as recheck-job).
import { useState } from "react";
import { getMechanism, patchMechanism, recheckMechanism, toApiError, type ApiError } from "../../api/client";
import type { MechanismOut, MechanismPatchIn } from "../../api/types.gen";
import { CapabilityNotice } from "../../components/CapabilityNotice";
import { ErrorBanner } from "../../components/ErrorBanner";
import { useCapability } from "../../state/store";
import { navigate, withParams, type Route } from "../../router";
import { MechanismStepList } from "./CurateStepList";
import { JobLine } from "./JobLine";
import { reactantsText } from "./MechanismList";
import { groupsFromToggles, groupsProblem, independenceHints, sameGroups, togglesFromGroups } from "./orderGroups";
import { isRecheckReport, RecheckPredictorSelect, RecheckReportView, RecheckRowsTable } from "./RecheckReport";
import { bumpRevision, createdWarnings } from "./session";
import { allowedTransitions, STATUS_MEANING, isStatus, statusBadgeClass, type MechanismStatus } from "./status";
import { fmtDate, parseStepDoc } from "./stepdoc";
import { useLoad } from "./useLoad";

interface Draft {
  notes: string | null;
  toggles: boolean[] | null;
}

/** The code of a server warning: they arrive as "CODE: message" (curation.py independence_issues
 *  and mechanism_out), or as a bare code. */
export function warningCode(w: string): string {
  return w.split(":", 1)[0].trim();
}

const WARNING_EXPLANATION: Record<string, string> = {
  INDEPENDENCE_UNCHECKED: "a merge broke map continuity, so the independence claim was accepted unchecked",
};

/** li[data-code] carries the bare code; the full "CODE: message" text is shown. */
export function MechanismWarnings({ warnings }: { warnings: readonly string[] }) {
  if (!warnings.length) return null;
  return (
    <ul className="cu-hints" data-testid="mechanism-warnings">
      {warnings.map((w) => {
        const code = warningCode(w);
        const why = WARNING_EXPLANATION[code];
        return (
          <li key={w} className="cu-hint-warn" data-code={code}>
            <code>{w}</code>
            {why ? ` (${why})` : ""}
          </li>
        );
      })}
    </ul>
  );
}

/** The stored versions are never rewritten (design §9.4), so a mechanism stays stale after a
 *  re-check: the banner then points at the re-check rows instead of asking for another one. */
export function StaleBanner({ m }: { m: Pick<MechanismOut, "stale_reasons" | "last_recheck"> }) {
  if (!m.stale_reasons.length) return null;
  const advice = m.last_recheck
    ? ` Last re-checked ${fmtDate(m.last_recheck)}; the stored versions are never rewritten, so see the re-check rows below.`
    : " Re-check it.";
  return (
    <div className="banner banner-warn" data-testid="mechanism-stale-reasons" data-rechecked={m.last_recheck ? "true" : "false"}>
      <strong>Stale</strong>: stored against another predictor, spec or physics version.{advice}
      <ul>
        {m.stale_reasons.map((r) => (
          <li key={r}>
            <code>{r}</code>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Header({ m, warnings }: { m: MechanismOut; warnings: readonly string[] }) {
  return (
    <>
      <h2 className="cu-title">
        Mechanism #{m.id} <span className={statusBadgeClass(m.status)} data-testid="mechanism-status">{m.status}</span>
        {m.stale && <span className="badge badge-warn">stale</span>}
      </h2>
      {isStatus(m.status) && <p className="muted">{STATUS_MEANING[m.status]}</p>}
      <dl className="kv">
        <dt>reactants</dt>
        <dd>
          <code>{reactantsText(m)}</code>
        </dd>
        <dt>conditions</dt>
        <dd>{m.conditions || <span className="muted">none recorded</span>}</dd>
        <dt>predictor</dt>
        <dd>
          {m.predictor} <code className="code">{m.checkpoint_hash}</code>
        </dd>
        <dt>spec</dt>
        <dd>
          <code>{m.spec_version}</code>
        </dd>
        <dt>created</dt>
        <dd>{fmtDate(m.created_at)}</dd>
        <dt>last re-check</dt>
        <dd>{fmtDate(m.last_recheck)}</dd>
      </dl>
      <StaleBanner m={m} />
      <MechanismWarnings warnings={warnings} />
    </>
  );
}

const MOVE_LABEL: Record<MechanismStatus, string> = {
  validated: "Validate",
  corrected: "Validate (as corrected)",
  rejected: "Reject",
  predicted: "Reopen",
};

const MOVE_TESTID: Record<MechanismStatus, string> = {
  validated: "curate-validate",
  corrected: "curate-validate",
  rejected: "curate-reject",
  predicted: "curate-reopen",
};

function Recheck({ route, m, onDone }: { route: Route; m: MechanismOut; onDone: () => void }) {
  const [predictor, setPredictor] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const recheck = useCapability("curate_recheck");
  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const created = await recheckMechanism(m.id, { predictor: predictor || null });
      navigate(withParams(route, { recheck: created.job_id }), { replace: true });
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="cu-section">
      <h3>Re-check</h3>
      <p className="muted">
        Re-scores every step with the current predictor through its dataset arrows and alignment. The status never changes
        automatically.
      </p>
      <div className="row cu-controls">
        <RecheckPredictorSelect value={predictor} onChange={setPredictor} testId="recheck-predictor" />
        <button
          type="button"
          className="btn"
          data-testid="recheck-button"
          disabled={busy || !recheck.enabled}
          title={recheck.enabled ? undefined : (recheck.reason ?? undefined)}
          onClick={start}
        >
          {busy ? "starting…" : "Re-check this mechanism"}
        </button>
      </div>
      <CapabilityNotice name="curate_recheck" title="Re-check is disabled" compact />
      <ErrorBanner error={error} title="Re-check not started" />
      <JobLine jobId={route.params.get("recheck")} testId="recheck-job" title="Re-check" onFinished={onDone}>
        {(job) => (isRecheckReport(job.result) ? <RecheckReportView report={job.result} /> : null)}
      </JobLine>
      <h4 className="cu-subhead">History</h4>
      <RecheckRowsTable rows={m.rechecks} testId="mechanism-rechecks" />
    </section>
  );
}

interface EditorProps {
  route: Route;
  m: MechanismOut;
  onSaved: (m: MechanismOut) => void;
}

/** The editable parts of a mechanism: a draft over the server values, and the PATCH sender. */
function useEditor(m: MechanismOut, onSaved: (m: MechanismOut) => void) {
  const [draft, setDraft] = useState<Draft>({ notes: null, toggles: null });
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const serverGroups = m.steps.map((s) => s.order_group);
  const toggles = draft.toggles ?? togglesFromGroups(serverGroups);
  const groups = groupsFromToggles(toggles, m.steps.length);
  const notes = draft.notes ?? m.notes;
  const rejected = m.status === "rejected";
  const notesDirty = notes !== m.notes;
  const groupsDirty = !rejected && !sameGroups(groups, serverGroups);
  const send = async (label: string, body: MechanismPatchIn) => {
    setBusy(label);
    setError(null);
    try {
      const out = await patchMechanism(m.id, body);
      setDraft({ notes: null, toggles: null });
      onSaved(out);
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(null);
    }
  };
  return {
    toggles,
    groups,
    notes,
    rejected,
    busy,
    error,
    dirty: notesDirty || groupsDirty,
    setNotes: (v: string) => setDraft((d) => ({ ...d, notes: v })),
    toggle: (i: number, v: boolean) => setDraft((d) => ({ ...d, toggles: toggles.map((t, k) => (k === i ? v : t)) })),
    save: () => send("save", { ...(notesDirty ? { notes } : {}), ...(groupsDirty ? { order_groups: groups } : {}) }),
    move: (to: MechanismStatus) => send(to, { status: to, ...(notesDirty ? { notes } : {}) }),
  };
}

function DrawDatasetToggle({ route, rejected }: { route: Route; rejected: boolean }) {
  const on = route.params.get("arrows") === "dataset";
  return (
    <div className="row cu-controls">
      <label>
        <input
          type="checkbox"
          data-testid="curate-draw-dataset"
          checked={on}
          onChange={(e) => navigate(withParams(route, { arrows: e.target.checked ? "dataset" : null }), { replace: true })}
        />{" "}
        draw the dataset arrows (atom level) instead of the token arrows
      </label>
      {rejected && <span className="muted">order groups are frozen on a rejected mechanism</span>}
    </div>
  );
}

interface ActionsProps {
  status: string;
  moves: readonly MechanismStatus[];
  busy: string | null;
  dirty: boolean;
  /** The curate_edit capability is off: every button disabled. */
  locked: boolean;
  onMove: (to: MechanismStatus) => void;
  onSave: () => void;
}

/** Only the moves of the transition table are offered (design §9.3). */
function Actions({ status, moves, busy, dirty, locked, onMove, onSave }: ActionsProps) {
  return (
    <div className="row cu-controls" data-testid="curate-actions" data-allowed={moves.join(",")}>
      {moves.map((to) => (
        <button
          key={to}
          type="button"
          className={to === "rejected" ? "btn" : "btn btn-primary"}
          data-testid={MOVE_TESTID[to]}
          data-target-status={to}
          disabled={busy !== null || locked}
          onClick={() => onMove(to)}
        >
          {busy === to ? "saving…" : MOVE_LABEL[to]}
        </button>
      ))}
      <button type="button" className="btn" data-testid="curate-save" disabled={busy !== null || !dirty || locked} onClick={onSave}>
        {busy === "save" ? "saving…" : "Save notes and groups"}
      </button>
      <span className="muted">
        allowed moves from {status}: {moves.join(", ") || "none"}
      </span>
    </div>
  );
}

function Editor({ route, m, onSaved }: EditorProps) {
  const ed = useEditor(m, onSaved);
  const docs = m.steps.map((s) => parseStepDoc(s.step));
  const hints = independenceHints(
    ed.groups,
    docs.map((d) => ({ reactantStateId: d.reactantStateId, productStateId: d.productStateId, touchedMaps: d.touchedMaps })),
  );
  const moves = allowedTransitions(m.status, m.steps.filter((s) => s.was_corrected).length);
  const problem = groupsProblem(ed.groups);
  const edit = useCapability("curate_edit");
  return (
    <>
      <DrawDatasetToggle route={route} rejected={ed.rejected} />
      <MechanismStepList
        steps={m.steps}
        groups={ed.groups}
        independent={ed.toggles}
        hints={hints}
        disabled={ed.rejected || ed.busy !== null}
        drawDataset={route.params.get("arrows") === "dataset"}
        onToggle={ed.toggle}
      />
      <label className="cu-notes-label">
        notes
        <textarea data-testid="curate-notes" value={ed.notes} rows={4} onChange={(e) => ed.setNotes(e.target.value)} />
      </label>
      {problem && <div className="cu-hint-error">{problem}</div>}
      <CapabilityNotice name="curate_edit" title="Editing a mechanism is disabled" compact />
      <Actions
        status={m.status}
        moves={moves}
        busy={ed.busy}
        dirty={ed.dirty}
        locked={!edit.enabled}
        onMove={ed.move}
        onSave={ed.save}
      />
      <ErrorBanner error={ed.error} title="Not saved" testId="curate-error" />
    </>
  );
}

export function MechanismView({ route, mid }: { route: Route; mid: number }) {
  const valid = Number.isInteger(mid) && mid >= 0;
  const m = useLoad<MechanismOut>(valid ? `/curate/${mid}` : null, (o) => getMechanism(mid, o));
  if (!valid) return <ErrorBanner error={`"${route.parts[1] ?? ""}" is not a mechanism id`} title="Not found" testId="curate-error" />;
  if (m.error && !m.data) return <ErrorBanner error={m.error} title={`Mechanism #${mid}`} testId="curate-error" />;
  if (!m.data) return <p className="muted">loading mechanism #{mid}…</p>;
  const mech = m.data;
  const warnings = [...new Set([...mech.warnings, ...createdWarnings(mech.id)])];
  const saved = (out: MechanismOut) => {
    m.set(out);
    bumpRevision();
  };
  return (
    <section
      className="panel cu-mechanism"
      data-testid="mechanism-view"
      data-id={mech.id}
      data-status={mech.status}
      data-stale={mech.stale ? "true" : "false"}
    >
      <p>
        <a href="#/curate">← all mechanisms</a>
      </p>
      <Header m={mech} warnings={warnings} />
      <Editor key={`${mech.id}`} route={route} m={mech} onSaved={saved} />
      <Recheck
        route={route}
        m={mech}
        onDone={() => {
          m.reload();
          bumpRevision();
        }}
      />
    </section>
  );
}
