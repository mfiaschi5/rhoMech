// CorrectEditor: the R editor on one pool state: the toolbar, the pool drawing with R, its
// arrows and the gestures (PoolCanvas), the entries table, the live check (CheckPanel), and
// Apply (POST /jobs/{id}/override: the new step and its product state, which onApplied opens).
//
// Props:
//   jobId       string                 the job whose network receives the step
//   stateId     string                 the parent pool state
//   initial     {matrix, arrows} | null   preloaded R (a predicted step being corrected)
//   onApplied   (edge, node) => void
//   extra       ReactNode              more actions beside Apply (Save to curation)
//
// The hidden H that R or its arrows touch are drawn; "show every H" draws them all (to give or
// take a proton).
//
// DOM: correct-editor[data-state-id], the EditorToolbar ids, correct-show-h, correct-canvas (PoolCanvas: atoms
// correct-canvas-atom-<i>, pairs correct-canvas-bond-<i>-<j>, marks correct-canvas-r-...),
// the EntryTable and CheckPanel ids, correct-apply, correct-apply-error.
import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from "react";
import { toApiError, type ApiError } from "../../api/client";
import type { PendingInfo } from "../../api/hooks";
import { ErrorBanner } from "../../components/ErrorBanner";
import { PendingNote } from "../../components/PendingNote";
import { CheckPanel } from "./CheckPanel";
import { editorReducer, initEditor, type EditorAction, type EditorState } from "./editor";
import { EditorToolbar } from "./EditorToolbar";
import { EntryTable } from "./EntryTable";
import { useDrawnStructure, usePool, useStepCheck } from "./hooks";
import { Panel } from "./Panel";
import { PoolCanvas, type CanvasInteraction } from "./PoolCanvas";
import type { PoolStructure } from "./poolStructure";
import { arrowFromGesture, isEmptyMatrix, localIssues } from "./rmatrix";
import { useEditorKeys } from "./useEditorKeys";
import { overrideStep } from "./v9api";
import type { ArrowReadingOut, EdgeOut, MatrixIO, NodeOut } from "./v9types";

export interface CorrectEditorProps {
  jobId: string;
  stateId: string;
  initial: { matrix: MatrixIO; arrows: ArrowReadingOut[] } | null;
  onApplied: (edge: EdgeOut, node: NodeOut) => void;
  extra?: ReactNode;
}

function useEditor(structure: PoolStructure | null, initial: CorrectEditorProps["initial"]) {
  const ctx = useRef<PoolStructure | null>(structure);
  ctx.current = structure;
  const reducer = useCallback((s: EditorState, a: EditorAction) => editorReducer(s, a, ctx.current), []);
  const [state, dispatch] = useReducer(reducer, initial, (i) => initEditor(i?.matrix, i?.arrows ?? []));
  return { state, dispatch };
}

/** The override call: 202 Pending (the parent's analysis is computing) is retried by request(),
 *  with its label shown; a 422 STEP_INVALID lists the server's issues. */
function useApply(jobId: string, stateId: string, matrix: MatrixIO, onApplied: CorrectEditorProps["onApplied"]) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [pending, setPending] = useState<PendingInfo | null>(null);
  const latest = useRef({ matrix, onApplied });
  latest.current = { matrix, onApplied };
  const apply = useCallback(async () => {
    if (busy || isEmptyMatrix(latest.current.matrix)) return;
    setBusy(true);
    setError(null);
    setPending(null);
    try {
      const onPending = (label: string, worker: PendingInfo["worker"]) => setPending({ label, worker });
      const out = await overrideStep(jobId, { parent_state_id: stateId, matrix: latest.current.matrix }, { onPending });
      latest.current.onApplied(out.edge, out.node);
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(false);
      setPending(null);
    }
  }, [busy, jobId, stateId]);
  return { busy, error, pending, apply };
}

/** The issues of a 422 STEP_INVALID override (error.detail.issues). */
function ServerIssues({ error, label }: { error: ApiError | null; label: (i: number) => string }) {
  const issues = (error?.detail?.issues ?? null) as { code: string; message: string; atoms: number[] }[] | null;
  if (!Array.isArray(issues) || !issues.length) return null;
  return (
    <ul className="pv-issues" data-testid="correct-apply-issues">
      {issues.map((x, k) => (
        <li key={k} data-code={x.code}>
          <code className="code">{x.code}</code> {x.message}
          {x.atoms?.length ? <span className="muted"> ({x.atoms.map(label).join(", ")})</span> : null}
        </li>
      ))}
    </ul>
  );
}

export function CorrectEditor({ jobId, stateId, initial, onApplied, extra }: CorrectEditorProps) {
  const pool = usePool(stateId);
  const s = pool.structure;
  const { state, dispatch } = useEditor(s, initial);
  const check = useStepCheck(stateId, state.matrix);
  const local = useMemo(() => localIssues(state.matrix, s), [state.matrix, s]);
  const [hover, setHover] = useState<number[] | null>(null);
  const { busy, error, pending, apply } = useApply(jobId, stateId, state.matrix, onApplied);
  useEditorKeys(dispatch, () => void apply());
  useEffect(() => setHover(null), [state.matrix]);
  const res = check.current ? check.result : null;
  const shownArrows = state.arrows.length ? state.arrows : (check.result?.arrows ?? []);
  const [allH, setAllH] = useState(false);
  const drawn = useDrawnStructure(pool, state.matrix, shownArrows, allH, initial?.matrix ?? null);
  const charges = useMemo(() => (res ? new Map(res.formal_charges) : null), [res]);
  const interaction: CanvasInteraction = {
    tool: state.tool,
    onAtom: (i) => dispatch({ type: "clickAtom", i }),
    onBond: (i, j) => dispatch({ type: "clickBond", i, j }),
    onArrow: (from, to) => {
      const g = arrowFromGesture(from, to, s);
      dispatch(g.ok ? { type: "arrow", arrow: g.arrow } : { type: "notice", message: g.error });
    },
    onArrowClick: (index) => dispatch({ type: "clickArrow", index }),
  };
  const blocked = isEmptyMatrix(state.matrix) ? "R is empty" : local.length ? "fix the issues first" : null;
  return (
    <div className="pv-correct-grid" data-testid="correct-editor" data-state-id={stateId}>
      <Panel title="Reaction matrix" note="edit R on the reactant pool; the arrows are read from it" className="pv-editor-panel">
        <div className="stack">
          <div className="row pv-inline-row">
            <EditorToolbar state={state} dispatch={dispatch} />
            <label className="field-inline pv-small">
              <input type="checkbox" data-testid="correct-show-h" checked={allH} onChange={(e) => setAllH(e.target.checked)} />
              show every H
            </label>
          </div>
          {drawn ? (
            <PoolCanvas structure={drawn} testId="correct-canvas" height={380} matrix={state.matrix} arrows={shownArrows} arrowsFaded={!state.arrows.length && !check.current} chargesAfter={charges} highlight={new Set(hover ?? [])} pending={state.pending} selectedKeys={state.selection} interaction={interaction} />
          ) : (
            <div className="pv-canvas-placeholder muted">{pool.loading ? "loading the depiction…" : "no depiction"}</div>
          )}
          <ErrorBanner error={pool.error} title="Cannot draw the pool" />
          <div className="row pv-inline-row">
            <button type="button" className="btn btn-primary" data-testid="correct-apply" disabled={busy || !!blocked} title={blocked ?? "apply R and open the product (Enter)"} onClick={() => void apply()}>
              {busy ? "applying…" : "Apply"}
            </button>
            {extra}
          </div>
          {busy && pending && (
            <div className="muted" data-testid="correct-apply-pending">
              waiting for the parent state's analysis: <PendingNote label={pending.label} worker={pending.worker} />
            </div>
          )}
          <ErrorBanner error={error} title="Cannot apply the step" testId="correct-apply-error" />
          <ServerIssues error={error} label={(i) => pool.structure?.label(i) ?? String(i)} />
        </div>
      </Panel>
      <div className="stack pv-side">
        <Panel title="Entries" note={`${state.matrix.bonds.length} bond, ${state.matrix.lone.length} lone`}>
          {s && <EntryTable state={state} dispatch={dispatch} structure={s} />}
        </Panel>
        <Panel title="Check">{s && <CheckPanel check={check} local={local} structure={s} stateId={stateId} empty={isEmptyMatrix(state.matrix)} onHover={setHover} />}</Panel>
      </div>
    </div>
  );
}
