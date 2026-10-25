// EditorToolbar: the R editor's tools (bond, lone pair, arrow, eraser), undo / redo / clear,
// and the last action's notice. Wraps in groups, like the workbench toolbars.
//
// Props: state (EditorState), dispatch.
// DOM: correct-toolbar, correct-tool-{bond|lone|arrow|eraser}[aria-pressed], correct-undo,
// correct-redo, correct-clear, correct-notice.
import type { Dispatch } from "react";
import { canRedo, canUndo, TOOLS, type EditorAction, type EditorState, type Tool } from "./editor";
import { isEmptyMatrix } from "./rmatrix";

const TOOL_INFO: Record<Tool, { label: string; key: string; title: string }> = {
  bond: { label: "Bond", key: "b", title: "click a bond: 0 → −1 → +1 → 0; click two atoms: form a bond" },
  lone: { label: "Lone pair", key: "l", title: "click an atom: 0 → −2 → +2 → 0" },
  arrow: { label: "Arrow", key: "a", title: "drag from an atom (its lone pair) or a bond to an atom or a bond" },
  eraser: { label: "Eraser", key: "e", title: "click an atom, a bond or an arrow to remove its entries" },
};

export function EditorToolbar({ state, dispatch }: { state: EditorState; dispatch: Dispatch<EditorAction> }) {
  return (
    <div className="row pv-toolbar" data-testid="correct-toolbar" data-tool={state.tool}>
      <span className="pv-tgroup" role="group" aria-label="tools">
        {TOOLS.map((t) => (
          <button key={t} type="button" className={`btn btn-small${state.tool === t ? " btn-primary" : ""}`} data-testid={`correct-tool-${t}`} aria-pressed={state.tool === t} title={`${TOOL_INFO[t].title} (${TOOL_INFO[t].key})`} onClick={() => dispatch({ type: "tool", tool: t })}>
            {TOOL_INFO[t].label} <kbd>{TOOL_INFO[t].key}</kbd>
          </button>
        ))}
      </span>
      <span className="pv-tgroup" role="group" aria-label="history">
        <button type="button" className="btn btn-small" data-testid="correct-undo" disabled={!canUndo(state)} title="undo (Ctrl+Z)" onClick={() => dispatch({ type: "undo" })}>
          Undo
        </button>
        <button type="button" className="btn btn-small" data-testid="correct-redo" disabled={!canRedo(state)} title="redo (Ctrl+Y)" onClick={() => dispatch({ type: "redo" })}>
          Redo
        </button>
        <button type="button" className="btn btn-small" data-testid="correct-clear" disabled={isEmptyMatrix(state.matrix)} onClick={() => dispatch({ type: "clear" })}>
          Clear
        </button>
      </span>
      {state.notice && (
        <span className="muted pv-notice" data-testid="correct-notice">
          {state.notice}
        </span>
      )}
    </div>
  );
}
