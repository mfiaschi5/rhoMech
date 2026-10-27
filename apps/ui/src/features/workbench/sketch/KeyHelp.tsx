// The sketcher's shortcut cheat sheet (the "?" key or the toolbar's "?" button): shortcuts.ts
// grouped by scope, each binding marked by origin (ChemDraw default, standard, ChemRho's own,
// with how it differs from ChemDraw).
//
// DOM: <div data-testid="sketch-keys-panel"> with one <tr data-testid="sketch-key-row"
// data-origin data-scope> per binding.
import { Fragment } from "react";
import { SCOPE_LABEL, shortcutsOf, type ShortcutOrigin, type ShortcutScope } from "./shortcuts";

const SCOPES: ShortcutScope[] = ["atom", "bond", "selection", "drawing", "drag"];
const ORIGIN: Record<ShortcutOrigin, string> = { chemdraw: "ChemDraw", standard: "standard", chemrho: "ChemRho" };

export function KeyHelp({ onClose }: { onClose(): void }) {
  return (
    <div className="sk-popover sk-panel sk-keys" data-testid="sketch-keys-panel" role="dialog" aria-label="Keyboard shortcuts">
      <div className="sk-panel-head">
        <strong>Keyboard shortcuts</strong>
        <span className="muted">keys act on what is under the pointer; Cmd for Ctrl on a Mac</span>
        <button type="button" className="btn btn-small" onClick={onClose}>
          Close
        </button>
      </div>
      <table className="sk-keys-table">
        <tbody>
          {SCOPES.map((scope) => (
            <Fragment key={scope}>
              <tr>
                <th colSpan={3}>{SCOPE_LABEL[scope]}</th>
              </tr>
              {shortcutsOf(scope).map((s, k) => (
                <tr key={k} data-testid="sketch-key-row" data-origin={s.origin} data-scope={scope}>
                  <td>
                    <kbd>{s.keys}</kbd>
                  </td>
                  <td>
                    {s.action}
                    {s.note && <div className="muted sk-key-note">ChemDraw: {s.note}</div>}
                  </td>
                  <td className="muted sk-key-origin">{ORIGIN[s.origin]}</td>
                </tr>
              ))}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
