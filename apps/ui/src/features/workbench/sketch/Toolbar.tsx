// Toolbar of the sketcher: the drawing tools, the element palette with its periodic picker,
// and the actions (undo, redo, clean, fit, clear, import, export).
//
// Props: tool / onTool, element / onElement (also the picker), the action callbacks with their
// enabled flags, and `panel` / onPanel for the import/export popovers.
//
// DOM: <div data-testid="sketch-toolbar">, buttons tool-<name>[data-active], element-<El>
// [data-active], element-more (opens <div data-testid="periodic-picker"> with periodic-<El>),
// sketch-undo, sketch-redo, sketch-clean, sketch-fit, sketch-clear, sketch-import, sketch-export.
import { useEffect, useState, type ReactNode } from "react";
import { PALETTE, PERIODIC_ROWS, elementClass } from "./elements";
import type { ToolName } from "./tools";

export type SketchPanel = "import" | "export" | null;

export interface ToolbarProps {
  tool: ToolName;
  onTool(t: ToolName): void;
  element: string;
  onElement(el: string): void;
  onUndo(): void;
  onRedo(): void;
  canUndo: boolean;
  canRedo: boolean;
  onClean(): void;
  canClean: boolean;
  onFit(): void;
  onClear(): void;
  panel: SketchPanel;
  onPanel(p: SketchPanel): void;
}

const ICON: Partial<Record<ToolName, ReactNode>> = {
  select: <rect x="3" y="3" width="12" height="12" strokeDasharray="2 2" />,
  bond: <line x1="3" y1="14" x2="15" y2="4" />,
  wedge: <polygon points="3,14 14,3 16,6" className="fill" />,
  hash: (
    <g>
      {[0.25, 0.45, 0.65, 0.85].map((t) => (
        <line key={t} x1={3 + 12 * t - 2 * t} y1={14 - 10 * t - 2 * t} x2={3 + 12 * t + 2 * t} y2={14 - 10 * t + 2 * t} />
      ))}
    </g>
  ),
  chain: <polyline points="2,12 6,6 10,12 14,6 17,10" />,
  benzene: (
    <g>
      <polygon points="9,2 15,5.5 15,12.5 9,16 3,12.5 3,5.5" />
      <circle cx="9" cy="9" r="3.6" />
    </g>
  ),
  cyclohexane: <polygon points="9,2 15,5.5 15,12.5 9,16 3,12.5 3,5.5" />,
  cyclopentane: <polygon points="9,2 15.7,6.9 13.1,14.8 4.9,14.8 2.3,6.9" />,
  eraser: <path d="M4 12 L10 4 L15 8 L9 16 Z M4 12 L9 16" />,
};

const TEXT: Partial<Record<ToolName, string>> = {
  atom: "A",
  "charge-plus": "⊕",
  "charge-minus": "⊖",
  "explicit-h": "H",
};

const TITLE: Record<ToolName, string> = {
  select: "Select / move (drag: rectangle; Shift: add)",
  atom: "Atom: click empty canvas to place, click an atom to change its element",
  bond: "Bond: click to draw or grow, click a bond to cycle its order, drag for a direction",
  wedge: "Wedge bond (again on the same bond: flip)",
  hash: "Hashed bond (again on the same bond: flip)",
  chain: "Chain: drag a zigzag chain",
  benzene: "Benzene ring (on an atom: spiro, on a bond: fused)",
  cyclohexane: "Cyclohexane ring",
  cyclopentane: "Cyclopentane ring",
  "charge-plus": "Charge +1",
  "charge-minus": "Charge −1",
  "explicit-h": "Show the label and H count of an atom",
  eraser: "Eraser",
};

const GROUPS: ToolName[][] = [
  ["select", "eraser"],
  ["bond", "wedge", "hash", "chain"],
  ["benzene", "cyclohexane", "cyclopentane"],
  ["atom", "charge-plus", "charge-minus", "explicit-h"],
];

function ToolButton({ name, active, onTool }: { name: ToolName; active: boolean; onTool(t: ToolName): void }) {
  return (
    <button
      type="button"
      data-testid={`tool-${name}`}
      data-active={active}
      className={`sk-tool${active ? " active" : ""}`}
      title={TITLE[name]}
      aria-pressed={active}
      onClick={() => onTool(name)}
    >
      {ICON[name] ? (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
          {ICON[name]}
        </svg>
      ) : (
        <span>{TEXT[name]}</span>
      )}
    </button>
  );
}

function PeriodicPicker({ current, onPick }: { current: string; onPick(el: string): void }) {
  return (
    <div className="sk-popover sk-periodic" data-testid="periodic-picker" role="dialog" aria-label="Periodic table">
      {PERIODIC_ROWS.map((row, r) =>
        row.map((el, c) =>
          el ? (
            <button
              key={`${r}-${c}`}
              type="button"
              data-testid={`periodic-${el}`}
              className={`sk-pt ${elementClass(el)}${el === current ? " active" : ""}`}
              style={{ gridRow: r + 1, gridColumn: c + 1 }}
              onClick={() => onPick(el)}
            >
              {el}
            </button>
          ) : null,
        ),
      )}
    </div>
  );
}

export function Toolbar(p: ToolbarProps) {
  const [picker, setPicker] = useState(false);
  useEffect(() => {
    if (!picker) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setPicker(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [picker]);
  const pick = (el: string) => {
    setPicker(false);
    p.onElement(el);
  };
  const act = (id: string, label: string, onClick: () => void, enabled = true, title?: string) => (
    <button type="button" className="btn btn-small" data-testid={id} disabled={!enabled} onClick={onClick} title={title}>
      {label}
    </button>
  );
  return (
    <div className="sk-toolbar" data-testid="sketch-toolbar">
      {GROUPS.map((g, i) => (
        <div key={i} className="sk-group">
          {g.map((t) => (
            <ToolButton key={t} name={t} active={p.tool === t} onTool={p.onTool} />
          ))}
        </div>
      ))}
      <div className="sk-group sk-palette">
        {PALETTE.map((el) => (
          <button
            key={el}
            type="button"
            data-testid={`element-${el}`}
            data-active={p.tool === "atom" && p.element === el}
            className={`sk-el ${elementClass(el)}${p.tool === "atom" && p.element === el ? " active" : ""}`}
            onClick={() => pick(el)}
          >
            {el}
          </button>
        ))}
        <span className="sk-picker-anchor">
          <button
            type="button"
            data-testid="element-more"
            className={`sk-el${!PALETTE.includes(p.element as (typeof PALETTE)[number]) ? " active" : ""}`}
            title="Periodic table"
            aria-expanded={picker}
            onClick={() => setPicker((v) => !v)}
          >
            {PALETTE.includes(p.element as (typeof PALETTE)[number]) ? "…" : p.element}
          </button>
          {picker && <PeriodicPicker current={p.element} onPick={pick} />}
        </span>
      </div>
      <div className="sk-group sk-actions">
        {act("sketch-undo", "Undo", p.onUndo, p.canUndo, "Ctrl+Z")}
        {act("sketch-redo", "Redo", p.onRedo, p.canRedo, "Ctrl+Shift+Z / Ctrl+Y")}
        {act("sketch-clean", "Clean", p.onClean, p.canClean, "RDKit 2D layout")}
        {act("sketch-fit", "Fit", p.onFit)}
        {act("sketch-clear", "Clear", p.onClear)}
        {act("sketch-import", "Import", () => p.onPanel(p.panel === "import" ? null : "import"), true, "Paste SMILES or a molfile, or drop a .mol/.sdf file")}
        {act("sketch-export", "Export", () => p.onPanel(p.panel === "export" ? null : "export"))}
      </div>
    </div>
  );
}
