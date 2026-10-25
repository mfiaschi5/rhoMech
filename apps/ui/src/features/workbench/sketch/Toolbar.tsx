// Toolbar of the sketcher: the drawing tools and the element palette with its periodic picker
// on the first row (wrapping groups of icon buttons), the action row below (ActionBar.tsx: undo,
// redo, clean, 3D->2D, crossing breaks, rotate/flip/scale, fit, clear, import, export).
//
// Props: tool / onTool, element / onElement (also the picker), the action callbacks with their
// enabled flags, onTransform with hasSelection (the transforms' scope), and `panel` / onPanel
// for the import/export popovers.
//
// DOM: <div data-testid="sketch-toolbar">, buttons tool-<name>[data-active] (tool-radical and
// tool-diradical give an atom one or two unpaired electrons; again: clear), element-<El>
// [data-active], element-more (opens <div data-testid="periodic-picker"> with periodic-<El>),
// the drawing style (sketch-style, sketch-style-edit: StyleControls.tsx), then <div data-testid="sketch-actions"> (its testids in ActionBar.tsx). There is no "Fix
// layout": the server's RDKit depiction replaced the home-made relaxation (docs/decisions.md,
// 2026-10-25).
import { useEffect, useState, type ReactNode } from "react";
import { PALETTE, PERIODIC_ROWS, elementClass } from "./elements";
import type { ToolName } from "./tools";
import type { CleanStyle } from "./depiction";
import type { TransformAction } from "./transform";
import { ActionBar } from "./ActionBar";
import { StyleControls, type StyleControlProps } from "./StyleControls";

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
  /** Spread touching fragments apart (fragments.ts); disabled when none touch. */
  onTidy(): void;
  canTidy: boolean;
  /** Connected fragments of the drawing (the Tidy button shows from two). */
  fragments: number;
  cleanStyle: CleanStyle;
  onCleanStyle(s: CleanStyle): void;
  /** A Clean or 3D->2D is waiting for the server. */
  busy: boolean;
  onFit(): void;
  onClear(): void;
  /** The projection button: null hides it (no 3D source). */
  projection: { enabled: boolean; reason: string | null; onProject(): void } | null;
  breakCrossings: boolean;
  onToggleBreaks(): void;
  panel: SketchPanel;
  onPanel(p: SketchPanel): void;
  /** Rotate, flip or scale the selection (the whole drawing without one). */
  onTransform(a: TransformAction): void;
  hasSelection: boolean;
  /** The R/S toggle: CIP labels drawn on the canvas and listed in the info bar. */
  showCip: boolean;
  onToggleCip(): void;
  /** The drawing-style select and editor (StyleControls.tsx). */
  styleControls: StyleControlProps;
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
  radical: (
    <g>
      <text x="2" y="15" className="fill sk-tool-glyph">C</text>
      <circle cx="14" cy="5" r="2" className="fill" />
    </g>
  ),
  diradical: (
    <g>
      <text x="2" y="15" className="fill sk-tool-glyph">C</text>
      <circle cx="11.5" cy="4.5" r="1.8" className="fill" />
      <circle cx="16" cy="4.5" r="1.8" className="fill" />
    </g>
  ),
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
  radical: "Radical: click an atom to give it one unpaired electron (again: clear). Key \".\" cycles none / one / two. Radicals can be drawn but not computed yet",
  diradical: "Diradical / carbene: click an atom to give it two unpaired electrons (again: clear). Radicals can be drawn but not computed yet",
  eraser: "Eraser",
};

const GROUPS: ToolName[][] = [
  ["select", "eraser"],
  ["bond", "wedge", "hash", "chain"],
  ["benzene", "cyclohexane", "cyclopentane"],
  ["atom", "charge-plus", "charge-minus", "explicit-h"],
  ["radical", "diradical"],
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

const inPalette = (el: string) => PALETTE.includes(el as (typeof PALETTE)[number]);

function Palette({ tool, element, onPick }: { tool: ToolName; element: string; onPick(el: string): void }) {
  const [picker, setPicker] = useState(false);
  useEffect(() => {
    if (!picker) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setPicker(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [picker]);
  const pick = (el: string) => {
    setPicker(false);
    onPick(el);
  };
  return (
    <div className="sk-group sk-palette">
      {PALETTE.map((el) => (
        <button
          key={el}
          type="button"
          data-testid={`element-${el}`}
          data-active={tool === "atom" && element === el}
          className={`sk-el ${elementClass(el)}${tool === "atom" && element === el ? " active" : ""}`}
          title={`Element ${el}`}
          onClick={() => pick(el)}
        >
          {el}
        </button>
      ))}
      <span className="sk-picker-anchor">
        <button type="button" data-testid="element-more" className={`sk-el${!inPalette(element) ? " active" : ""}`} title="Periodic table" aria-expanded={picker} onClick={() => setPicker((v) => !v)}>
          {inPalette(element) ? "…" : element}
        </button>
        {picker && <PeriodicPicker current={element} onPick={pick} />}
      </span>
    </div>
  );
}

export function Toolbar(p: ToolbarProps) {
  return (
    <div className="sk-toolbar" data-testid="sketch-toolbar">
      <div className="sk-tool-row">
        {GROUPS.map((g, i) => (
          <div key={i} className="sk-group">
            {g.map((t) => (
              <ToolButton key={t} name={t} active={p.tool === t} onTool={p.onTool} />
            ))}
          </div>
        ))}
        <Palette tool={p.tool} element={p.element} onPick={p.onElement} />
        <div className="sk-group sk-actions sk-style-group">
          <StyleControls {...p.styleControls} />
        </div>
      </div>
      <ActionBar {...p} />
    </div>
  );
}
