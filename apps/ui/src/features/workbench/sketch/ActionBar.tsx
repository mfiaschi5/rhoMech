// The sketcher's action row, in wrapping groups so it never spills out of its panel: history
// (undo, redo), layout (Clean with its style, 3D -> 2D, crossing breaks), transform (rotate
// 15 degrees either way, flip, and a "⋯" menu with 90-degree turns, scale +-10 % and reset bond
// length), and view/file (fit, clear, import, export). The common tools are icon buttons whose
// name and shortcut are in title / aria-label. Transforms apply to the selection, or to the whole
// drawing when nothing is selected (transform.ts).
//
// DOM: sketch-undo, sketch-redo, sketch-clean, <select data-testid="sketch-clean-style">
// (textbook | perspective), sketch-project (3D -> 2D; only with a projection source, disabled
// with the reason as data-reason and as the tooltip of sketch-project-wrap),
// sketch-break-toggle [data-active aria-pressed], sketch-cip-toggle [data-active aria-pressed]
// (R/S: CIP labels on the canvas and in the info bar), sketch-rotate-cw, sketch-rotate-ccw,
// sketch-flip-h, sketch-flip-v, sketch-more [aria-expanded] opening <div
// data-testid="sketch-more-menu"> with sketch-rotate-cw90, sketch-rotate-ccw90, sketch-scale-up,
// sketch-scale-down, sketch-reset-bond; sketch-fit, sketch-clear, sketch-import, sketch-export.
import { useEffect, useRef, useState, type ReactNode } from "react";
import { CLEAN_STYLES, type CleanStyle } from "./depiction";
import { Icon, type IconName } from "./icons";
import type { TransformAction } from "./transform";
import type { ToolbarProps } from "./Toolbar";

type Act = (id: string, label: string, onClick: () => void, opts?: { enabled?: boolean; title?: string; icon?: IconName }) => ReactNode;

const act: Act = (id, label, onClick, { enabled = true, title, icon } = {}) => (
  <button
    type="button"
    className={`btn btn-small${icon ? " sk-icon-btn" : ""}`}
    data-testid={id}
    disabled={!enabled}
    onClick={onClick}
    title={title ?? label}
    aria-label={icon ? label : undefined}
  >
    {icon ? <Icon name={icon} /> : label}
  </button>
);

/** The 3D -> 2D button; a disabled one sits in a span that carries the reason as its tooltip
 *  (a disabled button gets no pointer events, so its own title never shows). */
function ProjectButton({ enabled, reason, onProject }: NonNullable<ToolbarProps["projection"]>) {
  const title = enabled
    ? "Redraw the sketch as the 3D view shows the computed structure now (RDKit depiction matched to the view, or a perspective template; wedges keep the structure's stereo; crossings broken by depth) (Shift+D)"
    : `3D→2D unavailable: ${reason ?? "no 3D view"}`;
  return (
    <span className="sk-tip" title={title} data-testid="sketch-project-wrap">
      <button type="button" className="btn btn-small" data-testid="sketch-project" data-reason={enabled ? "" : (reason ?? "")} disabled={!enabled} onClick={onProject}>
        3D→2D
      </button>
    </span>
  );
}

const MENU: { action: TransformAction; id: string; label: string; keys?: string }[] = [
  { action: "rotate-cw90", id: "sketch-rotate-cw90", label: "Rotate 90° clockwise" },
  { action: "rotate-ccw90", id: "sketch-rotate-ccw90", label: "Rotate 90° counter-clockwise" },
  { action: "scale-up", id: "sketch-scale-up", label: "Scale up 10 %", keys: "Ctrl+=" },
  { action: "scale-down", id: "sketch-scale-down", label: "Scale down 10 %", keys: "Ctrl+−" },
  { action: "reset-bond", id: "sketch-reset-bond", label: "Reset bond length" },
];

/** The "⋯" menu of the rarely used transforms; closes on a pick, Escape or a click outside. */
function MoreMenu({ onTransform, scope }: { onTransform(a: TransformAction): void; scope: string }) {
  const [open, setOpen] = useState(false);
  const [alignLeft, setAlignLeft] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);
  const toggle = () => {
    // open towards the side with room: right-aligned unless the button is near the left edge
    setAlignLeft((ref.current?.getBoundingClientRect().left ?? 1e9) < 260);
    setOpen((v) => !v);
  };
  return (
    <span className="sk-menu-anchor" ref={ref}>
      <button type="button" className="btn btn-small sk-icon-btn" data-testid="sketch-more" aria-haspopup="menu" aria-expanded={open} title="More transforms" aria-label="More transforms" onClick={toggle}>
        ⋯
      </button>
      {open && (
        <div className={`sk-popover sk-menu${alignLeft ? " sk-menu-left" : ""}`} data-testid="sketch-more-menu" role="menu">
          <div className="sk-menu-scope muted">{scope}</div>
          {MENU.map((m) => (
            <button key={m.id} type="button" role="menuitem" className="sk-menu-item" data-testid={m.id} onClick={() => {
                setOpen(false);
                onTransform(m.action);
              }}>
              <span>{m.label}</span>
              {m.keys && <kbd>{m.keys}</kbd>}
            </button>
          ))}
        </div>
      )}
    </span>
  );
}

function CleanGroup(p: ToolbarProps) {
  const cleanTitle = "Server 2D layout (RDKit): textbook = ring templates or CoordGen, perspective = chairs, cages and bicycles drawn in perspective";
  return (
    <div className="sk-group sk-actions">
      {act("sketch-clean", p.busy ? "Clean…" : "Clean", p.onClean, { enabled: p.canClean, title: cleanTitle })}
      <select className="sk-style" data-testid="sketch-clean-style" value={p.cleanStyle} title="Clean style" aria-label="Clean style" onChange={(e) => p.onCleanStyle(e.target.value as CleanStyle)}>
        {CLEAN_STYLES.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
      {p.projection && <ProjectButton {...p.projection} />}
      <button
        type="button"
        className={`btn btn-small sk-icon-btn sk-toggle${p.breakCrossings ? " active" : ""}`}
        data-testid="sketch-break-toggle"
        data-active={p.breakCrossings}
        aria-pressed={p.breakCrossings}
        aria-label="Break crossings"
        onClick={p.onToggleBreaks}
        title="Break crossings: draw the bond behind a crossing with a gap (by depth after a projection, else the later bond is in front) (Shift+X)"
      >
        <Icon name="breaks" />
      </button>
      <button
        type="button"
        className={`btn btn-small sk-toggle sk-cip-toggle${p.showCip ? " active" : ""}`}
        data-testid="sketch-cip-toggle"
        data-active={p.showCip}
        aria-pressed={p.showCip}
        onClick={p.onToggleCip}
        title="R/S: show the CIP labels (R, S; r, s pseudo-asymmetric; ? undefined) of the stereocentres and E/Z of the double bonds, from RDKit"
      >
        R/S
      </button>
    </div>
  );
}

export function ActionBar(p: ToolbarProps) {
  const scope = p.hasSelection ? "applies to the selection" : "applies to the whole drawing";
  const tf = (a: TransformAction) => () => p.onTransform(a);
  return (
    <div className="sk-action-row" data-testid="sketch-actions">
      <div className="sk-group sk-actions">
        {act("sketch-undo", "Undo", p.onUndo, { enabled: p.canUndo, title: "Undo (Ctrl+Z)", icon: "undo" })}
        {act("sketch-redo", "Redo", p.onRedo, { enabled: p.canRedo, title: "Redo (Ctrl+Shift+Z / Ctrl+Y)", icon: "redo" })}
      </div>
      <CleanGroup {...p} />
      <div className="sk-group sk-actions">
        {act("sketch-rotate-ccw", "Rotate 15° counter-clockwise", tf("rotate-ccw15"), { title: `Rotate 15° counter-clockwise (Shift+R), ${scope}`, icon: "rotate-ccw" })}
        {act("sketch-rotate-cw", "Rotate 15° clockwise", tf("rotate-cw15"), { title: `Rotate 15° clockwise (R), ${scope}`, icon: "rotate-cw" })}
        {act("sketch-flip-h", "Flip horizontal", tf("flip-h"), { title: `Flip horizontal (Alt+H), ${scope}; wedges swap so the stereoisomer is kept`, icon: "flip-h" })}
        {act("sketch-flip-v", "Flip vertical", tf("flip-v"), { title: `Flip vertical (Alt+V), ${scope}; wedges swap so the stereoisomer is kept`, icon: "flip-v" })}
        <MoreMenu onTransform={p.onTransform} scope={scope} />
      </div>
      <div className="sk-group sk-actions">
        {act("sketch-fit", "Fit", p.onFit, { title: "Fit the drawing in view", icon: "fit" })}
        {act("sketch-clear", "Clear", p.onClear, { title: "Clear the drawing", icon: "clear" })}
        {act("sketch-import", "Import", () => p.onPanel(p.panel === "import" ? null : "import"), { title: "Import: paste SMILES or a molfile, or drop a .mol/.sdf file", icon: "import" })}
        {act("sketch-export", "Export", () => p.onPanel(p.panel === "export" ? null : "export"), { title: "Export: image (SVG, PNG, PDF at true size; copy for Word, LaTeX, chemfig), SMILES, InChI, molfile", icon: "export" })}
      </div>
    </div>
  );
}
