// LayoutBar: the workbench's "Layout" menu and the chips of hidden panels. The menu offers the
// presets (default, sketch-, 3D- and orbitals-focused), reset layout, a show/hide checkbox per
// panel and the preferences (sketch font size, bond line width, atom colouring, label style,
// default 3D representation, default orbital isovalue, theme). Closes on Escape or a click
// outside. Layout and preferences persist per browser (localStorage, storage.ts).
//
// DOM: <div data-testid="wb-layout-bar" data-preset>, wb-layout-button [aria-expanded], <div
// data-testid="wb-layout-menu"> with layout-preset-<name> [aria-pressed], layout-reset,
// layout-show-<panel> (checkbox), pref-font, pref-bond-width, pref-colour-atoms, pref-label-style,
// pref-rep, pref-iso, pref-theme, prefs-reset; outside the menu, layout-restore-<panel> per
// hidden panel.
import { useEffect, useRef, useState, type Dispatch, type ReactNode } from "react";
import { REPS, type Rep } from "../params";
import { PANELS, PANEL_TITLES, PRESETS, PRESET_TITLES, type LayoutAction, type LayoutState } from "./layoutState";
import { LABEL_STYLE_CHOICES, LABEL_STYLE_TITLES, PREF_BOUNDS, THEMES, type Prefs, type ThemeChoice } from "./prefs";
import type { LabelStyle } from "../sketch/drawing";

export interface LayoutBarProps {
  layout: LayoutState;
  dispatch: Dispatch<LayoutAction>;
  prefs: Prefs;
  setPrefs(patch: Partial<Prefs>): void;
  resetPrefs(): void;
}

const REP_TITLES: Record<Rep, string> = { "ball-and-stick": "ball and stick", sticks: "sticks", spacefill: "spacefill", wireframe: "wireframe", lines: "lines" };

function Field({ label, children, title }: { label: string; children: ReactNode; title?: string }) {
  return (
    <label className="wb-pref" title={title}>
      <span>{label}</span>
      {children}
    </label>
  );
}

function NumberPref({ id, value, bounds, onChange }: { id: string; value: number; bounds: { min: number; max: number; step: number }; onChange(v: number): void }) {
  return (
    <input
      type="number"
      className="wb-num-input"
      data-testid={id}
      min={bounds.min}
      max={bounds.max}
      step={bounds.step}
      value={value}
      onChange={(e) => {
        const v = Number(e.target.value);
        if (e.target.value !== "" && Number.isFinite(v)) onChange(v);
      }}
    />
  );
}

function PrefsSection({ prefs, setPrefs, resetPrefs }: Pick<LayoutBarProps, "prefs" | "setPrefs" | "resetPrefs">) {
  return (
    <fieldset className="wb-menu-section">
      <legend>Preferences</legend>
      <Field label="sketch font (px)">
        <NumberPref id="pref-font" value={prefs.fontPx} bounds={PREF_BOUNDS.fontPx} onChange={(fontPx) => setPrefs({ fontPx })} />
      </Field>
      <Field label="bond line width">
        <NumberPref id="pref-bond-width" value={prefs.bondWidth} bounds={PREF_BOUNDS.bondWidth} onChange={(bondWidth) => setPrefs({ bondWidth })} />
      </Field>
      <Field label="atom colours">
        <input type="checkbox" data-testid="pref-colour-atoms" checked={prefs.colourAtoms} onChange={(e) => setPrefs({ colourAtoms: e.target.checked })} />
      </Field>
      <Field label="labels">
        <select data-testid="pref-label-style" value={prefs.labelStyle} onChange={(e) => setPrefs({ labelStyle: e.target.value as LabelStyle })}>
          {LABEL_STYLE_CHOICES.map((s) => (
            <option key={s} value={s}>
              {LABEL_STYLE_TITLES[s]}
            </option>
          ))}
        </select>
      </Field>
      <Field label="3D default" title="the representation when the link names none">
        <select data-testid="pref-rep" value={prefs.rep} onChange={(e) => setPrefs({ rep: e.target.value as Rep })}>
          {REPS.map((r) => (
            <option key={r} value={r}>
              {REP_TITLES[r]}
            </option>
          ))}
        </select>
      </Field>
      <Field label="orbital isovalue" title="the isovalue of new orbital ψ surfaces (and of the selected-orbital layer)">
        <NumberPref id="pref-iso" value={prefs.orbitalIso} bounds={PREF_BOUNDS.orbitalIso} onChange={(orbitalIso) => setPrefs({ orbitalIso })} />
      </Field>
      <Field label="theme">
        <select data-testid="pref-theme" value={prefs.theme} onChange={(e) => setPrefs({ theme: e.target.value as ThemeChoice })}>
          {THEMES.map((t) => (
            <option key={t} value={t}>
              {t === "auto" ? "auto (system)" : t}
            </option>
          ))}
        </select>
      </Field>
      <button type="button" className="btn btn-small" data-testid="prefs-reset" onClick={resetPrefs}>
        Reset preferences
      </button>
    </fieldset>
  );
}

function LayoutSection({ layout, dispatch }: Pick<LayoutBarProps, "layout" | "dispatch">) {
  return (
    <>
      <fieldset className="wb-menu-section">
        <legend>Presets</legend>
        <div className="wb-menu-buttons">
          {PRESETS.map((p) => (
            <button key={p} type="button" className={`btn btn-small${layout.preset === p ? " active" : ""}`} data-testid={`layout-preset-${p}`} aria-pressed={layout.preset === p} onClick={() => dispatch({ type: "preset", name: p })}>
              {PRESET_TITLES[p]}
            </button>
          ))}
          <button type="button" className="btn btn-small" data-testid="layout-reset" title="Back to the default layout" onClick={() => dispatch({ type: "reset" })}>
            Reset layout
          </button>
        </div>
      </fieldset>
      <fieldset className="wb-menu-section">
        <legend>Panels</legend>
        {PANELS.map((p) => (
          <label key={p} className="wb-pref">
            <span>{PANEL_TITLES[p]}</span>
            <input type="checkbox" data-testid={`layout-show-${p}`} checked={!layout.hidden.includes(p)} onChange={(e) => dispatch({ type: e.target.checked ? "show" : "hide", panel: p })} />
          </label>
        ))}
      </fieldset>
    </>
  );
}

/** Open state that closes on Escape or a pointer press outside `ref`. */
function useDismiss(): [boolean, (v: boolean | ((o: boolean) => boolean)) => void, React.RefObject<HTMLDivElement | null>] {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
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
  return [open, setOpen, ref];
}

export function LayoutBar(p: LayoutBarProps) {
  const [open, setOpen, ref] = useDismiss();
  const hidden = PANELS.filter((x) => p.layout.hidden.includes(x));
  return (
    <div className="wb-layout-bar" data-testid="wb-layout-bar" data-preset={p.layout.preset}>
      <div className="wb-menu-anchor" ref={ref}>
        <button type="button" className="btn btn-small" data-testid="wb-layout-button" aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen((v) => !v)} title="Layout presets, panels and preferences">
          Layout ▾
        </button>
        {open && (
          <div className="wb-layout-menu" data-testid="wb-layout-menu" role="dialog" aria-label="Layout and preferences">
            <LayoutSection layout={p.layout} dispatch={p.dispatch} />
            <PrefsSection prefs={p.prefs} setPrefs={p.setPrefs} resetPrefs={p.resetPrefs} />
          </div>
        )}
      </div>
      <span className="muted wb-small">{p.layout.preset === "custom" ? "custom layout" : `${PRESET_TITLES[p.layout.preset]} layout`}</span>
      {hidden.map((x) => (
        <button key={x} type="button" className="btn btn-small wb-restore" data-testid={`layout-restore-${x}`} title={`Show ${PANEL_TITLES[x]} again`} onClick={() => p.dispatch({ type: "show", panel: x })}>
          + {PANEL_TITLES[x]}
        </button>
      ))}
    </div>
  );
}
