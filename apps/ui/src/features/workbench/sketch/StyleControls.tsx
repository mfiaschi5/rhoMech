// The drawing-style controls of the sketcher's action row: a select of the styles (Screen,
// ACS Document 1996, Total synthesis, Custom; the tooltip says what each is) and a button that
// opens the style editor, a popover with every field of the style (style.ts), a "start from"
// preset, and "Actual size" (zoom so a standard bond is drawn at the style's bond length on a
// 96 dpi screen). Editing any field switches to the custom style, started from the style shown.
// Render-only: nothing here touches the drawing, its coordinates, stereo or history.
//
// DOM: <select data-testid="sketch-style">, sketch-style-edit [aria-expanded] opening <div
// data-testid="sketch-style-editor"> with style-base (start from a preset), style-field-<key>
// for every field (inputs, selects, checkboxes) and sketch-actual-size.
import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  FONT_FAMILIES,
  NUMERIC_FIELDS,
  PRESETS,
  PRESET_IDS,
  STYLE_BOUNDS,
  STYLE_IDS,
  STYLE_TIPS,
  STYLE_TITLES,
  sanitizeStyle,
  type DrawingStyle,
  type PresetId,
  type StyleId,
} from "@ksmech/viz/sketch/style";

export interface StyleControlProps {
  styleId: StyleId;
  /** The style drawn now (the preset, Screen with the preferences, or the custom style). */
  style: DrawingStyle;
  onStyle(id: StyleId, custom?: DrawingStyle): void;
  onActualSize(): void;
}

const FONT_TITLES: Record<(typeof FONT_FAMILIES)[number], string> = { ui: "page font", arial: "Arial", helvetica: "Helvetica", times: "Times" };

type Choice<K extends keyof DrawingStyle> = { key: K; label: string; options: { value: DrawingStyle[K]; title: string }[] };

const CHOICES: Choice<"fontFamily" | "fontWeight" | "wedgeShape" | "chargePlacement" | "terminalDouble">[] = [
  { key: "fontFamily", label: "label font", options: FONT_FAMILIES.map((f) => ({ value: f, title: FONT_TITLES[f] })) },
  { key: "fontWeight", label: "font weight", options: [{ value: 400, title: "normal" }, { value: 700, title: "bold" }] },
  { key: "wedgeShape", label: "wedge", options: [{ value: "wedge", title: "classic wedge" }, { value: "tapered", title: "bold, tapered start" }] },
  { key: "chargePlacement", label: "charge", options: [{ value: "superscript", title: "superscript" }, { value: "circled", title: "circled" }] },
  { key: "terminalDouble", label: "terminal C=X", options: [{ value: "centred", title: "centred" }, { value: "offset", title: "offset" }] },
];

const FLAGS: { key: "colour" | "labelCarbons" | "labelMethyls" | "scaleWithZoom"; label: string; title: string }[] = [
  { key: "colour", label: "element colours", title: "colour the heteroatom labels (off: black and white)" },
  { key: "labelCarbons", label: "label carbons", title: "draw every carbon as C with its H count" },
  { key: "labelMethyls", label: "label terminal CH3", title: "draw terminal methyls as CH3 / H3C" },
  { key: "scaleWithZoom", label: "sizes follow zoom", title: "on: proportions to the bond fixed at any zoom (publication); off: fixed screen pixels" },
];

function Row({ label, title, children }: { label: string; title?: string; children: ReactNode }) {
  return (
    <label className="sk-style-row" title={title}>
      <span>{label}</span>
      {children}
    </label>
  );
}

/** The numeric fields of the editor, each with its unit. */
function NumberFields({ style, set }: { style: DrawingStyle; set(patch: Partial<DrawingStyle>): void }) {
  return (
    <>
      {NUMERIC_FIELDS.map((k) => {
        const b = STYLE_BOUNDS[k];
        return (
          <Row key={k} label={b.label} title={`${b.label} (${b.unit})`}>
            <input
              type="number"
              className="wb-num-input"
              data-testid={`style-field-${k}`}
              min={b.min}
              max={b.max}
              step={b.step}
              value={style[k]}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (e.target.value !== "" && Number.isFinite(v)) set({ [k]: v });
              }}
            />
            <small className="muted">{b.unit}</small>
          </Row>
        );
      })}
    </>
  );
}

function StyleEditor({ styleId, style, onStyle, onActualSize, alignLeft }: StyleControlProps & { alignLeft: boolean }) {
  const set = (patch: Partial<DrawingStyle>) => onStyle("custom", sanitizeStyle({ ...style, ...patch }, style));
  return (
    <div className={`sk-popover sk-style-editor${alignLeft ? " sk-menu-left" : ""}`} data-testid="sketch-style-editor" role="dialog" aria-label="Drawing style">
      <div className="sk-style-head">
        <strong>{STYLE_TITLES[styleId]}</strong>
        <button type="button" className="btn btn-small" data-testid="sketch-actual-size" onClick={onActualSize} title="Zoom so a standard bond is drawn at the style's bond length (true size on a 96 dpi screen)">
          Actual size
        </button>
      </div>
      <Row label="start from" title="copy a preset into the custom style">
        <select data-testid="style-base" value="" onChange={(e) => e.target.value && onStyle("custom", { ...PRESETS[e.target.value as PresetId] })}>
          <option value="">preset…</option>
          {PRESET_IDS.map((p) => (
            <option key={p} value={p}>
              {STYLE_TITLES[p]}
            </option>
          ))}
        </select>
      </Row>
      <div className="sk-style-grid">
        <NumberFields style={style} set={set} />
        {CHOICES.map((c) => (
          <Row key={c.key} label={c.label}>
            <select data-testid={`style-field-${c.key}`} value={String(style[c.key])} onChange={(e) => set({ [c.key]: c.key === "fontWeight" ? Number(e.target.value) : e.target.value } as Partial<DrawingStyle>)}>
              {c.options.map((o) => (
                <option key={String(o.value)} value={String(o.value)}>
                  {o.title}
                </option>
              ))}
            </select>
          </Row>
        ))}
        {FLAGS.map((f) => (
          <Row key={f.key} label={f.label} title={f.title}>
            <input type="checkbox" data-testid={`style-field-${f.key}`} checked={style[f.key]} onChange={(e) => set({ [f.key]: e.target.checked })} />
          </Row>
        ))}
      </div>
      <p className="muted sk-style-note">Fixed bond length = a standard bond; “Reset bond length” (⋯ menu) brings a scaled drawing back to it. Export image (Export panel) prints at true size.</p>
    </div>
  );
}

/** The style select and the editor popover (closes on Escape or a click outside). */
export function StyleControls(p: StyleControlProps) {
  const [open, setOpen] = useState(false);
  const [alignLeft, setAlignLeft] = useState(true);
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
  return (
    <span className="sk-menu-anchor sk-style-anchor" ref={ref}>
      <select className="sk-style" data-testid="sketch-style" value={p.styleId} title={`Drawing style: ${STYLE_TIPS[p.styleId]}`} aria-label="Drawing style" onChange={(e) => p.onStyle(e.target.value as StyleId)}>
        {STYLE_IDS.map((s) => (
          <option key={s} value={s} title={STYLE_TIPS[s]}>
            {STYLE_TITLES[s]}
          </option>
        ))}
      </select>
      <button type="button" className="btn btn-small sk-icon-btn" data-testid="sketch-style-edit" aria-haspopup="dialog" aria-expanded={open} title="Edit the drawing style (every field; edits make a custom style)" aria-label="Edit drawing style" onClick={() => {
          // open towards the side with room: left-aligned unless the controls sit far right
          const r = ref.current?.getBoundingClientRect();
          setAlignLeft(!r || r.left + 340 < window.innerWidth);
          setOpen((v) => !v);
        }}>
        ✎
      </button>
      {open && <StyleEditor {...p} alignLeft={alignLeft} />}
    </span>
  );
}
