// The atom label editor (ChemDraw: Enter on an atom): a small text field over the atom; Enter
// sets the typed element and charge (tools.parseAtomLabel), Escape or a click elsewhere cancels.
// DOM: <input data-testid="sketch-label-input" data-invalid> over the atom.
import { useState } from "react";
import type { SketchAtom } from "@ksmech/viz/sketch/model";
import { toScreen, type View } from "@ksmech/viz/sketch/geometry";
import { chargeLabel } from "@ksmech/viz/sketch/elements";
import { parseAtomLabel } from "./tools";

export function LabelEditor({ atom, view, onDone }: { atom: SketchAtom; view: View; onDone(label: { el: string; charge: number } | null): void }) {
  const [text, setText] = useState(`${atom.el}${atom.charge ? chargeLabel(atom.charge) : ""}`);
  const [invalid, setInvalid] = useState(false);
  const [x, y] = toScreen(view, [atom.x, atom.y]);
  return (
    <input
      className={`sk-label-input${invalid ? " invalid" : ""}`}
      data-testid="sketch-label-input"
      data-invalid={invalid ? "true" : "false"}
      style={{ left: x - 28, top: y - 13 }}
      aria-label="Atom label"
      title="Element symbol with an optional charge (N+, O-, Fe3+); Enter to set, Esc to cancel"
      autoFocus
      spellCheck={false}
      value={text}
      onFocus={(e) => e.currentTarget.select()}
      onChange={(e) => {
        setText(e.target.value);
        setInvalid(false);
      }}
      onBlur={() => onDone(null)}
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === "Escape") onDone(null);
        if (e.key !== "Enter") return;
        e.preventDefault();
        const label = parseAtomLabel(text);
        if (label) onDone(label);
        else setInvalid(true);
      }}
    />
  );
}
