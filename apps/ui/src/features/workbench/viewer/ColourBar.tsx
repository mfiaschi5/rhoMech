// ColourBar: the legend of a colour scale (atoms coloured by a property, the ESP on the density).
//
// Props: label, scale ("rwb" | "seq"), min, max, units?, note?
// DOM: <div data-testid="colour-bar" data-role data-min data-max data-scale>.
import { exact, fmtSig } from "../../../components/format";
import { cssGradient, type Scale } from "./colour";

export interface ColourBarProps {
  role: string;
  label: string;
  scale: Scale;
  min: number;
  max: number;
  units?: string;
  note?: string;
}

export function ColourBar({ role, label, scale, min, max, units, note }: ColourBarProps) {
  const mid = (min + max) / 2;
  return (
    <div className="wb-colour-bar" data-testid="colour-bar" data-role={role} data-min={exact(min)} data-max={exact(max)} data-scale={scale}>
      <div className="wb-colour-bar-label">
        {label}
        {units ? <span className="muted"> ({units})</span> : null}
      </div>
      <div className="wb-colour-bar-ramp" style={{ background: cssGradient(scale) }} aria-hidden="true" />
      <div className="wb-colour-bar-ticks num">
        <span>{fmtSig(min, 3)}</span>
        <span>{fmtSig(mid, 3)}</span>
        <span>{fmtSig(max, 3)}</span>
      </div>
      {note && <div className="muted wb-small">{note}</div>}
    </div>
  );
}
