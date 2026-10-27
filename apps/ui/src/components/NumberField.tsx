// NumberField: a labelled <input type="number"> that keeps the text the user types and reports
// a parsed, range-checked number (or null when empty and allowEmpty), so a cleared field does
// not snap back to a default while typing.
//
// Props:
//   label        string
//   testId       string                 data-testid of the <input>
//   value        number | null
//   onChange     (value: number | null) => void   called only with valid values (or null)
//   min, max     number                 inclusive bounds (min is exclusive with minExclusive)
//   step         number                 default 1
//   integer      boolean                default true
//   allowEmpty   boolean                empty input reports null (default false)
//   placeholder  string
//   hint         string                 small text after the input
import { useEffect, useState } from "react";

export interface NumberFieldProps {
  label: string;
  testId: string;
  value: number | null;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  minExclusive?: boolean;
  step?: number;
  integer?: boolean;
  allowEmpty?: boolean;
  placeholder?: string;
  hint?: string;
}

function parse(text: string, p: NumberFieldProps): number | null | undefined {
  const t = text.trim();
  if (!t) return p.allowEmpty ? null : undefined;
  const n = p.integer === false ? Number(t) : Number.parseInt(t, 10);
  if (!Number.isFinite(n)) return undefined;
  if (p.integer !== false && !/^[-+]?\d+$/.test(t)) return undefined;
  if (p.min !== undefined && (p.minExclusive ? n <= p.min : n < p.min)) return undefined;
  if (p.max !== undefined && n > p.max) return undefined;
  return n;
}

export function NumberField(props: NumberFieldProps) {
  const { value } = props;
  const [text, setText] = useState(value === null ? "" : String(value));
  useEffect(() => {
    setText((t) => (parse(t, props) === value ? t : value === null ? "" : String(value)));
    // only an outside change of `value` resyncs the text
  }, [value]);
  const parsed = parse(text, props);
  const invalid = parsed === undefined;
  return (
    <label className="field">
      <span className="field-label">{props.label}</span>
      <input
        type="number"
        data-testid={props.testId}
        value={text}
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        placeholder={props.placeholder}
        aria-invalid={invalid || undefined}
        className={invalid ? "invalid" : undefined}
        onChange={(e) => {
          setText(e.target.value);
          const v = parse(e.target.value, props);
          if (v !== undefined) props.onChange(v);
        }}
      />
      {props.hint && <span className="field-hint muted">{props.hint}</span>}
    </label>
  );
}
