// SmilesInput: one reactant SMILES field with its live RDKit.js depiction, redrawn 250 ms after
// the last keystroke (design §11.8).
//
// Props:
//   index     number                row index i (test ids smiles-input-{i}, input-depiction-{i})
//   value     string                the SMILES
//   onChange  (smiles) => void
//   width, height  number           depiction size (default 200 x 130)
import { SmilesDepiction } from "../../components/SmilesDepiction";
import { useDebounced } from "./hooks";

export interface SmilesInputProps {
  index: number;
  value: string;
  onChange: (smiles: string) => void;
  width?: number;
  height?: number;
}

export const DEPICTION_DEBOUNCE_MS = 250;

export function SmilesInput({ index, value, onChange, width = 200, height = 130 }: SmilesInputProps) {
  const settled = useDebounced(value, DEPICTION_DEBOUNCE_MS);
  return (
    <div className="smiles-input stack">
      <label className="field">
        <span className="field-label">SMILES {index + 1}</span>
        <input
          type="text"
          data-testid={`smiles-input-${index}`}
          value={value}
          spellCheck={false}
          autoComplete="off"
          placeholder={index === 0 ? "e.g. C[O-]" : "e.g. CCl"}
          onChange={(e) => onChange(e.target.value)}
        />
      </label>
      <SmilesDepiction smiles={settled.trim()} width={width} height={height} testId={`input-depiction-${index}`} />
    </div>
  );
}
