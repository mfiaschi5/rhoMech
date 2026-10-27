// SmilesDepiction: a live RDKit.js drawing of a SMILES string (reactant inputs, network nodes),
// without atom positions or arrows; use MolDepiction for pool states.
//
// Props:
//   smiles         string             the SMILES to draw ("" draws nothing)
//   width, height  number             pixel size (default 220 x 150)
//   testId         string             data-testid of the container (e.g. "input-depiction-0")
//   className      string
//
// DOM: <div data-testid={testId} data-state="empty|loading|ready|invalid"> holding the <svg>
// when the SMILES parses, or a short "cannot parse" note.
import { useMemo } from "react";
import { smilesToSvg, useRDKit } from "../chem/rdkit";

export interface SmilesDepictionProps {
  smiles: string;
  width?: number;
  height?: number;
  testId?: string;
  className?: string;
}

export function SmilesDepiction({ smiles, width = 220, height = 150, testId, className }: SmilesDepictionProps) {
  const { rdkit, error } = useRDKit();
  const svg = useMemo(
    () => (rdkit && smiles.trim() ? smilesToSvg(rdkit, smiles, { width, height }) : null),
    [rdkit, smiles, width, height],
  );
  const state = !smiles.trim() ? "empty" : !rdkit ? (error ? "invalid" : "loading") : svg ? "ready" : "invalid";
  return (
    <div
      data-testid={testId}
      className={`smiles-depiction${className ? ` ${className}` : ""}`}
      data-state={state}
      style={{ width, height }}
    >
      {svg && <div className="mol-svg" dangerouslySetInnerHTML={{ __html: svg }} />}
      {state === "invalid" && (
        <div className="mol-placeholder mol-error">{error ? "RDKit.js unavailable" : "cannot parse SMILES"}</div>
      )}
      {state === "loading" && <div className="mol-placeholder">loading RDKit…</div>}
    </div>
  );
}
