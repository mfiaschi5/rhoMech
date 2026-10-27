// DrawModal: the workbench sketcher in a dialog, to draw a reactant; "Use" hands back RDKit's
// canonical SMILES of the drawing. An initial SMILES is imported into the sketch.
//
// Props:
//   initial   string                   the row's SMILES (imported when it parses)
//   onUse     (smiles) => void         the drawing's SMILES
//   onClose   () => void
//   prefix    string                   test id prefix (default "predict")
//
// DOM: <prefix>-draw-modal (Modal) with the Sketcher (data-testid=sketch ...),
// <prefix>-draw-smiles (the SMILES that would be used), <prefix>-draw-use.
import { useMemo, useState } from "react";
import { depict2d } from "../../api/workbench";
import { useRDKit } from "@ksmech/viz/chem/rdkit";
import { importSketch, molblockSmiles, Sketcher, toMolblock, type Depictor, type SketchModel } from "../workbench/sketch";
import { Modal } from "../../components/Modal";

export interface DrawModalProps {
  initial: string;
  onUse: (smiles: string) => void;
  onClose: () => void;
  prefix?: string;
}

const depictor: Depictor = (req, signal) => depict2d(req, { signal });

function initialModel(rdkit: ReturnType<typeof useRDKit>["rdkit"], smiles: string): SketchModel | undefined {
  if (!rdkit || !smiles.trim()) return undefined;
  try {
    return importSketch(rdkit, smiles);
  } catch {
    return undefined;
  }
}

export function DrawModal({ initial, onUse, onClose, prefix = "predict" }: DrawModalProps) {
  const { rdkit } = useRDKit();
  const start = useMemo(() => initialModel(rdkit, initial), [rdkit, initial]);
  const [model, setModel] = useState<SketchModel | undefined>(undefined);
  const current = model ?? start;
  const smiles = useMemo(() => {
    if (!rdkit || !current || current.atoms.length === 0) return "";
    try {
      return molblockSmiles(rdkit, toMolblock(current));
    } catch {
      return "";
    }
  }, [rdkit, current]);
  const footer = (
    <>
      <span className="muted pv-grow">
        SMILES: <code className="code" data-testid={`${prefix}-draw-smiles`}>{smiles || "–"}</code>
      </span>
      <button type="button" className="btn" onClick={onClose}>
        Cancel
      </button>
      <button type="button" className="btn btn-primary" data-testid={`${prefix}-draw-use`} disabled={!smiles} onClick={() => onUse(smiles)}>
        Use
      </button>
    </>
  );
  return (
    <Modal title="Draw a reactant" onClose={onClose} testId={`${prefix}-draw-modal`} wide footer={footer}>
      {rdkit ? <Sketcher value={start} onChange={setModel} height={380} depictor={depictor} /> : <div className="muted">loading RDKit…</div>}
    </Modal>
  );
}
