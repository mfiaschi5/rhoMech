// EmbeddingDialog: EmbeddingView over the page (the Workbench's "Model trace" of the computed
// molecule). Escape and the backdrop close it.
//
// Props: smiles (the molecule's token key), onClose. DOM: <div role=dialog
// data-testid="wb-embedding-dialog">, its close button wb-embedding-close, the view
// "wb-embedding".
import { useEffect } from "react";
import { EmbeddingView } from "./EmbeddingView";
import "./embedding.css";

export function EmbeddingDialog({ smiles, onClose }: { smiles: string; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="emb-dialog-backdrop" onPointerDown={(e) => e.target === e.currentTarget && onClose()}>
      <div role="dialog" aria-modal="true" className="emb-dialog panel" data-testid="wb-embedding-dialog">
        <div className="emb-dialog-head">
          <h3>
            Model trace of <code className="code">{smiles}</code>
          </h3>
          <button type="button" className="btn btn-small" data-testid="wb-embedding-close" onClick={onClose} aria-label="close">
            ✕
          </button>
        </div>
        <EmbeddingView target={{ smiles }} testId="wb-embedding" />
      </div>
    </div>
  );
}
