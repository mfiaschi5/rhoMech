// ProductView: the product of a step drawn with RDKit.js (the product SMILES, one drawing for
// all species) with the SMILES and the sigma keys under it.
//
// DOM: <div data-testid="<prefix>-product" data-smiles> with <prefix>-sigma-<k>.
import { SmilesDepiction } from "../components/SmilesDepiction";

export function ProductView({ smiles, sigmaKeys, prefix }: { smiles: readonly string[]; sigmaKeys?: readonly string[]; prefix: string }) {
  const joined = smiles.join(".");
  return (
    <div className="pv-product stack" data-testid={`${prefix}-product`} data-smiles={joined}>
      {smiles.length ? <SmilesDepiction smiles={joined} width={300} height={220} className="pv-product-svg" /> : <div className="muted pv-product-empty">no product yet</div>}
      <code className="code pv-wrap">{joined || "–"}</code>
      {sigmaKeys?.map((k, i) => (
        <code key={i} className="code muted pv-wrap" data-testid={`${prefix}-sigma-${i}`} title="sigma key: the sigma skeleton with H counts frozen and charges zeroed">
          σ {k}
        </code>
      ))}
    </div>
  );
}
