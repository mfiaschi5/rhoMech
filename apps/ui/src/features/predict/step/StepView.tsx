// StepView: one elementary step drawn as its reaction matrix: the reactant pool (PoolCanvas)
// with R overlaid and the curved arrows of its reading, then the product next to it (RDKit.js
// drawing of the product SMILES, one per species) with the sigma keys under it.
//
// Props:
//   stateId        string             the reactant pool state
//   matrix         MatrixIO
//   arrows         ArrowReadingOut[]
//   productSmiles  string[]
//   sigmaKeys      string[]
//   chargesAfter   Map<atom, charge>  formal charges after R (when known)
//   highlight      Set<atom>
//   prefix         string             test id prefix ("predict-step", "correct-preview")
//   height         number             px (default 300)
//
// Hidden H that R or an arrow touches are drawn (poolStructure's `show`).
//
// DOM: <div data-testid="<prefix>"> with the PoolCanvas <prefix>-canvas, the arrow list
// <prefix>-arrow-text-<k>, <prefix>-product (data-smiles) and <prefix>-sigma-<k>.
import { SmilesDepiction } from "../../../components/SmilesDepiction";
import { ErrorBanner } from "../../../components/ErrorBanner";
import { useDrawnStructure, usePool } from "../hooks";
import { PoolCanvas } from "../PoolCanvas";
import { arrowText } from "../rmatrix";
import type { ArrowReadingOut, MatrixIO } from "../v9types";

export interface StepViewProps {
  stateId: string;
  matrix: MatrixIO;
  arrows: readonly ArrowReadingOut[];
  productSmiles: readonly string[];
  sigmaKeys?: readonly string[];
  chargesAfter?: ReadonlyMap<number, number> | null;
  highlight?: ReadonlySet<number>;
  prefix: string;
  height?: number;
}

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

export function StepView(props: StepViewProps) {
  const { stateId, matrix, arrows, prefix } = props;
  const pool = usePool(stateId);
  const s = useDrawnStructure(pool, matrix, arrows);
  return (
    <div className="pv-step" data-testid={prefix}>
      <div className="pv-step-grid">
        <div className="pv-step-reactant stack">
          {s ? <PoolCanvas structure={s} testId={`${prefix}-canvas`} height={props.height ?? 300} matrix={matrix} arrows={arrows} chargesAfter={props.chargesAfter} highlight={props.highlight} /> : <div className="pv-canvas-placeholder muted">{pool.loading ? "loading the depiction…" : "no depiction"}</div>}
          <ErrorBanner error={pool.error} title="Cannot draw the pool" />
          <ol className="pv-arrow-list">
            {arrows.map((a, k) => (
              <li key={k} data-testid={`${prefix}-arrow-text-${k}`} data-kind={a.kind}>
                {arrowText(a, (i) => s?.label(i) ?? String(i))} <span className="muted">({a.kind.replace(/_/g, " ")})</span>
              </li>
            ))}
          </ol>
        </div>
        <div className="pv-step-arrow" aria-hidden="true">
          →
        </div>
        <ProductView smiles={props.productSmiles} sigmaKeys={props.sigmaKeys} prefix={prefix} />
      </div>
    </div>
  );
}
