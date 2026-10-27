// StepPanel: the selected edge of the Predict job view: its kind, rank, predictor and score (a
// log-probability when the predictor's scores are probabilities, else the raw score), the anchor,
// the step viewport in one of four variants (a segmented control in the header, remembered in
// localStorage: StepTextbook, StepAnimated, StepOrbital, and Embedding: the parent state traced
// through a trained checkpoint, features/embedding/EmbeddingView; greyed with the reason while
// the embedding capability is off), the facts (R entries, arrows, product,
// sigma keys, PMechDB code, beam weight, alternatives), the orbital reading table and "Correct
// this step" (the Correct page on the parent state with this R preloaded).
//
// Alternatives (other R with the same product) are browsed with ‹ › in the header or picked from
// the list; an alternative's arrows and readings come from POST /step/check.
//
// Props: jobId, edge, net, probabilities.
// DOM: <section data-testid="predict-step-panel" data-edge data-kind data-variant data-alt> with
// predict-step-score [data-unit="logp|score"], predict-step-anchor, predict-step-variant >
// button[data-testid=predict-step-variant-<textbook|animated|orbital|embedding>][aria-pressed],
// predict-step-alt-prev / -next / -index, the variant's view inside <div data-testid=
// "predict-step" data-variant> (its canvas "predict-step-canvas" with the ROverlay ids; the
// embedding variant: EmbeddingView "predict-step-embedding"),
// predict-step-entries, predict-step-arrow-text-<k>, predict-step-product (data-smiles),
// predict-step-sigma-<k>, predict-step-pmechdb, predict-step-alternatives >
// li[data-testid=predict-step-alternative-<k>], the ReadingTable "predict-step",
// predict-correct-step.
import { useEffect, useState } from "react";
import { exact, fmt, fmtSig } from "../../../components/format";
import { useCapability } from "../../../state/store";
import { EmbeddingView } from "../../embedding/EmbeddingView";
import { correctHash } from "../../../router";
import { matrixText, rankOf, speciesLabel } from "../network";
import { Panel } from "../../../components/Panel";
import { ReadingTable } from "../ReadingTable";
import { arrowText } from "../rmatrix";
import { StepAnimated } from "./StepAnimated";
import { STEP_VARIANTS, STEP_VARIANT_TITLES, NO_HOVER, useShownStep, useStepSides, useStepVariant, type HoverSet, type ShownStep, type StepVariant } from "./stepHooks";
import { StepOrbital } from "./StepOrbital";
import { stepAlternative } from "./stepText";
import { StepTextbook, type StepVariantProps } from "./StepTextbook";
import type { EdgeOut, NetworkOut } from "../v9types";
import "./step.css";

export interface StepPanelProps {
  jobId: string;
  edge: EdgeOut;
  net: NetworkOut;
  probabilities: boolean;
}

function Score({ edge, probabilities }: { edge: EdgeOut; probabilities: boolean }) {
  if (edge.kind === "corrected") return <span className="badge badge-corrected" data-testid="predict-step-score" data-unit="user">user step</span>;
  const useLogp = probabilities && edge.logp !== null;
  return (
    <span className="badge" data-testid="predict-step-score" data-unit={useLogp ? "logp" : "score"} data-value={exact(useLogp ? edge.logp : edge.score)} title={useLogp ? "log-probability of the step" : "the predictor's score (not a probability)"}>
      {useLogp ? `log p ${fmt(edge.logp, 3)} (p ${fmt(Math.exp(edge.logp ?? 0), 3)})` : `score ${fmtSig(edge.score, 4)}`}
    </span>
  );
}

const scoreWords = (s: { score: number | null; logp: number | null }, probabilities: boolean) => (probabilities && s.logp !== null ? `log p ${fmt(s.logp, 2)}` : `score ${fmtSig(s.score, 4)}`);

function VariantSwitch({ value, onChange }: { value: StepVariant; onChange: (v: StepVariant) => void }) {
  const embedding = useCapability("embedding");
  const off = (v: StepVariant) => v === "embedding" && embedding.known && !embedding.enabled;
  const titleOf = (v: StepVariant) => (off(v) ? `embedding trace unavailable: ${embedding.reason ?? ""}` : v === "embedding" ? "the parent state traced through a trained checkpoint" : undefined);
  return (
    <span className="ss-seg" role="group" aria-label="step view" data-testid="predict-step-variant">
      {STEP_VARIANTS.map((v) => (
        <button key={v} type="button" className={`ss-seg-btn${v === value ? " ss-seg-on" : ""}`} aria-pressed={v === value} data-testid={`predict-step-variant-${v}`} disabled={off(v)} title={titleOf(v)} onClick={() => onChange(v)}>
          {STEP_VARIANT_TITLES[v]}
        </button>
      ))}
    </span>
  );
}

function AltNav({ edge, index, onIndex }: { edge: EdgeOut; index: number; onIndex: (k: number) => void }) {
  const n = edge.alternatives.length;
  if (!n) return null;
  return (
    <span className="ss-altnav" role="group" aria-label="alternatives with the same product">
      <button type="button" className="btn btn-small" data-testid="predict-step-alt-prev" onClick={() => onIndex(stepAlternative(index, n, -1))} title="previous reaction matrix with the same product">
        ‹
      </button>
      <span className="ss-alt-index" data-testid="predict-step-alt-index" data-index={index}>
        {index < 0 ? "predicted" : `alt ${index + 1}`} <span className="muted">/ {n + 1}</span>
      </span>
      <button type="button" className="btn btn-small" data-testid="predict-step-alt-next" onClick={() => onIndex(stepAlternative(index, n, 1))} title="next reaction matrix with the same product">
        ›
      </button>
    </span>
  );
}

function StepFacts({ edge, step, label, probabilities, onAlt }: { edge: EdgeOut; step: ShownStep; label: (i: number) => string; probabilities: boolean; onAlt: (k: number) => void }) {
  const joined = edge.product_smiles.join(".");
  return (
    <div className="ss-facts">
      <div className="ss-fact" data-testid="predict-step-entries">
        <span className="ss-fact-k">R</span> <code className="code">{matrixText(step.matrix, label)}</code>
      </div>
      <div className="ss-fact">
        <span className="ss-fact-k">arrows</span>
        <ol className="ss-arrow-list">
          {step.arrows.map((a, k) => (
            <li key={k} data-testid={`predict-step-arrow-text-${k}`} data-kind={a.kind}>
              {arrowText(a, label)} <span className="muted">({a.kind.replace(/_/g, " ")})</span>
            </li>
          ))}
          {!step.arrows.length && <li className="muted">{step.note ?? "none"}</li>}
        </ol>
      </div>
      <div className="ss-fact" data-testid="predict-step-product" data-smiles={joined}>
        <span className="ss-fact-k">product</span> <code className="code pv-wrap">{joined || "–"}</code>
        {edge.sigma_keys.map((k, i) => (
          <code key={i} className="code muted pv-wrap" data-testid={`predict-step-sigma-${i}`} title="sigma key: the sigma skeleton with H counts frozen and charges zeroed">
            σ {k}
          </code>
        ))}
      </div>
      <div className="ss-fact muted">
        beam weight <code className="code">{fmt(edge.weight, 3)}</code>
        {edge.pmechdb_code && (
          <>
            {" "}
            · PMechDB <code className="code" data-testid="predict-step-pmechdb">{edge.pmechdb_code}</code>
          </>
        )}
      </div>
      {edge.alternatives.length > 0 && (
        <details className="ss-fact" data-testid="predict-step-alternatives">
          <summary>{edge.alternatives.length} other reaction matri{edge.alternatives.length === 1 ? "x" : "ces"} with the same product</summary>
          <ol className="pv-steps">
            {edge.alternatives.map((a, k) => (
              <li key={k} data-testid={`predict-step-alternative-${k}`} className={k === step.index ? "ss-alt-on" : undefined}>
                <button type="button" className="ss-link" onClick={() => onAlt(k)}>
                  <code className="code">{matrixText(a.matrix, label)}</code>
                </button>{" "}
                <span className="muted">{scoreWords(a, probabilities)}{a.rank > 0 ? `, rank ${a.rank}` : ""}</span>
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}

export function StepPanel({ jobId, edge, net, probabilities }: StepPanelProps) {
  const [variant, setVariant] = useStepVariant();
  const [alt, setAlt] = useState(-1);
  const [hover, setHover] = useState<HoverSet>(NO_HOVER);
  useEffect(() => setAlt(-1), [edge.id]);
  const step = useShownStep(edge, alt);
  const sides = useStepSides(edge, step);
  const s = sides.structure;
  const label = (i: number) => s?.label(i) ?? String(i);
  const src = net.nodes.find((n) => n.id === edge.source);
  const tools = (
    <>
      <VariantSwitch value={variant} onChange={setVariant} />
      <AltNav edge={edge} index={alt} onIndex={setAlt} />
      <a className="btn btn-small btn-primary" data-testid="predict-correct-step" href={correctHash(jobId, { edge: edge.id })} title="edit this step's reaction matrix on its parent state">
        Correct this step
      </a>
    </>
  );
  const note = (
    <>
      <span className="badge">{edge.kind}</span> {rankOf(edge) !== null && <span className="badge">rank {rankOf(edge)}</span>} <Score edge={edge} probabilities={probabilities} /> {edge.predictor && <span className="muted">{edge.predictor}</span>}
      {edge.status === "superseded" && <span className="badge">superseded</span>}
    </>
  );
  const vp: StepVariantProps = {
    sides,
    step,
    prefix: "predict-step",
    productSmiles: edge.product_smiles,
    sigmaKeys: edge.sigma_keys,
    label,
    above: edge.kind === "corrected" ? "user step" : edge.predictor,
    below: [scoreWords(step, probabilities), step.index < 0 && rankOf(edge) !== null ? `rank ${rankOf(edge)}` : step.index >= 0 ? `alternative ${step.index + 1}` : null].filter(Boolean).join(" · "),
    hover,
    onHover: (h) => setHover(h ?? NO_HOVER),
  };
  return (
    <section data-testid="predict-step-panel" data-edge={edge.id} data-kind={edge.kind} data-variant={variant} data-alt={alt}>
      <Panel title="Step" note={note} tools={tools} className="ss-panel">
        <div className="stack">
          <div className="muted pv-wrap">
            from {src ? speciesLabel(src) : edge.source}
            {edge.anchor && (
              <>
                {" "}
                · anchor{" "}
                <span data-testid="predict-step-anchor" data-i={edge.anchor.i} data-j={edge.anchor.j} data-sign={edge.anchor.sign}>
                  {edge.anchor.sign === "form" ? "forms" : "breaks"} {label(edge.anchor.i)}–{label(edge.anchor.j)}
                </span>
              </>
            )}
            {step.note && <span className="ss-note"> · {step.note}</span>}
          </div>
          <div className="ss-view" data-testid="predict-step" data-variant={variant}>
            {variant === "textbook" && <StepTextbook {...vp} />}
            {variant === "animated" && <StepAnimated {...vp} />}
            {variant === "orbital" && <StepOrbital {...vp} stateId={edge.source_state_id} />}
            {variant === "embedding" && <EmbeddingView target={{ stateId: edge.source_state_id }} testId="predict-step-embedding" />}
          </div>
          <StepFacts edge={edge} step={step} label={label} probabilities={probabilities} onAlt={setAlt} />
          <h4 className="pv-subtitle">Orbital reading</h4>
          <ReadingTable readings={step.readings} structure={s} stateId={edge.source_state_id} prefix="predict-step" onHover={vp.onHover} />
        </div>
      </Panel>
    </section>
  );
}
