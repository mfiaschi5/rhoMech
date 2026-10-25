// StepView (design §10, §11.8): one stored step (GET /browse/step/{ref}) with both sides drawn by
// the same MolDepiction / ArrowOverlay as Predict, the dataset arrows in map mode on the
// reactants, the source, split badges, the resonance flag, what a substructure search matched,
// the curated mechanism with its status and latest re-check, and "predict from these reactants"
// (#/predict?r=...&n=... with the reactant species unmapped).
// Depictions of curated steps that the browser cannot place fall back to the server depiction of
// the step's saved pool states (found through GET /curate/{mid}); PMechDB steps need the
// server_svg of StepViewOut (w2-geometry-browse).
import { useMemo } from "react";
import { getDepiction, getMechanism, getBrowseStep } from "../../api/client";
import type {
  DatasetArrowOut,
  DepictionOut,
  MechanismSummaryOut,
  RecheckRowOut,
  StepSummaryOut,
  StepViewOut,
} from "../../api/types.gen";
import { useRDKit } from "../../chem/rdkit";
import { CapabilityNotice } from "../../components/CapabilityNotice";
import { ErrorBanner } from "../../components/ErrorBanner";
import { useCapability } from "../../state/store";
import { MolDepiction } from "../../components/MolDepiction";
import { PendingNote } from "../../components/PendingNote";
import { predictPrefillHash, type PrefillSpecies } from "../../router";
import { SplitBadges, stepLabel } from "./ResultsTable";
import { speciesCounts, speciesText } from "./smiles";
import { useLoad } from "./useLoad";

type Side = "reactants" | "products";

/** Tier-3 depiction for a curated step: the server depiction of its saved pool state. */
function curatedFetcher(step: StepSummaryOut, side: Side, dep: DepictionOut): (() => Promise<DepictionOut>) | undefined {
  const mid = step.mechanism_id;
  const index = step.step_index;
  if (step.source !== "user_curated" || mid === null || index === null) return undefined;
  return async () => {
    const m = await getMechanism(mid);
    const doc = m.steps.find((s) => s.step_index === index)?.step;
    const stateId = doc?.[side === "reactants" ? "reactant_state_id" : "product_state_id"];
    if (typeof stateId !== "string") throw new Error(`${step.ref}: the StepDoc has no ${side} state id`);
    const keepMaps = dep.atoms.filter((a) => a.element === "H").map((a) => a.map);
    return getDepiction(stateId, { keepMaps, serverSvg: true });
  };
}

function datasetText(a: DatasetArrowOut): string {
  return `${a.source_kind} [${a.source_atoms.join(",")}] → ${a.sink_kind} [${a.sink_atoms.join(",")}]`;
}

function Mechanism({ m, recheck }: { m: MechanismSummaryOut | null; recheck: RecheckRowOut | null }) {
  if (!m) return null;
  return (
    <div className="br-mechanism" data-testid="browse-step-mechanism" data-status={m.status} data-stale={m.stale ? "true" : "false"}>
      <strong>Curated mechanism</strong>{" "}
      <a href={`#/curate/m/${m.id}`} data-testid="browse-mechanism-link">
        #{m.id}
      </a>{" "}
      <span className={m.status === "rejected" ? "badge badge-error" : "badge badge-ok"}>{m.status}</span>
      {m.stale && <span className="badge badge-warn">stale</span>} <span className="muted">predictor {m.predictor}</span>
      {recheck ? (
        <div className="muted" data-testid="browse-step-recheck" data-top1={String(recheck.top1)}>
          latest re-check ({recheck.predictor}): first-arrow rank {recheck.first_arrow_rank ?? "–"} / {recheck.n_valid ?? "–"},
          product {recheck.product_match}
          {recheck.top1_reproduces_product === false ? ", top-1 does not reproduce the product" : ""}
          {recheck.reason ? `, ${recheck.reason}` : ""}
        </div>
      ) : (
        <div className="muted">not re-checked yet</div>
      )}
    </div>
  );
}

function Title({ s }: { s: StepSummaryOut }) {
  return (
    <h3 className="br-title">
      {stepLabel(s)} <span className="badge">{s.source === "pmechdb" ? "PMechDB" : "curated"}</span>{" "}
      <span className="badge">{s.status}</span>
      {s.resonance_step && (
        <span className="badge badge-warn" data-testid="browse-step-resonance" title="a resonance step: no net bond change">
          resonance step
        </span>
      )}
      <SplitBadges splits={s.splits} />
    </h3>
  );
}

/** "matched via `[CH2-]C=O`": the recorded form of a species a substructure query matched (§10). */
function MatchedForms({ forms, species }: { forms: readonly string[]; species: readonly string[] }) {
  if (!forms.length) return null;
  return (
    <p data-testid="browse-matched-forms">
      matched via{" "}
      {forms.map((f, i) => (
        <span key={f}>
          {i ? ", " : ""}
          <code>{f}</code>
        </span>
      ))}
      {species.length > 0 && <span className="muted"> (species {species.join(", ")})</span>}
    </p>
  );
}

interface SidesProps {
  v: StepViewOut;
  reactants: readonly PrefillSpecies[];
  products: readonly PrefillSpecies[];
}

/** Both sides, drawn like Predict's pools; the dataset arrows sit on the reactants (map mode). */
function Sides({ v, reactants, products }: SidesProps) {
  return (
    <div className="br-sides">
      <div data-testid="browse-step-reactants">
        <div className="muted">reactants: {speciesText(reactants)}</div>
        <MolDepiction
          depiction={v.reactants_depiction}
          width={380}
          height={250}
          datasetArrows={v.dataset_arrows}
          fetchServer={curatedFetcher(v.step, "reactants", v.reactants_depiction)}
        />
      </div>
      <div className="br-arrow" aria-hidden="true">
        →
      </div>
      <div data-testid="browse-step-products">
        <div className="muted">products: {speciesText(products)}</div>
        <MolDepiction
          depiction={v.products_depiction}
          width={380}
          height={250}
          fetchServer={curatedFetcher(v.step, "products", v.products_depiction)}
        />
      </div>
    </div>
  );
}

function Details({ v }: { v: StepViewOut }) {
  const s = v.step;
  return (
    <dl className="kv">
      <dt>arrows</dt>
      <dd>{v.dataset_arrows.length ? v.dataset_arrows.map(datasetText).join("; ") : "none"}</dd>
      <dt>PMechDB code</dt>
      <dd>{s.pmechdb_code ? <code>{s.pmechdb_code}</code> : "–"}</dd>
      <dt>pools</dt>
      <dd>
        <code>{s.reactant_pool_key}</code> → <code>{s.product_pool_key}</code>
      </dd>
      {s.exported_step_id && (
        <>
          <dt>exported as</dt>
          <dd>
            <code>{s.exported_step_id}</code>
          </dd>
        </>
      )}
    </dl>
  );
}

export interface StepViewProps {
  stepRef: string;
  /** The search-result row of this step, if any: its matched forms belong to the current query. */
  resultRow?: StepSummaryOut;
}

export function StepView({ stepRef, resultRow }: StepViewProps) {
  const view = useLoad(`/browse/step/${stepRef}`, (o) => getBrowseStep(stepRef, o));
  const { rdkit } = useRDKit();
  const v = view.data;
  const reactants = useMemo(() => (v ? speciesCounts(rdkit, v.step.reactants_smiles) : []), [rdkit, v]);
  const products = useMemo(() => (v ? speciesCounts(rdkit, v.step.products_smiles) : []), [rdkit, v]);
  if (view.error) return <ErrorBanner error={view.error} title={`Step ${stepRef}`} testId="browse-step-error" />;
  if (!v) {
    return (
      <p className="muted">
        loading step {stepRef}… {view.pending && <PendingNote label={view.pending.label} worker={view.pending.worker} />}
      </p>
    );
  }
  const s = v.step;
  const predict = useCapability("predict");
  const forms = resultRow?.matched_forms.length ? resultRow.matched_forms : s.matched_forms;
  const species = resultRow?.matched_species.length ? resultRow.matched_species : s.matched_species;
  return (
    <section className="panel br-step" data-testid="browse-step-view" data-ref={s.ref} data-source={s.source} data-status={s.status}>
      <Title s={s} />
      <MatchedForms forms={forms} species={species} />
      <Sides v={v} reactants={reactants} products={products} />
      <Details v={v} />
      <Mechanism m={v.mechanism} recheck={v.recheck} />
      {predict.enabled ? (
        <p>
          <a className="btn btn-primary" data-testid="browse-predict-from" href={predictPrefillHash(reactants)}>
            Predict from these reactants
          </a>
        </p>
      ) : (
        <CapabilityNotice name="predict" title="Predict from these reactants is disabled" compact />
      )}
    </section>
  );
}
