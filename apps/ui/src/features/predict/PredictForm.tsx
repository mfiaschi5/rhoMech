// PredictForm: the inputs of the Predict page (spec v9 "Inference"): reactant rows with
// stoichiometry, live depictions, a Draw button (the workbench sketcher) and the /normalize
// preflight; the predictor (the network with its checkpoint picker, CheckpointPicker); beam
// (default 5) and max steps (default 8). Submit creates the job and opens #/predict/job/<id>. Rows can be prefilled by #/predict?r=<smiles>&n=<count>&...
// "Correct from scratch" opens the Correct page on the same reactants with no prediction.
//
// Props:
//   route   Route     the #/predict route (its r / n params prefill the rows)
//
// DOM: predict-form, the ReactantRows ids, predict-conditions, the PredictorSelect ids, the
// CheckpointPicker ids (network only), predict-beam, predict-max-steps, predict-submit, predict-correct-scratch, predict-error,
// predict-recent.
import { useState } from "react";
import type { CheckpointOut } from "../../api/types.gen";
import { toApiError, type ApiError } from "../../api/http";
import { listJobs, predict } from "../../api/predict";
import { useCached } from "../../api/cache";
import { ErrorBanner } from "../../components/ErrorBanner";
import { formatRoute, jobHash, navigate, parsePredictPrefill, type Route } from "../../router";
import { metaStore, useStore } from "../../state/store";
import { CheckpointPicker, useNetworkCheckpoint } from "./CheckpointPicker";
import { refOf } from "./checkpoints";
import { NumberField } from "../../components/NumberField";
import { Panel } from "../../components/Panel";
import { defaultPredictor, PredictorSelect } from "./PredictorSelect";
import { ReactantRows, usePreflight, type ReactantRow } from "./ReactantRows";
import { usePredictors } from "./hooks";
import type { PredictIn } from "@ksmech/viz/v9types";

export const DEFAULT_BEAM = 5;
export const DEFAULT_MAX_STEPS = 8;

interface Settings {
  conditions: string;
  predictor: string | null;
  checkpoint: string | null; // network: the chosen checkpoint id; null: the default
  beam: number | null;
  maxSteps: number | null;
}

const INITIAL: Settings = { conditions: "", predictor: null, checkpoint: null, beam: DEFAULT_BEAM, maxSteps: DEFAULT_MAX_STEPS };

function RecentJobs() {
  const [mounted] = useState(() => Date.now());
  const jobs = useCached(`/jobs?kind=predict&limit=10#${mounted}`, () => listJobs({ kind: "predict", limit: 10 }));
  if (!jobs.data?.length) return null;
  return (
    <Panel title="Recent predictions" testId="predict-recent">
      <ul className="pv-recent">
        {jobs.data.map((j) => (
          <li key={j.job_id}>
            <a href={jobHash(j.job_id)}>{j.label || j.job_id}</a> <span className="badge">{j.status}</span> <span className="muted">{new Date(j.created_at).toLocaleString()}</span>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

/** POST /predict, then open the job; the error stays in place. */
function usePredictSubmit() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const submit = async (body: PredictIn) => {
    setBusy(true);
    setError(null);
    try {
      const created = await predict(body);
      navigate(jobHash(created.job_id));
    } catch (err) {
      setError(toApiError(err));
      setBusy(false);
    }
  };
  return { busy, error, submit };
}

function requestOf(rows: ReactantRow[], s: Settings, predictor: string, checkpoint: CheckpointOut | null): PredictIn {
  return {
    reactants_smiles: rows.map((r) => r.smiles.trim()),
    stoichiometry: rows.map((r) => r.count),
    conditions: s.conditions.trim() || null,
    predictor,
    beam: s.beam ?? DEFAULT_BEAM,
    max_steps: s.maxSteps ?? DEFAULT_MAX_STEPS,
    ...(predictor === "network" && checkpoint ? { checkpoint: refOf(checkpoint) } : {}),
  };
}

function blockedBy(filled: number, predictor: string, available: boolean | undefined, checkpoint: CheckpointOut | null): string | null {
  if (!filled) return "enter at least one reactant";
  if (available === false) return `predictor ${predictor} is unavailable`;
  if (predictor === "network" && !checkpoint) return "choose a usable checkpoint";
  return null;
}

/** "#/correct?r=..&n=..": the Correct page on bare reactants. */
export const correctScratchHash = (rows: ReactantRow[]) =>
  formatRoute({ page: "correct", params: { r: rows.map((r) => r.smiles.trim()), n: rows.map((r) => r.count) } });

export function PredictForm({ route }: { route: Route }) {
  const prefill = parsePredictPrefill(route);
  const [rows, setRows] = useState<ReactantRow[]>(prefill.length ? prefill : [{ smiles: "", count: 1 }]);
  const [s, setS] = useState<Settings>(INITIAL);
  const { busy, error, submit } = usePredictSubmit();
  const preflight = usePreflight(rows);
  const meta = useStore(metaStore, (m) => m.data);
  const predictors = usePredictors();
  const set = (p: Partial<Settings>) => setS((prev) => ({ ...prev, ...p }));
  const predictor = s.predictor ?? defaultPredictor(predictors.data);
  const info = predictors.data?.find((p) => p.name === predictor);
  const network = predictor === "network";
  const ck = useNetworkCheckpoint(network, s.checkpoint).chosen;
  const filled = rows.filter((r) => r.smiles.trim());
  const blocked = blockedBy(filled.length, predictor, info?.available, network ? ck : null);
  return (
    <div className="pv-page pv-form stack" data-testid="predict-form">
      <Panel title="Predict a mechanism" note="each step is a reaction matrix R = ΔBE; the arrows are its reading">
        <div className="stack">
          <ReactantRows rows={rows} onChange={setRows} preflight={preflight} maxRows={meta?.limits.max_species ?? 6} maxCount={meta?.limits.max_count ?? 4} />
          <label className="field">
            <span className="field-label">
              conditions <span className="muted">(recorded only: no predictor reads them)</span>
            </span>
            <input type="text" data-testid="predict-conditions" maxLength={500} value={s.conditions} placeholder="e.g. MeOH, 25 °C" onChange={(e) => set({ conditions: e.target.value })} />
          </label>
          <PredictorSelect value={predictor} onChange={(name) => set({ predictor: name })} predictors={predictors.data ?? null} error={predictors.error} />
          {network && info?.available && <CheckpointPicker value={s.checkpoint} onChange={(checkpoint) => set({ checkpoint })} />}
          <div className="row pv-inline-row">
            <NumberField label="beam" testId="predict-beam" value={s.beam} min={1} max={meta?.limits.max_beam ?? 10} onChange={(beam) => set({ beam })} />
            <NumberField label="max steps" testId="predict-max-steps" value={s.maxSteps} min={1} max={meta?.limits.max_steps ?? 12} onChange={(maxSteps) => set({ maxSteps })} />
          </div>
          <div className="row pv-inline-row">
            <button type="button" className="btn btn-primary" data-testid="predict-submit" disabled={busy || !!blocked} title={blocked ?? undefined} onClick={() => void submit(requestOf(filled, s, predictor, network ? ck : null))}>
              {busy ? "submitting…" : "Predict"}
            </button>
            <a className={`btn${filled.length ? "" : " pv-link-disabled"}`} data-testid="predict-correct-scratch" href={filled.length ? correctScratchHash(filled) : undefined} title="draw the steps yourself on these reactants, without a prediction">
              Correct from scratch
            </a>
            {!preflight.ok && filled.length > 0 && <span className="muted">preflight: checking or refused (see the rows)</span>}
          </div>
          <ErrorBanner error={error} title="Cannot start the prediction" testId="predict-error" />
        </div>
      </Panel>
      <RecentJobs />
    </div>
  );
}
