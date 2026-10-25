// PredictForm: the inputs of the Predict page (design §11.8): reactant rows with live depictions
// and /normalize preflight, conditions (recorded only), the predictor, beam (default 3), max
// steps (default 4), temperature, candidate scope and turnover stop; submit creates the job and
// opens #/predict/job/<id>. Rows can be prefilled by #/predict?r=<smiles>&n=<count>&...
//
// Props:
//   route   Route     the #/predict route (its r / n params prefill the rows)
//
// DOM: smiles-input-{i}, stoich-input-{i}, input-depiction-{i}, preflight-{i}, add-reactant,
// conditions-input, predictor-select, beam-input, max-steps-input, temperature-input,
// scope-select, predict-submit.
import { useEffect, useState } from "react";
import { listJobs, predict, toApiError, type ApiError } from "../../api/client";
import { useCached } from "../../api/cache";
import type { PredictIn } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import { jobHash, navigate, parsePredictPrefill, type Route } from "../../router";
import { loadPredictors, metaStore, predictorsStore, useStore } from "../../state/store";
import { NumberField } from "./NumberField";
import { defaultPredictor, PredictorSelect } from "./PredictorSelect";
import { ReactantRows, usePreflight, type Preflight, type ReactantRow } from "./ReactantRows";

export const DEFAULT_BEAM = 3;
export const DEFAULT_MAX_STEPS = 4;

interface Settings {
  conditions: string;
  predictor: string | null;
  beam: number | null;
  maxSteps: number | null;
  temperature: number | null;
  scope: "all" | "inter";
  stopAtTurnover: boolean;
}

const INITIAL: Settings = { conditions: "", predictor: null, beam: DEFAULT_BEAM, maxSteps: DEFAULT_MAX_STEPS, temperature: null, scope: "all", stopAtTurnover: true };

function RecentJobs() {
  const [mounted] = useState(() => Date.now());
  const jobs = useCached(`/jobs?kind=predict&limit=10#${mounted}`, () => listJobs({ kind: "predict", limit: 10 }));
  if (!jobs.data?.length) return null;
  return (
    <section className="panel recent-jobs">
      <h3>Recent predictions</h3>
      <ul>
        {jobs.data.map((j) => (
          <li key={j.job_id}>
            <a href={jobHash(j.job_id)}>{j.label || j.job_id}</a> <span className="badge">{j.status}</span>{" "}
            <span className="muted">{new Date(j.created_at).toLocaleString()}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Options({ s, set, limits }: { s: Settings; set: (p: Partial<Settings>) => void; limits: { beam: number; steps: number }; }) {
  return (
    <div className="row predict-options">
      <NumberField label="beam" testId="beam-input" value={s.beam} min={1} max={limits.beam} onChange={(beam) => set({ beam })} />
      <NumberField label="max steps" testId="max-steps-input" value={s.maxSteps} min={1} max={limits.steps} onChange={(maxSteps) => set({ maxSteps })} />
      <NumberField label="temperature" testId="temperature-input" value={s.temperature} min={0} minExclusive integer={false} step={0.05} allowEmpty placeholder="predictor default" onChange={(temperature) => set({ temperature })} />
      <label className="field">
        <span className="field-label">scope</span>
        <select data-testid="scope-select" value={s.scope} onChange={(e) => set({ scope: e.target.value === "inter" ? "inter" : "all" })}>
          <option value="all">all pairs</option>
          <option value="inter">intermolecular only</option>
        </select>
      </label>
      <label className="field-inline">
        <input type="checkbox" data-testid="stop-at-turnover" checked={s.stopAtTurnover} onChange={(e) => set({ stopAtTurnover: e.target.checked })} />
        stop at turnover
      </label>
    </div>
  );
}

function preflightNote(p: Preflight): string {
  if (p.rows.some((r) => r.state === "error" || r.state === "unchecked")) return "has errors (the server will refuse them)";
  if (p.result && !p.result.ok) return "the pool as a whole was refused";
  return "checking…";
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

function requestOf(rows: ReactantRow[], s: Settings, predictor: string): PredictIn {
  return {
    reactants_smiles: rows.map((r) => r.smiles.trim()),
    stoichiometry: rows.map((r) => r.count),
    conditions: s.conditions.trim() || null,
    predictor,
    beam: s.beam ?? DEFAULT_BEAM,
    max_steps: s.maxSteps ?? DEFAULT_MAX_STEPS,
    temperature: s.temperature,
    candidate_scope: s.scope,
    stop_at_turnover: s.stopAtTurnover,
  };
}

export function PredictForm({ route }: { route: Route }) {
  const prefill = parsePredictPrefill(route);
  const [rows, setRows] = useState<ReactantRow[]>(prefill.length ? prefill : [{ smiles: "", count: 1 }]);
  const [s, setS] = useState<Settings>(INITIAL);
  const { busy, error, submit } = usePredictSubmit();
  const preflight = usePreflight(rows);
  const meta = useStore(metaStore, (m) => m.data);
  const predictors = useStore(predictorsStore);
  useEffect(() => {
    void loadPredictors();
  }, []);
  const set = (p: Partial<Settings>) => setS((prev) => ({ ...prev, ...p }));
  const predictor = s.predictor ?? defaultPredictor(predictors.data);
  const info = predictors.data?.find((p) => p.name === predictor);
  const filled = rows.filter((r) => r.smiles.trim());
  const blocked = !filled.length ? "enter at least one reactant" : info && !info.available ? `predictor ${predictor} is unavailable` : null;
  return (
    <div className="predict-form stack" data-testid="predict-form">
      <section className="panel stack">
        <h2>Predict a mechanism</h2>
        <ReactantRows rows={rows} onChange={setRows} preflight={preflight} maxRows={meta?.limits.max_species ?? 6} maxCount={meta?.limits.max_count ?? 4} />
        <label className="field">
          <span className="field-label">
            conditions <span className="muted">(recorded only: no predictor reads them)</span>
          </span>
          <input type="text" data-testid="conditions-input" maxLength={500} value={s.conditions} placeholder="e.g. MeOH, 25 °C" onChange={(e) => set({ conditions: e.target.value })} />
        </label>
        <PredictorSelect value={predictor} onChange={(name) => set({ predictor: name, temperature: null })} predictors={predictors.data} error={predictors.error} />
        <Options s={s} set={set} limits={{ beam: meta?.limits.max_beam ?? 10, steps: meta?.limits.max_steps ?? 12 }} />
        {info && <div className="muted">default temperature for {predictor}: {info.default_temperature}</div>}
        <div className="row">
          <button type="button" className="btn btn-primary" data-testid="predict-submit" disabled={busy || !!blocked} title={blocked ?? undefined} onClick={() => void submit(requestOf(filled, s, predictor))}>
            {busy ? "submitting…" : "predict"}
          </button>
          {!preflight.ok && filled.length > 0 && <span className="muted">preflight: {preflightNote(preflight)}</span>}
        </div>
        <ErrorBanner error={error} title="Cannot start the prediction" testId="predict-error" />
      </section>
      <RecentJobs />
    </div>
  );
}
