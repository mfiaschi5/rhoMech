// PredictorSelect: the predictor <select> from GET /predictors (spec v9). An unavailable
// predictor (the network without a trained checkpoint) stays listed but disabled, with its
// reason as the option's title and in the list under the select. A predictor whose scores are not
// probabilities (the untrained physics floor) gets a badge saying so.
//
// Props:
//   value      string                 the selected predictor name
//   onChange   (name) => void
//   predictors PredictorOut[] | null  null while loading
//   error      ApiError | null        the /predictors load error
//
// DOM: select[data-testid=predict-predictor] > option[data-testid=predict-predictor-option-<name>]
// [data-available]; predict-score-badge (scores, not probabilities); per unavailable predictor
// [data-testid=predict-predictor-reason-<name>].
import type { ApiError } from "../../api/http";
import { ErrorBanner } from "../../components/ErrorBanner";
import type { PredictorOut } from "./v9types";

export interface PredictorSelectProps {
  value: string;
  onChange: (name: string) => void;
  predictors: PredictorOut[] | null;
  error?: ApiError | null;
}

/** The default choice: the first available predictor in the order network, gbdt, floor. */
export function defaultPredictor(predictors: readonly PredictorOut[] | null | undefined): string {
  const list = predictors ?? [];
  for (const kind of ["network", "gbdt", "floor"]) {
    const hit = list.find((p) => p.kind === kind && p.available);
    if (hit) return hit.name;
  }
  return list.find((p) => p.available)?.name ?? list[0]?.name ?? "floor";
}

/** The badge text of a predictor whose scores are not probabilities, else null. */
export function scoreBadge(p: PredictorOut | undefined): string | null {
  if (!p || p.scores_are_probabilities) return null;
  return p.kind === "floor" ? "untrained physics floor: scores, not probabilities" : "scores, not probabilities";
}

export function PredictorSelect({ value, onChange, predictors, error }: PredictorSelectProps) {
  const list = predictors ?? [];
  const selected = list.find((p) => p.name === value);
  const badge = scoreBadge(selected);
  return (
    <div className="pv-predictor stack">
      <div className="row pv-inline-row">
        <label className="field">
          <span className="field-label">predictor</span>
          <select data-testid="predict-predictor" value={value} onChange={(e) => onChange(e.target.value)}>
            {list.length === 0 && <option value={value}>{value} {predictors === null ? "(loading…)" : ""}</option>}
            {list.map((p) => (
              <option key={p.name} value={p.name} disabled={!p.available} title={p.available ? (p.description ?? p.kind) : (p.reason ?? "unavailable")} data-testid={`predict-predictor-option-${p.name}`} data-available={p.available ? "true" : "false"}>
                {`${p.label || p.name}${p.kind !== p.name ? ` (${p.kind})` : ""}${p.available ? "" : " (unavailable)"}`}
              </option>
            ))}
          </select>
        </label>
        {badge && (
          <span className="badge badge-warn pv-score-badge" data-testid="predict-score-badge">
            {badge}
          </span>
        )}
        {selected?.version && <code className="code muted">{selected.version}</code>}
      </div>
      {list
        .filter((p) => !p.available)
        .map((p) => (
          <div key={p.name} className="muted pv-unavailable" data-testid={`predict-predictor-reason-${p.name}`}>
            {p.label || p.name}: unavailable{p.reason ? `: ${p.reason}` : ""}
          </div>
        ))}
      <ErrorBanner error={error ?? null} title="Cannot list predictors" testId="predict-predictors-error" />
    </div>
  );
}
