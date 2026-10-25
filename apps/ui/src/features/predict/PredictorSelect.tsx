// PredictorSelect: the predictor <select> from GET /predictors (design §11.8). An unavailable
// predictor (transformer without a checkpoint, a GBDT whose features do not match) stays listed
// but disabled, with its reason as the option's title and under the select.
//
// Props:
//   value      string                     the selected predictor name
//   onChange   (name) => void
//   predictors PredictorInfoOut[] | null  null while loading
//   error      ApiError | null            the /predictors load error
//
// DOM: select[data-testid=predictor-select] > option[data-testid=predictor-option-{name}][data-available].
import type { ApiError } from "../../api/client";
import type { PredictorInfoOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";

export interface PredictorSelectProps {
  value: string;
  onChange: (name: string) => void;
  predictors: PredictorInfoOut[] | null;
  error?: ApiError | null;
}

/** The default choice: floor when available, else the first available predictor. */
export function defaultPredictor(predictors: readonly PredictorInfoOut[] | null): string {
  if (!predictors?.length) return "floor";
  const floor = predictors.find((p) => p.name === "floor" && p.available);
  return floor?.name ?? predictors.find((p) => p.available)?.name ?? predictors[0].name;
}

export function PredictorSelect({ value, onChange, predictors, error }: PredictorSelectProps) {
  const list = predictors ?? [];
  const selected = list.find((p) => p.name === value);
  return (
    <div className="predictor-select stack">
      <label className="field">
        <span className="field-label">predictor</span>
        <select data-testid="predictor-select" value={value} onChange={(e) => onChange(e.target.value)}>
          {list.length === 0 && (
            <option value={value} data-testid={`predictor-option-${value}`} data-available="unknown">
              {value} {predictors === null ? "(loading…)" : ""}
            </option>
          )}
          {list.map((p) => (
            <option
              key={p.name}
              value={p.name}
              disabled={!p.available}
              title={p.available ? p.description : (p.reason ?? p.code ?? "unavailable")}
              data-testid={`predictor-option-${p.name}`}
              data-available={p.available ? "true" : "false"}
            >
              {p.label || p.name}
              {p.available ? "" : " (unavailable)"}
            </option>
          ))}
        </select>
      </label>
      {selected && (
        <div className="predictor-info muted">
          <span className="badge">{selected.kind}</span> {selected.description}
          {selected.version && (
            <>
              {" "}
              <code className="code">{selected.version}</code>
            </>
          )}
          {selected.warnings.map((w) => (
            <div key={w} className="badge badge-warn">
              {w}
            </div>
          ))}
        </div>
      )}
      {list
        .filter((p) => !p.available)
        .map((p) => (
          <div key={p.name} className="muted predictor-unavailable" data-predictor={p.name}>
            {p.label || p.name}: unavailable{p.code ? ` (${p.code})` : ""}
            {p.reason ? `: ${p.reason}` : ""}
          </div>
        ))}
      <ErrorBanner error={error} title="Cannot list predictors" testId="predictors-error" />
    </div>
  );
}
