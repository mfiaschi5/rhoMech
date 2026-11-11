// CheckpointPicker: the network predictor's checkpoint on the Predict form, from GET /api/models
// ("source/run/file.pt", encoder, epoch, val top-1; smoke runs marked "smoke"; refused and backward
// runs disabled). The default is the server's own (ModelsOut.default_checkpoint: a forward run of
// the development source first, then val top-1), never recomputed here.
//
// Props:
//   value     string | null              the chosen checkpoint id (source/run/file); null: default
//   onChange  (id: string) => void
//
// DOM: predict-checkpoint (select) > option[data-testid=predict-checkpoint-option-<source>-<run>-
// <file>][data-smoke][data-usable]; predict-checkpoint-error; predict-checkpoint-none.
import { useCached } from "../../api/cache";
import type { CheckpointOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import { getModels } from "../../api/predict";
import { checkpointId, chooseCheckpoint, pickerLabel, predictable } from "./checkpoints";

/** GET /models (cached), the chosen checkpoint (or the default) and the list. */
export function useNetworkCheckpoint(on: boolean, value: string | null) {
  const models = useCached(on ? "/models#predict" : null, () => getModels());
  const list: CheckpointOut[] = models.data?.checkpoints ?? [];
  const chosen = chooseCheckpoint(list, value, models.data?.default_checkpoint);
  return { list, chosen, loading: models.loading, error: models.error ?? null };
}

export function CheckpointPicker({ value, onChange }: { value: string | null; onChange: (id: string) => void }) {
  const { list, chosen, loading, error } = useNetworkCheckpoint(true, value);
  return (
    <div className="stack pv-checkpoint">
      <label className="field">
        <span className="field-label">checkpoint</span>
        <select data-testid="predict-checkpoint" value={chosen ? checkpointId(chosen) : ""} onChange={(e) => onChange(e.target.value)} disabled={!list.length}>
          {!list.length && <option value="">{loading ? "listing the checkpoints…" : "no local checkpoint"}</option>}
          {list.map((c) => (
            <option key={checkpointId(c)} value={checkpointId(c)} disabled={!predictable(c)} title={c.reason ?? (c.direction === "backward" ? "a backward run predicts the previous step" : c.path)} data-testid={`predict-checkpoint-option-${c.source}-${c.run}-${c.file}`} data-smoke={c.smoke ? "true" : "false"} data-usable={c.usable ? "true" : "false"}>
              {c.source}/{pickerLabel(c)}
            </option>
          ))}
        </select>
      </label>
      {!loading && list.length > 0 && !chosen && (
        <div className="muted" data-testid="predict-checkpoint-none">
          no usable checkpoint
        </div>
      )}
      <ErrorBanner error={error} title="Cannot list the checkpoints" testId="predict-checkpoint-error" />
    </div>
  );
}
