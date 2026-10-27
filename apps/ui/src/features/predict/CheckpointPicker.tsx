// CheckpointPicker: the network predictor's checkpoint on the Predict form, from GET /api/models
// (run, file, encoder, epoch, val top-1; smoke runs marked "smoke", refused ones disabled). The
// default is the server's (checkpoints.ts: the best usable non-smoke run by val top-1).
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
import { checkpointId, defaultNetworkCheckpoint, pickerLabel } from "./checkpoints";

/** GET /models (cached), the chosen checkpoint (or the default) and the list. */
export function useNetworkCheckpoint(on: boolean, value: string | null) {
  const models = useCached(on ? "/models#predict" : null, () => getModels());
  const list: CheckpointOut[] = models.data?.checkpoints ?? [];
  const chosen = (value ? list.find((c) => checkpointId(c) === value && c.usable) : null) ?? defaultNetworkCheckpoint(list);
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
            <option key={checkpointId(c)} value={checkpointId(c)} disabled={!c.usable} title={c.reason ?? c.path} data-testid={`predict-checkpoint-option-${c.source}-${c.run}-${c.file}`} data-smoke={c.smoke ? "true" : "false"} data-usable={c.usable ? "true" : "false"}>
              {pickerLabel(c)}
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
