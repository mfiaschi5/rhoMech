// StateDepiction: the depiction of one saved pool state (GET /pool/{state}/depiction) with token or
// dataset arrows on top. The H atoms the arrows use are kept (keep_maps), every other H is folded
// into its parent. A state that cannot be loaded (404 after a cleanup, say) falls back to a plain
// RDKit.js drawing of `fallbackSmiles` with the error beside it, never to an empty box.
import { useState } from "react";
import { type RequestOptions } from "../../api/http";
import { getDepiction } from "../../api/pool";
import { useCached } from "../../api/cache";
import type { DatasetArrowOut, DepictionOut, WorkerStatusOut } from "../../api/types.gen";
import type { TokenArrowLike } from "../../chem/arrowGeom";
import { ErrorBanner } from "../../components/ErrorBanner";
import { MolDepiction } from "../../components/MolDepiction";
import { PendingNote } from "../../components/PendingNote";
import { SmilesDepiction } from "../../components/SmilesDepiction";
import { arrowMaps } from "./stepdoc";

export interface StateDepictionProps {
  stateId: string | null;
  fallbackSmiles?: string;
  arrows?: readonly TokenArrowLike[] | null;
  datasetArrows?: readonly DatasetArrowOut[] | null;
  /** Extra maps to keep (e.g. the original arrows drawn on another copy). */
  extraMaps?: readonly number[];
  edgeId?: string;
  faded?: boolean;
  width?: number;
  height?: number;
  testId?: string;
}

export function depictionKey(stateId: string, keepMaps: readonly number[]): string {
  return `/pool/${stateId}/depiction?keep_maps=${keepMaps.join(",")}`;
}

export function StateDepiction(props: StateDepictionProps) {
  const { stateId, width = 300, height = 200 } = props;
  const [pending, setPending] = useState<{ label: string; worker: WorkerStatusOut } | null>(null);
  const keep = [...new Set([...arrowMaps(props.arrows, props.datasetArrows), ...(props.extraMaps ?? [])])].sort(
    (a, b) => a - b,
  );
  const key = stateId ? depictionKey(stateId, keep) : null;
  const dep = useCached<DepictionOut>(key, () => {
    const o: RequestOptions = { onPending: (label, worker) => setPending({ label, worker }) };
    return getDepiction(stateId ?? "", { keepMaps: keep }, o).finally(() => setPending(null));
  });
  if (!stateId || dep.error) {
    return (
      <div className="cu-depiction-fallback" data-testid={props.testId ? `${props.testId}-fallback` : undefined}>
        {props.fallbackSmiles ? (
          <SmilesDepiction smiles={props.fallbackSmiles} width={width} height={height} />
        ) : (
          <div className="mol-depiction cu-empty" style={{ width, height }}>
            <div className="mol-placeholder">no state to draw</div>
          </div>
        )}
        {dep.error && <ErrorBanner error={dep.error} title="Depiction unavailable (drawn without arrows)" />}
      </div>
    );
  }
  return (
    <div className="cu-depiction">
      <MolDepiction
        depiction={dep.data ?? null}
        stateId={stateId}
        width={width}
        height={height}
        arrows={props.arrows}
        datasetArrows={props.datasetArrows}
        edgeId={props.edgeId}
        fadedArrows={props.faded}
        testId={props.testId}
      />
      {pending && <PendingNote label={pending.label} worker={pending.worker} />}
    </div>
  );
}
