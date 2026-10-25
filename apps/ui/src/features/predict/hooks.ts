// Predict-local React hooks: the predictors list, a pool state's depiction read as its Lewis
// structure, and the debounced live /step/check of a matrix being edited.
import { useEffect, useMemo, useState } from "react";
import { getDepiction, isAbort, toApiError, type ApiError } from "../../api/client";
import { useCached, type CachedState } from "../../api/cache";
import { useDebounced } from "../../api/hooks";
import type { DepictionOut } from "../../api/types.gen";
import { poolStructure, touchedHydrogens, type PoolStructure } from "./poolStructure";
import { inContractRange, isEmptyMatrix } from "./rmatrix";
import { checkStep, getPredictorsV9 } from "./v9api";
import type { ArrowReadingOut, MatrixIO, PredictorOut, StepCheckOut } from "./v9types";

export function usePredictors(): CachedState<PredictorOut[]> {
  return useCached<PredictorOut[]>("/predictors#v9", () => getPredictorsV9());
}

export interface PoolState {
  depiction: DepictionOut | undefined;
  structure: PoolStructure | null;
  error: ApiError | null;
  loading: boolean;
}

/** GET /pool/{state}/depiction, read as the reactant structure (every pool atom, hidden H
 *  included; none of the hidden H drawn). */
export function usePool(stateId: string | null): PoolState {
  const dep = useCached<DepictionOut>(stateId ? `/pool/${stateId}/depiction#v9` : null, () => getDepiction(stateId ?? ""));
  const parsed = useMemo(() => parse(dep.data, new Set()), [dep.data]);
  return { depiction: dep.data, structure: parsed.structure, error: dep.error ?? parsed.error, loading: dep.loading };
}

function parse(dep: DepictionOut | undefined, show: ReadonlySet<number> | "all"): { structure: PoolStructure | null; error: ApiError | null } {
  if (!dep) return { structure: null, error: null };
  try {
    return { structure: poolStructure(dep, show), error: null };
  } catch (err) {
    return { structure: null, error: toApiError(err) };
  }
}

/** The pool structure that draws the hidden H an R or its arrows touch (all with `allH`). */
export function useDrawnStructure(pool: PoolState, matrix: MatrixIO | null, arrows: readonly ArrowReadingOut[], allH = false): PoolStructure | null {
  const dep = pool.depiction;
  const touched = dep ? [...touchedHydrogens(dep, matrix, arrows)].sort((a, b) => a - b) : [];
  const key = allH ? "all" : touched.join(",");
  return useMemo(() => {
    if (!dep || (!allH && !touched.length)) return pool.structure;
    return parse(dep, allH ? "all" : new Set(touched)).structure;
    // key stands for allH and touched
  }, [dep, pool.structure, key]);
}

export const CHECK_DEBOUNCE_MS = 300;
/** A check whose readings are pending (the state's tokens are computing) is asked again. */
export const READINGS_RETRY_MS = 3000;

export interface StepCheck {
  result: StepCheckOut | null;
  /** The matrix the result belongs to (JSON), to tell a stale answer. */
  forKey: string | null;
  error: ApiError | null;
  busy: boolean;
}

/** The live POST /step/check of `matrix` on `stateId`, 300 ms after its last change; an empty
 *  matrix or one outside the contract's range is not sent. Stale answers are dropped. */
export function useStepCheck(stateId: string | null, matrix: MatrixIO): StepCheck & { current: boolean } {
  const key = JSON.stringify(matrix);
  const settledKey = useDebounced(key, CHECK_DEBOUNCE_MS);
  const [state, setState] = useState<StepCheck>({ result: null, forKey: null, error: null, busy: false });
  const [retry, setRetry] = useState(0);
  const pending = state.result?.readings_status === "pending" && state.forKey === settledKey;
  useEffect(() => {
    if (!pending) return;
    const t = setTimeout(() => setRetry((n) => n + 1), READINGS_RETRY_MS);
    return () => clearTimeout(t);
  }, [pending, retry]);
  useEffect(() => {
    const m = JSON.parse(settledKey) as MatrixIO;
    if (!stateId || isEmptyMatrix(m) || !inContractRange(m)) {
      setState({ result: null, forKey: settledKey, error: null, busy: false });
      return;
    }
    const ctl = new AbortController();
    setState((s) => ({ ...s, busy: s.forKey !== settledKey || !s.result }));
    checkStep({ state_id: stateId, matrix: m }, { signal: ctl.signal }).then(
      (result) => setState({ result, forKey: settledKey, error: null, busy: false }),
      (err: unknown) => {
        if (!isAbort(err)) setState({ result: null, forKey: settledKey, error: toApiError(err), busy: false });
      },
    );
    return () => ctl.abort();
  }, [stateId, settledKey, retry]);
  return { ...state, current: state.forKey === key && !state.busy };
}
