// Feature-local React hooks: cached reads that report the 202 pending label, pool depictions
// keyed by the map numbers they must keep, debounced values, and the job poll with a reload.
import { useEffect, useRef, useState } from "react";
import { getDepiction, getTokens, type RequestOptions } from "../../api/client";
import { useCached, type CachedState } from "../../api/cache";
import type { DepictionOut, TokensOut, WorkerStatusOut } from "../../api/types.gen";

export interface PendingInfo {
  label: string;
  worker: WorkerStatusOut;
}

export interface PendingCached<T> extends CachedState<T> {
  /** The last PendingOut of the running load (null once it settled). */
  pending: PendingInfo | null;
}

/** useCached plus the pending protocol: `load` receives RequestOptions with onPending, and
 *  the hook exposes the label and worker status of the last 202 for <PendingNote>. */
export function usePendingCached<T>(key: string | null, load: (o: RequestOptions) => Promise<T>): PendingCached<T> {
  const [pending, setPending] = useState<{ key: string; info: PendingInfo } | null>(null);
  const state = useCached<T>(key, () => {
    const k = key ?? "";
    return load({ onPending: (label, worker) => setPending({ key: k, info: { label, worker } }) });
  });
  const live = state.loading && pending !== null && pending.key === key;
  return { ...state, pending: live ? pending.info : null };
}

/** The depiction of a pool state that keeps the explicit H atoms among `maps`. */
export function usePoolDepiction(stateId: string | null, maps: readonly number[] = []): CachedState<DepictionOut> {
  const keep = [...new Set(maps)].sort((a, b) => a - b);
  const key = stateId ? `/pool/${stateId}/depiction?keep_maps=${keep.join(",")}` : null;
  return useCached<DepictionOut>(key, () => getDepiction(stateId ?? "", { keepMaps: keep }));
}

/** The tokens of a pool state (pending while the featurization runs). */
export function usePoolTokens(stateId: string | null): PendingCached<TokensOut> {
  const key = stateId ? `/pool/${stateId}/tokens` : null;
  return usePendingCached<TokensOut>(key, (o) => getTokens(stateId ?? "", o));
}

/** `value`, settled for `ms` milliseconds. */
export function useDebounced<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return settled;
}

/** A ref that always holds the latest value (for callbacks registered once). */
export function useLatest<T>(value: T) {
  const ref = useRef(value);
  ref.current = value;
  return ref;
}
