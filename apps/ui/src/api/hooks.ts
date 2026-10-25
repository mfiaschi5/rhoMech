// Shared React hooks over the API client: cached reads that report the 202 pending label,
// debounced values and a latest-value ref. Feature-specific hooks (pool depictions, tokens)
// stay in their feature.
import { useEffect, useRef, useState } from "react";
import type { RequestOptions } from "./client";
import { useCached, type CachedState } from "./cache";
import type { WorkerStatusOut } from "./types.gen";

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
