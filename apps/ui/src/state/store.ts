// A minimal external store (no state library): createStore<T>() + useStore via
// useSyncExternalStore. Server state lives in api/cache.ts; selection state lives in the hash.
// The app-wide stores here hold /api/meta, /api/predictors and the shell's error banner.
import { useSyncExternalStore } from "react";
import { ApiError, toApiError } from "../api/http";
import { getMeta } from "../api/meta";
import { getPredictors } from "../api/predict";
import type { MetaOut, PredictorInfoOut } from "../api/types.gen";

export interface Store<T> {
  get(): T;
  set(next: T): void;
  update(fn: (prev: T) => T): void;
  subscribe(listener: () => void): () => void;
}

export function createStore<T>(initial: T): Store<T> {
  let value = initial;
  const listeners = new Set<() => void>();
  return {
    get: () => value,
    set(next) {
      if (Object.is(next, value)) return;
      value = next;
      for (const l of listeners) l();
    },
    update(fn) {
      this.set(fn(value));
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

/** The store's value (or a slice of it); re-renders when it changes. Keep selectors pure and
 *  returning existing references (or primitives), or the component re-renders every time. */
export function useStore<T>(store: Store<T>): T;
export function useStore<T, S>(store: Store<T>, select: (value: T) => S): S;
export function useStore<T, S>(store: Store<T>, select?: (value: T) => S): T | S {
  const snapshot = () => (select ? select(store.get()) : store.get());
  return useSyncExternalStore(store.subscribe, snapshot, snapshot);
}

// ---- server metadata

export interface Loadable<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
}

export const metaStore = createStore<Loadable<MetaOut>>({ data: null, error: null, loading: false });
export const predictorsStore = createStore<Loadable<PredictorInfoOut[]>>({
  data: null,
  error: null,
  loading: false,
});

async function load<T>(store: Store<Loadable<T>>, fetcher: () => Promise<T>, force: boolean) {
  const cur = store.get();
  if (cur.loading || (cur.data && !force)) return;
  store.set({ ...cur, loading: true });
  try {
    store.set({ data: await fetcher(), error: null, loading: false });
  } catch (err) {
    store.set({ data: cur.data, error: toApiError(err), loading: false });
  }
}

/** Fetch /api/meta once (force: again). Errors land in metaStore.error, never thrown. */
export function loadMeta(force = false): Promise<void> {
  return load(metaStore, () => getMeta(), force);
}

/** Fetch /api/predictors once (force: again). Nothing is fetched while /api/meta says the
 *  predict capability is off (the route answers 503 FEATURE_DISABLED): the list stays empty. */
export function loadPredictors(force = false): Promise<void> {
  const predict = metaStore.get().data?.capabilities?.predict;
  if (predict && !predict.enabled) {
    if (!predictorsStore.get().data) predictorsStore.set({ data: [], error: null, loading: false });
    return Promise.resolve();
  }
  return load(predictorsStore, () => getPredictors(), force);
}

// ---- capabilities (/api/meta.capabilities: the features the spec v8 port switched off)

export interface Capability {
  /** /api/meta has loaded (before that, every capability reads as enabled). */
  known: boolean;
  enabled: boolean;
  /** Why it is off; null when on. */
  reason: string | null;
}

export function capabilityOf(meta: MetaOut | null, name: string): Capability {
  if (!meta) return { known: false, enabled: true, reason: null };
  const cap = meta.capabilities?.[name];
  return cap ? { known: true, enabled: cap.enabled, reason: cap.reason } : { known: true, enabled: true, reason: null };
}

/** One capability from /api/meta; an unknown name reads as enabled. */
export function useCapability(name: string): Capability {
  return capabilityOf(useStore(metaStore, (s) => s.data), name);
}

// ---- the shell's error banner (one app-wide error; pages show their own errors in place)

export const appErrorStore = createStore<ApiError | null>(null);

export function reportError(err: unknown): void {
  appErrorStore.set(toApiError(err));
}

export function clearError(): void {
  appErrorStore.set(null);
}
