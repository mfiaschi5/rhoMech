// useLoad: a fresh (uncached) read of mutable server state, keyed by `key` (null: idle).
// Curation rows change under every POST / PATCH, so these reads bypass api/cache.ts; depictions,
// which never change for a state id, go through useCached instead. A 202 PendingOut is retried by
// the client (§11.3) and surfaces here as `pending` (shown with components/PendingNote).
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, isAbort, toApiError, type RequestOptions } from "../../api/http";
import type { WorkerStatusOut } from "../../api/types.gen";

export interface PendingInfo {
  label: string;
  worker: WorkerStatusOut;
}

export interface LoadState<T> {
  data: T | undefined;
  error: ApiError | null;
  loading: boolean;
  pending: PendingInfo | null;
  /** Load the same key again (the old data stays visible meanwhile). */
  reload: () => void;
  /** Replace the data (e.g. with the response of a mutation). */
  set: (data: T) => void;
}

interface Inner<T> {
  key: string | null;
  data: T | undefined;
  error: ApiError | null;
  pending: PendingInfo | null;
  inflight: boolean;
}

export function useLoad<T>(key: string | null, load: (o: RequestOptions) => Promise<T>): LoadState<T> {
  const loadRef = useRef(load);
  loadRef.current = load;
  const [nonce, setNonce] = useState(0);
  const [inner, setInner] = useState<Inner<T>>({ key: null, data: undefined, error: null, pending: null, inflight: false });

  useEffect(() => {
    if (!key) {
      setInner({ key, data: undefined, error: null, pending: null, inflight: false });
      return;
    }
    const ctl = new AbortController();
    setInner((s) => ({ key, data: s.key === key ? s.data : undefined, error: null, pending: null, inflight: true }));
    const onPending = (label: string, worker: WorkerStatusOut) =>
      setInner((s) => (s.key === key ? { ...s, pending: { label, worker } } : s));
    loadRef.current({ signal: ctl.signal, onPending }).then(
      (data) => setInner({ key, data, error: null, pending: null, inflight: false }),
      (err: unknown) => {
        if (isAbort(err)) return;
        setInner((s) => ({ ...s, key, error: toApiError(err), pending: null, inflight: false }));
      },
    );
    return () => ctl.abort();
  }, [key, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  const set = useCallback((data: T) => setInner((s) => ({ ...s, data, error: null })), []);
  const current = inner.key === key;
  return {
    data: current ? inner.data : undefined,
    error: current ? inner.error : null,
    loading: !!key && (!current || inner.inflight),
    pending: current ? inner.pending : null,
    reload,
    set,
  };
}
