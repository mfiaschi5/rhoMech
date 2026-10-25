// Server-state cache (design §11.2): a key (normally the request URL) -> promise map. A
// settled value is kept until a mutation invalidates it; a rejected load is dropped, so the
// next read retries. Use it for immutable or slowly changing reads (depictions, tokens,
// geometry, meta); poll jobs with api/poll.ts instead.
//
//   const dep = useCached(`/pool/${id}/depiction`, () => getDepiction(id));
//   ...after a curation mutation: invalidate("/curate");
//   ...a "retry" button after an error: dep.reload() (refresh(key): reloads even though the
//   failed entry was already dropped, which invalidate() would not notice).
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, isAbort, toApiError } from "./client";

const entries = new Map<string, Promise<unknown>>();
const values = new Map<string, unknown>();
const listeners = new Set<(key: string) => void>();

/** The cached promise for `key`, starting `load()` on a miss. */
export function cached<T>(key: string, load: () => Promise<T>): Promise<T> {
  let p = entries.get(key) as Promise<T> | undefined;
  if (!p) {
    const fresh = load();
    p = fresh;
    entries.set(key, fresh);
    fresh.then(
      (v) => {
        if (entries.get(key) === fresh) values.set(key, v);
      },
      () => {
        if (entries.get(key) === fresh) entries.delete(key);
      },
    );
  }
  return p;
}

/** The settled value of `key`, if any (no load). */
export function peek<T>(key: string): T | undefined {
  return values.get(key) as T | undefined;
}

/** Put a value in the cache (e.g. the response of a mutation that returns the new state). */
export function prime<T>(key: string, value: T): void {
  entries.set(key, Promise.resolve(value));
  values.set(key, value);
  for (const l of listeners) l(key);
}

export type KeyMatch = string | RegExp | ((key: string) => boolean);

function matches(match: KeyMatch | undefined, key: string): boolean {
  if (match === undefined) return true;
  if (typeof match === "string") return key.startsWith(match);
  if (match instanceof RegExp) return match.test(key);
  return match(key);
}

/** Drop every entry whose key matches (a string is a prefix; nothing: everything). Mounted
 *  useCached readers of a dropped key reload. Returns the number of entries dropped. */
export function invalidate(match?: KeyMatch): number {
  const dropped = [...entries.keys()].filter((k) => matches(match, k));
  for (const k of dropped) {
    entries.delete(k);
    values.delete(k);
  }
  for (const k of dropped) for (const l of listeners) l(k);
  return dropped.length;
}

/** Load `key` again: drop it if cached and tell every mounted reader of it to reload. Unlike
 *  invalidate(), it notifies also when the key is no longer cached, which is the state a failed
 *  load leaves behind (cached() drops a rejected entry), so a "retry" after an error reloads. */
export function refresh(key: string): void {
  entries.delete(key);
  values.delete(key);
  for (const l of listeners) l(key);
}

/** Call `listener(key)` whenever `key` is invalidated or refreshed; returns the unsubscribe. */
export function onInvalidate(listener: (key: string) => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export interface CachedState<T> {
  data: T | undefined;
  error: ApiError | null;
  loading: boolean;
  /** Load this key again (refresh): also after a failed load, whose entry is already gone. */
  reload: () => void;
}

/** React hook over `cached`: `key` null means idle. `load` may change between renders; only
 *  `key` decides when to reload. */
export function useCached<T>(key: string | null, load: () => Promise<T>): CachedState<T> {
  const loadRef = useRef(load);
  loadRef.current = load;
  const [version, setVersion] = useState(0);
  const [state, setState] = useState<{ key: string | null; data: T | undefined; error: ApiError | null }>(
    () => ({ key, data: key ? peek<T>(key) : undefined, error: null }),
  );
  useEffect(() => {
    if (!key) return;
    return onInvalidate((k) => {
      if (k === key) setVersion((v) => v + 1);
    });
  }, [key]);
  useEffect(() => {
    let live = true;
    if (!key) {
      setState({ key, data: undefined, error: null });
      return;
    }
    const hit = peek<T>(key);
    setState({ key, data: hit, error: null });
    cached(key, () => loadRef.current()).then(
      (data) => live && setState({ key, data, error: null }),
      (err) => live && !isAbort(err) && setState({ key, data: undefined, error: toApiError(err) }),
    );
    return () => {
      live = false;
    };
  }, [key, version]);
  const reload = useCallback(() => {
    if (key) refresh(key);
  }, [key]);
  const current = state.key === key;
  return {
    data: current ? state.data : undefined,
    error: current ? state.error : null,
    loading: !!key && (!current || (state.data === undefined && state.error === null)),
    reload,
  };
}
