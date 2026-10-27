// The value of an async load, keyed: `key` null is idle, a new key loads again (and drops the
// answer of the previous one). The loader is read when the key changes, never a dependency.
import { useEffect, useRef, useState } from "react";

export interface Loaded<T> {
  data: T | undefined;
  error: string | null;
}

export function useLoaded<T>(key: string | null, load: () => Promise<T>): Loaded<T> {
  const loadRef = useRef(load);
  loadRef.current = load;
  const [state, setState] = useState<Loaded<T> & { key: string | null }>({ key: null, data: undefined, error: null });
  useEffect(() => {
    let live = true;
    setState({ key, data: undefined, error: null });
    if (key === null) return;
    loadRef.current().then(
      (data) => live && setState({ key, data, error: null }),
      (err: unknown) => live && setState({ key, data: undefined, error: err instanceof Error ? err.message : String(err) }),
    );
    return () => {
      live = false;
    };
  }, [key]);
  return state.key === key ? { data: state.data, error: state.error } : { data: undefined, error: null };
}
