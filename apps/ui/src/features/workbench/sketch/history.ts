// Undo/redo history of immutable values, capped at HISTORY_CAP past states. Pure.

export const HISTORY_CAP = 200;

export interface History<T> {
  readonly past: readonly T[];
  readonly present: T;
  readonly future: readonly T[];
}

export function initHistory<T>(present: T): History<T> {
  return { past: [], present, future: [] };
}

/** Record a new present (no-op when it is the current one); the redo stack is dropped and the
 *  oldest states fall off beyond `cap`. */
export function pushHistory<T>(h: History<T>, next: T, cap = HISTORY_CAP): History<T> {
  if (next === h.present) return h;
  const past = [...h.past, h.present];
  return { past: past.length > cap ? past.slice(past.length - cap) : past, present: next, future: [] };
}

export function undo<T>(h: History<T>): History<T> {
  if (h.past.length === 0) return h;
  return { past: h.past.slice(0, -1), present: h.past[h.past.length - 1], future: [h.present, ...h.future] };
}

export function redo<T>(h: History<T>): History<T> {
  if (h.future.length === 0) return h;
  return { past: [...h.past, h.present], present: h.future[0], future: h.future.slice(1) };
}

export const canUndo = <T>(h: History<T>): boolean => h.past.length > 0;
export const canRedo = <T>(h: History<T>): boolean => h.future.length > 0;
