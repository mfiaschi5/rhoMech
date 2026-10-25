// The layout and the preferences of the workbench as React state, read from and written to
// localStorage (storage.ts; defaults when it is unavailable), and the theme preference applied
// to <html data-theme>.
import { useCallback, useEffect, useReducer, useState, type Dispatch } from "react";
import { layoutReducer, sanitizeLayout, type LayoutAction, type LayoutState } from "./layoutState";
import { DEFAULT_PREFS, sanitizePrefs, themeAttribute, updatePrefs, type Prefs, type ThemeChoice } from "./prefs";
import { LAYOUT_KEY, PREFS_KEY, readJson, writeJson } from "./storage";

export function useLayoutState(): [LayoutState, Dispatch<LayoutAction>] {
  const [state, dispatch] = useReducer(layoutReducer, null, () => sanitizeLayout(readJson(LAYOUT_KEY)));
  useEffect(() => writeJson(LAYOUT_KEY, state), [state]);
  return [state, dispatch];
}

/** Set or clear <html data-theme> (styles.css: light / dark override, none = the system's). */
export function applyTheme(t: ThemeChoice): void {
  const v = themeAttribute(t);
  if (v) document.documentElement.dataset.theme = v;
  else delete document.documentElement.dataset.theme;
}

/** The stored theme applied before the first render (main.tsx), so no page flashes. */
export function applyStoredTheme(): void {
  applyTheme(sanitizePrefs(readJson(PREFS_KEY)).theme);
}

export interface PrefsState {
  prefs: Prefs;
  setPrefs(patch: Partial<Prefs>): void;
  resetPrefs(): void;
}

export function usePrefs(): PrefsState {
  const [prefs, set] = useState<Prefs>(() => sanitizePrefs(readJson(PREFS_KEY)));
  useEffect(() => {
    writeJson(PREFS_KEY, prefs);
    applyTheme(prefs.theme);
  }, [prefs]);
  const setPrefs = useCallback((patch: Partial<Prefs>) => set((p) => updatePrefs(p, patch)), []);
  const resetPrefs = useCallback(() => set(DEFAULT_PREFS), []);
  return { prefs, setPrefs, resetPrefs };
}
