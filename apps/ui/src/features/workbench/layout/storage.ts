// localStorage access of the workbench layout and preferences (per browser, survives reloads
// and tabs). Every access is in try/catch: storage may be missing, blocked (private mode) or
// full, and then the page runs on the defaults and nothing persists. Values are JSON; a value
// that does not parse reads as null. The readers sanitize, so a stale entry never breaks a page.

export const LAYOUT_KEY = "ksmech.workbench.layout.v1";
export const PREFS_KEY = "ksmech.workbench.prefs.v1";

export function readJson(key: string): unknown {
  try {
    const text = window.localStorage.getItem(key);
    return text === null ? null : (JSON.parse(text) as unknown);
  } catch {
    return null;
  }
}

export function writeJson(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // storage unavailable: the layout lasts as long as the page
  }
}
