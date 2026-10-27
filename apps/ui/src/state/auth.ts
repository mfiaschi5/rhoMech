// Who is signed in (/api/auth/me): the header's user menu, the login page and the users page
// read it. Reloaded after every sign-in and sign-out; an error leaves the last value in place.
import { getAuthMe } from "../api/auth";
import { toApiError } from "../api/http";
import type { AuthStatusOut } from "../api/types.gen";
import { createStore, type Loadable } from "./store";

export const authStore = createStore<Loadable<AuthStatusOut>>({ data: null, error: null, loading: false });

export async function loadAuth(): Promise<void> {
  const cur = authStore.get();
  authStore.set({ ...cur, loading: true });
  try {
    authStore.set({ data: await getAuthMe(), error: null, loading: false });
  } catch (err) {
    authStore.set({ data: cur.data, error: toApiError(err), loading: false });
  }
}

export function setAuth(status: AuthStatusOut): void {
  authStore.set({ data: status, error: null, loading: false });
}

/** Whether the request is an admin's: a signed-in admin, an admin token, or the lab's local admin. */
export function isAdmin(status: AuthStatusOut | null): boolean {
  return status?.principal.role === "admin";
}
