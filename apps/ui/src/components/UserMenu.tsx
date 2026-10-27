// UserMenu: the right end of the header (serve/accounts). Signed in: the user's name and role,
// "Users" for an admin, "Sign out". Not signed in: who the request is (the lab's local admin,
// or anonymous) and "Sign in", which comes back to the current page.
import { useState } from "react";
import { signOut } from "../api/auth";
import { toApiError } from "../api/http";
import { useHashRoute } from "../router";
import { authStore, isAdmin, setAuth } from "../state/auth";
import { appErrorStore, useStore } from "../state/store";

export function UserMenu() {
  const status = useStore(authStore, (s) => s.data);
  const route = useHashRoute();
  const [busy, setBusy] = useState(false);
  if (!status) return null;
  const { principal, user, quota } = status;
  const returnTo = route.page === "login" ? "#/" : route.hash || "#/";
  const onSignOut = async () => {
    setBusy(true);
    try {
      setAuth(await signOut());
    } catch (err) {
      appErrorStore.set(toApiError(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="user-menu" data-testid="user-menu" data-role={principal.role} data-signed-in={user ? "true" : "false"}>
      {user ? (
        <a href="#/login" className="user-name" title={`signed in as ${user.login}`} data-testid="user-name">
          {user.name}
        </a>
      ) : (
        <span className="muted" data-testid="user-name">
          {principal.name}
        </span>
      )}
      <span className={`badge${principal.role === "admin" ? " badge-ok" : ""}`} data-testid="user-role">
        {principal.role}
        {quota.unlimited ? " · no limits" : ""}
      </span>
      {isAdmin(status) && (
        <a href="#/users" className="nav-link" data-testid="nav-users">
          Users
        </a>
      )}
      {user ? (
        <button type="button" className="btn btn-small" onClick={onSignOut} disabled={busy} data-testid="sign-out">
          Sign out
        </button>
      ) : (
        <a
          className="btn btn-small btn-primary"
          href={`#/login?${new URLSearchParams({ return_to: returnTo }).toString()}`}
          data-testid="sign-in"
        >
          Sign in
        </a>
      )}
    </div>
  );
}
