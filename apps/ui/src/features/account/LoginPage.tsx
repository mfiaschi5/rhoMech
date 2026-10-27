// #/login?return_to=<route>&error=<message>: sign in (a local password, or a provider of the
// auth file), and once signed in the account: role, providers, job quota, password change.
// An OIDC sign-in leaves the app and comes back through /api/auth/oidc/<id>/callback, which
// lands on return_to, or here with ?error= when it failed.
import { useState, type FormEvent } from "react";
import { changePassword, providerStartUrl, signIn } from "../../api/auth";
import { toApiError, type ApiError } from "../../api/http";
import type { AuthStatusOut } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import type { PageProps } from "../../registry";
import { navigate } from "../../router";
import { authStore, loadAuth, setAuth } from "../../state/auth";
import { useStore } from "../../state/store";

function safeReturn(value: string | null): string {
  return value && value.startsWith("#/") && !value.startsWith("#/login") ? value : "#/";
}

function SignInForm({ status, returnTo }: { status: AuthStatusOut; returnTo: string }) {
  const [login, setLogin] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setAuth(await signIn({ login, password }));
      navigate(returnTo, { replace: true });
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="stack account-signin">
      {status.password_login && (
        <form className="stack" onSubmit={submit} data-testid="login-form">
          <label className="stack account-field">
            <span>Login or email</span>
            <input autoComplete="username" value={login} onChange={(e) => setLogin(e.target.value)} data-testid="login-login" required />
          </label>
          <label className="stack account-field">
            <span>Password</span>
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              data-testid="login-password"
              required
            />
          </label>
          <button type="submit" className="btn btn-primary" disabled={busy} data-testid="login-submit">
            Sign in
          </button>
        </form>
      )}
      {status.providers.length > 0 && (
        <div className="stack" data-testid="login-providers">
          {status.password_login && <div className="muted account-or">or</div>}
          {status.providers.map((p) => (
            <a key={p.id} className="btn" href={providerStartUrl(p, returnTo)} data-testid={`login-provider-${p.id}`}>
              Sign in with {p.label}
            </a>
          ))}
        </div>
      )}
      {!status.password_login && status.providers.length === 0 && (
        <p className="muted">No sign-in method is configured on this server (KSMECH_SERVE_AUTH).</p>
      )}
      <ErrorBanner error={error} title="Sign-in failed" testId="login-error" />
    </div>
  );
}

function PasswordForm() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [error, setError] = useState<ApiError | null>(null);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      await changePassword({ current, new: next });
      await loadAuth(); // every session ended, this one too: sign in again
      navigate("#/login", { replace: true });
    } catch (err) {
      setError(toApiError(err));
    }
  };
  return (
    <form className="stack" onSubmit={submit} data-testid="password-form">
      <h3>Change password</h3>
      <input type="password" autoComplete="current-password" placeholder="current password" value={current} onChange={(e) => setCurrent(e.target.value)} required />
      <input type="password" autoComplete="new-password" placeholder="new password (10 characters or more)" value={next} minLength={10} onChange={(e) => setNext(e.target.value)} required />
      <button type="submit" className="btn">
        Change password
      </button>
      <ErrorBanner error={error} title="Password not changed" />
    </form>
  );
}

function Account({ status }: { status: AuthStatusOut }) {
  const { user, quota } = status;
  if (!user) return null;
  const limits = quota.unlimited
    ? "no limits (admin)"
    : `${quota.used_today} of ${quota.per_day ?? "∞"} jobs in the last 24 h, ${quota.per_minute ?? "∞"} per minute`;
  return (
    <div className="stack" data-testid="account">
      <dl className="kv">
        <dt>Name</dt>
        <dd>{user.name}</dd>
        <dt>Login</dt>
        <dd>
          <code>{user.login}</code>
        </dd>
        <dt>Email</dt>
        <dd>{user.email ?? "—"}</dd>
        <dt>Role</dt>
        <dd data-testid="account-role">{user.role}</dd>
        <dt>Sign-in</dt>
        <dd>{[user.has_password ? "password" : null, ...user.providers].filter(Boolean).join(", ")}</dd>
        <dt>Jobs</dt>
        <dd data-testid="account-quota">{limits}</dd>
      </dl>
      {user.has_password && <PasswordForm />}
    </div>
  );
}

export function LoginPage({ route }: PageProps) {
  const { data: status, error } = useStore(authStore);
  const returnTo = safeReturn(route.params.get("return_to"));
  const failed = route.params.get("error");
  return (
    <section className="panel account-page" data-testid="login-page">
      <h2>{status?.user ? "Account" : "Sign in"}</h2>
      <ErrorBanner error={error} title="Cannot read /api/auth/me" />
      {failed && <ErrorBanner error={failed} title="Sign-in failed" testId="login-callback-error" />}
      {status && !status.user && status.principal.name === "local" && (
        <p className="muted">
          This is the lab profile: without signing in you are the local admin. Sign in to act as one of the platform&apos;s
          users.
        </p>
      )}
      {status && (status.user ? <Account status={status} /> : <SignInForm status={status} returnTo={returnTo} />)}
    </section>
  );
}
