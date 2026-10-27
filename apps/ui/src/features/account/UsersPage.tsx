// #/users: the admins' user management (/api/admin/users): every account with its role and
// state, a role and an enable / disable switch per row, a new password per row, and a form for
// a new local account. The server refuses a change that would leave no active admin
// (LAST_ADMIN) and shows it here.
import { useCallback, useEffect, useState, type FormEvent } from "react";
import { createUser, listUsers, updateUser } from "../../api/auth";
import { toApiError, type ApiError } from "../../api/http";
import type { UserCreateIn, UserOut, UserPatchIn } from "../../api/types.gen";
import { ErrorBanner } from "../../components/ErrorBanner";
import { authStore, isAdmin, loadAuth } from "../../state/auth";
import { useStore } from "../../state/store";

const ROLES = ["reader", "user", "curator", "admin"] as const;
type RoleName = (typeof ROLES)[number];

function UserRow({ user, onPatch }: { user: UserOut; onPatch: (id: number, patch: UserPatchIn) => void }) {
  const [password, setPassword] = useState("");
  return (
    <tr data-testid={`user-row-${user.login}`} data-disabled={user.disabled ? "true" : "false"}>
      <td className="num">{user.id}</td>
      <td>
        <code>{user.login}</code>
      </td>
      <td>{user.name}</td>
      <td>{user.email ?? "—"}</td>
      <td>
        <select
          aria-label={`role of ${user.login}`}
          value={user.role}
          onChange={(e) => onPatch(user.id, { role: e.target.value as RoleName })}
          data-testid={`user-role-${user.login}`}
        >
          {ROLES.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </td>
      <td>{[user.has_password ? "password" : null, ...user.providers].filter(Boolean).join(", ") || "—"}</td>
      <td>{user.last_login_at ? new Date(user.last_login_at).toLocaleString() : "never"}</td>
      <td>
        <button type="button" className="btn btn-small" onClick={() => onPatch(user.id, { disabled: !user.disabled })}>
          {user.disabled ? "Enable" : "Disable"}
        </button>
      </td>
      <td>
        <form
          className="row users-password"
          onSubmit={(e) => {
            e.preventDefault();
            onPatch(user.id, { password });
            setPassword("");
          }}
        >
          <input type="password" autoComplete="new-password" placeholder="new password" minLength={10} value={password} onChange={(e) => setPassword(e.target.value)} required />
          <button type="submit" className="btn btn-small">
            Set
          </button>
        </form>
      </td>
    </tr>
  );
}

const EMPTY: UserCreateIn = { login: "", name: "", email: null, role: "user", password: "" };

function NewUser({ onCreate }: { onCreate: (body: UserCreateIn) => Promise<boolean> }) {
  const [body, setBody] = useState<UserCreateIn>(EMPTY);
  const set = (patch: Partial<UserCreateIn>) => setBody((b) => ({ ...b, ...patch }));
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (await onCreate({ ...body, email: body.email || null })) setBody(EMPTY);
  };
  return (
    <form className="row users-new" onSubmit={submit} data-testid="user-new">
      <input placeholder="login" pattern="[a-z0-9][a-z0-9_.\-]{1,39}" value={body.login} onChange={(e) => set({ login: e.target.value })} required data-testid="user-new-login" />
      <input placeholder="name" value={body.name} onChange={(e) => set({ name: e.target.value })} required data-testid="user-new-name" />
      <input placeholder="email (optional)" type="email" value={body.email ?? ""} onChange={(e) => set({ email: e.target.value })} />
      <select aria-label="role" value={body.role} onChange={(e) => set({ role: e.target.value as RoleName })} data-testid="user-new-role">
        {ROLES.map((r) => (
          <option key={r} value={r}>
            {r}
          </option>
        ))}
      </select>
      <input type="password" autoComplete="new-password" placeholder="password (10+)" minLength={10} value={body.password} onChange={(e) => set({ password: e.target.value })} required data-testid="user-new-password" />
      <button type="submit" className="btn btn-primary" data-testid="user-new-submit">
        Add user
      </button>
    </form>
  );
}

export function UsersPage() {
  const status = useStore(authStore, (s) => s.data);
  const [users, setUsers] = useState<UserOut[] | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const reload = useCallback(async () => {
    try {
      setUsers((await listUsers()).users);
    } catch (err) {
      setError(toApiError(err));
    }
  }, []);
  const admin = isAdmin(status);
  useEffect(() => {
    if (admin) void reload();
  }, [admin, reload]);
  const onPatch = async (id: number, patch: UserPatchIn) => {
    setError(null);
    try {
      await updateUser(id, patch);
      await Promise.all([reload(), loadAuth()]); // the change may be the admin's own
    } catch (err) {
      setError(toApiError(err));
    }
  };
  const onCreate = async (body: UserCreateIn) => {
    setError(null);
    try {
      await createUser(body);
      await reload();
      return true;
    } catch (err) {
      setError(toApiError(err));
      return false;
    }
  };
  if (status && !admin) {
    return (
      <section className="panel" data-testid="users-page">
        <h2>Users</h2>
        <p className="muted">Only an admin manages the users.</p>
      </section>
    );
  }
  return (
    <section className="panel stack" data-testid="users-page">
      <h2>Users</h2>
      <p className="muted">
        Roles: reader (reads), user (predicts and corrects), curator (curation and ML), admin (everything, no job limits).
      </p>
      <ErrorBanner error={error} onDismiss={() => setError(null)} />
      <NewUser onCreate={onCreate} />
      {users && (
        <table className="table" data-testid="users-table">
          <thead>
            <tr>
              <th>id</th>
              <th>login</th>
              <th>name</th>
              <th>email</th>
              <th>role</th>
              <th>sign-in</th>
              <th>last sign-in</th>
              <th>state</th>
              <th>password</th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <UserRow key={u.id} user={u} onPatch={onPatch} />
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
