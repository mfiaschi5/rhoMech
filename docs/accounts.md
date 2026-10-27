# Accounts: sign-in, roles and job quotas

`ksmech.serve.accounts` gives the platform its users. A user has a role (reader < user <
curator < admin, the table in `docs/serve_architecture.md`) and signs in with a local password
or through an OIDC provider (ORCID, Google, or any other issuer). The browser then holds an
HttpOnly session cookie. The CLI and the MCP server keep using bearer tokens
(`KSMECH_SERVE_TOKENS`).

Users live in `data/serve/accounts/accounts.sqlite`. It is a database of its own, separate from
the curation database. It stores no password in clear (scrypt hashes) and no session in clear
(sha256 of the cookie value).

## The first admin

```bash
uv run ksmech-serve users add jacopo --name "Jacopo" --email you@example.org --role admin --generate
```

`--generate` prints a random password once. Without it the password is asked on the terminal,
or read from stdin with `--password-stdin`. The command works while the server runs. After
that, the admin manages everyone from the app: header → **Users** (`#/users`). An admin can add
local accounts, change roles, disable or enable an account, and set a new password. Disabling
an account or changing its password signs it out everywhere. Neither the app nor the CLI lets
you demote or disable the last active admin (LAST_ADMIN).

```bash
uv run ksmech-serve users list
uv run ksmech-serve users set ada --role curator
uv run ksmech-serve users set ada --disable
```

In the `lab` profile a browser without a session is still the local admin, so nothing changes
for the lab. Signing in there acts as that user, with that user's role and quota. In `public`
a browser without a session is an anonymous reader. It can look at molecules, predictors and
jobs, and it needs to sign in to submit a job.

## Admins have no limits

Every role except admin has a job quota. It counts submitted jobs: `POST /predict`,
`/jobs/{id}/expand` and `/embedding`. Each role has a limit per day (the last 24 hours) and a
limit per minute.

| Role | Per day | Per minute |
|---|---|---|
| reader | 0 | 0 |
| user | 100 | 10 |
| curator | 1000 | 30 |
| admin | no limit | no limit |

A refused job is 429 QUOTA_EXCEEDED. A job is counted when its request arrives, before the
body is checked, so an invalid request counts too. The account page (`#/login` once signed in)
shows the quota and how many jobs were used. The admin quota cannot be configured: admins are
never counted.

The per-request size limits (`max_species`, `max_atoms`, `max_beam`, `max_steps`,
`max_nodes`, in `/api/meta`) protect the single compute worker. They are server settings and
apply to everyone. Change them in `ServeSettings` for the whole server.

## The auth file: OIDC providers, quotas, sessions

`KSMECH_SERVE_AUTH=/path/auth.toml` (keep it, and above all the secrets, out of git):

```toml
public_url = "https://ksmech.example.org"   # base of the redirect URIs (default: the request's URL)
session_days = 14
cookie_secure = "auto"         # true when public_url is https
password_login = true
default_role = "user"          # the role a first OIDC sign-in gets
admin_emails = ["you@example.org"]               # a provider-verified email: admin at first sign-in
admin_subjects = ["orcid:0000-0002-1825-0097"]   # <provider>:<sub>: admin at first sign-in

[quota.user]
per_day = 100
per_minute = 10

[providers.orcid]
client_id = "APP-XXXXXXXXXXXXXXXX"
client_secret_env = "KSMECH_ORCID_SECRET"

[providers.google]
client_id = "1234-abc.apps.googleusercontent.com"
client_secret_env = "KSMECH_GOOGLE_SECRET"
```

`orcid` and `google` are presets that set the issuer, the label and the scopes. Any other id
needs `issuer = "https://..."`. That covers managed providers that also offer email and
password sign-in (Auth0, Clerk, Keycloak, Supabase, …): each is an OIDC issuer, so adding one
is configuration, not code.

The redirect URI to register with each provider is
`<public_url>/api/auth/oidc/<id>/callback`. For example:

- `https://ksmech.example.org/api/auth/oidc/orcid/callback`
- `https://ksmech.example.org/api/auth/oidc/google/callback`

- **ORCID:** register at <https://orcid.org/developer-tools>; the sandbox is
  `issuer = "https://sandbox.orcid.org"`. An ORCID ID token carries the ORCID iD and the name,
  not the email.
- **Google:** register at Google Cloud Console → APIs & Services → Credentials → OAuth client ID
  (Web application). Google accepts `http://localhost` redirect URIs but not a LAN address such
  as `http://192.168.1.196:8000`. To try Google sign-in in the LAN, open the app through
  `http://localhost:8000` (an SSH tunnel) or a host name with https.

A first sign-in creates a user, unless the provider reports a verified email that an existing
user already has; then it links to that user. Later sign-ins of the same provider subject are
the same user. ID tokens are taken straight from the provider's token endpoint over TLS and
their claims are checked (issuer, audience, expiry, nonce), with PKCE and a state cookie (see
the `serve.accounts.oidc` docstring).

## Routes

| Route | Who | What |
|---|---|---|
| `GET /api/auth/me` | anyone | the principal, the signed-in user, the providers, the quota |
| `POST /api/auth/login` | anyone | `{login, password}`, sets the session cookie; 10 failures in 15 min → 429 LOGIN_THROTTLED |
| `POST /api/auth/logout` | anyone | ends this browser's session |
| `POST /api/auth/password` | signed in | `{current, new}`; ends every session of the user |
| `GET /api/auth/oidc/{id}/start?return_to=#/...` | anyone | redirects to the provider |
| `GET /api/auth/oidc/{id}/callback` | the provider | back to `return_to`, or `#/login?error=` |
| `GET /api/admin/users`, `POST /api/admin/users`, `PATCH /api/admin/users/{id}` | admin | user management |
