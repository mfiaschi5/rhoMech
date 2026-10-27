"""Accounts of the platform: users with a role, sign-in by a local password or an OIDC provider
(ORCID, Google, or any issuer of the auth file), browser sessions and the job quota of each
role. It stands on `core` alone: `core.auth` resolves a session cookie through the lookup the
app injects (`AccountService.session_principal`), and `core.capabilities` charges a job to the
quota through `app.state.charge`.

The users live in their own database, `data/serve/accounts/accounts.sqlite`
(`ServeSettings.accounts_db`), never in the curation database. The configuration is the
optional TOML `KSMECH_SERVE_AUTH` (`config.AuthConfig`); without it password sign-in works for
the users that `ksmech-serve users add` creates, and no OIDC provider is offered.

Admins have no quota. In the `lab` profile a request without a session stays the local admin,
as before; a session there names its user.
"""
