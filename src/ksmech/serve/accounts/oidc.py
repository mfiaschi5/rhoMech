"""OIDC sign-in (authorization code flow with PKCE) against the providers of the auth file.

`start` builds the provider's authorization URL with a random `state`, a `nonce` and a S256
code challenge, and keeps them in memory for `PENDING_S` (one server process; a restart
only means signing in again). `finish` takes the provider's `code` back, exchanges it at the
token endpoint (the client secret and the code verifier) and checks the ID token's claims:
`iss` the provider's issuer, `aud` our client id, `exp` in the future, `nonce` ours.

The ID token's signature is not verified, as OIDC Core 3.1.3.7 allows for a token received
straight from the token endpoint over TLS: the issuer and every endpoint must be https, and the
TLS server check authenticates the provider, so no JWT library is needed. The discovery document
(`/.well-known/openid-configuration`) is cached for an hour.

Any failure is OIDC_FAILED (502) with a reason a person can act on.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from ksmech.serve.accounts.config import AuthConfig, Provider
from ksmech.serve.core.errors import Code, ServeError

PENDING_S = 10 * 60
DISCOVERY_TTL_S = 3600
CLOCK_SKEW_S = 120
TIMEOUT_S = 10.0


def _fail(message: str, **detail: Any) -> ServeError:
    return ServeError(Code.OIDC_FAILED, message, detail=detail or None)


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def jwt_claims(token: str) -> dict[str, Any]:
    """The payload of a compact JWT, unverified (module docstring)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError) as exc:
        raise _fail("the provider's ID token is not a JWT") from exc
    if not isinstance(claims, dict):
        raise _fail("the provider's ID token payload is not an object")
    return claims


@dataclass(frozen=True)
class PendingLogin:
    provider: str
    verifier: str
    nonce: str
    redirect_uri: str
    return_to: str
    created: float


def _issuer_matches(claimed: Any, issuer: str) -> bool:
    # Google's ID tokens may name the issuer without the scheme
    return str(claimed).rstrip("/") in (issuer, issuer.removeprefix("https://"))


def check_claims(claims: dict[str, Any], provider: Provider, nonce: str, now: float) -> None:
    if not _issuer_matches(claims.get("iss"), provider.issuer):
        raise _fail("the ID token names another issuer", iss=str(claims.get("iss"))[:200])
    aud = claims.get("aud")
    if provider.client_id not in (aud if isinstance(aud, list) else [aud]):
        raise _fail("the ID token is for another client")
    exp = claims.get("exp")
    if not isinstance(exp, int | float) or exp < now - CLOCK_SKEW_S:
        raise _fail("the ID token has expired")
    if claims.get("nonce") != nonce:
        raise _fail("the ID token's nonce does not match this sign-in")
    if not claims.get("sub"):
        raise _fail("the ID token has no subject")


class OidcClient:
    def __init__(
        self, config: AuthConfig, transport: httpx.BaseTransport | None = None, clock=time.time
    ) -> None:
        self.config = config
        self._clock = clock
        self._http = httpx.Client(timeout=TIMEOUT_S, transport=transport)
        self._discovery: dict[str, tuple[dict, float]] = {}
        self._pending: dict[str, PendingLogin] = {}
        self._lock = threading.Lock()

    def provider(self, provider_id: str) -> Provider:
        found = self.config.providers.get(provider_id)
        if found is None:
            msg = f"no sign-in provider {provider_id!r}"
            raise ServeError(Code.PROVIDER_NOT_FOUND, msg, status=404)
        return found

    def _get_json(self, url: str, **kwargs: Any) -> dict:
        try:
            response = self._http.get(url, **kwargs)
            response.raise_for_status()
            doc = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise _fail(f"cannot read {url}: {exc}") from exc
        if not isinstance(doc, dict):
            raise _fail(f"{url} did not answer a JSON object")
        return doc

    def discovery(self, provider: Provider) -> dict:
        cached = self._discovery.get(provider.id)
        if cached is not None and cached[1] > self._clock() - DISCOVERY_TTL_S:
            return cached[0]
        doc = self._get_json(f"{provider.issuer}/.well-known/openid-configuration")
        if not _issuer_matches(doc.get("issuer"), provider.issuer):
            raise _fail("the discovery document names another issuer")
        for key in ("authorization_endpoint", "token_endpoint"):
            if not str(doc.get(key, "")).startswith("https://"):
                raise _fail(f"the provider's {key} is not an https URL")
        self._discovery[provider.id] = (doc, self._clock())
        return doc

    def start(self, provider_id: str, redirect_uri: str, return_to: str) -> tuple[str, str]:
        """(the authorization URL, the state) of a new sign-in."""
        provider = self.provider(provider_id)
        doc = self.discovery(provider)
        state, nonce, verifier = (secrets.token_urlsafe(24) for _ in range(3))
        challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        now = self._clock()
        with self._lock:
            self._pending = {k: v for k, v in self._pending.items() if v.created > now - PENDING_S}
            self._pending[state] = PendingLogin(
                provider.id, verifier, nonce, redirect_uri, return_to, now
            )
        query = {
            "response_type": "code",
            "client_id": provider.client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(provider.scopes),
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return f"{doc['authorization_endpoint']}?{urlencode(query)}", state

    def _take(self, provider_id: str, state: str) -> PendingLogin:
        with self._lock:
            pending = self._pending.pop(state, None)
        if pending is None or pending.provider != provider_id:
            raise _fail("this sign-in is unknown or was already used: start again")
        if pending.created <= self._clock() - PENDING_S:
            raise _fail("this sign-in took too long: start again")
        return pending

    def _token(self, provider: Provider, doc: dict, pending: PendingLogin, code: str) -> dict:
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": pending.redirect_uri,
            "code_verifier": pending.verifier,
            "client_id": provider.client_id,
        }
        methods = doc.get("token_endpoint_auth_methods_supported") or ["client_secret_basic"]
        auth = None
        if provider.client_secret is not None:
            if "client_secret_post" in methods:
                form["client_secret"] = provider.client_secret
            else:
                auth = (provider.client_id, provider.client_secret)
        try:
            response = self._http.post(doc["token_endpoint"], data=form, auth=auth)
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise _fail(f"the token endpoint failed: {exc}") from exc
        if response.status_code != 200 or not isinstance(body, dict) or "id_token" not in body:
            reason = body.get("error") if isinstance(body, dict) else None
            raise _fail("the provider refused the sign-in code", error=str(reason)[:200])
        return body

    def _userinfo(self, doc: dict, access_token: str | None, sub: str) -> dict[str, Any]:
        url = doc.get("userinfo_endpoint")
        if not access_token or not str(url or "").startswith("https://"):
            return {}
        try:
            info = self._get_json(url, headers={"Authorization": f"Bearer {access_token}"})
        except ServeError:
            return {}  # the ID token is enough; userinfo only fills a name or an email
        return info if str(info.get("sub")) == sub else {}

    def finish(self, provider_id: str, state: str, code: str) -> tuple[dict[str, Any], str]:
        """(the checked claims, the return path) of a provider's callback."""
        pending = self._take(provider_id, state)
        provider = self.provider(provider_id)
        doc = self.discovery(provider)
        tokens = self._token(provider, doc, pending, code)
        claims = jwt_claims(str(tokens["id_token"]))
        check_claims(claims, provider, pending.nonce, self._clock())
        extra = self._userinfo(doc, tokens.get("access_token"), str(claims["sub"]))
        merged = {**{k: v for k, v in extra.items() if k != "sub"}, **claims}
        return merged, pending.return_to

    def close(self) -> None:
        self._http.close()
