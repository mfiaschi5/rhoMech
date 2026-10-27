"""Password hashes with the standard library's scrypt (no new dependency).

A hash is `scrypt$<n>$<r>$<p>$<salt b64>$<key b64>`: the parameters travel with it, so raising
`N` later still verifies the old hashes. Verification compares in constant time.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

N, R, P = 2**15, 8, 1  # 32 MiB per hash, about 0.1 s here
KEY_BYTES = 32
MAXMEM = 64 * 1024 * 1024
MIN_LENGTH = 10


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _key(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    data = password.encode("utf-8")
    return hashlib.scrypt(data, salt=salt, n=n, r=r, p=p, maxmem=MAXMEM, dklen=KEY_BYTES)


def hash_password(password: str) -> str:
    """The stored form of `password` (a fresh 16-byte salt)."""
    if len(password) < MIN_LENGTH:
        raise ValueError(f"a password needs at least {MIN_LENGTH} characters")
    salt = secrets.token_bytes(16)
    return f"scrypt${N}${R}${P}${_b64(salt)}${_b64(_key(password, salt, N, R, P))}"


def verify_password(password: str, stored: str | None) -> bool:
    """Whether `password` matches `stored`; False for None or a malformed hash."""
    if not stored:
        return False
    try:
        scheme, n, r, p, salt, key = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(key)
        got = _key(password, base64.b64decode(salt), int(n), int(r), int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, expected)


def generate_password() -> str:
    """A random password for `ksmech-serve users add --generate` (about 120 bits)."""
    return secrets.token_urlsafe(15)
