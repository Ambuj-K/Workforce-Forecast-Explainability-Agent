"""Who is asking, which stores they may see, and how often they may ask.

Users are listed in a YAML file (no secrets in it, only token hashes):

    users:
      - id: planner-north
        role: planner              # planner: ask + runs; admin: also monitoring
        stores: [S001, S002]       # or "*" for every store
        token_sha256: 3f1c...      # sha256 of the bearer token; the token itself lives in a secret store

Three ways to authenticate, chosen explicitly (there is no implicit default):

* ``token``: ``Authorization: Bearer <token>``; the token's hash is compared in constant time.
* ``header``: an identity proxy in front of the service (cloud IAP, a load balancer with OIDC,
  an app-platform auth layer) has already authenticated the user and passes their id in a
  header. Only safe when the service is reachable *only* through that proxy.
* ``none``: local development; everyone is a local admin. ``serve.py`` refuses it on any
  address other than loopback.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

Role = Literal["planner", "admin"]
Mode = Literal["token", "header", "none"]


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: Role
    stores: frozenset[str] | None  # None = every store

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


LOCAL_ADMIN = Principal("local", "admin", None)


def check_bind(mode: Mode, host: str) -> None:
    """Refuse to serve without sign-in on anything but loopback."""
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if mode == "none" and not loopback:
        raise ValueError(f"auth mode 'none' is only allowed on loopback, not {host}: use 'token' or 'header'")


@dataclass(frozen=True)
class _User:
    principal: Principal
    token_sha256: str | None


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def load_users(path: Path) -> list[_User]:
    entries = yaml.safe_load(path.read_text()).get("users", [])
    users = []
    for e in entries:
        if e.get("role") not in ("planner", "admin"):
            raise ValueError(f"user {e.get('id')!r}: role must be planner or admin")
        stores = e.get("stores")
        if stores != "*" and not isinstance(stores, list):
            raise ValueError(f"user {e.get('id')!r}: stores must be a list or '*'")
        users.append(_User(Principal(str(e["id"]), e["role"], None if stores == "*" else frozenset(map(str, stores))), e.get("token_sha256")))
    return users


class Authenticator:
    def __init__(self, mode: Mode, users: list[_User] | None = None, header: str = "X-Authenticated-User") -> None:
        if mode != "none" and not users:
            raise ValueError(f"auth mode {mode!r} needs a users file")
        self.mode = mode
        self.header = header
        self._users = users or []

    @classmethod
    def from_file(cls, mode: Mode, path: Path | None, header: str = "X-Authenticated-User") -> Authenticator:
        return cls(mode, load_users(path) if path else None, header)

    def authenticate(self, authorization: str | None, headers: dict[str, str]) -> Principal | None:
        """The caller's principal, or None if they aren't a known user."""
        if self.mode == "none":
            return LOCAL_ADMIN
        if self.mode == "header":
            user_id = headers.get(self.header.lower())
            return next((u.principal for u in self._users if user_id and u.principal.user_id == user_id), None)
        if not authorization or not authorization.lower().startswith("bearer "):
            return None
        presented = hash_token(authorization[7:].strip())
        match = None
        for u in self._users:  # compare against every user so timing doesn't reveal which hash matched
            if u.token_sha256 and hmac.compare_digest(presented, u.token_sha256):
                match = u.principal
        return match


class RateLimiter:
    """Token bucket per user: ``per_minute`` questions per minute, bursts up to ``burst``.

    In-process, so it limits one instance. On a platform with several instances, put the
    platform's gateway limit in front; this one still caps what a single instance spends.
    """

    def __init__(self, per_minute: float = 6, burst: int = 3, clock=time.monotonic) -> None:
        self._rate = per_minute / 60
        self._burst = burst
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str) -> float:
        """0 if allowed (and a token is spent), else seconds until the next token."""
        with self._lock:
            now = self._clock()
            tokens, last = self._buckets.get(key, (float(self._burst), now))
            tokens = min(self._burst, tokens + (now - last) * self._rate)
            if tokens >= 1:
                self._buckets[key] = (tokens - 1, now)
                return 0.0
            self._buckets[key] = (tokens, now)
            return (1 - tokens) / self._rate
