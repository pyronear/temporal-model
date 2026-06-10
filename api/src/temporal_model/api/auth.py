"""Shared-token bearer auth for protected endpoints.

A single secret in ``settings.token`` (env ``TEMPORAL_API_TOKEN``) guards the
routes that depend on ``require_token``. When the secret is unset/empty, auth is
disabled and the dependency is a no-op.
"""

import secrets

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .errors import Unauthorized
from .settings import settings

# auto_error=False so we render our own {detail, code} 401 instead of FastAPI's
# default 403 on a missing/garbled Authorization header.
_bearer = HTTPBearer(auto_error=False)
# Module-level Depends instance — keeps the call out of the argument default
# (ruff B008) while preserving the standard FastAPI dependency wiring.
_bearer_dep = Depends(_bearer)


def require_token(
    creds: HTTPAuthorizationCredentials | None = _bearer_dep,
) -> None:
    """Reject requests lacking a valid bearer token (no-op when auth is off)."""
    expected = settings.token
    if not expected:
        return
    if creds is None or not secrets.compare_digest(creds.credentials, expected):
        raise Unauthorized("missing or invalid token")
