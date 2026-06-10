# API auth: shared-token bearer guard on `/predict`

**Date:** 2026-06-10
**Status:** Approved (brainstorm)
**Scope:** `api` (settings, a new `auth` module, `app.py` wiring, `errors.py`)
plus the `benchmark` HTTP client (`run_api.py`), which must present the token
when calling `/predict`. No model behavior change. No change to
request/response payloads.

## Goal

Stop anonymous callers from hitting `POST /predict` when the API sits on a
public URL. A single shared secret, supplied via an env var, must be presented
as an HTTP bearer token. `GET /health` stays open so load balancers and uptime
monitors can probe readiness without a credential.

This is deliberately the simplest thing that works ("casual exposure" threat
model): one shared token, no per-client identity, no revocation story, no
expiry. If we later need to tell callers apart or revoke one without rotating
everyone, that is a separate design.

## Background / current state

`api/app.py` is a FastAPI app exposing two routes: `GET /health` (open, used for
readiness) and `POST /predict` (the inference path). Runtime config lives in
`api/settings.py` as a `pydantic-settings` `BaseSettings` with
`env_prefix="TEMPORAL_API_"`, so a field named `token` is populated from
`TEMPORAL_API_TOKEN`.

Errors are modeled as `ApiError` subclasses in `api/errors.py` (each carries a
`status_code`, `detail`, and `code`) and rendered by a single registered
`_api_error_handler` into `{"detail": ..., "code": ...}`. Auth must reuse this
shape so error responses stay consistent.

## Design

### Setting

Add one field to `Settings` (`api/settings.py`):

```python
token: str | None = None  # TEMPORAL_API_TOKEN; unset/empty disables auth
```

When unset or empty, auth is **disabled** (open `/predict`). This keeps local
dev frictionless. The risk — shipping to prod with auth accidentally off — is
mitigated by a startup log line (below), not by failing closed.

### Error type

Add to `api/errors.py`:

```python
class Unauthorized(ApiError):
    status_code = 401
    code = "unauthorized"
```

It flows through the existing `_api_error_handler`, yielding
`{"detail": ..., "code": "unauthorized"}` with status 401.

The one nuance is the `WWW-Authenticate: Bearer` response header (expected on a
401 for a bearer scheme). The existing handler does not set custom headers, so
we handle this in the auth dependency by raising in a way that carries the
header — simplest is to set `headers={"WWW-Authenticate": "Bearer"}` on the
response. Implementation detail to settle during coding: either (a) give
`ApiError` an optional `headers` attribute the handler passes through, or (b)
special-case `Unauthorized` in the handler. Prefer (a) if it stays small.

### Auth dependency

New module `api/auth.py` exposing a FastAPI dependency:

```python
security = HTTPBearer(auto_error=False)

def require_token(
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    expected = settings.token
    if not expected:          # auth disabled
        return
    if creds is None or not secrets.compare_digest(creds.credentials, expected):
        raise Unauthorized("missing or invalid token")
```

- `HTTPBearer(auto_error=False)` so we control the error shape rather than
  letting FastAPI emit its default 403.
- `secrets.compare_digest` for constant-time comparison (avoids timing leaks).
- Returns `None` on success; the dependency is used only for its side effect.

### Wiring

In `api/app.py`, protect `/predict` only:

```python
@app.post("/predict", ..., dependencies=[Depends(require_token)])
```

`/health` is left untouched. Using a per-route dependency (rather than
middleware that pattern-matches paths to skip `/health`) keeps the guard
explicit, unit-testable in isolation, and visible in the OpenAPI schema — so
`/docs` gets the "Authorize" button.

### Startup visibility

In the `lifespan` startup, log one line via the existing module logger:

- token set → `INFO  auth enabled`
- token unset → `WARNING auth disabled: TEMPORAL_API_TOKEN not set`

So an operator can see at a glance whether the deployment is protected.

### Benchmark client (consumer)

The `benchmark` package's `api` subcommand drives `/predict` over HTTP via
`run_api.py::_http_post` (the in-process `core` subcommand is unaffected — it
never touches the API). With auth enabled, those requests must carry the token
or every one returns 401.

`_http_post` reads `TEMPORAL_API_TOKEN` from the environment (the **same** var
the server uses) and, when set, sends `Authorization: Bearer <token>`. When
unset, no header is added and behavior is identical to today — so this is a
no-op against an auth-disabled server. No new CLI flag: the operator exports the
same env var they already set on the server. The token is read per request via
`os.environ.get`, keeping the existing `post=` injection seam intact for tests.

## Testing

New `api/tests/test_auth.py` (plus, if natural, a case or two in `test_app.py`):

1. token set + correct `Authorization: Bearer <token>` → 200
2. token set + wrong token → 401, body `code == "unauthorized"`,
   `WWW-Authenticate: Bearer` header present
3. token set + missing `Authorization` header → 401
4. token unset → `/predict` reachable (auth off)
5. `/health` returns 200 with no credential, regardless of token setting

Tests override the setting (monkeypatch `settings.token`) rather than
relying on process env, matching the existing `test_settings.py` style.

## Docs

- Add `TEMPORAL_API_TOKEN` to the `api/README.md` env-var table with a one-line
  description.
- Add a commented `TEMPORAL_API_TOKEN=` example to `.envrc` and to the
  `environment:` block in `docker-compose.yml`.
- Note in `benchmark/README.md` (under the `api` subcommand) that
  `TEMPORAL_API_TOKEN` must be exported to benchmark an auth-protected server.

## Out of scope

- Multiple keys / per-client identity / revocation.
- Token rotation, expiry, or refresh.
- Rate limiting, IP allow-listing, mTLS.
- Protecting `/health`.
