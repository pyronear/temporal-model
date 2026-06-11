# API Version in `/predict` Responses — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every `/predict` response self-describes model + code via a grouped
`version: {api, model}` block, with the code version baked into the Docker
image from the git tag.

**Architecture:** A new `settings.api_version` (env `TEMPORAL_API_VERSION`)
is the single runtime source of the code version. CI passes the git-tag
version as a Docker build arg → env var. The response DTO replaces the
`model: {name, version}` block with a flat `version: {api, model}` block
(breaking change, agreed). `/health` additively gains `api_version`.

**Spec:** `docs/specs/2026-06-11-api-version-in-response-design.md`

**Tech Stack:** FastAPI, pydantic / pydantic-settings, pytest. All work is in
`api/` (plus CI workflow + docs). Run all commands from `api/` unless noted.

---

### Task 1: `settings.api_version` with empty-string normalization

The Dockerfile will always `ENV TEMPORAL_API_VERSION=${VERSION}`; when the
build arg is absent (local `docker build`), that sets the env var to the
**empty string**, not unset. The setting must normalize `""` → `None` so a
local image still reports `null`, not `""`.

**Files:**
- Modify: `api/src/temporal_model/api/settings.py`
- Test: `api/tests/test_settings.py`

- [ ] **Step 1: Write the failing tests**

Append to `api/tests/test_settings.py`:

```python
def test_api_version_default_none():
    assert Settings(_env_file=None).api_version is None


def test_api_version_env_override(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_VERSION", "0.3.0")
    assert Settings(_env_file=None).api_version == "0.3.0"


def test_api_version_empty_env_is_none(monkeypatch):
    # The Dockerfile always sets ENV TEMPORAL_API_VERSION=${VERSION}; a build
    # without the arg yields "" — must normalize to None (not a release).
    monkeypatch.setenv("TEMPORAL_API_VERSION", "")
    assert Settings(_env_file=None).api_version is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_settings.py -v`
Expected: the three new tests FAIL with `AttributeError: 'Settings' object has no attribute 'api_version'` (or pydantic unknown-field error); all others PASS.

- [ ] **Step 3: Implement the setting**

In `api/src/temporal_model/api/settings.py`:

Change the import line:

```python
from pydantic import Field, field_validator
```

Add the field after `token` (keeping the grouping: service identity vs S3).
**Naming gotcha:** with `env_prefix="TEMPORAL_API_"`, a plain `api_version`
field would read env `TEMPORAL_API_API_VERSION`. The spec'd env var is
`TEMPORAL_API_VERSION`, so the field needs an explicit `validation_alias`
(pydantic-settings does **not** apply the prefix to aliased fields):

```python
    # Release version of the serving code, stamped into the Docker image from
    # the git tag (env TEMPORAL_API_VERSION via a build arg). None on
    # non-release builds → surfaced as null in responses. The alias avoids the
    # env_prefix doubling ("TEMPORAL_API_API_VERSION").
    api_version: str | None = Field(
        default=None, validation_alias="TEMPORAL_API_VERSION"
    )
```

Add the validator at the end of the class (after the `port` field):

```python
    @field_validator("api_version")
    @classmethod
    def _empty_api_version_is_none(cls, v: str | None) -> str | None:
        # ENV TEMPORAL_API_VERSION=${VERSION} with an absent build arg sets "".
        return v or None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_settings.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/temporal_model/api/settings.py tests/test_settings.py
git commit -m "feat(api): add api_version setting from TEMPORAL_API_VERSION"
```

---

### Task 2: Response schema — `version: {api, model}` replaces `model: {name, version}`

Breaking DTO change in `schemas.py`: `ModelInfo` becomes `Version` (two flat
`str | None` fields), and `to_response()` takes `api_version`/`model_version`
instead of `name`/`version`.

**Files:**
- Modify: `api/src/temporal_model/api/schemas.py`
- Test: `api/tests/test_schemas.py`

- [ ] **Step 1: Update the schema tests**

In `api/tests/test_schemas.py`, update every `to_response` call site — the
old keyword pair `name="m", version=<X>` becomes `model_version=<X>`, and
`api_version` is passed explicitly where the test asserts on it (it has no
default, so mechanical call sites pass `api_version=None`).

Replace `test_smoke_uses_max_kept_probability` (full new body — note the
expected dump):

```python
def test_smoke_uses_max_kept_probability():
    # Trigger tube (id 7) has the LOWER prob; reported value is the max (0.91).
    out = SimpleNamespace(
        is_positive=True,
        trigger_frame_index=3,
        details=_details([_tube(7, 0.62), _tube(2, 0.91)]),
    )
    resp = to_response(
        out, api_version="0.3.0", model_version="1.2.0", calibrated=True, verbose=False
    )
    dumped = resp.model_dump(exclude_unset=True)
    assert dumped == {
        "is_smoke": True,
        "probability": 0.91,
        "version": {"api": "0.3.0", "model": "1.2.0"},
    }
```

Mechanically update the remaining call sites (same line, same test
otherwise). The exact replacements:

| Test | Old call fragment | New call fragment |
|---|---|---|
| `test_negative_uses_max_kept_probability` | `name="m", version="1.2.0"` | `api_version=None, model_version="1.2.0"` |
| `test_negative_no_tubes_is_zero_when_calibrated` | `name="m", version="1.2.0"` | `api_version=None, model_version="1.2.0"` |
| `test_uncalibrated_probability_is_null` | `name="m", version=None` | `api_version=None, model_version=None` |
| `test_verbose_adds_details_block` | `name="m", version="1.2.0"` | `api_version=None, model_version="1.2.0"` |
| `test_verbose_surfaces_threshold_override` | `name="m",\n        version="1.2.0",` | `api_version=None,\n        model_version="1.2.0",` |
| `test_to_response_includes_profiling_when_verbose` (3 calls) | `name="m", version="1"` | `api_version=None, model_version="1"` |
| `test_verbose_details_map_num_tubes_outside_roi` | `name="m", version="1"` | `api_version=None, model_version="1"` |
| `test_verbose_details_num_tubes_outside_roi_is_strict` | `name="m", version="1"` | `api_version=None, model_version="1"` |

Append two new tests:

```python
def test_version_block_carries_nulls_independently():
    # Each identity is null on its own: api when not a release build, model
    # when the package is a legacy unstamped one.
    out = SimpleNamespace(
        is_positive=False, trigger_frame_index=None, details=_details([])
    )
    resp = to_response(
        out, api_version=None, model_version="1.2.0", calibrated=True, verbose=False
    )
    assert resp.version.api is None
    assert resp.version.model == "1.2.0"
    resp2 = to_response(
        out, api_version="0.3.0", model_version=None, calibrated=True, verbose=False
    )
    assert resp2.version.api == "0.3.0"
    assert resp2.version.model is None


def test_response_has_no_top_level_model_key():
    # The old model: {name, version} block is gone (breaking change, agreed
    # in the spec) — its content lives at version.model.
    out = SimpleNamespace(
        is_positive=False, trigger_frame_index=None, details=_details([])
    )
    resp = to_response(
        out, api_version="0.3.0", model_version="1.2.0", calibrated=True, verbose=False
    )
    assert "model" not in resp.model_dump(exclude_unset=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: every updated/new test FAILS with `TypeError: to_response() got an unexpected keyword argument 'api_version'`. The request-validation tests still PASS.

- [ ] **Step 3: Implement the schema change**

In `api/src/temporal_model/api/schemas.py`:

Replace the `ModelInfo` class (keep its position in the file):

```python
class Version(BaseModel):
    """Provenance of a prediction: the code release + the model release.

    ``api`` equals the Docker image tag (null on non-release builds);
    ``model`` is the packaged ``manifest.model_version`` (null on legacy
    unstamped packages). Together they fully identify what produced a result.
    """

    model_config = ConfigDict(protected_namespaces=())

    api: str | None
    model: str | None
```

Replace `PredictResponse` (the `model` field becomes `version`; the
`ConfigDict` moves to `Version`, which now owns the `model` field name):

```python
class PredictResponse(BaseModel):
    is_smoke: bool
    probability: float | None
    version: Version
    details: Details | None = None
```

Replace the `to_response` signature and the `kwargs` block (the `verbose`
branch and `_to_details` are unchanged):

```python
def to_response(
    out: Any,
    *,
    api_version: str | None,
    model_version: str | None,
    calibrated: bool,
    verbose: bool,
    threshold_overridden: bool = False,
    packaged_threshold: float | None = None,
    profiling: dict[str, Any] | None = None,
) -> PredictResponse:
    """Reshape a core model output into the public response DTO."""
    kwargs: dict[str, Any] = {
        "is_smoke": out.is_positive,
        "probability": _decision_probability(out.details, calibrated),
        "version": Version(api=api_version, model=model_version),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: all PASS. (`tests/test_app.py` is now broken — fixed in Task 3; do
not run the full suite yet.)

- [ ] **Step 5: Commit**

```bash
git add src/temporal_model/api/schemas.py tests/test_schemas.py
git commit -m "feat(api)!: group code+model provenance under version in /predict"
```

---

### Task 3: Wire the version through `app.py` (`/predict`, `/health`, OpenAPI)

**Files:**
- Modify: `api/src/temporal_model/api/app.py`
- Test: `api/tests/test_app.py`

- [ ] **Step 1: Update the app tests**

In `api/tests/test_app.py`:

Update `test_health_loaded`'s expected JSON:

```python
    assert r.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "bbox-tube-vit-dinov2",
        "model_version": "1.2.0",
        "api_version": None,
    }
```

Update `test_health_unavailable`'s expected JSON:

```python
    assert r.json() == {
        "status": "unavailable",
        "model_loaded": False,
        "model_name": None,
        "model_version": None,
        "api_version": None,
    }
```

Update `test_predict_default`'s expected JSON:

```python
    assert r.json() == {
        "is_smoke": True,
        "probability": 0.98,
        "version": {"api": None, "model": "1.2.0"},
    }
```

In `test_predict_verbose`, replace the `body["model"]` assertion line with:

```python
    assert body["version"] == {"api": None, "model": "1.2.0"}
```

Append two new tests (after `test_predict_verbose_surfaces_override`):

```python
def test_predict_reports_api_version(client, monkeypatch):
    # settings.api_version is read per request, so a monkeypatched value
    # must show up as version.api.
    monkeypatch.setattr(settings, "api_version", "0.3.0")
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 200
    assert r.json()["version"] == {"api": "0.3.0", "model": "1.2.0"}


def test_health_reports_api_version(client, monkeypatch):
    monkeypatch.setattr(settings, "api_version", "0.3.0")
    r = client.get("/health")
    assert r.json()["api_version"] == "0.3.0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py -v`
Expected: the updated and new tests FAIL (the route still calls
`to_response(name=..., version=...)`, which now raises `TypeError` →
surfaces as 500 `inference_error`; health lacks `api_version`). Unrelated
tests (auth, buckets, ROI, errors) PASS.

- [ ] **Step 3: Implement the wiring**

In `api/src/temporal_model/api/app.py`:

`HealthResponse` gains the field:

```python
class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    api_version: str | None = None
```

The app constructor stops hardcoding the stale version (OpenAPI requires a
string, hence the `"dev"` fallback; evaluated at import, which is fine —
the env var is baked into the image):

```python
app = FastAPI(
    title="Temporal Model API",
    version=settings.api_version or "dev",
    lifespan=lifespan,
)
```

`health()` reports it on both branches (it describes the service, not the
model):

```python
@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        return HealthResponse(
            status="unavailable",
            model_loaded=False,
            api_version=settings.api_version,
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=runner.name,
        model_version=runner.version,
        api_version=settings.api_version,
    )
```

The `/predict` route's `to_response` call (only the two version kwargs
change; `runner.name` is no longer passed — it stays in use for `/health`):

```python
            return to_response(
                out,
                api_version=settings.api_version,
                model_version=runner.version,
                calibrated=runner.calibrated,
                verbose=verbose,
                threshold_overridden=runner.threshold_overridden,
                packaged_threshold=runner.packaged_threshold,
                profiling=profiling,
            )
```

- [ ] **Step 4: Run the full API suite**

Run: `uv run pytest tests/ -v`
Expected: all PASS (integration tests skip without
`TEMPORAL_API_TEST_MODEL_PATH`).

- [ ] **Step 5: Lint**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/temporal_model/api/app.py tests/test_app.py
git commit -m "feat(api): report code version in /predict, /health and OpenAPI"
```

---

### Task 4: Bake the version into the image (Dockerfile + release workflow)

No runtime test covers this seam (spec: verified by hitting `/health` on the
next released image). Verification here is a local image build.

**Files:**
- Modify: `api/Dockerfile`
- Modify: `.github/workflows/push.yml` (repo root)

- [ ] **Step 1: Add the build arg to the Dockerfile**

In `api/Dockerfile`, after the `COPY api/models/model.zip /models/model.zip`
line and before `EXPOSE 8000`:

```dockerfile
# Stamp the release version (the git tag, == the image tag) into the runtime
# so the API reports it as version.api. Absent on local builds → the env var
# is empty and settings normalize it to null.
ARG VERSION
ENV TEMPORAL_API_VERSION=${VERSION}
```

- [ ] **Step 2: Pass the build arg in the release workflow**

In `.github/workflows/push.yml`, extend the `Build and push` step's `with:`
block (the `VERSION` env is already resolved from the git tag by the
"Resolve versions" step):

```yaml
      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: api/Dockerfile
          push: true
          build-args: |
            VERSION=${{ env.VERSION }}
          tags: |
            ${{ env.BACKEND_IMAGE_NAME }}:${{ env.VERSION }}
            ${{ env.BACKEND_IMAGE_NAME }}:latest
```

- [ ] **Step 3: Verify with a local build (requires `api/models/model.zip`; run `make fetch-model` from the repo root first if missing)**

From the repo root:

```bash
docker build -f api/Dockerfile --build-arg VERSION=9.9.9-test -t tm-api-version-test .
docker run --rm tm-api-version-test uv run --no-dev python -c \
  "from temporal_model.api.settings import Settings; print(Settings(_env_file=None).api_version)"
```

Expected output: `9.9.9-test`

Then the no-arg case:

```bash
docker build -f api/Dockerfile -t tm-api-version-test-noarg .
docker run --rm tm-api-version-test-noarg uv run --no-dev python -c \
  "from temporal_model.api.settings import Settings; print(Settings(_env_file=None).api_version)"
```

Expected output: `None`

Clean up: `docker rmi tm-api-version-test tm-api-version-test-noarg`

(If Docker or `model.zip` is unavailable in the execution environment, note
it and rely on the next release's `/health` check, per the spec.)

- [ ] **Step 4: Commit**

```bash
git add api/Dockerfile .github/workflows/push.yml
git commit -m "feat(ci): bake the release version into the API image"
```

---

### Task 5: Documentation

**Files:**
- Modify: `docs/model-versioning.md` (§1 table)
- Modify: `api/README.md` (endpoint summary)

- [ ] **Step 1: Update `docs/model-versioning.md`**

In §1 ("The version of a served model"), replace the table with:

```markdown
| Where | Value |
|---|---|
| Git tag / Docker image tag | repo version — `vX.Y.Z` / `pyronear/temporal-model-api:X.Y.Z` |
| `/predict` → `version.api` | repo version — `X.Y.Z`, baked into the image from the git tag (`null` on non-release builds) |
| `api/MODEL_VERSION` | **model** version the repo ships — `X.Y.Z` |
| `model.zip` manifest / `/predict` → `version.model` | `model_version: "X.Y.Z"` (matches `api/MODEL_VERSION`) |
```

And append to the end of the §1 prose paragraph (after "...recoverable from
the manifest's `provenance` block (below).")

```markdown
At runtime, every `/predict` response reports both identities in one block —
`version: {api, model}` — so a stored result is traceable to the exact
image and model that produced it (see
[`docs/specs/2026-06-11-api-version-in-response-design.md`](specs/2026-06-11-api-version-in-response-design.md)).
```

- [ ] **Step 2: Update `api/README.md`**

Line 10 (the `/health` bullet) becomes:

```markdown
- `GET /health` — readiness + loaded model name/version + API code version.
```

Line 16's response-shape fragment changes from
`returns { is_smoke, probability, model }` to:

```markdown
  returns `{ is_smoke, probability, version }` (`probability` = max kept-tube
```

with the surrounding sentence kept intact, and add after that sentence:

```markdown
  `version` is `{api, model}` — the code release (== the Docker image tag,
  `null` on non-release builds) and the packaged model release.
```

(Adjust the exact splice to the README's current sentence flow — keep the
existing `probability` explanation untouched.)

- [ ] **Step 3: Commit**

```bash
git add docs/model-versioning.md api/README.md
git commit -m "docs: document the version block in /predict responses"
```

---

## Done criteria

- `uv run pytest tests/ -v` green in `api/`.
- `/predict` body has `version: {api, model}`, no `model` key.
- `/health` includes `api_version`.
- Local Docker build with `--build-arg VERSION=x` reports `x`; without, `None`.
- Docs updated.
