# API Local Frame Files Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `/predict` serve frames from the local filesystem (shared volume on an edge box) instead of S3, via an optional `source` request field and a settings-only `frames_root`.

**Architecture:** A new optional `source: "s3" | "local"` on `PredictRequest` defaults to a new `settings.frame_source` (default `"s3"` → zero behavior change for existing deployments). A new `local.py` module (sibling of `s3.py`) resolves relative frame paths under `settings.frames_root` with a symlink-aware containment check and hands real file paths to the runner — no temp dir, no copy. The route branches on the effective source; response schema is untouched. Spec: `docs/specs/2026-06-11-api-local-frames-design.md`.

**Tech Stack:** FastAPI, Pydantic v2 (+ pydantic-settings), pytest, uv. All work is in the `api/` package.

**Context for the engineer:**
- Work from the git worktree at `.claude/worktrees/arthur+feat-api-local-frames` (branch `arthur/feat-api-local-frames`). All paths below are relative to the worktree root.
- Run tests from the `api/` directory: `cd api && uv run pytest ...`.
- `settings` is a module-level singleton (`temporal_model.api.settings.settings`); app tests monkeypatch attributes on it (see existing fixtures in `api/tests/test_app.py`).
- Commit messages: conventional commits (`feat(api): ...`), **no Claude co-author trailer**.

---

### Task 1: Settings — `frame_source` and `frames_root`

**Files:**
- Modify: `api/src/temporal_model/api/settings.py`
- Test: `api/tests/test_settings.py`

- [ ] **Step 1: Write the failing tests**

Append to `api/tests/test_settings.py`:

```python
def test_frame_source_default_s3():
    assert Settings(_env_file=None).frame_source == "s3"


def test_frame_source_env_override(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_FRAME_SOURCE", "local")
    assert Settings(_env_file=None).frame_source == "local"


def test_frame_source_rejects_unknown(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_FRAME_SOURCE", "ftp")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_frames_root_default_empty():
    assert Settings(_env_file=None).frames_root == ""


def test_frames_root_env_override(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_FRAMES_ROOT", "/data/frames")
    assert Settings(_env_file=None).frames_root == "/data/frames"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/test_settings.py -v`
Expected: the five new tests FAIL with `AttributeError: 'Settings' object has no attribute 'frame_source'` (or `ValidationError` not raised); existing tests PASS.

- [ ] **Step 3: Implement the settings fields**

In `api/src/temporal_model/api/settings.py`, add the import at the top:

```python
from typing import Literal
```

and add these fields after the `s3_endpoint_url` line (before `host`):

```python
    # Where /predict frames come from when a request omits its `source` field:
    # "s3" downloads keys from a bucket; "local" resolves relative paths under
    # `frames_root` (see docs/specs/2026-06-11-api-local-frames-design.md).
    frame_source: Literal["s3", "local"] = "s3"

    # Root directory for local frames. Required when serving local frames.
    # Settings-only by design — a request-supplied root would let callers
    # probe arbitrary server paths.
    frames_root: str = ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/test_settings.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/settings.py api/tests/test_settings.py
git commit -m "feat(api): add frame_source and frames_root settings"
```

---

### Task 2: Schema — optional `source` field on `PredictRequest`

**Files:**
- Modify: `api/src/temporal_model/api/schemas.py`
- Test: `api/tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Append to `api/tests/test_schemas.py`:

```python
def test_request_source_defaults_to_none():
    assert PredictRequest(frames=["a.jpg"]).source is None


@pytest.mark.parametrize("source", ["s3", "local"])
def test_request_accepts_source(source):
    assert PredictRequest(frames=["a.jpg"], source=source).source == source


def test_request_rejects_unknown_source():
    with pytest.raises(ValidationError):
        PredictRequest(frames=["a.jpg"], source="ftp")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/test_schemas.py -v`
Expected: `test_request_source_defaults_to_none` and `test_request_accepts_source` FAIL with `AttributeError: 'PredictRequest' object has no attribute 'source'`; `test_request_rejects_unknown_source` FAILS because Pydantic ignores the unknown field instead of raising. Existing tests PASS.

- [ ] **Step 3: Implement the field**

In `api/src/temporal_model/api/schemas.py` (`Literal` is already imported), add to `PredictRequest` directly after the `frames` field:

```python
    # Where `frames` live: "s3" (keys in a bucket) or "local" (relative paths
    # under the server's frames_root). None → the server's configured default
    # (settings.frame_source).
    source: Literal["s3", "local"] | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/test_schemas.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/schemas.py api/tests/test_schemas.py
git commit -m "feat(api): add optional source field to PredictRequest"
```

---

### Task 3: `local.py` — frame resolver

**Files:**
- Create: `api/src/temporal_model/api/local.py`
- Create: `api/tests/test_local.py`

- [ ] **Step 1: Write the failing tests**

Create `api/tests/test_local.py`:

```python
from pathlib import Path

import pytest

from temporal_model.api.errors import FrameNotFound, InvalidRequest
from temporal_model.api.local import resolve_frames

FRAMES = [
    "cam12/2023-05-23/adf_2023-05-23T17-18-01.jpg",
    "cam12/2023-05-23/adf_2023-05-23T17-18-31.jpg",
]


def _make_frames(root: Path, frames=FRAMES) -> None:
    for frame in frames:
        p = root / frame
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff\xe0jpegbytes")


def test_resolve_preserves_order_and_points_at_real_files(tmp_path):
    _make_frames(tmp_path)
    paths = resolve_frames(tmp_path, FRAMES)
    # The real files, in request order — no copy.
    assert paths == [(tmp_path / f).resolve() for f in FRAMES]
    assert all(p.read_bytes() == b"\xff\xd8\xff\xe0jpegbytes" for p in paths)


def test_missing_file_raises_frame_not_found(tmp_path):
    with pytest.raises(FrameNotFound):
        resolve_frames(tmp_path, ["cam12/missing.jpg"])


def test_directory_raises_frame_not_found(tmp_path):
    (tmp_path / "cam12").mkdir()
    with pytest.raises(FrameNotFound):
        resolve_frames(tmp_path, ["cam12"])


def test_absolute_path_raises_invalid_request(tmp_path):
    _make_frames(tmp_path)
    inside = tmp_path / FRAMES[0]
    # Even an absolute path pointing inside the root is rejected: the contract
    # is relative identifiers only.
    with pytest.raises(InvalidRequest):
        resolve_frames(tmp_path, [str(inside)])


def test_dotdot_raises_invalid_request_even_when_inside(tmp_path):
    _make_frames(tmp_path)
    # Resolves inside the root, but `..` segments are rejected outright.
    with pytest.raises(InvalidRequest):
        resolve_frames(tmp_path, [f"cam12/../{FRAMES[0]}"])


def test_escaping_dotdot_raises_invalid_request(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.jpg").write_bytes(b"x")
    with pytest.raises(InvalidRequest):
        resolve_frames(root, ["../secret.jpg"])


def test_symlink_escaping_root_raises_invalid_request(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")
    (root / "link.jpg").symlink_to(outside)
    with pytest.raises(InvalidRequest):
        resolve_frames(root, ["link.jpg"])


def test_error_message_echoes_request_string_not_server_path(tmp_path):
    with pytest.raises(FrameNotFound) as exc_info:
        resolve_frames(tmp_path, ["cam12/missing.jpg"])
    assert "cam12/missing.jpg" in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/test_local.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'temporal_model.api.local'`.

- [ ] **Step 3: Implement the resolver**

Create `api/src/temporal_model/api/local.py`:

```python
"""Local-filesystem frame resolution.

Resolves request frame paths (relative identifiers) under the
server-configured ``frames_root`` and returns them in request order. Unlike
the S3 path, nothing is copied: the returned paths point at the real files.
The root comes from settings only — never from the request — so a request
cannot reference paths outside it (see
docs/specs/2026-06-11-api-local-frames-design.md, decision 2).
"""

from pathlib import Path

from .errors import FrameNotFound, InvalidRequest


def resolve_frames(root: Path, frames: list[str]) -> list[Path]:
    """Resolve ``frames`` under ``root``, in request order.

    Rejects absolute paths and ``..`` segments outright, and anything whose
    resolved path (symlinks followed) lands outside ``root``. A missing file
    raises :class:`FrameNotFound` — the same error a missing S3 key maps to.
    Error messages echo the request string, never the resolved server path.
    """
    root = root.resolve()
    paths: list[Path] = []
    for frame in frames:
        rel = Path(frame)
        if rel.is_absolute() or ".." in rel.parts:
            raise InvalidRequest(
                f"frame must be a relative path without '..': {frame!r}"
            )
        resolved = (root / rel).resolve()
        if not resolved.is_relative_to(root):
            raise InvalidRequest(f"frame escapes the frames root: {frame!r}")
        if not resolved.is_file():
            raise FrameNotFound(f"frame not found: {frame}")
        paths.append(resolved)
    return paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/test_local.py -v`
Expected: all 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/local.py api/tests/test_local.py
git commit -m "feat(api): add local frame resolver"
```

---

### Task 4: Route — branch `/predict` on the effective source

**Files:**
- Modify: `api/src/temporal_model/api/app.py`
- Test: `api/tests/test_app.py`

- [ ] **Step 1: Extend `FakeRunner` to record paths**

In `api/tests/test_app.py`, modify `FakeRunner` so tests can assert which paths reached the runner. In `__init__`, after `self.roi = None`, add:

```python
        self.paths = None
```

and in `predict`, after `self.roi = roi`, add:

```python
        self.paths = paths
```

- [ ] **Step 2: Write the failing app tests**

Append to `api/tests/test_app.py`:

```python
@pytest.fixture
def local_client(monkeypatch, tmp_path):
    # An edge-box style deployment: frame_source=local, frames on a shared
    # volume (tmp_path), no S3 involved.
    monkeypatch.setattr(settings, "frame_source", "local")
    monkeypatch.setattr(settings, "frames_root", str(tmp_path))
    for key in KEYS:
        p = tmp_path / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff\xe0jpeg")
    with TestClient(app) as c:
        c.app.state.runner = FakeRunner(output=_smoke_output())
        yield c


def test_predict_local_default_source(local_client, tmp_path):
    # `source` omitted follows settings.frame_source="local"; the runner gets
    # the real files under the root, in request order — no copy.
    r = local_client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 200
    assert r.json()["is_smoke"] is True
    runner = local_client.app.state.runner
    assert runner.paths == [(tmp_path / k).resolve() for k in KEYS]


def test_predict_local_explicit_source(local_client):
    r = local_client.post("/predict", json={"frames": KEYS, "source": "local"})
    assert r.status_code == 200


def test_predict_local_rejects_bucket(local_client):
    r = local_client.post("/predict", json={"frames": KEYS, "bucket": "some-bucket"})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request"
    assert "bucket" in r.json()["detail"]


def test_predict_local_without_root_400(local_client, monkeypatch):
    monkeypatch.setattr(settings, "frames_root", "")
    r = local_client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request"
    assert "TEMPORAL_API_FRAMES_ROOT" in r.json()["detail"]


def test_predict_local_no_root_400_takes_precedence_over_model(
    local_client, monkeypatch
):
    # Mirrors test_predict_no_bucket_400_takes_precedence_over_model: the
    # prerequisite check runs before the model-loaded check.
    monkeypatch.setattr(settings, "frames_root", "")
    local_client.app.state.runner = None
    r = local_client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request"


def test_predict_local_missing_frame_404(local_client):
    r = local_client.post("/predict", json={"frames": ["cam12/missing.jpg"]})
    assert r.status_code == 404
    assert r.json()["code"] == "frame_not_found"


def test_predict_local_traversal_400(local_client):
    r = local_client.post("/predict", json={"frames": ["../etc/passwd"]})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request"


def test_predict_explicit_s3_same_as_omitted(client):
    # On an s3-default server, explicit source="s3" behaves identically to
    # omitting it.
    r = client.post("/predict", json={"frames": KEYS, "source": "s3"})
    assert r.status_code == 200
    assert r.json()["is_smoke"] is True


def test_predict_source_local_on_s3_server_needs_root(client):
    # The s3-mode `client` fixture has no frames_root configured: an explicit
    # local request is a clear 400, not a confusing fallback.
    r = client.post("/predict", json={"frames": KEYS, "source": "local"})
    assert r.status_code == 400
    assert "TEMPORAL_API_FRAMES_ROOT" in r.json()["detail"]


def test_predict_local_profiling_stage(local_client, monkeypatch):
    monkeypatch.setattr(settings, "profile", True)
    r = local_client.post("/predict?verbose=true", json={"frames": KEYS})
    assert r.status_code == 200
    prof = r.json()["details"]["profiling"]
    assert "local_resolve" in prof["stages_ms"]
    assert "s3_fetch" not in prof["stages_ms"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/test_app.py -v`
Expected: the new `local_*` tests FAIL (local requests are treated as S3 — 400 "no S3 bucket" or S3 errors); `test_predict_explicit_s3_same_as_omitted` may already PASS (source field exists and s3 is the default path). All pre-existing tests PASS.

- [ ] **Step 4: Implement the route branch**

In `api/src/temporal_model/api/app.py`:

1. Change the contextlib import (line 6) to:

```python
from contextlib import ExitStack, asynccontextmanager
```

2. Add the resolver import next to the s3 import:

```python
from .local import resolve_frames
```

3. Replace the body of `predict` (everything from `bucket = body.bucket or settings.s3_bucket` through the end of the function) with:

```python
    source = body.source or settings.frame_source
    if source == "local":
        if body.bucket is not None:
            raise InvalidRequest("bucket is not valid with local frames")
        if not settings.frames_root:
            raise InvalidRequest(
                "local frames not enabled: set TEMPORAL_API_FRAMES_ROOT"
            )
        bucket = None
    else:
        bucket = body.bucket or settings.s3_bucket
        if not bucket:
            raise InvalidRequest(
                "no S3 bucket: set request 'bucket' or TEMPORAL_API_S3_BUCKET"
            )

    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        raise ModelNotLoaded("model is not loaded")

    with ExitStack() as stack:
        try:
            # Timer carries the model device so GPU/MPS stages are synced for
            # honest timing; on the CPU serving target this is a no-op.
            timer = StageTimer(settings.device) if settings.profile else None
            profile: dict | None = {} if settings.profile else None

            if source == "local":
                # Local frames are read in place — no temp dir, no copy.
                with stage_ctx(timer, "local_resolve"):
                    paths = resolve_frames(Path(settings.frames_root), body.frames)
            else:
                # The temp dir must outlive runner.predict — frames are read
                # during inference.
                tmp = stack.enter_context(tempfile.TemporaryDirectory())
                with stage_ctx(timer, "s3_fetch"):
                    # fetch_frames is blocking boto3 I/O — run it off the
                    # event loop.
                    paths = await run_in_threadpool(
                        fetch_frames,
                        request.app.state.s3_client,
                        bucket,
                        body.frames,
                        Path(tmp),
                    )

            out = await runner.predict(
                paths, roi=body.roi_xyxyn, timer=timer, profile=profile
            )

            profiling = None
            if timer is not None:
                stages = timer.as_dict()
                profiling = {
                    "stages_ms": stages,
                    "total_ms": round(sum(stages.values()), 3),
                    **(profile or {}),
                }
                logger.info("profile %s", json.dumps(profiling))

            return to_response(
                out,
                name=runner.name,
                version=runner.version,
                calibrated=runner.calibrated,
                verbose=verbose,
                threshold_overridden=runner.threshold_overridden,
                packaged_threshold=runner.packaged_threshold,
                profiling=profiling,
            )
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface as inference_error
            raise InferenceError(str(exc)) from exc
```

Notes for the implementer:
- The `s3_client = request.app.state.s3_client` local variable from the old body is gone — the client is read inline in the s3 branch only. Remove the now-unused assignment; do not remove the `make_s3_client` call in `lifespan` (the client stays constructed at startup on purpose — boto3 does not touch the network at construction).
- The prerequisite checks stay **before** the model-loaded check, preserving the precedence the existing `test_predict_no_bucket_400_takes_precedence_over_model` asserts.
- The `from .s3 import fetch_frames, make_s3_client` import is unchanged.

- [ ] **Step 5: Run the full app test file**

Run: `cd api && uv run pytest tests/test_app.py -v`
Expected: all PASS — every pre-existing test unmodified (the back-compat proof from the spec) plus the new local tests.

- [ ] **Step 6: Commit**

```bash
git add api/src/temporal_model/api/app.py api/tests/test_app.py
git commit -m "feat(api): serve /predict frames from the local filesystem"
```

---

### Task 5: Document the local frame source

**Files:**
- Modify: `api/README.md`

- [ ] **Step 1: Update the endpoints bullet**

In `api/README.md`, replace the `POST /predict` bullet (lines 11–20) with:

```markdown
- `POST /predict` — body `{ "frames": [...], "source": "s3" | "local",
  "bucket": "<name>", "roi_xyxyn": [x_min, y_min, x_max, y_max] }`
  (ordered frames; `source` optional, falls back to `FRAME_SOURCE` — with
  `s3`, frames are S3 keys and `bucket` optionally overrides `S3_BUCKET`;
  with `local`, frames are relative paths under `FRAMES_ROOT` and `bucket`
  is invalid; `roi_xyxyn` optional normalized region of interest — tubes
  with no real detection intersecting it are dropped before scoring);
  returns `{ is_smoke, probability, model }` (`probability` = max kept-tube
  calibrated probability, `null` if uncalibrated).
  `POST /predict?verbose=true` adds a `details` block (decision, preprocessing,
  per-tube tracks). See `docs/specs/2026-06-02-api-service-design.md` and
  `docs/specs/2026-06-11-api-local-frames-design.md` for the full contract.
```

- [ ] **Step 2: Update the configuration section**

Replace the "Configuration via env vars" paragraph (lines 36–42) with:

```markdown
Configuration via env vars (prefix `TEMPORAL_API_`): `MODEL_PATH`, `DEVICE`,
`CALIBRATOR_THRESHOLD`, `TOKEN`, `FRAME_SOURCE`, `FRAMES_ROOT`, `S3_BUCKET`,
`S3_REGION`, `S3_ENDPOINT_URL` (empty = real AWS; set for OVH or MinIO),
`HOST`, `PORT`. AWS/OVH/MinIO credentials come from the standard boto3 chain
(env vars / IAM role). `S3_BUCKET` is an optional default; a request may
override it per call with its `bucket` field (needed for alert-api stacks
whose per-org bucket names are not known ahead of time). A request with
neither is rejected with `400 invalid_request`.

`FRAME_SOURCE` (default `s3`) selects where `/predict` frames come from when
a request omits its optional `source` field. With `local` (an edge box whose
frames sit on a shared volume), `frames` are relative paths resolved under
`FRAMES_ROOT`; `FRAMES_ROOT` is settings-only by design — a request-supplied
root would let callers probe arbitrary server paths — and absolute paths or
`..` segments are rejected with `400 invalid_request`. A missing file is the
same `404 frame_not_found` as a missing S3 key, and local requests skip the
S3 download entirely (frames are read in place).
```

- [ ] **Step 3: Commit**

```bash
git add api/README.md
git commit -m "docs(api): document local frame source"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run the full api suite**

Run: `make -C api test`
Expected: all tests pass (115 pre-existing + 1 skipped, plus the ~26 added in Tasks 1–4), zero failures, no pre-existing test modified other than the `FakeRunner` path-recording extension.

- [ ] **Step 2: Lint**

Run: `make -C api lint`
Expected: clean (ruff reports no issues).

- [ ] **Step 3: Verify the diff is surgical**

Run: `git diff main --stat`
Expected: only these files changed —
`api/src/temporal_model/api/{settings,schemas,app,local}.py`,
`api/tests/{test_settings,test_schemas,test_app,test_local}.py`,
`api/README.md`, `docs/specs/2026-06-11-api-local-frames-design.md`,
`docs/plans/2026-06-11-api-local-frames.md`.
