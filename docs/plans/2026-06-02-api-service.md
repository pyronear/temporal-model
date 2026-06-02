# Temporal Model API Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the `api` package scaffold into a runnable FastAPI service that loads a packaged `model.zip`, fetches an ordered frame sequence from S3 by key, runs the bundled temporal smoke pipeline, and returns a reshaped JSON verdict (with opt-in `?verbose=true` details).

**Architecture:** Small, layered modules inside `temporal_model.api`: `settings` (config), `errors` (domain exceptions → HTTP), `s3` (fetch frames to a temp dir), `model_runner` (load `model.zip` + serialized inference), `schemas` (request/response DTOs + mapper), and `app` (FastAPI wiring). The real model class from `temporal_model.core` is imported lazily so the API builds and unit-tests now (model mocked); a real end-to-end test is gated on a `model.zip` being present.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, pydantic / pydantic-settings, boto3 (S3, works against AWS/OVH/MinIO via `endpoint_url`), pyyaml (manifest read); tests use pytest + `moto` (mocked S3) + FastAPI `TestClient`.

**Spec:** `docs/specs/2026-06-02-api-service-design.md`

All commands run from the `api/` directory unless noted. Each task ends with a commit. Do **not** add Claude co-authorship trailers to commits.

---

### Task 1: Dependencies and settings

**Files:**
- Modify: `api/pyproject.toml`
- Modify: `api/src/temporal_model/api/settings.py`
- Test: `api/tests/test_settings.py`

- [ ] **Step 1: Add runtime and dev dependencies**

In `api/pyproject.toml`, extend the `dependencies` list and the `dev` group:

```toml
dependencies = [
    "temporal-model-core",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pydantic-settings>=2.2",
    "boto3>=1.34",
    "pyyaml>=6.0",
]
```

```toml
[dependency-groups]
dev = [
    "httpx>=0.27",
    "pytest>=8.0",
    "ruff>=0.9",
    "moto[s3]>=5.0",
]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: resolves and installs boto3, pyyaml, moto without error.

- [ ] **Step 3: Write the failing settings test**

Create `api/tests/test_settings.py`:

```python
from temporal_model.api.settings import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.model_path == "/models/model.zip"
    assert s.device is None
    assert s.s3_bucket == ""
    assert s.s3_region is None
    assert s.s3_endpoint_url is None
    assert s.port == 8000


def test_env_override(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_S3_BUCKET", "pyro-frames")
    monkeypatch.setenv("TEMPORAL_API_S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("TEMPORAL_API_DEVICE", "cpu")
    s = Settings(_env_file=None)
    assert s.s3_bucket == "pyro-frames"
    assert s.s3_endpoint_url == "http://minio:9000"
    assert s.device == "cpu"
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/test_settings.py -v`
Expected: FAIL — `Settings` has no `device` / `s3_bucket` attributes.

- [ ] **Step 5: Extend `Settings`**

Replace the body of `api/src/temporal_model/api/settings.py` with:

```python
"""Runtime configuration for the API, read from ``TEMPORAL_API_*`` env vars."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TEMPORAL_API_",
        protected_namespaces=(),
    )

    model_path: str = "/models/model.zip"
    device: str | None = None

    s3_bucket: str = ""
    s3_region: str | None = None
    s3_endpoint_url: str | None = None

    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_settings.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add api/pyproject.toml api/uv.lock api/src/temporal_model/api/settings.py api/tests/test_settings.py
git commit -m "feat(api): add boto3/pyyaml deps and S3/device settings"
```

---

### Task 2: Domain errors

**Files:**
- Create: `api/src/temporal_model/api/errors.py`
- Test: `api/tests/test_errors.py`

- [ ] **Step 1: Write the failing test**

Create `api/tests/test_errors.py`:

```python
from temporal_model.api.errors import (
    ApiError,
    FrameNotFound,
    InferenceError,
    ModelNotLoaded,
    S3Unavailable,
)


def test_error_codes_and_status():
    assert (FrameNotFound("x").status_code, FrameNotFound("x").code) == (404, "frame_not_found")
    assert (S3Unavailable("x").status_code, S3Unavailable("x").code) == (502, "s3_unavailable")
    assert (ModelNotLoaded("x").status_code, ModelNotLoaded("x").code) == (503, "model_not_loaded")
    assert (InferenceError("x").status_code, InferenceError("x").code) == (500, "inference_error")


def test_detail_is_carried():
    err = FrameNotFound("missing key cam12/a.jpg")
    assert err.detail == "missing key cam12/a.jpg"
    assert isinstance(err, ApiError)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_errors.py -v`
Expected: FAIL — module `errors` does not exist.

- [ ] **Step 3: Implement `errors.py`**

Create `api/src/temporal_model/api/errors.py`:

```python
"""Domain errors mapped to HTTP responses by the FastAPI app.

Each error carries a stable machine-readable ``code`` and a human ``detail``.
The app's exception handler renders them as ``{"detail": ..., "code": ...}``.
"""


class ApiError(Exception):
    """Base class for errors that map to a specific HTTP status + code."""

    status_code: int = 500
    code: str = "error"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class FrameNotFound(ApiError):
    status_code = 404
    code = "frame_not_found"


class S3Unavailable(ApiError):
    status_code = 502
    code = "s3_unavailable"


class ModelNotLoaded(ApiError):
    status_code = 503
    code = "model_not_loaded"


class InferenceError(ApiError):
    status_code = 500
    code = "inference_error"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_errors.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/errors.py api/tests/test_errors.py
git commit -m "feat(api): add domain error types with HTTP status + codes"
```

---

### Task 3: S3 frame fetching

**Files:**
- Create: `api/src/temporal_model/api/s3.py`
- Test: `api/tests/test_s3.py`

The fetcher downloads each key into a per-index subdirectory of a temp dir
(`<tmp>/0000/<basename>`), preserving the key's exact basename (the model parses
timestamps from it) and returning paths in the request order.

- [ ] **Step 1: Write the failing test**

Create `api/tests/test_s3.py`:

```python
from pathlib import Path

import boto3
import botocore.exceptions as botoexc
import pytest
from moto import mock_aws

from temporal_model.api.errors import FrameNotFound, S3Unavailable
from temporal_model.api.s3 import fetch_frames

BUCKET = "test-frames"
KEYS = [
    "cam12/2023-05-23/adf_2023-05-23T17-18-01.jpg",
    "cam12/2023-05-23/adf_2023-05-23T17-18-31.jpg",
]


@mock_aws
def test_fetch_preserves_order_and_basenames(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)
    for key in KEYS:
        client.put_object(Bucket=BUCKET, Key=key, Body=b"\xff\xd8\xff\xe0jpegbytes")

    paths = fetch_frames(client, BUCKET, KEYS, tmp_path)

    assert [p.name for p in paths] == [
        "adf_2023-05-23T17-18-01.jpg",
        "adf_2023-05-23T17-18-31.jpg",
    ]
    assert all(p.exists() and p.read_bytes() for p in paths)
    # order is preserved exactly as requested
    assert paths == sorted(paths, key=lambda p: p.parent.name)


@mock_aws
def test_missing_key_raises_frame_not_found(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)
    with pytest.raises(FrameNotFound):
        fetch_frames(client, BUCKET, ["cam12/missing.jpg"], tmp_path)


def test_endpoint_failure_raises_s3_unavailable(tmp_path, monkeypatch):
    class _Boom:
        def download_file(self, *a, **k):
            raise botoexc.EndpointConnectionError(endpoint_url="http://nope")

    with pytest.raises(S3Unavailable):
        fetch_frames(_Boom(), BUCKET, ["cam12/a.jpg"], tmp_path)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_s3.py -v`
Expected: FAIL — module `s3` does not exist.

- [ ] **Step 3: Implement `s3.py`**

Create `api/src/temporal_model/api/s3.py`:

```python
"""S3 frame fetching.

Downloads frame objects (by key) into a per-request temp directory, preserving
each key's basename and the requested order. Credentials and bucket come from
settings / the boto3 chain — never from the request.
"""

from pathlib import Path

import boto3
import botocore.exceptions as botoexc

from .errors import FrameNotFound, S3Unavailable
from .settings import Settings

_NOT_FOUND_CODES = {"404", "NoSuchKey", "NoSuchBucket"}


def make_s3_client(settings: Settings):
    """Build a boto3 S3 client. ``endpoint_url`` empty → real AWS."""
    return boto3.client(
        "s3",
        region_name=settings.s3_region or None,
        endpoint_url=settings.s3_endpoint_url or None,
    )


def fetch_frames(
    s3_client, bucket: str, keys: list[str], dest_dir: Path
) -> list[Path]:
    """Download ``keys`` from ``bucket`` into ``dest_dir``, in order.

    Each key is written to ``dest_dir/<NNNN>/<basename>`` so basenames are
    preserved exactly and never collide. Returns local paths in request order.
    """
    paths: list[Path] = []
    for i, key in enumerate(keys):
        basename = key.rsplit("/", 1)[-1]
        subdir = dest_dir / f"{i:04d}"
        subdir.mkdir(parents=True, exist_ok=True)
        dst = subdir / basename
        try:
            s3_client.download_file(bucket, key, str(dst))
        except botoexc.ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in _NOT_FOUND_CODES:
                raise FrameNotFound(f"frame not found: {key}") from exc
            raise S3Unavailable(f"S3 error fetching {key}: {error_code}") from exc
        except botoexc.EndpointConnectionError as exc:
            raise S3Unavailable(f"S3 endpoint unreachable: {exc}") from exc
        paths.append(dst)
    return paths
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_s3.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/s3.py api/tests/test_s3.py
git commit -m "feat(api): add S3 frame fetcher with moto-tested error mapping"
```

---

### Task 4: Model runner (load + serialized inference)

**Files:**
- Create: `api/src/temporal_model/api/model_runner.py`
- Test: `api/tests/test_model_runner.py`

`read_manifest` reads `manifest.yaml` from the zip for `name` (variant),
`version` (model_version, `None` if absent), and `calibrated` (whether a
`logistic_calibrator` pointer exists). The real model class is imported lazily
in `_load_core_model` so this module imports cleanly while `core` is still a
stub, and tests can patch it.

- [ ] **Step 1: Write the failing test**

Create `api/tests/test_model_runner.py`:

```python
import asyncio
import zipfile
from types import SimpleNamespace

import yaml

from temporal_model.api import model_runner as mr
from temporal_model.api.model_runner import ModelRunner, read_manifest


def _make_package(tmp_path, manifest: dict):
    path = tmp_path / "model.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.yaml", yaml.safe_dump(manifest))
    return path


def test_read_manifest_calibrated(tmp_path):
    path = _make_package(
        tmp_path,
        {"variant": "bbox-tube-vit-dinov2", "model_version": "1.2.0",
         "logistic_calibrator": "logistic_calibrator.json"},
    )
    meta = read_manifest(path)
    assert meta == {"name": "bbox-tube-vit-dinov2", "version": "1.2.0", "calibrated": True}


def test_read_manifest_legacy_uncalibrated(tmp_path):
    path = _make_package(tmp_path, {"variant": "old-model"})
    meta = read_manifest(path)
    assert meta == {"name": "old-model", "version": None, "calibrated": False}


def test_load_uses_lazy_core_model(tmp_path, monkeypatch):
    path = _make_package(
        tmp_path, {"variant": "m", "model_version": "9", "logistic_calibrator": "c.json"}
    )
    fake_model = SimpleNamespace(name="fake")
    monkeypatch.setattr(mr, "_load_core_model", lambda p, d: fake_model)

    runner = ModelRunner.load(path, device="cpu")

    assert runner.name == "m"
    assert runner.version == "9"
    assert runner.calibrated is True
    assert runner._model is fake_model


def test_predict_delegates_to_model():
    captured = {}

    class FakeModel:
        def predict_sequence(self, paths):
            captured["paths"] = paths
            return "OUT"

    runner = ModelRunner(FakeModel(), name="m", version="1", calibrated=True)
    result = asyncio.run(runner.predict(["a.jpg", "b.jpg"]))

    assert result == "OUT"
    assert captured["paths"] == ["a.jpg", "b.jpg"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_model_runner.py -v`
Expected: FAIL — module `model_runner` does not exist.

- [ ] **Step 3: Implement `model_runner.py`**

Create `api/src/temporal_model/api/model_runner.py`:

```python
"""Model lifecycle and serialized inference.

Loads a packaged ``model.zip`` once, reads its manifest for display metadata,
and runs ``predict_sequence`` off the event loop behind a lock (GPU inference is
not reentrant). The concrete model class lives in ``temporal_model.core`` and is
imported lazily so this module loads while ``core`` is still being migrated.
"""

import asyncio
import zipfile
from pathlib import Path
from typing import Any

import yaml
from starlette.concurrency import run_in_threadpool


def read_manifest(package_path: Path) -> dict[str, Any]:
    """Read display metadata from the package manifest.

    Returns ``{"name", "version", "calibrated"}``. ``version`` is ``None`` for
    legacy packages without a ``model_version`` field.
    """
    with zipfile.ZipFile(package_path) as zf:
        manifest = yaml.safe_load(zf.read("manifest.yaml"))
    return {
        "name": manifest.get("variant"),
        "version": manifest.get("model_version"),
        "calibrated": "logistic_calibrator" in manifest,
    }


def _load_core_model(package_path: Path, device: str | None) -> Any:
    """Lazily import and instantiate the core model from a package."""
    from temporal_model.core.model import BboxTubeTemporalModel  # noqa: PLC0415

    return BboxTubeTemporalModel.from_package(package_path, device=device)


class ModelRunner:
    """Holds the loaded model and serializes inference calls."""

    def __init__(self, model: Any, *, name: str, version: str | None, calibrated: bool) -> None:
        self._model = model
        self.name = name
        self.version = version
        self.calibrated = calibrated
        self._lock = asyncio.Lock()

    @classmethod
    def load(cls, package_path: Path, device: str | None) -> "ModelRunner":
        meta = read_manifest(package_path)
        model = _load_core_model(package_path, device)
        return cls(model, **meta)

    async def predict(self, frame_paths: list[Path]) -> Any:
        """Run ``predict_sequence`` in a worker thread, one call at a time."""
        async with self._lock:
            return await run_in_threadpool(self._model.predict_sequence, frame_paths)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_model_runner.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/model_runner.py api/tests/test_model_runner.py
git commit -m "feat(api): add model runner (manifest read + lazy core load + locked inference)"
```

---

### Task 5: Request/response schemas and mapper

**Files:**
- Create: `api/src/temporal_model/api/schemas.py`
- Test: `api/tests/test_schemas.py`

The mapper consumes a model output object exposing `is_positive`,
`trigger_frame_index`, and `details` (a dict matching `core`'s
`BboxTubeDetails`). It computes the top-level `probability` per the spec rule and
builds the verbose `details` block only when requested.

- [ ] **Step 1: Write the failing test**

Create `api/tests/test_schemas.py`:

```python
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from temporal_model.api.schemas import PredictRequest, to_response


def _details(kept):
    return {
        "decision": {"aggregation": "max_logit", "threshold": 0.5,
                     "trigger_tube_id": kept[0]["tube_id"] if kept else None},
        "preprocessing": {"num_frames_input": 30, "num_truncated": 0,
                          "padded_frame_indices": []},
        "tubes": {"num_candidates": len(kept) + 1, "kept": kept},
    }


def _tube(tube_id, prob):
    return {
        "tube_id": tube_id, "start_frame": 2, "end_frame": 12, "logit": 3.4,
        "probability": prob, "first_crossing_frame": 3,
        "entries": [
            {"frame_idx": 2, "bbox": [1.0, 2.0, 3.0, 4.0], "is_gap": False, "confidence": 0.8},
            {"frame_idx": 3, "bbox": None, "is_gap": True, "confidence": None},
        ],
    }


def test_request_rejects_empty():
    with pytest.raises(ValidationError):
        PredictRequest(frames=[])


def test_request_rejects_scheme():
    with pytest.raises(ValidationError):
        PredictRequest(frames=["s3://bucket/a.jpg"])


def test_smoke_default_uses_trigger_tube_probability():
    out = SimpleNamespace(is_positive=True, trigger_frame_index=3,
                          details=_details([_tube(7, 0.98)]))
    resp = to_response(out, name="m", version="1.2.0", calibrated=True, verbose=False)
    dumped = resp.model_dump(exclude_unset=True)
    assert dumped == {
        "is_smoke": True, "probability": 0.98, "trigger_frame_index": 3,
        "model": {"name": "m", "version": "1.2.0"},
    }


def test_negative_uses_max_kept_probability():
    out = SimpleNamespace(is_positive=False, trigger_frame_index=None,
                          details=_details([_tube(1, 0.1), _tube(2, 0.41)]))
    resp = to_response(out, name="m", version="1.2.0", calibrated=True, verbose=False)
    assert resp.probability == 0.41
    assert resp.is_smoke is False
    assert resp.trigger_frame_index is None


def test_negative_no_tubes_is_zero_when_calibrated():
    out = SimpleNamespace(is_positive=False, trigger_frame_index=None, details=_details([]))
    resp = to_response(out, name="m", version="1.2.0", calibrated=True, verbose=False)
    assert resp.probability == 0.0


def test_uncalibrated_probability_is_null():
    out = SimpleNamespace(is_positive=False, trigger_frame_index=None,
                          details=_details([_tube(1, None)]))
    resp = to_response(out, name="m", version=None, calibrated=False, verbose=False)
    assert resp.probability is None


def test_verbose_adds_details_block():
    out = SimpleNamespace(is_positive=True, trigger_frame_index=3,
                          details=_details([_tube(7, 0.98)]))
    resp = to_response(out, name="m", version="1.2.0", calibrated=True, verbose=True)
    dumped = resp.model_dump(exclude_unset=True)
    assert dumped["details"]["decision"] == {
        "aggregation": "max_logit", "threshold": 0.5, "trigger_tube_id": 7}
    assert dumped["details"]["preprocessing"]["num_tube_candidates"] == 2
    assert dumped["details"]["tubes"][0]["tube_id"] == 7
    assert dumped["details"]["tubes"][0]["entries"][1]["bbox"] is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL — module `schemas` does not exist.

- [ ] **Step 3: Implement `schemas.py`**

Create `api/src/temporal_model/api/schemas.py`:

```python
"""Public request/response DTOs and the mapper from the core model output.

The default response is the lean verdict; ``?verbose=true`` adds a ``details``
block. ``details`` is only set when verbose, so the route serializes with
``exclude_unset=True`` to omit it otherwise (while keeping explicit ``null``s).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


class PredictRequest(BaseModel):
    frames: list[str]

    @field_validator("frames")
    @classmethod
    def _validate_frames(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("frames must contain at least one S3 key")
        for key in v:
            if "://" in key:
                raise ValueError(
                    f"frame key must be a bare S3 key, not a URL: {key!r}"
                )
        return v


class FrameEntry(BaseModel):
    frame_idx: int
    bbox: tuple[float, float, float, float] | None
    is_gap: bool
    confidence: float | None


class Tube(BaseModel):
    tube_id: int
    start_frame: int
    end_frame: int
    logit: float
    probability: float | None
    first_crossing_frame: int | None
    entries: list[FrameEntry]


class Decision(BaseModel):
    aggregation: Literal["max_logit", "logistic"]
    threshold: float
    trigger_tube_id: int | None


class Preprocessing(BaseModel):
    num_frames_input: int
    num_truncated: int
    padded_frame_indices: list[int]
    num_tube_candidates: int


class Details(BaseModel):
    decision: Decision
    preprocessing: Preprocessing
    tubes: list[Tube]


class ModelInfo(BaseModel):
    name: str
    version: str | None


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    is_smoke: bool
    probability: float | None
    trigger_frame_index: int | None
    model: ModelInfo
    details: Details | None = None


def _decision_probability(
    details: dict[str, Any], is_smoke: bool, calibrated: bool
) -> float | None:
    if not calibrated:
        return None
    kept = details["tubes"]["kept"]
    if is_smoke:
        trigger_id = details["decision"]["trigger_tube_id"]
        for tube in kept:
            if tube["tube_id"] == trigger_id:
                return tube["probability"]
        return None
    probs = [t["probability"] for t in kept if t.get("probability") is not None]
    return max(probs) if probs else 0.0


def _to_details(details: dict[str, Any]) -> Details:
    tubes_block = details["tubes"]
    pre = details["preprocessing"]
    return Details(
        decision=Decision(**details["decision"]),
        preprocessing=Preprocessing(
            num_frames_input=pre["num_frames_input"],
            num_truncated=pre["num_truncated"],
            padded_frame_indices=pre["padded_frame_indices"],
            num_tube_candidates=tubes_block["num_candidates"],
        ),
        tubes=[Tube(**t) for t in tubes_block["kept"]],
    )


def to_response(
    out: Any, *, name: str, version: str | None, calibrated: bool, verbose: bool
) -> PredictResponse:
    """Reshape a core model output into the public response DTO."""
    kwargs: dict[str, Any] = {
        "is_smoke": out.is_positive,
        "probability": _decision_probability(out.details, out.is_positive, calibrated),
        "trigger_frame_index": out.trigger_frame_index,
        "model": ModelInfo(name=name, version=version),
    }
    if verbose:
        kwargs["details"] = _to_details(out.details)
    return PredictResponse(**kwargs)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/schemas.py api/tests/test_schemas.py
git commit -m "feat(api): add request/response schemas and core-output mapper"
```

---

### Task 6: FastAPI app wiring

**Files:**
- Modify: `api/src/temporal_model/api/app.py`
- Test: `api/tests/test_app.py` (replace existing)

The app loads the model + S3 client in `lifespan`, exposes `/health` and
`POST /predict`, and translates errors. Tests inject a fake runner and a
moto-backed S3 client via app state (lifespan tolerates a missing model file by
leaving `runner = None`).

- [ ] **Step 1: Write the failing tests**

Replace the contents of `api/tests/test_app.py` with:

```python
from types import SimpleNamespace

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from temporal_model.api.app import app
from temporal_model.api.settings import settings

BUCKET = "frames"
KEYS = ["cam12/adf_2023-05-23T17-18-01.jpg", "cam12/adf_2023-05-23T17-18-31.jpg"]


def _details(kept, trigger):
    return {
        "decision": {"aggregation": "max_logit", "threshold": 0.5, "trigger_tube_id": trigger},
        "preprocessing": {"num_frames_input": 30, "num_truncated": 0, "padded_frame_indices": []},
        "tubes": {"num_candidates": 2, "kept": kept},
    }


def _smoke_output():
    kept = [{
        "tube_id": 7, "start_frame": 2, "end_frame": 12, "logit": 3.4,
        "probability": 0.98, "first_crossing_frame": 3,
        "entries": [{"frame_idx": 2, "bbox": [1.0, 2.0, 3.0, 4.0], "is_gap": False, "confidence": 0.8}],
    }]
    return SimpleNamespace(is_positive=True, trigger_frame_index=3, details=_details(kept, 7))


class FakeRunner:
    name = "bbox-tube-vit-dinov2"
    version = "1.2.0"
    calibrated = True

    def __init__(self, output=None, error=None):
        self._output = output
        self._error = error

    async def predict(self, paths):
        if self._error:
            raise self._error
        return self._output


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "s3_bucket", BUCKET)
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        for key in KEYS:
            s3.put_object(Bucket=BUCKET, Key=key, Body=b"\xff\xd8\xff\xe0jpeg")
        with TestClient(app) as c:
            c.app.state.s3_client = s3
            c.app.state.runner = FakeRunner(output=_smoke_output())
            yield c


def test_health_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok", "model_loaded": True,
        "model_name": "bbox-tube-vit-dinov2", "model_version": "1.2.0",
    }


def test_predict_default(client):
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 200
    assert r.json() == {
        "is_smoke": True, "probability": 0.98, "trigger_frame_index": 3,
        "model": {"name": "bbox-tube-vit-dinov2", "version": "1.2.0"},
    }


def test_predict_verbose(client):
    r = client.post("/predict?verbose=true", json={"frames": KEYS})
    body = r.json()
    assert r.status_code == 200
    assert body["details"]["preprocessing"]["num_tube_candidates"] == 2
    assert body["details"]["tubes"][0]["tube_id"] == 7


def test_predict_empty_frames_400(client):
    r = client.post("/predict", json={"frames": []})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request"


def test_predict_scheme_key_400(client):
    r = client.post("/predict", json={"frames": ["s3://frames/a.jpg"]})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request"


def test_predict_missing_key_404(client):
    r = client.post("/predict", json={"frames": ["cam12/missing.jpg"]})
    assert r.status_code == 404
    assert r.json()["code"] == "frame_not_found"


def test_predict_inference_error_500(client):
    client.app.state.runner = FakeRunner(error=RuntimeError("boom"))
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 500
    assert r.json()["code"] == "inference_error"


def test_predict_model_not_loaded_503(client):
    client.app.state.runner = None
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 503
    assert r.json()["code"] == "model_not_loaded"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_app.py -v`
Expected: FAIL — current `app.py` has the stub `/predict` (501) and the old `/health` shape.

- [ ] **Step 3: Rewrite `app.py`**

Replace the contents of `api/src/temporal_model/api/app.py` with:

```python
"""FastAPI application: load a packaged model and serve smoke predictions."""

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .errors import ApiError, InferenceError, ModelNotLoaded
from .model_runner import ModelRunner
from .s3 import fetch_frames, make_s3_client
from .schemas import PredictRequest, PredictResponse, to_response
from .settings import settings

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.s3_client = make_s3_client(settings)
    try:
        app.state.runner = ModelRunner.load(Path(settings.model_path), settings.device)
    except Exception as exc:  # noqa: BLE001 — degrade to not-ready, report via /health
        logger.warning("model load failed: %s", exc)
        app.state.runner = None
    yield


app = FastAPI(title="Temporal Model API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(ApiError)
async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail, "code": exc.code}
    )


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    detail = errors[0]["msg"] if errors else "invalid request"
    return JSONResponse(status_code=400, content={"detail": detail, "code": "invalid_request"})


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        return HealthResponse(status="unavailable", model_loaded=False)
    return HealthResponse(
        status="ok", model_loaded=True,
        model_name=runner.name, model_version=runner.version,
    )


@app.post("/predict", response_model=PredictResponse, response_model_exclude_unset=True)
async def predict(
    body: PredictRequest, request: Request, verbose: bool = False
) -> PredictResponse:
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        raise ModelNotLoaded("model is not loaded")
    s3_client = request.app.state.s3_client

    with tempfile.TemporaryDirectory() as tmp:
        paths = fetch_frames(s3_client, settings.s3_bucket, body.frames, Path(tmp))
        try:
            out = await runner.predict(paths)
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface as inference_error
            raise InferenceError(str(exc)) from exc
        return to_response(
            out, name=runner.name, version=runner.version,
            calibrated=runner.calibrated, verbose=verbose,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_app.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest tests/ -v && uv run ruff check . && uv run ruff format --check .`
Expected: all tests pass; ruff reports no issues. (If `ruff format --check` fails, run `uv run ruff format .` and re-check, then include the formatting in the commit.)

- [ ] **Step 6: Commit**

```bash
git add api/src/temporal_model/api/app.py api/tests/test_app.py
git commit -m "feat(api): implement /predict and /health with model + S3 wiring"
```

---

### Task 7: Local dev (MinIO), README, and gated integration test

**Files:**
- Modify: `api/docker-compose.yml`
- Modify: `api/README.md`
- Create: `api/tests/test_integration.py`

- [ ] **Step 1: Add a MinIO service for manual local dev**

Replace `api/docker-compose.yml` with:

```yaml
services:
  api:
    build:
      context: ..
      dockerfile: api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - TEMPORAL_API_MODEL_PATH=/models/model.zip
      - TEMPORAL_API_S3_BUCKET=frames
      - TEMPORAL_API_S3_ENDPOINT_URL=http://minio:9000
      - TEMPORAL_API_S3_REGION=us-east-1
      - AWS_ACCESS_KEY_ID=minioadmin
      - AWS_SECRET_ACCESS_KEY=minioadmin
    volumes:
      - ./models:/models
    depends_on:
      - minio

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio-data:/data

volumes:
  minio-data:
```

- [ ] **Step 2: Verify the compose file parses**

Run (from `api/`): `docker compose config >/dev/null && echo OK`
Expected: prints `OK` (no YAML/schema errors). If `docker` is unavailable in the environment, skip and note it.

- [ ] **Step 3: Add the gated integration test**

Create `api/tests/test_integration.py`:

```python
"""End-to-end test gated on a real model package.

Set TEMPORAL_API_TEST_MODEL_PATH to a real model.zip to run it; otherwise it is
skipped. This is the only test that loads the actual core model + weights.
"""

import os
from pathlib import Path

import pytest

from temporal_model.api.model_runner import ModelRunner

MODEL_PATH = os.environ.get("TEMPORAL_API_TEST_MODEL_PATH")


@pytest.mark.skipif(not MODEL_PATH, reason="no real model.zip provided")
def test_real_model_loads_and_reports_metadata():
    runner = ModelRunner.load(Path(MODEL_PATH), device="cpu")
    assert runner.name
    assert isinstance(runner.calibrated, bool)
```

- [ ] **Step 4: Run the suite (integration skips)**

Run: `uv run pytest tests/ -v`
Expected: all prior tests pass; `test_integration.py` is skipped (1 skipped).

- [ ] **Step 5: Update the README**

In `api/README.md`, replace the scaffold-stage description and add usage. Replace the body after the title with:

```markdown
FastAPI serving layer for the temporal smoke classifier, packaged as a
Docker service.

Import as `temporal_model.api`. Depends on `temporal-model-core`.

## Endpoints

- `GET /health` — readiness + loaded model name/version.
- `POST /predict` — body `{ "frames": ["<s3-key>", ...] }` (ordered S3 keys);
  returns `{ is_smoke, probability, trigger_frame_index, model }`.
  `POST /predict?verbose=true` adds a `details` block (decision, preprocessing,
  per-tube tracks). See `docs/specs/2026-06-02-api-service-design.md` for the
  full contract.

## Run

```bash
make serve                  # local dev, http://localhost:8000
docker compose up --build   # API + MinIO (S3) locally
```

Configuration via env vars (prefix `TEMPORAL_API_`): `MODEL_PATH`, `DEVICE`,
`S3_BUCKET`, `S3_REGION`, `S3_ENDPOINT_URL` (empty = real AWS; set for OVH or
MinIO), `HOST`, `PORT`. AWS/OVH/MinIO credentials come from the standard boto3
chain (env vars / IAM role).

## Test

```bash
make test                   # fast, hermetic (model mocked, S3 via moto)
TEMPORAL_API_TEST_MODEL_PATH=/path/to/model.zip make test   # + integration
```
```

- [ ] **Step 6: Final full verification**

Run: `uv run pytest tests/ -v && uv run ruff check . && uv run ruff format --check .`
Expected: tests pass (integration skipped), ruff clean.

- [ ] **Step 7: Commit**

```bash
git add api/docker-compose.yml api/README.md api/tests/test_integration.py
git commit -m "feat(api): add MinIO dev compose, README usage, gated integration test"
```

---

## Notes for the implementer

- **Why the model is mocked:** `temporal_model.core` is still a stub. The API
  depends on `BboxTubeTemporalModel.from_package(...).predict_sequence(paths)`
  via a lazy import in `model_runner._load_core_model`. Until `core` is migrated,
  only `test_integration.py` (gated) exercises the real path; everything else
  mocks it. **Do not** add a hard top-level import of `temporal_model.core.model`
  anywhere — it would break import at app startup while `core` is a stub.
- **`probability` rule (Task 5):** calibrated ⇒ always a number — trigger tube's
  probability when `is_smoke`, else max kept-tube probability, else `0.0` (no
  tubes). `null` ⇒ uncalibrated only. The "no tube found" case is visible only
  via `?verbose=true` (`num_tube_candidates: 0`, empty `tubes`) — by design.
- **Filename preservation (Task 3):** keys are downloaded to
  `<tmp>/<NNNN>/<basename>` so the model can parse timestamps from the basename
  and order is preserved exactly. Never re-sort `frames`.
- **`exclude_unset` (Tasks 5–6):** `details` is set only when verbose, so the
  route's `response_model_exclude_unset=True` omits it otherwise while keeping
  explicit `null`s for `probability` / `trigger_frame_index`.
- Run every command from `api/`. Keep commits per task. No Claude co-author
  trailers.

## Spec coverage check

| Spec requirement | Task |
|---|---|
| `POST /predict` request (`frames`, bare keys, ≥1, reject scheme) | 5 (validation), 6 (route) |
| Default response (`is_smoke`, `probability`, `trigger_frame_index`, `model`) | 5, 6 |
| `?verbose=true` `details` (decision, preprocessing+`num_tube_candidates`, tubes) | 5, 6 |
| `probability` rule (calibrated/uncalibrated/no-tube) | 5 |
| `GET /health` (status, model_loaded, name, version) | 6 |
| Error table (400/404/502/503/500 + codes) | 2 (types), 3 (404/502), 6 (400/503/500) |
| Model loaded once via lifespan singleton; device setting | 4, 6 |
| Inference serialized behind lock in threadpool | 4 |
| Settings (`MODEL_PATH`, `DEVICE`, `S3_*`, `HOST`, `PORT`) | 1 |
| S3 client with `endpoint_url` (AWS/OVH/MinIO) | 1, 3 |
| Basenames + order preserved into `predict_sequence` | 3 |
| `model_version` read from manifest; `null` for legacy | 4 |
| Testing: model mocked + moto; gated integration; MinIO compose | 3–7 |
```
