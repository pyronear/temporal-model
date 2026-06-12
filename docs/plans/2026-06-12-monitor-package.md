# monitor/ Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `monitor/` package per `docs/specs/2026-06-12-monitor-design.md`: import production sequences from alert-api, replay them through the exact pinned api+model Docker release with verbose tube details, and write the eval-viewer reporting contract, all DVC-tracked.

**Architecture:** A sixth uv package `temporal_model.monitor` (no `core`/torch dependency) with a `temporal-monitor` CLI. `import` fetches scored sequences + frames from alert-api into a `dvc add`-tracked store; `replay` (a `dvc.yaml` stage) groups sequences by recorded `temporal_api_version`, runs the matching `pyronear/temporal-model-api:<tag>` image + MinIO via docker compose, reconstructs the exact production call, and writes `data/08_reporting/<org>/vit_dinov2_finetune/` in the eval-viewer contract. The existing `viewer/` reads it via `DATA_ROOT=../monitor`.

**Tech Stack:** Python 3.11, uv, pydantic, requests, boto3, docker compose (subprocess), DVC (s3remote), pytest, Next.js viewer (one small additive change).

**Key references (read before each task):**
- Spec: `docs/specs/2026-06-12-monitor-design.md`
- Production logic to mirror: `../pyro-api/src/app/services/validation.py:99-148` (`_sequence_frames_and_roi`), `../pyro-api/src/app/services/temporal.py:60-61` (`MIN_FRAMES=4`, `MAX_FRAMES=10`), `../pyro-api/src/app/schemas/detections.py:19-25` (`FLOAT_PATTERN`/`BOX_PATTERN`)
- Eval contract to reproduce: `eval/src/temporal_model/eval/evaluate.py:180-196` (results row), `eval/src/temporal_model/eval/view_store.py` (SequenceView), `eval/src/temporal_model/eval/outcomes.py` (decision/outcome), `core/src/temporal_model/core/details_schema.py` (details shape), `core/src/temporal_model/core/stabilize.py` (window geometry)
- Verbose API response: `api/src/temporal_model/api/schemas.py`
- Viewer contract types: `viewer/lib/types.ts`, paths: `viewer/lib/paths.ts`

---

### Task 1: Package scaffold + repo wiring

**Files:**
- Create: `monitor/pyproject.toml`
- Create: `monitor/Makefile`
- Create: `monitor/.envrc.example`
- Create: `monitor/src/temporal_model/monitor/__init__.py`
- Create: `monitor/tests/test_scaffold.py`
- Modify: `Makefile:1` (root — PACKAGES list)
- Modify: `.github/workflows/ci.yml:15` (matrix)
- Modify: `.gitignore` (root — add `.envrc`)
- Modify: `README.md` (root — packages table)

- [ ] **Step 1: Create `monitor/pyproject.toml`** (house style from `benchmark/pyproject.toml`, minus core/torch):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "temporal-model-monitor"
version = "0.1.0"
description = "Production decision replay for the temporal smoke classifier: import scored sequences from alert-api, re-run them with verbose details through the exact pinned api+model release, view tubes in the eval viewer"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7",
    "requests>=2.31",
    "boto3>=1.34",
]

[project.scripts]
temporal-monitor = "temporal_model.monitor.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/temporal_model"]

[dependency-groups]
dev = [
    "dvc[s3]>=3.56",
    "pytest>=8.0",
    "ruff>=0.9",
]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM", "PLC0415"]

[tool.ruff.lint.isort]
known-first-party = ["temporal_model"]

[tool.ruff.format]
quote-style = "double"
```

- [ ] **Step 2: Create `monitor/Makefile`** (copy of `benchmark/Makefile`; the `import` target is added later in Task 10):

```make
.PHONY: install lint format test

install: ## uv sync
	uv sync

lint: ## ruff check
	uv run ruff check .

format: ## ruff format
	uv run ruff format .

test: ## pytest
	uv run pytest tests/ -v
```

- [ ] **Step 3: Create `monitor/.envrc.example`**:

```bash
# Copy to monitor/.envrc (untracked) and fill in; direnv loads it automatically.
# Credentials for the alert-api (the platform that calls /predict in production).
export ALERT_API_URL=https://alertapi.pyronear.org
export ALERT_API_LOGIN=changeme
export ALERT_API_PASSWORD=changeme
```

- [ ] **Step 4: Create the package and a scaffold test**

`monitor/src/temporal_model/monitor/__init__.py`:

```python
"""Production decision replay: alert-api import + pinned-release re-run."""
```

`monitor/tests/test_scaffold.py`:

```python
from temporal_model.monitor import __doc__ as pkg_doc


def test_package_importable():
    assert "replay" in pkg_doc
```

- [ ] **Step 5: Install and run the test**

Run: `cd monitor && make install && uv run pytest tests/ -v`
Expected: `test_package_importable PASSED`

- [ ] **Step 6: Wire the repo**
- Root `Makefile` line 1: `PACKAGES := core train eval api benchmark` → `PACKAGES := core train eval api benchmark monitor`
- `.github/workflows/ci.yml` line 15: `package: [core, train, eval, api]` → `package: [core, train, eval, api, monitor]`
- Root `.gitignore`: add under the `# OS` section a new section:

```
# direnv credentials (per-package .envrc.example documents the variables)
.envrc
```

- Root `README.md` packages table: add the row
  `| `monitor/` | `temporal_model.monitor` | Production decision replay: import scored sequences from alert-api, re-run them through the pinned api+model release, view tubes in the eval viewer. |`
  and change the sentence "Five independent packages" to "Six independent packages".

- [ ] **Step 7: Lint and commit**

Run: `cd monitor && make lint` — expected: no errors.

```bash
git add monitor/ Makefile .github/workflows/ci.yml .gitignore README.md
git commit -m "feat(monitor): scaffold the monitor package"
```

---

### Task 2: Sequence store (`store.py`)

The on-disk store: `data/01_raw/sequences/<org_slug>/<camera_slug>/seq_<id>/{meta.json, images/}`. Mirrors the vision-rd explorer's store, extended with replay provenance (recorded score/versions, `bucket_key`, verbatim `bbox` string per detection).

**Files:**
- Create: `monitor/src/temporal_model/monitor/store.py`
- Test: `monitor/tests/test_store.py`

- [ ] **Step 1: Write the failing tests**

`monitor/tests/test_store.py`:

```python
from temporal_model.monitor.store import (
    FrameMeta,
    SequenceMeta,
    iter_metas,
    label_from_is_wildfire,
    read_meta,
    sequence_dir,
    sequence_exists,
    slugify,
    write_meta,
)


def make_meta(sequence_id: int = 42307) -> SequenceMeta:
    return SequenceMeta(
        key=f"platform_{sequence_id}",
        sequence_id=sequence_id,
        label="smoke",
        label_detail="wildfire_smoke",
        camera_id=122,
        camera_name="donon-sarrebourg-01",
        organization_id=11,
        organization_name="sis-67",
        started_at="2026-05-15T13:08:18.671072",
        temporal_model_score=0.9867,
        temporal_model_version="0.1.0",
        temporal_api_version="0.3.1",
        frames=[
            FrameMeta(
                file="images/detection_100.jpg",
                detection_id=100,
                created_at="2026-05-15T13:08:18.671072",
                bucket_key="cam122/frame-100.jpg",
                bbox="[(0.1,0.2,0.3,0.4,0.9)]",
            )
        ],
    )


def test_slugify():
    assert slugify("SIS 67") == "sis-67"
    assert slugify("Donon/Sarrebourg_01") == "donon-sarrebourg-01"
    assert slugify("") == "unknown"
    assert slugify(None) == "unknown"


def test_label_from_is_wildfire():
    assert label_from_is_wildfire("wildfire_smoke") == ("smoke", "wildfire_smoke")
    assert label_from_is_wildfire("other_smoke") == ("smoke", "other_smoke")
    assert label_from_is_wildfire("other") == ("fp", "other")
    assert label_from_is_wildfire(None) == ("unknown", None)


def test_meta_roundtrip(tmp_path):
    meta = make_meta()
    seq_dir = sequence_dir(tmp_path, meta)
    write_meta(seq_dir, meta)
    assert seq_dir == tmp_path / "sis-67" / "donon-sarrebourg-01" / "seq_42307"
    assert read_meta(seq_dir) == meta


def test_sequence_exists(tmp_path):
    meta = make_meta()
    assert not sequence_exists(tmp_path, 42307)
    write_meta(sequence_dir(tmp_path, meta), meta)
    assert sequence_exists(tmp_path, 42307)
    assert not sequence_exists(tmp_path, 999)


def test_iter_metas(tmp_path):
    for sid in (1, 2):
        meta = make_meta(sid)
        write_meta(sequence_dir(tmp_path, meta), meta)
    found = list(iter_metas(tmp_path))
    assert sorted(m.sequence_id for _, m in found) == [1, 2]
    assert all(d.name == f"seq_{m.sequence_id}" for d, m in found)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd monitor && uv run pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'temporal_model.monitor.store'`

- [ ] **Step 3: Implement `monitor/src/temporal_model/monitor/store.py`**

```python
"""Sequence store: data/01_raw/sequences/<org>/<camera>/seq_<id>/{meta.json, images/}.

Mirrors the vision-rd explorer's store layout, extended with everything replay
needs: the recorded temporal score + version provenance and, per detection,
the original S3 ``bucket_key`` and the verbatim ``bbox`` string (parsed later
by ``reconstruct``).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel

META_NAME = "meta.json"
IMAGES_DIR = "images"

# alert-api AnnotationType -> (label, label_detail)
_SMOKE_VALUES = {"wildfire_smoke", "other_smoke"}
_FP_VALUES = {"other"}


class FrameMeta(BaseModel):
    file: str  # relative to the sequence dir, e.g. "images/detection_100.jpg"
    detection_id: int
    created_at: str
    bucket_key: str
    bbox: str  # verbatim alert-api bbox string, e.g. "[(0.1,0.2,0.3,0.4,0.9)]"


class SequenceMeta(BaseModel):
    key: str  # "platform_<sequence_id>" — the viewer join key
    sequence_id: int
    source: str = "platform"
    label: str  # "smoke" | "fp" | "unknown"
    label_detail: str | None = None
    camera_id: int | None = None
    camera_name: str | None = None
    organization_id: int | None = None
    organization_name: str | None = None
    started_at: str | None = None
    temporal_model_score: float | None = None
    temporal_model_version: str | None = None
    temporal_api_version: str | None = None
    frames: list[FrameMeta] = []


def slugify(value: str | None) -> str:
    """Filesystem-safe lowercase slug; 'unknown' when there is nothing to slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return slug or "unknown"


def label_from_is_wildfire(is_wildfire: str | None) -> tuple[str, str | None]:
    """Map alert-api's ``is_wildfire`` annotation to (label, label_detail)."""
    if is_wildfire in _SMOKE_VALUES:
        return "smoke", is_wildfire
    if is_wildfire in _FP_VALUES:
        return "fp", is_wildfire
    return "unknown", is_wildfire


def sequence_dir(store_dir: Path, meta: SequenceMeta) -> Path:
    return (
        store_dir
        / slugify(meta.organization_name)
        / slugify(meta.camera_name)
        / f"seq_{meta.sequence_id}"
    )


def write_meta(seq_dir: Path, meta: SequenceMeta) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / META_NAME).write_text(meta.model_dump_json(indent=2))


def read_meta(seq_dir: Path) -> SequenceMeta:
    return SequenceMeta.model_validate_json((seq_dir / META_NAME).read_text())


def sequence_exists(store_dir: Path, sequence_id: int) -> bool:
    return any(store_dir.glob(f"*/*/seq_{sequence_id}/{META_NAME}"))


def iter_metas(store_dir: Path) -> Iterator[tuple[Path, SequenceMeta]]:
    """Yield (sequence_dir, meta) for every sequence in the store."""
    for meta_path in sorted(store_dir.glob(f"*/*/seq_*/{META_NAME}")):
        yield meta_path.parent, read_meta(meta_path.parent)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd monitor && uv run pytest tests/test_store.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
cd monitor && make lint
git add monitor/src/temporal_model/monitor/store.py monitor/tests/test_store.py
git commit -m "feat(monitor): sequence store with replay provenance"
```

---

### Task 3: alert-api client (`alert_api.py`)

Thin HTTP client for the four endpoints monitor needs. Auth is OAuth2 form-encoded; detections are paginated (server cap `le=100`) and must be fetched **completely** — production ROI uses all detections (`fetch_all` in `validation.py:116`), so import must too.

**Files:**
- Create: `monitor/src/temporal_model/monitor/alert_api.py`
- Test: `monitor/tests/test_alert_api.py`

- [ ] **Step 1: Write the failing tests**

`monitor/tests/test_alert_api.py`:

```python
import pytest

from temporal_model.monitor.alert_api import AlertApiClient, AlertApiConfig


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    """Records requests; pops queued responses in FIFO order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, data=None, timeout=None):
        self.calls.append(("POST", url, data))
        return self.responses.pop(0)

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        return self.responses.pop(0)


CONFIG = AlertApiConfig(url="https://api.test", login="user", password="pw")


def make_client(responses):
    session = FakeSession([FakeResponse(r) for r in responses])
    client = AlertApiClient(CONFIG, session=session)
    return client, session


def test_login_posts_form_credentials():
    client, session = make_client([{"access_token": "tok"}])
    client.login()
    method, url, data = session.calls[0]
    assert (method, url) == ("POST", "https://api.test/api/v1/login/creds")
    assert data == {"username": "user", "password": "pw"}
    assert client.token == "tok"


def test_list_sequences_for_date_paginates_until_short_page():
    page1 = [{"id": i} for i in range(100)]
    page2 = [{"id": 100}]
    client, session = make_client([{"access_token": "t"}, page1, page2])
    client.login()
    seqs = client.list_sequences_for_date("2026-06-11")
    assert len(seqs) == 101
    # two GET calls with increasing offset
    gets = [c for c in session.calls if c[0] == "GET"]
    assert gets[0][2]["from_date"] == "2026-06-11"
    assert gets[0][2]["offset"] == 0
    assert gets[1][2]["offset"] == 100


def test_list_sequence_detections_paginates_ascending():
    page1 = [{"id": i} for i in range(100)]
    page2 = [{"id": 100}]
    client, session = make_client([{"access_token": "t"}, page1, page2])
    client.login()
    dets = client.list_sequence_detections(42)
    assert len(dets) == 101
    gets = [c for c in session.calls if c[0] == "GET"]
    assert gets[0][1].endswith("/api/v1/sequences/42/detections")
    assert gets[0][2] == {"limit": 100, "offset": 0, "desc": False}
    assert gets[1][2]["offset"] == 100


def test_requests_require_login():
    client, _ = make_client([])
    with pytest.raises(RuntimeError, match="login"):
        client.list_cameras()


def test_list_cameras_and_organizations():
    client, session = make_client(
        [{"access_token": "t"}, [{"id": 1, "name": "cam"}], [{"id": 11, "name": "org"}]]
    )
    client.login()
    assert client.list_cameras() == [{"id": 1, "name": "cam"}]
    assert client.list_organizations() == [{"id": 11, "name": "org"}]
    gets = [c for c in session.calls if c[0] == "GET"]
    assert gets[0][1].endswith("/api/v1/cameras/")
    assert gets[0][2] == {"include_non_trustable": True}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd monitor && uv run pytest tests/test_alert_api.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `monitor/src/temporal_model/monitor/alert_api.py`**

```python
"""Thin alert-api HTTP client: login, sequences, detections, cameras, orgs.

Pagination: list endpoints are server-capped at 100 rows; helpers loop with
``offset`` until a short page. Detections are fetched COMPLETELY and oldest
first — production ROI reconstruction uses every detection of a sequence
(pyro-api ``validation._sequence_frames_and_roi`` runs an unbounded
``fetch_all``), so a truncated import would change the replayed call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

PAGE_SIZE = 100  # server-side cap (alert-api Query(..., le=100))
TIMEOUT_S = 60


@dataclass(frozen=True)
class AlertApiConfig:
    url: str
    login: str
    password: str

    @classmethod
    def from_env(cls) -> AlertApiConfig:
        try:
            return cls(
                url=os.environ["ALERT_API_URL"].rstrip("/"),
                login=os.environ["ALERT_API_LOGIN"],
                password=os.environ["ALERT_API_PASSWORD"],
            )
        except KeyError as exc:
            raise SystemExit(
                f"missing env var {exc.args[0]} (see monitor/.envrc.example)"
            ) from exc


class AlertApiClient:
    def __init__(
        self, config: AlertApiConfig, session: requests.Session | None = None
    ) -> None:
        self.config = config
        self.session = session if session is not None else requests.Session()
        self.token: str | None = None

    def login(self) -> None:
        resp = self.session.post(
            f"{self.config.url}/api/v1/login/creds",
            data={"username": self.config.login, "password": self.config.password},
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        self.token = resp.json()["access_token"]

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self.token is None:
            raise RuntimeError("call login() first")
        resp = self.session.get(
            f"{self.config.url}{path}",
            headers={"Authorization": f"Bearer {self.token}"},
            params=params,
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_paginated(self, path: str, params: dict[str, Any]) -> list[dict]:
        rows: list[dict] = []
        offset = 0
        while True:
            page = self._get(path, {**params, "limit": PAGE_SIZE, "offset": offset})
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                return rows
            offset += PAGE_SIZE

    def list_sequences_for_date(self, day: str) -> list[dict]:
        """All sequences started on ``day`` (YYYY-MM-DD)."""
        return self._get_paginated(
            "/api/v1/sequences/all/fromdate", {"from_date": day}
        )

    def list_sequence_detections(self, sequence_id: int) -> list[dict]:
        """ALL detections of a sequence, oldest first (see module docstring)."""
        return self._get_paginated(
            f"/api/v1/sequences/{sequence_id}/detections", {"desc": False}
        )

    def list_cameras(self) -> list[dict]:
        return self._get("/api/v1/cameras/", {"include_non_trustable": True})

    def list_organizations(self) -> list[dict]:
        return self._get("/api/v1/organizations/")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd monitor && uv run pytest tests/test_alert_api.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
cd monitor && make lint
git add monitor/src/temporal_model/monitor/alert_api.py monitor/tests/test_alert_api.py
git commit -m "feat(monitor): alert-api client with full-detection pagination"
```

---

### Task 4: Import command (`import_platform.py` + `cli.py`)

Orchestrates client + store: for each day in the range, fetch sequences, skip already-stored ones (incremental), download frames, write `meta.json`.

**Files:**
- Create: `monitor/src/temporal_model/monitor/import_platform.py`
- Create: `monitor/src/temporal_model/monitor/cli.py`
- Test: `monitor/tests/test_import_platform.py`

- [ ] **Step 1: Write the failing tests**

`monitor/tests/test_import_platform.py`:

```python
from temporal_model.monitor.import_platform import import_platform
from temporal_model.monitor.store import read_meta, sequence_exists

SEQ = {
    "id": 42307,
    "camera_id": 122,
    "is_wildfire": "wildfire_smoke",
    "started_at": "2026-05-15T13:08:18.671072",
    "temporal_model_score": 0.9867,
    "temporal_model_version": "0.1.0",
    "temporal_api_version": "0.3.1",
}
DETS = [
    {
        "id": 100,
        "created_at": "2026-05-15T13:08:18.671072",
        "bucket_key": "cam122/frame-100.jpg",
        "bbox": "[(0.1,0.2,0.3,0.4,0.9)]",
        "url": "https://s3.test/frame-100.jpg?sig=x",
    },
    {
        "id": 101,
        "created_at": "2026-05-15T13:09:18.671072",
        "bucket_key": "cam122/frame-101.jpg",
        "bbox": "[(0.11,0.2,0.31,0.4,0.8)]",
        "url": "https://s3.test/frame-101.jpg?sig=x",
    },
]


class FakeClient:
    def __init__(self):
        self.detection_calls = 0

    def list_sequences_for_date(self, day):
        return [SEQ] if day == "2026-05-15" else []

    def list_sequence_detections(self, sequence_id):
        assert sequence_id == 42307
        self.detection_calls += 1
        return DETS

    def list_cameras(self):
        return [{"id": 122, "name": "donon-sarrebourg-01", "organization_id": 11}]

    def list_organizations(self):
        return [{"id": 11, "name": "sis-67"}]


def fake_download(url: str) -> bytes:
    return b"jpegbytes:" + url.encode()


def test_import_writes_store(tmp_path):
    client = FakeClient()
    stats = import_platform(
        client, tmp_path, "2026-05-15", "2026-05-16", download=fake_download
    )
    assert stats == {"imported": 1, "skipped": 0}
    assert sequence_exists(tmp_path, 42307)
    seq_dir = tmp_path / "sis-67" / "donon-sarrebourg-01" / "seq_42307"
    meta = read_meta(seq_dir)
    assert meta.key == "platform_42307"
    assert meta.label == "smoke"
    assert meta.temporal_api_version == "0.3.1"
    assert [f.bucket_key for f in meta.frames] == [
        "cam122/frame-100.jpg",
        "cam122/frame-101.jpg",
    ]
    assert (seq_dir / "images" / "detection_100.jpg").read_bytes().startswith(
        b"jpegbytes:"
    )


def test_import_is_incremental(tmp_path):
    client = FakeClient()
    import_platform(client, tmp_path, "2026-05-15", "2026-05-16", download=fake_download)
    stats = import_platform(
        client, tmp_path, "2026-05-15", "2026-05-16", download=fake_download
    )
    assert stats == {"imported": 0, "skipped": 1}
    assert client.detection_calls == 1  # second run never re-fetched detections


def test_import_handles_missing_org_names(tmp_path):
    class NoOrgClient(FakeClient):
        def list_organizations(self):
            raise PermissionError("403")

    import_platform(
        NoOrgClient(), tmp_path, "2026-05-15", "2026-05-16", download=fake_download
    )
    assert (tmp_path / "org-11").is_dir()  # falls back to org-<id>
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd monitor && uv run pytest tests/test_import_platform.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `monitor/src/temporal_model/monitor/import_platform.py`**

```python
"""Import scored sequences (frames + provenance) from alert-api into the store.

Incremental by design: a sequence already in the store is skipped, so a
recurring import only pays for new sequences. Detections arrive oldest first
and are stored one FrameMeta per detection (a bucket_key can repeat when
several detections share a frame; replay deduplicates, mirroring production).
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from pathlib import Path

import requests

from temporal_model.monitor.store import (
    IMAGES_DIR,
    FrameMeta,
    SequenceMeta,
    label_from_is_wildfire,
    sequence_dir,
    sequence_exists,
    write_meta,
)

logger = logging.getLogger(__name__)


def _default_download(url: str) -> bytes:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def _date_range(day_from: str, day_to: str) -> list[str]:
    start = dt.date.fromisoformat(day_from)
    end = dt.date.fromisoformat(day_to)
    return [
        (start + dt.timedelta(days=i)).isoformat()
        for i in range((end - start).days + 1)
    ]


def _camera_index(client) -> dict[int, dict]:
    return {cam["id"]: cam for cam in client.list_cameras()}


def _org_names(client) -> dict[int, str]:
    try:
        return {org["id"]: org["name"] for org in client.list_organizations()}
    except Exception:  # noqa: BLE001 — org listing may need admin scope
        logger.warning("organizations endpoint unavailable; using org-<id> names")
        return {}


def import_platform(
    client,
    store_dir: Path,
    day_from: str,
    day_to: str,
    *,
    force: bool = False,
    download: Callable[[str], bytes] = _default_download,
) -> dict[str, int]:
    """Import all sequences in [day_from, day_to] (inclusive). Returns counts."""
    cameras = _camera_index(client)
    org_names = _org_names(client)
    imported = skipped = 0
    for day in _date_range(day_from, day_to):
        for seq in client.list_sequences_for_date(day):
            if not force and sequence_exists(store_dir, seq["id"]):
                skipped += 1
                continue
            _import_one(client, store_dir, seq, cameras, org_names, download)
            imported += 1
    logger.info("import done: %d imported, %d skipped", imported, skipped)
    return {"imported": imported, "skipped": skipped}


def _import_one(
    client,
    store_dir: Path,
    seq: dict,
    cameras: dict[int, dict],
    org_names: dict[int, str],
    download: Callable[[str], bytes],
) -> None:
    dets = client.list_sequence_detections(seq["id"])
    dets = sorted(dets, key=lambda d: d["created_at"])
    camera = cameras.get(seq.get("camera_id")) or {}
    org_id = camera.get("organization_id")
    label, label_detail = label_from_is_wildfire(seq.get("is_wildfire"))
    meta = SequenceMeta(
        key=f"platform_{seq['id']}",
        sequence_id=seq["id"],
        label=label,
        label_detail=label_detail,
        camera_id=seq.get("camera_id"),
        camera_name=camera.get("name"),
        organization_id=org_id,
        organization_name=org_names.get(org_id)
        or (f"org-{org_id}" if org_id is not None else None),
        started_at=seq.get("started_at"),
        temporal_model_score=seq.get("temporal_model_score"),
        temporal_model_version=seq.get("temporal_model_version"),
        temporal_api_version=seq.get("temporal_api_version"),
        frames=[
            FrameMeta(
                file=f"{IMAGES_DIR}/detection_{d['id']}.jpg",
                detection_id=d["id"],
                created_at=d["created_at"],
                bucket_key=d["bucket_key"],
                bbox=d.get("bbox") or "",
            )
            for d in dets
        ],
    )
    seq_dir = sequence_dir(store_dir, meta)
    images_dir = seq_dir / IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    for det in dets:
        (images_dir / f"detection_{det['id']}.jpg").write_bytes(download(det["url"]))
    # meta.json last: its presence marks the sequence complete, so a crashed
    # download is re-fetched (not skipped) on the next run.
    write_meta(seq_dir, meta)
```

- [ ] **Step 4: Implement `monitor/src/temporal_model/monitor/cli.py`** (the `replay` subcommand is wired in Task 9 — for now it exits with a message):

```python
"""temporal-monitor CLI: import sequences from alert-api, replay them locally."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

DEFAULT_STORE = Path("data/01_raw/sequences")
DEFAULT_OUTPUT = Path("data/08_reporting")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="temporal-monitor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import", help="fetch scored sequences from alert-api")
    today = dt.date.today().isoformat()
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    imp.add_argument("--date-from", default=yesterday, help="YYYY-MM-DD (inclusive)")
    imp.add_argument("--date-to", default=today, help="YYYY-MM-DD (inclusive)")
    imp.add_argument("--store", type=Path, default=DEFAULT_STORE)
    imp.add_argument(
        "--force", action="store_true", help="re-download already-stored sequences"
    )

    rep = sub.add_parser(
        "replay", help="re-run stored sequences through their pinned api release"
    )
    rep.add_argument("--store", type=Path, default=DEFAULT_STORE)
    rep.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    rep.add_argument(
        "--compose-file",
        type=Path,
        # cli.py -> monitor(pkg) -> temporal_model -> src -> monitor/ root
        default=Path(__file__).resolve().parents[3] / "docker-compose.yml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.command == "import":
        # Imported lazily so `replay` does not need alert-api credentials.
        from temporal_model.monitor.alert_api import AlertApiClient, AlertApiConfig
        from temporal_model.monitor.import_platform import import_platform

        client = AlertApiClient(AlertApiConfig.from_env())
        client.login()
        import_platform(
            client, args.store, args.date_from, args.date_to, force=args.force
        )
    elif args.command == "replay":
        raise SystemExit("replay is not implemented yet")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests**

Run: `cd monitor && uv run pytest tests/test_import_platform.py -v`
Expected: all PASS

Run: `cd monitor && uv run temporal-monitor import --help`
Expected: usage text with `--date-from/--date-to/--store/--force`

- [ ] **Step 6: Lint and commit**

```bash
cd monitor && make lint
git add monitor/src/temporal_model/monitor/import_platform.py monitor/src/temporal_model/monitor/cli.py monitor/tests/test_import_platform.py
git commit -m "feat(monitor): incremental alert-api import command"
```

---

### Task 5: Production call reconstruction (`reconstruct.py`)

Port of pyro-api's `_sequence_frames_and_roi` (`validation.py:99-148`) plus its bbox parsing (`detections.py` `BOX_PATTERN` + `_parse_bbox`). Must behave identically — fixture values below are derived by hand-executing the pyro-api code.

**Files:**
- Create: `monitor/src/temporal_model/monitor/reconstruct.py`
- Test: `monitor/tests/test_reconstruct.py`

- [ ] **Step 1: Write the failing tests**

`monitor/tests/test_reconstruct.py`:

```python
from temporal_model.monitor.reconstruct import (
    MAX_FRAMES,
    MIN_FRAMES,
    extract_bbox_strings,
    frames_and_roi,
    parse_bbox,
)
from temporal_model.monitor.store import FrameMeta


def frame(i: int, bucket_key: str, bbox: str) -> FrameMeta:
    return FrameMeta(
        file=f"images/detection_{i}.jpg",
        detection_id=i,
        created_at=f"2026-05-15T13:{i:02d}:00",
        bucket_key=bucket_key,
        bbox=bbox,
    )


def test_constants_match_pyro_api():
    assert MIN_FRAMES == 4
    assert MAX_FRAMES == 10


def test_extract_and_parse_bbox():
    strs = extract_bbox_strings("[(0.1,0.2,0.3,0.4,0.9),(0.5,0.5,0.6,0.6,0.2)]")
    assert strs == ["(0.1,0.2,0.3,0.4,0.9)", "(0.5,0.5,0.6,0.6,0.2)"]
    assert parse_bbox(strs[0]) == (0.1, 0.2, 0.3, 0.4, 0.9)
    assert parse_bbox("garbage") is None
    assert parse_bbox("(0.1,0.2)") is None


def test_frames_distinct_and_ordered():
    frames = [
        frame(0, "k0.jpg", "[(0.1,0.1,0.2,0.2,0.9)]"),
        frame(1, "k0.jpg", "[(0.15,0.1,0.25,0.2,0.9)]"),  # same frame, 2nd detection
        frame(2, "k1.jpg", "[(0.2,0.1,0.3,0.2,0.9)]"),
    ]
    total, kept, roi = frames_and_roi(frames)
    assert total == 2
    assert kept == ["k0.jpg", "k1.jpg"]
    # envelope of the three primary bboxes
    assert roi == [0.1, 0.1, 0.3, 0.2]


def test_truncates_to_last_max_frames_and_roi_covers_kept_only():
    frames = [
        frame(i, f"k{i}.jpg", f"[(0.{i},0.1,0.{i + 1},0.2,0.9)]") for i in range(1, 9)
    ] + [
        frame(20, "k20.jpg", "[(0.5,0.5,0.6,0.6,0.9)]"),
        frame(21, "k21.jpg", "[(0.5,0.5,0.6,0.6,0.9)]"),
        frame(22, "k22.jpg", "[(0.5,0.5,0.6,0.6,0.9)]"),
    ]
    total, kept, roi = frames_and_roi(frames)
    assert total == 11
    assert len(kept) == MAX_FRAMES
    assert kept[0] == "k2.jpg"  # k1 truncated away
    # k1's bbox (xmin 0.1) is truncated away; the kept envelope starts at k2's 0.2
    assert roi[0] == 0.2


def test_roi_none_when_no_bbox_parses():
    frames = [frame(i, f"k{i}.jpg", "") for i in range(5)]
    total, kept, roi = frames_and_roi(frames)
    assert total == 5
    assert roi is None


def test_roi_none_when_degenerate():
    # a single zero-area box -> xmin == xmax -> degenerate -> None
    frames = [frame(0, "k0.jpg", "[(0.5,0.5,0.5,0.5,0.9)]")]
    _, _, roi = frames_and_roi(frames)
    assert roi is None


def test_roi_clamped_to_unit_square():
    # parseable floats are already in [0,1] per the regex, but clamping is kept
    # for parity with pyro-api (max(0,...), min(1,...))
    frames = [frame(0, "k0.jpg", "[(0,0,1,1,0.9)]")]
    _, _, roi = frames_and_roi(frames)
    assert roi == [0.0, 0.0, 1.0, 1.0]


def test_unparseable_primary_bbox_skipped_not_fatal():
    frames = [
        frame(0, "k0.jpg", "[(0.1,0.1,0.2,0.2,0.9)]"),
        frame(1, "k1.jpg", "not-a-bbox"),
    ]
    total, kept, roi = frames_and_roi(frames)
    assert kept == ["k0.jpg", "k1.jpg"]
    assert roi == [0.1, 0.1, 0.2, 0.2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd monitor && uv run pytest tests/test_reconstruct.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `monitor/src/temporal_model/monitor/reconstruct.py`**

```python
"""Reconstruct the exact /predict call alert-api made for a sequence.

Line-for-line port of pyro-api's ``validation._sequence_frames_and_roi``
(pyro-api ``src/app/services/validation.py``) and its bbox parsing
(``src/app/api/api_v1/endpoints/detections.py`` + ``schemas/detections.py``):
distinct bucket_keys oldest first, truncated to the most recent MAX_FRAMES;
ROI = union envelope of the kept detections' PRIMARY bboxes (the first box in
each detection's bbox string), clamped to [0, 1], None when nothing parses or
the envelope is degenerate. Keep this in sync with pyro-api — parity is the
whole point of replay.

Known, accepted limit (spec): detections that arrived after the last
production scoring can shift the reconstruction; the replay_matches check
makes that visible.
"""

from __future__ import annotations

import re
from ast import literal_eval

from temporal_model.monitor.store import FrameMeta

# pyro-api src/app/services/temporal.py
MIN_FRAMES = 4
MAX_FRAMES = 10

# pyro-api src/app/schemas/detections.py (verbatim)
FLOAT_PATTERN = r"(0?\.[0-9]{1,3}|0|1)"
BOX_PATTERN = (
    rf"\({FLOAT_PATTERN},{FLOAT_PATTERN},{FLOAT_PATTERN},"
    rf"{FLOAT_PATTERN},{FLOAT_PATTERN}\)"
)


def extract_bbox_strings(bboxes: str) -> list[str]:
    return [match.group(0) for match in re.finditer(BOX_PATTERN, bboxes)]


def parse_bbox(bbox_str: str) -> tuple[float, float, float, float, float] | None:
    """Parse one '(xmin,ymin,xmax,ymax,conf)' string; None when malformed.

    pyro-api raises HTTP 422 here; the worker catches it and skips the bbox —
    returning None reproduces the skip without the exception plumbing.
    """
    try:
        bbox = literal_eval(bbox_str)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(bbox, tuple) or len(bbox) != 5:
        return None
    return bbox


def frames_and_roi(
    frames: list[FrameMeta], last_n: int | None = MAX_FRAMES
) -> tuple[int, list[str], list[float] | None]:
    """(total_distinct, kept_frame_keys, roi_xyxyn) for a stored sequence.

    ``frames`` must be ordered by created_at ascending (the store writes them
    that way), matching the worker's ``order_by="created_at"`` fetch.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    corners_by_frame: dict[str, list[tuple[float, float, float, float]]] = {}
    for f in frames:
        if f.bucket_key not in seen:
            seen.add(f.bucket_key)
            ordered.append(f.bucket_key)
        bbox_strs = extract_bbox_strings(f.bbox)
        if bbox_strs:
            parsed = parse_bbox(bbox_strs[0])
            if parsed is not None:
                xmin, ymin, xmax, ymax, _ = parsed
                corners_by_frame.setdefault(f.bucket_key, []).append(
                    (xmin, ymin, xmax, ymax)
                )
    total = len(ordered)
    kept = ordered if last_n is None or total <= last_n else ordered[-last_n:]
    corners = [c for fr in kept for c in corners_by_frame.get(fr, [])]
    if not corners:
        return total, kept, None
    roi = [
        max(0.0, min(c[0] for c in corners)),
        max(0.0, min(c[1] for c in corners)),
        min(1.0, max(c[2] for c in corners)),
        min(1.0, max(c[3] for c in corners)),
    ]
    if not (roi[0] < roi[2] and roi[1] < roi[3]):
        return total, kept, None
    return total, kept, roi
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd monitor && uv run pytest tests/test_reconstruct.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
cd monitor && make lint
git add monitor/src/temporal_model/monitor/reconstruct.py monitor/tests/test_reconstruct.py
git commit -m "feat(monitor): port pyro-api frame/ROI reconstruction"
```

---

### Task 6: Stabilized-window geometry (`geometry.py`)

Port of `core/src/temporal_model/core/stabilize.py` operating on the API's tube-entry **dicts** (`{"bbox": [cx,cy,w,h] | null, "is_gap": bool, ...}`) instead of core's dataclasses. Window is normalized `(cx, cy, w, h)` — the union of observed (non-gap) boxes, falling back to all available boxes.

**Files:**
- Create: `monitor/src/temporal_model/monitor/geometry.py`
- Test: `monitor/tests/test_geometry.py`

- [ ] **Step 1: Write the failing tests**

`monitor/tests/test_geometry.py`:

```python
import pytest

from temporal_model.monitor.geometry import tube_stabilized_window, union_window


def entry(bbox, is_gap=False):
    return {"frame_idx": 0, "bbox": bbox, "is_gap": is_gap, "confidence": 0.5}


def test_union_window_single_box_is_identity():
    assert union_window([(0.5, 0.5, 0.2, 0.1)]) == (0.5, 0.5, 0.2, 0.1)


def test_union_window_encloses():
    # box A spans x [0.1, 0.3], box B spans x [0.4, 0.6] -> union x [0.1, 0.6]
    a = (0.2, 0.2, 0.2, 0.2)
    b = (0.5, 0.5, 0.2, 0.2)
    cx, cy, w, h = union_window([a, b])
    assert (round(cx, 6), round(cy, 6)) == (0.35, 0.35)
    assert (round(w, 6), round(h, 6)) == (0.5, 0.5)


def test_union_window_empty_raises():
    with pytest.raises(ValueError):
        union_window([])


def test_stabilized_window_ignores_gap_boxes():
    observed = (0.2, 0.2, 0.2, 0.2)
    gap = (0.8, 0.8, 0.1, 0.1)  # interpolated — must not widen the window
    win = tube_stabilized_window([entry(list(observed)), entry(list(gap), is_gap=True)])
    assert win == observed


def test_stabilized_window_falls_back_to_gap_boxes():
    gap = (0.8, 0.8, 0.1, 0.1)
    win = tube_stabilized_window([entry(None), entry(list(gap), is_gap=True)])
    assert win == gap


def test_stabilized_window_none_without_any_box():
    assert tube_stabilized_window([entry(None), entry(None, is_gap=True)]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd monitor && uv run pytest tests/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `monitor/src/temporal_model/monitor/geometry.py`**

```python
"""Stabilized crop window, recomputed from verbose-response tube entries.

Port of ``core/src/temporal_model/core/stabilize.py`` (the API's verbose
details omit ``stabilized_window``; the viewer's crop panel needs it). Same
policy: union of the tube's observed (non-gap) boxes, falling back to the
union of all available boxes; None when the tube has no usable detection.
Operates on the API's entry dicts rather than core dataclasses.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]  # normalized (cx, cy, w, h)


def union_window(boxes: list[Box]) -> Box:
    """Enclosing box of ``boxes``; raises ValueError on empty input."""
    if not boxes:
        raise ValueError("union_window requires at least one box")
    x0 = min(cx - w / 2 for cx, _, w, _ in boxes)
    y0 = min(cy - h / 2 for _, cy, _, h in boxes)
    x1 = max(cx + w / 2 for cx, _, w, _ in boxes)
    y1 = max(cy + h / 2 for _, cy, _, h in boxes)
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def tube_stabilized_window(entries: list[dict]) -> Box | None:
    """Fixed crop window for one verbose-response tube, or None."""
    boxes = [
        (tuple(e["bbox"]) if e.get("bbox") is not None else None, bool(e["is_gap"]))
        for e in entries
    ]
    available = [b for b, _ in boxes if b is not None]
    observed = [b for b, is_gap in boxes if b is not None and not is_gap]
    chosen = observed or available
    return union_window(chosen) if chosen else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd monitor && uv run pytest tests/test_geometry.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
cd monitor && make lint
git add monitor/src/temporal_model/monitor/geometry.py monitor/tests/test_geometry.py
git commit -m "feat(monitor): stabilized-window geometry port"
```

---

### Task 7: Eval-viewer contract writers (`report.py`)

Reshape verbose API responses into the eval reporting contract (`viewer/lib/types.ts` is the source of truth) and write the per-org tree. The API's verbose `details.tubes` is a flat **list** without `first_crossing_frame`/`stabilized_window`, and `decision` lacks `trigger_tube_id` on releases ≤ v0.3.1 — the reshape fills/derives them.

**Files:**
- Create: `monitor/src/temporal_model/monitor/report.py`
- Test: `monitor/tests/test_report.py`

- [ ] **Step 1: Write the failing tests**

`monitor/tests/test_report.py`:

```python
import json

from temporal_model.monitor.report import (
    OrgReport,
    compute_outcome,
    reshape_details,
    result_row,
    write_report,
)
from temporal_model.monitor.store import FrameMeta, SequenceMeta

VERBOSE_RESPONSE = {
    "is_smoke": True,
    "probability": 0.93,
    "version": {"api": "0.3.1", "model": "0.1.0"},
    "details": {
        "decision": {
            "aggregation": "logistic",
            "threshold": 0.52,
            "threshold_overridden": False,
            "packaged_threshold": None,
        },
        "preprocessing": {
            "num_frames_input": 10,
            "num_truncated": 0,
            "padded_frame_indices": [],
            "num_tube_candidates": 3,
            "num_tubes_outside_roi": 1,
        },
        "tubes": [
            {
                "tube_id": 7,
                "start_frame": 2,
                "end_frame": 4,
                "logit": 3.41,
                "probability": 0.93,
                "entries": [
                    {
                        "frame_idx": 2,
                        "bbox": [0.2, 0.2, 0.2, 0.2],
                        "is_gap": False,
                        "confidence": 0.81,
                    },
                    {
                        "frame_idx": 3,
                        "bbox": [0.5, 0.5, 0.2, 0.2],
                        "is_gap": False,
                        "confidence": 0.7,
                    },
                ],
            }
        ],
        "profiling": None,
    },
}


def make_meta() -> SequenceMeta:
    return SequenceMeta(
        key="platform_42307",
        sequence_id=42307,
        label="smoke",
        camera_name="donon-sarrebourg-01",
        organization_name="sis-67",
        started_at="2026-05-15T13:08:18",
        temporal_model_score=0.93,
        temporal_model_version="0.1.0",
        temporal_api_version="0.3.1",
        frames=[
            FrameMeta(
                file="images/detection_100.jpg",
                detection_id=100,
                created_at="2026-05-15T13:08:18",
                bucket_key="cam122/frame-100.jpg",
                bbox="[(0.1,0.2,0.3,0.4,0.9)]",
            )
        ],
    )


def test_compute_outcome_matches_eval():
    assert compute_outcome("keep", "smoke") == "kept-smoke"
    assert compute_outcome("discard", "smoke") == "discarded-smoke"
    assert compute_outcome("keep", "fp") == "kept-fp"
    assert compute_outcome("discard", "fp") == "discarded-fp"
    assert compute_outcome("keep", "unknown") == "n/a"


def test_reshape_details_to_eval_shape():
    details = reshape_details(VERBOSE_RESPONSE["details"])
    assert set(details) == {"preprocessing", "tubes", "decision"}
    assert details["preprocessing"] == {
        "num_frames_input": 10,
        "num_truncated": 0,
        "padded_frame_indices": [],
    }
    assert details["tubes"]["num_candidates"] == 3
    assert details["tubes"]["num_outside_roi"] == 1
    kept = details["tubes"]["kept"]
    assert len(kept) == 1
    tube = kept[0]
    # absent on <= v0.3.1 -> filled with None / derived
    assert tube["first_crossing_frame"] is None
    cx, cy, w, h = tube["stabilized_window"]
    assert (round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)) == (
        0.35,
        0.35,
        0.5,
        0.5,
    )
    assert details["decision"] == {
        "aggregation": "logistic",
        "threshold": 0.52,
        "trigger_tube_id": None,
    }


def test_reshape_preserves_trigger_fields_when_present():
    resp = json.loads(json.dumps(VERBOSE_RESPONSE))  # deep copy
    resp["details"]["decision"]["trigger_tube_id"] = 7
    resp["details"]["tubes"][0]["first_crossing_frame"] = 3
    details = reshape_details(resp["details"])
    assert details["decision"]["trigger_tube_id"] == 7
    assert details["tubes"]["kept"][0]["first_crossing_frame"] == 3


def test_result_row():
    meta = make_meta()
    details = reshape_details(VERBOSE_RESPONSE["details"])
    row = result_row(
        meta=meta,
        response=VERBOSE_RESPONSE,
        details=details,
        replay_matches=True,
    )
    assert row == {
        "key": "platform_42307",
        "source": "sis-67",
        "label": "smoke",
        "decision": "keep",
        "outcome": "kept-smoke",
        "score": 3.41,
        "probability": 0.93,
        "num_tubes_kept": 1,
        "trigger_frame_index": None,
        "organization_name": "sis-67",
        "camera_name": "donon-sarrebourg-01",
        "started_at": "2026-05-15T13:08:18",
        "recorded_probability": 0.93,
        "replay_matches": True,
        "temporal_model_version": "0.1.0",
        "temporal_api_version": "0.3.1",
    }


def test_write_report_tree(tmp_path):
    meta = make_meta()
    details = reshape_details(VERBOSE_RESPONSE["details"])
    row = result_row(
        meta=meta, response=VERBOSE_RESPONSE, details=details, replay_matches=True
    )
    report = OrgReport(org_slug="sis-67")
    report.add(
        row=row,
        details=details,
        view={
            "key": "platform_42307",
            "source": "sis-67",
            "label": "smoke",
            "organization_name": "sis-67",
            "camera_name": "donon-sarrebourg-01",
            "started_at": "2026-05-15T13:08:18",
            "frames": [
                "data/01_raw/sequences/sis-67/donon-sarrebourg-01/seq_42307/images/detection_100.jpg"
            ],
        },
        model_config={"variant": "vit_dinov2_finetune"},
    )
    report.drop("platform_999", "no_temporal_version")
    write_report(tmp_path, report)

    out = tmp_path / "sis-67" / "vit_dinov2_finetune"
    rows = json.loads((out / "results.json").read_text())
    assert rows[0]["key"] == "platform_42307"
    written = json.loads((out / "details" / "platform_42307.json").read_text())
    assert written["tubes"]["kept"][0]["tube_id"] == 7
    view = json.loads((out / "sequences" / "platform_42307.json").read_text())
    assert view["frames"][0].startswith("data/01_raw/sequences/")
    assert json.loads((out / "model_config.json").read_text()) == {
        "variant": "vit_dinov2_finetune"
    }
    assert json.loads((out / "dropped.json").read_text()) == [
        {"sequence_id": "platform_999", "reason": "no_temporal_version"}
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd monitor && uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `monitor/src/temporal_model/monitor/report.py`**

```python
"""Writers for the eval-viewer reporting contract, one tree per organization.

Layout (read by viewer/ with DATA_ROOT=../monitor — see viewer/lib/paths.ts):
``data/08_reporting/<org_slug>/vit_dinov2_finetune/{results.json, details/,
sequences/, model_config.json, dropped.json}``. Shapes mirror
``eval/src/temporal_model/eval/evaluate.py`` rows and
``core/src/temporal_model/core/details_schema.py`` details; results rows add
the monitor-only provenance columns (recorded_probability, replay_matches,
temporal_*_version), which the viewer treats as optional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from temporal_model.monitor.geometry import tube_stabilized_window
from temporal_model.monitor.store import SequenceMeta, slugify

MODEL_DIR = "vit_dinov2_finetune"  # viewer/lib/paths.ts MODEL_NAME


def decision_from_output(is_smoke: bool) -> str:
    return "keep" if is_smoke else "discard"


def compute_outcome(decision: str, label: str) -> str:
    """Copy of eval's outcomes.compute_outcome (same strings, same n/a rule)."""
    if label == "smoke":
        return "kept-smoke" if decision == "keep" else "discarded-smoke"
    if label == "fp":
        return "kept-fp" if decision == "keep" else "discarded-fp"
    return "n/a"


def reshape_details(api_details: dict[str, Any]) -> dict[str, Any]:
    """Verbose /predict ``details`` -> eval ``BboxTubeDetails`` shape.

    Differences bridged: the API nests tube counters under preprocessing and
    flattens tubes to a list; trigger fields exist only on releases with
    compute_trigger (absent -> None); stabilized_window is never in the API
    response (derived here).
    """
    pre = api_details["preprocessing"]
    dec = api_details["decision"]
    kept = [
        {
            "tube_id": t["tube_id"],
            "start_frame": t["start_frame"],
            "end_frame": t["end_frame"],
            "logit": t["logit"],
            "probability": t.get("probability"),
            "first_crossing_frame": t.get("first_crossing_frame"),
            "entries": t["entries"],
            "stabilized_window": tube_stabilized_window(t["entries"]),
        }
        for t in api_details["tubes"]
    ]
    return {
        "preprocessing": {
            "num_frames_input": pre["num_frames_input"],
            "num_truncated": pre["num_truncated"],
            "padded_frame_indices": pre["padded_frame_indices"],
        },
        "tubes": {
            "num_candidates": pre.get("num_tube_candidates", 0),
            "num_outside_roi": pre.get("num_tubes_outside_roi", 0),
            "kept": kept,
        },
        "decision": {
            "aggregation": dec["aggregation"],
            "threshold": dec["threshold"],
            "trigger_tube_id": dec.get("trigger_tube_id"),
        },
    }


def result_row(
    *,
    meta: SequenceMeta,
    response: dict[str, Any],
    details: dict[str, Any],
    replay_matches: bool | None,
) -> dict[str, Any]:
    """One results.json row: eval columns + monitor provenance extras."""
    kept = details["tubes"]["kept"]
    decision = decision_from_output(response["is_smoke"])
    return {
        "key": meta.key,
        # must equal the reporting tree's <org_slug> dir — the viewer filters
        # rows by string equality with the directory-derived source name
        "source": slugify(meta.organization_name),
        "label": meta.label,
        "decision": decision,
        "outcome": compute_outcome(decision, meta.label),
        "score": max(t["logit"] for t in kept) if kept else None,
        "probability": response.get("probability"),
        "num_tubes_kept": len(kept),
        "trigger_frame_index": response.get("trigger_frame_index"),
        "organization_name": meta.organization_name,
        "camera_name": meta.camera_name,
        "started_at": meta.started_at,
        "recorded_probability": meta.temporal_model_score,
        "replay_matches": replay_matches,
        "temporal_model_version": meta.temporal_model_version,
        "temporal_api_version": meta.temporal_api_version,
    }


@dataclass
class OrgReport:
    """Accumulates one organization's rows/details/views before writing."""

    org_slug: str
    rows: list[dict] = field(default_factory=list)
    details_by_key: dict[str, dict] = field(default_factory=dict)
    views_by_key: dict[str, dict] = field(default_factory=dict)
    model_config: dict | None = None
    dropped: list[dict] = field(default_factory=list)

    def add(self, *, row: dict, details: dict, view: dict, model_config: dict) -> None:
        self.rows.append(row)
        self.details_by_key[row["key"]] = details
        self.views_by_key[row["key"]] = view
        if self.model_config is None:
            self.model_config = model_config

    def drop(self, key: str, reason: str) -> None:
        # eval's dropped.json field name is sequence_id; keep it for tooling parity
        self.dropped.append({"sequence_id": key, "reason": reason})


def write_report(output_dir: Path, report: OrgReport) -> None:
    out = output_dir / report.org_slug / MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report.rows, indent=2))
    (out / "model_config.json").write_text(
        json.dumps(report.model_config or {}, indent=2)
    )
    (out / "dropped.json").write_text(json.dumps(report.dropped, indent=2))
    details_dir = out / "details"
    details_dir.mkdir(exist_ok=True)
    for key, details in report.details_by_key.items():
        (details_dir / f"{key}.json").write_text(json.dumps(details, indent=2))
    sequences_dir = out / "sequences"
    sequences_dir.mkdir(exist_ok=True)
    for key, view in report.views_by_key.items():
        (sequences_dir / f"{key}.json").write_text(json.dumps(view, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd monitor && uv run pytest tests/test_report.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
cd monitor && make lint
git add monitor/src/temporal_model/monitor/report.py monitor/tests/test_report.py
git commit -m "feat(monitor): eval-viewer contract writers"
```

---

### Task 8: Replay stack (`stack.py` + `docker-compose.yml`)

Pinned api image + MinIO via a monitor-owned compose file. Ports 18000/19000 to avoid colliding with `api/docker-compose.yml` (8000/9000). The model is **baked into the image** (`api/Dockerfile` `COPY api/models/model.zip /models/model.zip` — `TEMPORAL_API_MODEL_PATH` defaults there), so no model volume.

**Files:**
- Create: `monitor/docker-compose.yml`
- Create: `monitor/src/temporal_model/monitor/stack.py`
- Test: `monitor/tests/test_stack.py`

- [ ] **Step 1: Create `monitor/docker-compose.yml`**

```yaml
# Replay stack: the pinned release image (model.zip baked in) + throwaway MinIO.
# Image tag comes from MONITOR_API_IMAGE (set by `temporal-monitor replay` per
# api-version group). Ports are offset from api/docker-compose.yml so both
# stacks can run side by side.
services:
  api:
    image: ${MONITOR_API_IMAGE:?set by temporal-monitor replay}
    ports:
      - "18000:8000"
    environment:
      - TEMPORAL_API_S3_BUCKET=frames
      - TEMPORAL_API_S3_ENDPOINT_URL=http://minio:9000
      - TEMPORAL_API_S3_REGION=us-east-1
      # local replay only — never reused for shared environments
      - AWS_ACCESS_KEY_ID=minioadmin
      - AWS_SECRET_ACCESS_KEY=minioadmin
    depends_on:
      - minio

  minio:
    image: minio/minio:latest
    command: server /data
    ports:
      - "19000:9000"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin

  # One-shot: wait for MinIO, then create the bucket the API reads from.
  createbuckets:
    image: minio/mc:latest
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
      until mc alias set local http://minio:9000 minioadmin minioadmin; do
        echo 'waiting for minio...'; sleep 1;
      done;
      mc mb --ignore-existing local/frames;
      echo 'bucket frames ready';
      "
```

- [ ] **Step 2: Write the failing tests**

`monitor/tests/test_stack.py`:

```python
from pathlib import Path
from unittest.mock import patch

import pytest

from temporal_model.monitor.stack import (
    API_URL,
    BUCKET,
    ReplayStack,
    StackError,
)


def test_compose_commands_pin_the_image_tag():
    stack = ReplayStack(Path("dc.yml"), version="0.3.1")
    with patch("temporal_model.monitor.stack.subprocess.run") as run:
        stack.up()
        stack.down()
    up_call, down_call = run.call_args_list
    assert up_call.args[0] == [
        "docker",
        "compose",
        "-f",
        "dc.yml",
        "-p",
        "temporal-monitor-replay",
        "up",
        "-d",
    ]
    assert (
        up_call.kwargs["env"]["MONITOR_API_IMAGE"]
        == "pyronear/temporal-model-api:0.3.1"
    )
    assert up_call.kwargs["check"] is True
    assert down_call.args[0][-2:] == ["down", "-v"]


def test_wait_healthy_polls_until_model_loaded():
    stack = ReplayStack(Path("dc.yml"), version="0.3.1")
    responses = iter(
        [
            ConnectionError("not up yet"),
            {"status": "ok", "model_loaded": False},
            {"status": "ok", "model_loaded": True, "model_version": "0.1.0"},
        ]
    )

    def fake_health():
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    with patch.object(stack, "_fetch_health", side_effect=fake_health):
        health = stack.wait_healthy(timeout_s=5, poll_s=0)
    assert health["model_version"] == "0.1.0"


def test_wait_healthy_times_out():
    stack = ReplayStack(Path("dc.yml"), version="0.3.1")
    with (
        patch.object(stack, "_fetch_health", side_effect=ConnectionError),
        pytest.raises(StackError, match="health"),
    ):
        stack.wait_healthy(timeout_s=0.05, poll_s=0.01)


def test_upload_frames_puts_each_key_once(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"a")
    (tmp_path / "b.jpg").write_bytes(b"b")
    stack = ReplayStack(Path("dc.yml"), version="0.3.1")
    with patch.object(stack, "_s3_client") as make_client:
        stack.upload_frames(
            {"cam/k1.jpg": tmp_path / "a.jpg", "cam/k2.jpg": tmp_path / "b.jpg"}
        )
    uploaded = [c.args for c in make_client.return_value.upload_file.call_args_list]
    assert sorted(u[2] for u in uploaded) == ["cam/k1.jpg", "cam/k2.jpg"]
    assert all(u[1] == BUCKET for u in uploaded)


def test_api_url_uses_offset_port():
    assert API_URL == "http://localhost:18000"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd monitor && uv run pytest tests/test_stack.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `monitor/src/temporal_model/monitor/stack.py`**

```python
"""Docker compose lifecycle for the pinned-release replay stack.

One ReplayStack per api-version group: ``up()`` starts
``pyronear/temporal-model-api:<version>`` (model.zip baked into the image) +
a throwaway MinIO; ``upload_frames`` puts a sequence's frames under their
ORIGINAL bucket_keys so the request body is byte-identical to production's;
``down()`` removes containers and the MinIO volume.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import boto3
import requests

IMAGE_REPO = "pyronear/temporal-model-api"
COMPOSE_PROJECT = "temporal-monitor-replay"
API_URL = "http://localhost:18000"
MINIO_URL = "http://localhost:19000"
BUCKET = "frames"
_MINIO_CREDS = {"aws_access_key_id": "minioadmin", "aws_secret_access_key": "minioadmin"}


class StackError(RuntimeError):
    pass


class ReplayStack:
    def __init__(self, compose_file: Path, version: str) -> None:
        self.compose_file = compose_file
        self.version = version

    def _compose(self, *args: str) -> None:
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(self.compose_file),
                "-p",
                COMPOSE_PROJECT,
                *args,
            ],
            env={**os.environ, "MONITOR_API_IMAGE": f"{IMAGE_REPO}:{self.version}"},
            check=True,
        )

    def up(self) -> None:
        """Start the stack; raises CalledProcessError on pull/start failure."""
        self._compose("up", "-d")

    def down(self) -> None:
        self._compose("down", "-v")

    def _fetch_health(self) -> dict:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def wait_healthy(self, timeout_s: float = 300, poll_s: float = 2) -> dict:
        """Poll /health until model_loaded; returns the health payload."""
        deadline = time.monotonic() + timeout_s
        last: str = "no response"
        while time.monotonic() < deadline:
            try:
                health = self._fetch_health()
            except Exception as exc:  # noqa: BLE001 — keep polling until deadline
                last = repr(exc)
            else:
                if health.get("model_loaded"):
                    return health
                last = str(health)
            time.sleep(poll_s)
        raise StackError(f"api at {API_URL} never became healthy: {last}")

    def _s3_client(self):
        return boto3.client("s3", endpoint_url=MINIO_URL, **_MINIO_CREDS)

    def upload_frames(self, files_by_key: dict[str, Path]) -> None:
        """Upload local frame files under their original S3 bucket_keys."""
        client = self._s3_client()
        for key, path in files_by_key.items():
            client.upload_file(str(path), BUCKET, key)
```

(`_s3_client` is a method, not a property, so the test's
`patch.object(stack, "_s3_client")` can intercept it.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd monitor && uv run pytest tests/test_stack.py -v`
Expected: all PASS

- [ ] **Step 6: Lint and commit**

```bash
cd monitor && make lint
git add monitor/docker-compose.yml monitor/src/temporal_model/monitor/stack.py monitor/tests/test_stack.py
git commit -m "feat(monitor): pinned-release replay stack"
```

---

### Task 9: Replay orchestration (`replay.py` + cli wiring)

Group → stack per version → health/model check → upload → predict → consistency check → per-org report. All drop reasons from the spec: `no_temporal_version`, `image_pull_failed`, `model_version_mismatch`, `too_few_frames`, `no_images`, `predict_failed`.

**Files:**
- Create: `monitor/src/temporal_model/monitor/replay.py`
- Modify: `monitor/src/temporal_model/monitor/cli.py` (replace the `replay` stub)
- Test: `monitor/tests/test_replay.py`

- [ ] **Step 1: Write the failing tests**

`monitor/tests/test_replay.py`:

```python
import json
import subprocess
from pathlib import Path

from temporal_model.monitor.replay import SCORE_TOLERANCE, run_replay
from temporal_model.monitor.store import (
    FrameMeta,
    SequenceMeta,
    sequence_dir,
    write_meta,
)

VERBOSE_DETAILS = {
    "decision": {"aggregation": "logistic", "threshold": 0.52},
    "preprocessing": {
        "num_frames_input": 4,
        "num_truncated": 0,
        "padded_frame_indices": [],
        "num_tube_candidates": 1,
        "num_tubes_outside_roi": 0,
    },
    "tubes": [
        {
            "tube_id": 0,
            "start_frame": 0,
            "end_frame": 3,
            "logit": 3.41,
            "probability": 0.93,
            "entries": [
                {"frame_idx": 0, "bbox": [0.2, 0.2, 0.2, 0.2], "is_gap": False, "confidence": 0.8}
            ],
        }
    ],
}


def store_sequence(
    store: Path,
    sequence_id: int,
    *,
    api_version: str | None = "0.3.1",
    model_version: str | None = "0.1.0",
    score: float | None = 0.93,
    n_frames: int = 4,
    with_images: bool = True,
) -> SequenceMeta:
    meta = SequenceMeta(
        key=f"platform_{sequence_id}",
        sequence_id=sequence_id,
        label="smoke",
        label_detail="wildfire_smoke",
        camera_name="cam-01",
        organization_name="sis-67",
        started_at="2026-05-15T13:08:18",
        temporal_model_score=score,
        temporal_model_version=model_version,
        temporal_api_version=api_version,
        frames=[
            FrameMeta(
                file=f"images/detection_{i}.jpg",
                detection_id=i,
                created_at=f"2026-05-15T13:{i:02d}:00",
                bucket_key=f"cam/seq{sequence_id}-f{i}.jpg",
                bbox="[(0.1,0.1,0.3,0.3,0.9)]",
            )
            for i in range(n_frames)
        ],
    )
    seq_dir = sequence_dir(store, meta)
    write_meta(seq_dir, meta)
    if with_images:
        (seq_dir / "images").mkdir(exist_ok=True)
        for i in range(n_frames):
            (seq_dir / "images" / f"detection_{i}.jpg").write_bytes(b"jpg")
    return meta


class FakeStack:
    instances: list["FakeStack"] = []
    fail_versions: set[str] = set()

    def __init__(self, compose_file, version):
        self.version = version
        self.uploaded: list[dict] = []
        self.up_called = self.down_called = False
        FakeStack.instances.append(self)

    def up(self):
        self.up_called = True
        if self.version in FakeStack.fail_versions:
            raise subprocess.CalledProcessError(1, ["docker"])

    def down(self):
        self.down_called = True

    def wait_healthy(self, **kwargs):
        return {"status": "ok", "model_loaded": True, "model_version": "0.1.0"}

    def upload_frames(self, files_by_key):
        self.uploaded.append(dict(files_by_key))


def fake_predict_ok(frames, roi_xyxyn):
    return {
        "is_smoke": True,
        "probability": 0.93,
        "version": {"api": "0.3.1", "model": "0.1.0"},
        "details": VERBOSE_DETAILS,
    }


def run(store, out, predict=fake_predict_ok):
    FakeStack.instances = []
    return run_replay(
        store_dir=store,
        output_dir=out,
        compose_file=Path("dc.yml"),
        stack_factory=FakeStack,
        predict=predict,
    )


def test_happy_path_writes_org_tree(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1)
    summary = run(store, out)
    assert summary["replayed"] == 1
    assert summary["mismatched"] == 0
    rows = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "results.json").read_text()
    )
    assert rows[0]["replay_matches"] is True
    assert rows[0]["recorded_probability"] == 0.93
    stack = FakeStack.instances[0]
    assert stack.up_called and stack.down_called
    # 4 distinct keys uploaded once
    assert len(stack.uploaded[0]) == 4
    view = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "sequences" / "platform_1.json").read_text()
    )
    # viewer frames = the kept (replayed) frames, relative to monitor/
    assert view["frames"] == [
        f"data/01_raw/sequences/sis-67/cam-01/seq_1/images/detection_{i}.jpg"
        for i in range(4)
    ]


def test_groups_by_api_version_one_stack_each(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1, api_version="0.3.0")
    store_sequence(store, 2, api_version="0.3.1")
    store_sequence(store, 3, api_version="0.3.1")
    run(store, out)
    assert sorted(s.version for s in FakeStack.instances) == ["0.3.0", "0.3.1"]


def test_drop_reasons(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1, api_version=None)  # no_temporal_version
    store_sequence(store, 2, n_frames=2)  # too_few_frames
    store_sequence(store, 3, with_images=False)  # no_images
    store_sequence(store, 4, model_version="9.9.9")  # model_version_mismatch
    store_sequence(store, 5)  # ok
    summary = run(store, out)
    assert summary["replayed"] == 1
    dropped = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "dropped.json").read_text()
    )
    reasons = {d["sequence_id"]: d["reason"] for d in dropped}
    assert reasons == {
        "platform_1": "no_temporal_version",
        "platform_2": "too_few_frames",
        "platform_3": "no_images",
        "platform_4": "model_version_mismatch",
    }


def test_image_pull_failure_drops_whole_group(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1, api_version="0.0.9")
    FakeStack.fail_versions = {"0.0.9"}
    try:
        summary = run(store, out)
    finally:
        FakeStack.fail_versions = set()
    assert summary["replayed"] == 0
    dropped = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "dropped.json").read_text()
    )
    assert dropped[0]["reason"] == "image_pull_failed"


def test_predict_failure_drops_sequence_and_continues(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1)
    store_sequence(store, 2)
    calls = []

    def flaky_predict(frames, roi_xyxyn):
        calls.append(frames)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return fake_predict_ok(frames, roi_xyxyn)

    summary = run(store, out, predict=flaky_predict)
    assert summary["replayed"] == 1
    dropped = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "dropped.json").read_text()
    )
    assert dropped[0]["reason"] == "predict_failed"


def test_score_mismatch_flagged(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store_sequence(store, 1, score=0.5)  # recorded 0.5, replay says 0.93
    summary = run(store, out)
    assert summary["mismatched"] == 1
    rows = json.loads(
        (out / "sis-67" / "vit_dinov2_finetune" / "results.json").read_text()
    )
    assert rows[0]["replay_matches"] is False
    assert SCORE_TOLERANCE == 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd monitor && uv run pytest tests/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `monitor/src/temporal_model/monitor/replay.py`**

```python
"""Replay stored sequences through their pinned api release, write reports.

Flow: group sequences by recorded temporal_api_version -> one compose stack
per group (image tag == version; model.zip is baked into the image) -> verify
/health model_version matches each sequence's recorded one -> upload frames
under their original bucket_keys -> POST /predict?verbose=true&
compute_trigger=true (older releases ignore the unknown params; the trigger
fields are then simply absent) -> compare the replayed probability to the
recorded score -> write one eval-viewer tree per organization.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Callable, Protocol

import requests

from temporal_model.monitor import reconstruct
from temporal_model.monitor.report import (
    MODEL_DIR,
    OrgReport,
    reshape_details,
    result_row,
    write_report,
)
from temporal_model.monitor.stack import API_URL, BUCKET, ReplayStack
from temporal_model.monitor.store import (
    SequenceMeta,
    iter_metas,
    slugify,
)

logger = logging.getLogger(__name__)

SCORE_TOLERANCE = 1e-6
STORE_REL = "data/01_raw/sequences"  # viewer frame paths are relative to monitor/


class _StackFactory(Protocol):
    def __call__(self, compose_file: Path, version: str) -> Any: ...


def _default_predict(frames: list[str], roi_xyxyn: list[float] | None) -> dict:
    body: dict[str, Any] = {"bucket": BUCKET, "frames": frames}
    if roi_xyxyn is not None:
        body["roi_xyxyn"] = roi_xyxyn
    resp = requests.post(
        f"{API_URL}/predict",
        params={"verbose": "true", "compute_trigger": "true"},
        json=body,
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def _org_report(reports: dict[str, OrgReport], meta: SequenceMeta) -> OrgReport:
    org = slugify(meta.organization_name)
    if org not in reports:
        reports[org] = OrgReport(org_slug=org)
    return reports[org]


def _files_by_key(seq_dir: Path, meta: SequenceMeta, kept: list[str]) -> dict[str, Path]:
    """First stored file per kept bucket_key (several detections may share one)."""
    files: dict[str, Path] = {}
    for f in meta.frames:
        if f.bucket_key in kept and f.bucket_key not in files:
            files[f.bucket_key] = seq_dir / f.file
    return files


def run_replay(
    *,
    store_dir: Path,
    output_dir: Path,
    compose_file: Path,
    stack_factory: _StackFactory = ReplayStack,
    predict: Callable[[list[str], list[float] | None], dict] = _default_predict,
) -> dict[str, int]:
    reports: dict[str, OrgReport] = {}
    groups: dict[str, list[tuple[Path, SequenceMeta]]] = {}
    replayed = mismatched = dropped = 0

    for seq_dir, meta in iter_metas(store_dir):
        if not meta.temporal_api_version:
            _org_report(reports, meta).drop(meta.key, "no_temporal_version")
            dropped += 1
            continue
        groups.setdefault(meta.temporal_api_version, []).append((seq_dir, meta))

    for version in sorted(groups):
        items = groups[version]
        stack = stack_factory(compose_file, version)
        try:
            stack.up()
        except subprocess.CalledProcessError:
            logger.error("could not start image for api version %s", version)
            for _, meta in items:
                _org_report(reports, meta).drop(meta.key, "image_pull_failed")
                dropped += 1
            continue
        try:
            health = stack.wait_healthy()
            for seq_dir, meta in items:
                outcome = _replay_one(stack, health, seq_dir, meta, reports, predict)
                if outcome == "ok":
                    replayed += 1
                elif outcome == "mismatch":
                    replayed += 1
                    mismatched += 1
                else:
                    dropped += 1
        finally:
            stack.down()

    for report in reports.values():
        write_report(output_dir, report)
    summary = {"replayed": replayed, "mismatched": mismatched, "dropped": dropped}
    logger.info(
        "replay done: %(replayed)d replayed (%(mismatched)d score mismatches), "
        "%(dropped)d dropped",
        summary,
    )
    return summary


def _replay_one(
    stack: Any,
    health: dict,
    seq_dir: Path,
    meta: SequenceMeta,
    reports: dict[str, OrgReport],
    predict: Callable[[list[str], list[float] | None], dict],
) -> str:
    report = _org_report(reports, meta)
    if health.get("model_version") != meta.temporal_model_version:
        report.drop(meta.key, "model_version_mismatch")
        return "model_version_mismatch"
    total, kept, roi = reconstruct.frames_and_roi(meta.frames)
    if total < reconstruct.MIN_FRAMES:
        report.drop(meta.key, "too_few_frames")
        return "too_few_frames"
    files = _files_by_key(seq_dir, meta, kept)
    if len(files) < len(kept) or not all(p.is_file() for p in files.values()):
        report.drop(meta.key, "no_images")
        return "no_images"
    try:
        stack.upload_frames(files)
        response = predict(kept, roi)
    except Exception:  # noqa: BLE001 — one bad sequence must not stop the run
        logger.exception("predict failed for %s", meta.key)
        report.drop(meta.key, "predict_failed")
        return "predict_failed"

    details = reshape_details(response["details"])
    matches = _score_matches(meta.temporal_model_score, response.get("probability"))
    org = slugify(meta.organization_name)
    cam = slugify(meta.camera_name)
    # The frames the model actually saw, in request order, as paths relative
    # to monitor/ (the viewer resolves them against DATA_ROOT).
    frames_rel = []
    for key in kept:
        rel = files[key].relative_to(seq_dir)  # e.g. images/detection_100.jpg
        frames_rel.append(
            f"{STORE_REL}/{org}/{cam}/seq_{meta.sequence_id}/{rel.as_posix()}"
        )
    view = {
        "key": meta.key,
        # like result_row's source: must equal the reporting tree's <org_slug>
        "source": org,
        "label": meta.label,
        "organization_name": meta.organization_name,
        "camera_name": meta.camera_name,
        "started_at": meta.started_at,
        "frames": frames_rel,
    }
    report.add(
        row=result_row(
            meta=meta, response=response, details=details, replay_matches=matches
        ),
        details=details,
        view=view,
        model_config={
            "variant": MODEL_DIR,
            "decision": details["decision"],
            "model_version": health.get("model_version"),
            "api_version": health.get("api_version"),
        },
    )
    return "ok" if (matches is None or matches) else "mismatch"


def _score_matches(recorded: float | None, replayed: float | None) -> bool | None:
    if recorded is None:
        return None
    if replayed is None:
        return False
    return abs(recorded - replayed) <= SCORE_TOLERANCE
```

- [ ] **Step 4: Wire the cli `replay` subcommand** — in `monitor/src/temporal_model/monitor/cli.py` replace

```python
    elif args.command == "replay":
        raise SystemExit("replay is not implemented yet")
```

with

```python
    elif args.command == "replay":
        from temporal_model.monitor.replay import run_replay

        run_replay(
            store_dir=args.store,
            output_dir=args.output_dir,
            compose_file=args.compose_file,
        )
```

The `--compose-file` default from Task 4 already points at `monitor/docker-compose.yml` (created in Task 8). Guard it with a test:

```python
# append to monitor/tests/test_replay.py
from temporal_model.monitor.cli import _parse_args


def test_cli_replay_defaults_point_at_package_compose_file():
    args = _parse_args(["replay"])
    assert args.compose_file.name == "docker-compose.yml"
    assert args.compose_file.parent.name == "monitor"
    assert args.compose_file.is_file()
```

- [ ] **Step 5: Run the tests**

Run: `cd monitor && uv run pytest tests/test_replay.py -v`
Expected: all PASS

Run: `cd monitor && uv run pytest tests/ -v`
Expected: full suite PASS

- [ ] **Step 6: Lint and commit**

```bash
cd monitor && make lint
git add monitor/src/temporal_model/monitor/replay.py monitor/src/temporal_model/monitor/cli.py monitor/tests/test_replay.py
git commit -m "feat(monitor): version-pinned replay orchestration"
```

---

### Task 10: DVC repo, replay stage, Makefile import target, README

**Files:**
- Create: `monitor/.dvc/config` (+ `.dvcignore`) via `dvc init`
- Create: `monitor/dvc.yaml`
- Modify: `monitor/Makefile`
- Create: `monitor/README.md`

- [ ] **Step 1: Initialize the DVC repo** (same shape as `eval/.dvc/config`):

```bash
cd monitor
uv run dvc init --subdir
uv run dvc remote add -d s3remote s3://pyro-vision-rd/dvc/temporal-model/monitor/
uv run dvc config core.analytics false
```

Verify: `cat .dvc/config` shows

```
[core]
    remote = s3remote
    analytics = false
['remote "s3remote"']
    url = s3://pyro-vision-rd/dvc/temporal-model/monitor/
```

- [ ] **Step 2: Create `monitor/dvc.yaml`** (replay as a stage; outs CACHED — unlike eval's `cache: false`, monitor artifacts are pushed so teammates `dvc pull` instead of re-running Docker):

```yaml
stages:
  replay:
    cmd: >-
      uv run python -m temporal_model.monitor.cli replay
      --store data/01_raw/sequences
      --output-dir data/08_reporting
    deps:
      - src/temporal_model/monitor/cli.py
      - src/temporal_model/monitor/store.py
      - src/temporal_model/monitor/reconstruct.py
      - src/temporal_model/monitor/stack.py
      - src/temporal_model/monitor/replay.py
      - src/temporal_model/monitor/geometry.py
      - src/temporal_model/monitor/report.py
      - docker-compose.yml
      - data/01_raw/sequences
    outs:
      - data/08_reporting
```

- [ ] **Step 3: Add the `import` convenience target to `monitor/Makefile`**:

```make
.PHONY: install lint format test import

# ... existing targets unchanged ...

import: ## import new sequences from alert-api, then dvc add + push the store
	uv run temporal-monitor import $(ARGS)
	uv run dvc add data/01_raw/sequences
	uv run dvc push
```

(Usage: `make import` for yesterday..today, `make import ARGS="--date-from 2026-06-01 --date-to 2026-06-10"`.)

- [ ] **Step 4: Write `monitor/README.md`**:

````markdown
# 🔎🔥 monitor — production decision replay

Answers “what did the temporal model decide in production, and why?”.
Imports sequences scored by the deployed API (alert-api records the
probability + `version` provenance per sequence), replays them through the
**exact** pinned release image with `verbose=true&compute_trigger=true`, and
writes the eval-viewer contract so tubes/boxes/crops are explorable in
[`viewer/`](../viewer).

Design: [`docs/specs/2026-06-12-monitor-design.md`](../docs/specs/2026-06-12-monitor-design.md)

## Setup

```bash
make install                 # uv sync
cp .envrc.example .envrc     # fill in alert-api credentials (direnv loads it)
dvc pull                     # optional: fetch the shared store + reports
```

## Workflow

```bash
make import                                  # 1. fetch new sequences (incremental),
                                             #    dvc add + push the store
make import ARGS="--date-from 2026-06-01 --date-to 2026-06-10"  # backfill a range
uv run dvc repro                             # 2. replay through pinned releases
uv run dvc push                              # 3. share the artifacts
cd ../viewer && DATA_ROOT=../monitor npm run dev   # 4. browse at localhost:3000
```

`replay` groups sequences by their recorded `temporal_api_version`, runs
`pyronear/temporal-model-api:<tag>` (model.zip baked in) + a throwaway MinIO
on ports 18000/19000 (offset from `api/`'s 8000/9000), uploads each
sequence's frames under their original S3 keys, and reconstructs the exact
production call (last ≤10 distinct frames oldest-first, ROI = envelope of
the detections' primary bboxes — mirrors pyro-api's validation worker).
Docker must be running; each version group costs one image pull.

## Outputs (`data/08_reporting/<org>/vit_dinov2_finetune/`)

- `results.json` — eval columns + monitor extras: `recorded_probability`
  (what production stored), `replay_matches` (|Δ| ≤ 1e-6),
  `temporal_model_version`, `temporal_api_version`.
- `details/<key>.json` — tubes in the eval shape; `stabilized_window` is
  recomputed client-side; trigger fields appear for releases shipping
  `compute_trigger` (older images ignore the flag → “no trigger” in the
  viewer).
- `sequences/<key>.json`, `model_config.json`, `dropped.json` (skip reasons:
  `no_temporal_version`, `image_pull_failed`, `stack_unhealthy`,
  `model_version_mismatch`, `too_few_frames`, `no_images`, `predict_failed`).

A `replay_matches: false` row means the reconstruction diverged from the
recorded score — usually detections that arrived after the last production
scoring (known limitation; see the spec).

## Tests

```bash
make test    # offline: mocked HTTP, fake docker stack — no Docker needed
```
````

- [ ] **Step 5: Smoke-check the stage wiring** (no data yet, so just validate the yaml):

Run: `cd monitor && uv run dvc status`
Expected: complains about missing dep `data/01_raw/sequences` (or reports the stage as never run) — NOT a yaml parse error.

Create the store dir so the repo layout exists: `mkdir -p monitor/data/01_raw monitor/data/08_reporting` — the root `.gitignore` (`**/data/**` with `.dvc`-file exceptions) already keeps contents out of git.

- [ ] **Step 6: Commit**

```bash
git add monitor/.dvc/config monitor/.dvcignore monitor/dvc.yaml monitor/Makefile monitor/README.md
git commit -m "feat(monitor): dvc pipeline (replay stage) and workflow docs"
```

(`dvc init` may also create `.dvc/.gitignore` — add it too if present.)

---

### Task 11: Viewer — show monitor provenance columns when present

Additive only: eval data (no `recorded_probability` key) renders exactly as before. With monitor data, the table gains `prod prob` and `match` columns and the detail pane shows recorded-vs-replay + versions.

**Files:**
- Modify: `viewer/lib/types.ts` (ResultRow)
- Modify: `viewer/components/SequenceTable.tsx`
- Modify: `viewer/components/detail/DetailPanel.tsx`
- Test: `viewer/components/__tests__/SequenceTable.test.tsx` (extend)

- [ ] **Step 1: Extend `ResultRow` in `viewer/lib/types.ts`** — add optional monitor fields after `started_at`:

```ts
export interface ResultRow {
  key: string;
  source: string;
  label: Label;
  decision: Decision;
  outcome: Outcome;
  score: number | null;
  probability: number | null;
  num_tubes_kept: number;
  trigger_frame_index: number | null;
  organization_name: string | null;
  camera_name: string | null;
  started_at: string | null;
  // Monitor-only provenance (absent in eval reporting trees).
  recorded_probability?: number | null;
  replay_matches?: boolean | null;
  temporal_model_version?: string | null;
  temporal_api_version?: string | null;
}
```

- [ ] **Step 2: Write the failing test** — append to `viewer/components/__tests__/SequenceTable.test.tsx` (follow the existing render/assert style in that file; it uses vitest + testing-library):

```tsx
const monitorRow = {
  ...rows[0], // reuse an existing fixture row from this test file
  key: "platform_1",
  recorded_probability: 0.931,
  replay_matches: false,
};

it("shows provenance columns only for monitor rows", () => {
  // eval rows: no prod prob column
  render(
    <SequenceTable rows={rows} selectedKey={null} onSelect={() => {}} />,
  );
  expect(screen.queryByText("prod prob")).toBeNull();
  cleanup();
  // monitor rows: prod prob + match columns appear
  render(
    <SequenceTable
      rows={[monitorRow]}
      selectedKey={null}
      onSelect={() => {}}
    />,
  );
  expect(screen.getByText("prod prob")).toBeTruthy();
  expect(screen.getByText("0.931")).toBeTruthy();
  expect(screen.getByText("≠")).toBeTruthy(); // mismatch marker
});
```

Run: `cd viewer && npm test -- SequenceTable`
Expected: FAIL (`prod prob` not found for monitor rows)

- [ ] **Step 3: Implement the conditional columns in `viewer/components/SequenceTable.tsx`** — `COLUMNS` is a module-level const (`SequenceTable.tsx:16-36`); make the rendered set depend on the rows:

```tsx
const MONITOR_COLUMNS: Column[] = [
  {
    header: "prod prob",
    sortCol: null,
    render: (r) => num(r.recorded_probability ?? null),
  },
  {
    header: "match",
    sortCol: null,
    render: (r) =>
      r.replay_matches == null ? "—" : r.replay_matches ? "=" : "≠",
    cellStyle: (r) => ({
      color: r.replay_matches === false ? "#b91c1c" : undefined,
    }),
  },
];
```

and inside the component, before the return:

```tsx
const hasProvenance = rows.some((r) => r.recorded_probability !== undefined);
const columns = hasProvenance ? [...COLUMNS, ...MONITOR_COLUMNS] : COLUMNS;
```

then replace both `COLUMNS.map(...)` usages in the JSX (`SequenceTable.tsx:82` header row and `:109` body row) with `columns.map(...)`.

- [ ] **Step 4: Add provenance stats to `viewer/components/detail/DetailPanel.tsx`** — the header already renders `Stat`-style entries for decision/correctness/probability (around `DetailPanel.tsx:140-167`). After the `probability` stat, add (matching the local `Stat` component's actual props — check its signature at the top of the file and mirror the `probability` usage exactly):

```tsx
{row.recorded_probability !== undefined && (
  <>
    <Stat
      label="prod prob"
      value={
        row.recorded_probability == null
          ? "—"
          : row.recorded_probability.toFixed(3)
      }
      hint="probability recorded by production (alert-api)"
    />
    <Stat
      label="replay"
      value={
        row.replay_matches == null
          ? "—"
          : row.replay_matches
            ? "matches"
            : "MISMATCH"
      }
      color={row.replay_matches === false ? "#b91c1c" : undefined}
      hint={`api ${row.temporal_api_version ?? "?"} · model ${row.temporal_model_version ?? "?"}`}
    />
  </>
)}
```

If `Stat` lacks `color`/`hint` props, drop them rather than extending `Stat` — match whatever the existing component supports.

- [ ] **Step 5: Run viewer checks**

Run: `cd viewer && npm test -- SequenceTable && npm run lint && npm run format:check && npx tsc --noEmit`
Expected: all green

- [ ] **Step 6: Commit**

```bash
git add viewer/lib/types.ts viewer/components/SequenceTable.tsx viewer/components/detail/DetailPanel.tsx viewer/components/__tests__/SequenceTable.test.tsx
git commit -m "feat(viewer): show monitor provenance columns when present"
```

---

### Task 12: Final verification

- [ ] **Step 1: Full repo checks**

Run from the repo root:

```bash
cd monitor && make lint && make test && cd ..
cd viewer && npm run lint && npm test && cd ..
make lint   # all packages — confirms monitor joined PACKAGES cleanly
```

Expected: everything green.

- [ ] **Step 2: Spec checklist** — re-read `docs/specs/2026-06-12-monitor-design.md` section by section and confirm:
- [ ] import: endpoints, incremental skip, meta.json fields incl. provenance + bbox + bucket_key (Tasks 3-4)
- [ ] replay: grouping, pinned image, health/model check, MinIO upload under original keys, reconstruction (last ≤10/min 4/ROI envelope), `verbose+compute_trigger`, 1e-6 consistency check, all six drop reasons (Tasks 5, 8, 9)
- [ ] reporting: per-org tree, eval columns + 4 monitor extras, reshaped details with recomputed `stabilized_window`, SequenceView frame paths relative to `monitor/` (Tasks 6, 7, 9)
- [ ] DVC: `.dvc/config` mirrors eval, store as `data/01_raw/sequences.dvc` (created by the first real `make import`), replay stage with cached outs (Task 10)
- [ ] conventions: PACKAGES + CI matrix + root README + `.envrc.example` (Task 1)
- [ ] viewer additive change (Task 11)

- [ ] **Step 3: Manual e2e (documented, optional here — needs Docker + creds)**

```bash
cd monitor
make import ARGS="--date-from <recent-day> --date-to <recent-day>"
uv run dvc repro
cd ../viewer && DATA_ROOT=../monitor npm run dev
```

Check: a sequence opens with tubes + bbox overlay; `replay_matches` is true for most rows; the run summary printed drop counts.

- [ ] **Step 4: Commit any fixes, then hand off** for review (superpowers:finishing-a-development-branch).
