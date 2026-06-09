# Enforce Model Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make calibration a core contract so that, by default, you cannot load or build an uncalibrated model package — `api` and `eval` inherit a hard guarantee, while `benchmark` and `train` opt out explicitly.

**Architecture:** A single source of truth in `core/.../package.py` (`is_calibrated` / `require_calibrated` / `UncalibratedModelError`) gates both `load_model_package` and `build_model_package` with a `allow_uncalibrated: bool = False` flag. `BboxTubeTemporalModel.from_package`/`from_archive` thread the flag through. Consumers wire in: api/eval use the strict default; benchmark exposes a CLI opt-in; train opts out only for intentional `max_logit` variants.

**Tech Stack:** Python 3.13, pytest, `uv` per-package workspaces. Tests run with `uv run pytest tests/ -v` from each package dir.

**Definition of "calibrated":** the package bundles a `LogisticCalibrator` **and** `decision.aggregation == "logistic"`. A calibrator-present-but-`max_logit` package counts as **uncalibrated**.

Spec: `docs/specs/2026-06-09-enforce-model-calibration-design.md`

---

## File Structure

- `core/src/temporal_model/core/package.py` — **modify**: add `UncalibratedModelError`, `is_calibrated`, `require_calibrated`; gate `load_model_package` and `build_model_package`; extend `__all__`.
- `core/src/temporal_model/core/model.py` — **modify**: add `allow_uncalibrated` to `from_package`/`from_archive`.
- `core/tests/test_package.py` — **modify**: new gate tests; add `allow_uncalibrated=True` to existing intentionally-uncalibrated build/load calls.
- `benchmark/src/temporal_model/benchmark/run_core.py` — **modify**: add `allow_uncalibrated` param, thread into `from_package`.
- `benchmark/src/temporal_model/benchmark/cli.py` — **modify**: add `--allow-uncalibrated` flag, thread into `run_core`.
- `benchmark/tests/test_run_core_gate.py` — **create**: gate fires by default, bypassed by flag.
- `train/src/temporal_model/train/package.py` — **modify**: pass `allow_uncalibrated=(aggregation != "logistic")` to `build_model_package`.
- `api/tests/test_app.py` — **modify**: add a boundary test that an `UncalibratedModelError` at load degrades `/health` to `unavailable`.

---

## Task 1: Core calibration helpers (single source of truth)

**Files:**
- Modify: `core/src/temporal_model/core/package.py`
- Test: `core/tests/test_package.py`

- [ ] **Step 1: Write the failing test**

Append to `core/tests/test_package.py` (it already imports from `temporal_model.core.package` and `LogisticCalibrator`; add the new names to that import and `_make_calibrator` already exists in the file):

```python
class TestIsCalibrated:
    def test_calibrator_and_logistic_is_calibrated(self) -> None:
        assert is_calibrated(_make_calibrator(), "logistic") is True

    def test_calibrator_but_max_logit_is_uncalibrated(self) -> None:
        assert is_calibrated(_make_calibrator(), "max_logit") is False

    def test_no_calibrator_logistic_is_uncalibrated(self) -> None:
        assert is_calibrated(None, "logistic") is False

    def test_no_calibrator_max_logit_is_uncalibrated(self) -> None:
        assert is_calibrated(None, "max_logit") is False


class TestRequireCalibrated:
    def test_passes_when_calibrated(self) -> None:
        require_calibrated(_make_calibrator(), "logistic", context="x")  # no raise

    def test_raises_when_uncalibrated(self) -> None:
        with pytest.raises(UncalibratedModelError, match="not calibrated"):
            require_calibrated(None, "max_logit", context="ctx")

    def test_error_is_a_valueerror(self) -> None:
        assert issubclass(UncalibratedModelError, ValueError)
```

Update the existing `from temporal_model.core.package import (...)` block to also import
`UncalibratedModelError`, `is_calibrated`, `require_calibrated`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd core && uv run pytest tests/test_package.py::TestIsCalibrated tests/test_package.py::TestRequireCalibrated -v`
Expected: FAIL — `ImportError: cannot import name 'is_calibrated'`.

- [ ] **Step 3: Write minimal implementation**

In `core/src/temporal_model/core/package.py`, after the imports / constants (e.g. just below `DEFAULT_EXTRACT_DIR`), add:

```python
class UncalibratedModelError(ValueError):
    """Raised when a model package is not calibrated and the caller did not opt out."""


def is_calibrated(calibrator: "LogisticCalibrator | None", aggregation: str) -> bool:
    """A package is calibrated iff a calibrator is bundled AND the decision is logistic."""
    return calibrator is not None and aggregation == "logistic"


def require_calibrated(
    calibrator: "LogisticCalibrator | None",
    aggregation: str,
    *,
    context: str,
) -> None:
    """Raise :class:`UncalibratedModelError` unless the package is calibrated."""
    if not is_calibrated(calibrator, aggregation):
        raise UncalibratedModelError(
            f"{context}: model is not calibrated "
            f"(calibrator={'present' if calibrator is not None else 'missing'}, "
            f"aggregation={aggregation!r}); pass allow_uncalibrated=True to override"
        )
```

Extend `__all__` (currently `["ModelPackage", "build_model_package", "load_model_package", "load_yolo"]`) to add `"UncalibratedModelError"`, `"is_calibrated"`, `"require_calibrated"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd core && uv run pytest tests/test_package.py::TestIsCalibrated tests/test_package.py::TestRequireCalibrated -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add core/src/temporal_model/core/package.py core/tests/test_package.py
git commit -m "feat(core): add is_calibrated/require_calibrated + UncalibratedModelError"
```

---

## Task 2: Gate `build_model_package` (strict-by-default + opt-in)

**Files:**
- Modify: `core/src/temporal_model/core/package.py` (`build_model_package`)
- Test: `core/tests/test_package.py`

- [ ] **Step 1: Write the failing test**

Append to `core/tests/test_package.py`:

```python
class TestBuildCalibrationGate:
    def test_build_rejects_uncalibrated_by_default(
        self, tmp_path: Path, dummy_yolo_weights: Path, dummy_classifier_ckpt: Path
    ) -> None:
        # SAMPLE_CONFIG is max_logit and no calibrator -> uncalibrated.
        with pytest.raises(UncalibratedModelError, match="not calibrated"):
            build_model_package(
                yolo_weights_path=dummy_yolo_weights,
                classifier_ckpt_path=dummy_classifier_ckpt,
                config=SAMPLE_CONFIG,
                variant="vit_dinov2_finetune",
                output_path=tmp_path / "out.zip",
            )

    def test_build_rejects_calibrator_with_max_logit(
        self, tmp_path: Path, dummy_yolo_weights: Path, dummy_classifier_ckpt: Path
    ) -> None:
        # Calibrator present but aggregation is max_logit -> still uncalibrated.
        with pytest.raises(UncalibratedModelError):
            build_model_package(
                yolo_weights_path=dummy_yolo_weights,
                classifier_ckpt_path=dummy_classifier_ckpt,
                config=SAMPLE_CONFIG,  # aggregation == "max_logit"
                variant="v",
                output_path=tmp_path / "out.zip",
                calibrator=_make_calibrator(),
            )

    def test_build_allows_uncalibrated_when_opted_in(
        self, tmp_path: Path, dummy_yolo_weights: Path, dummy_classifier_ckpt: Path
    ) -> None:
        out = build_model_package(
            yolo_weights_path=dummy_yolo_weights,
            classifier_ckpt_path=dummy_classifier_ckpt,
            config=SAMPLE_CONFIG,
            variant="v",
            output_path=tmp_path / "out.zip",
            allow_uncalibrated=True,
        )
        assert out.exists()

    def test_build_allows_calibrated_by_default(
        self, tmp_path: Path, dummy_yolo_weights: Path, dummy_classifier_ckpt: Path
    ) -> None:
        cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in SAMPLE_CONFIG.items()}
        cfg["decision"] = dict(cfg["decision"])
        cfg["decision"]["aggregation"] = "logistic"
        out = build_model_package(
            yolo_weights_path=dummy_yolo_weights,
            classifier_ckpt_path=dummy_classifier_ckpt,
            config=cfg,
            variant="v",
            output_path=tmp_path / "out.zip",
            calibrator=_make_calibrator(),
        )
        assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd core && uv run pytest tests/test_package.py::TestBuildCalibrationGate -v`
Expected: FAIL — `build_model_package() got an unexpected keyword argument 'allow_uncalibrated'` and the "rejects" tests fail (no raise).

- [ ] **Step 3: Write minimal implementation**

In `build_model_package`, add the parameter to the signature (keyword-only, after `calibrator`):

```python
    calibrator: LogisticCalibrator | None = None,
    allow_uncalibrated: bool = False,
) -> Path:
```

Then, immediately **after** the two `FileNotFoundError` existence checks (so a genuinely missing file still raises `FileNotFoundError` first) and before `manifest = {...}`, insert:

```python
    if not allow_uncalibrated:
        aggregation = config.get("decision", {}).get("aggregation", "max_logit")
        require_calibrated(calibrator, aggregation, context="build_model_package")
```

- [ ] **Step 4: Run new test to verify it passes**

Run: `cd core && uv run pytest tests/test_package.py::TestBuildCalibrationGate -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Fix existing build calls broken by the gate**

The gate now breaks the existing fixtures/tests that intentionally build uncalibrated packages. Add `allow_uncalibrated=True` to these `build_model_package(...)` calls in `core/tests/test_package.py`:

- `built_archive` fixture (the call inside it) — add `allow_uncalibrated=True`.
- `real_tiny_archive` fixture — add `allow_uncalibrated=True`.
- `TestCalibratorBundling.test_package_with_calibrator_round_trips` — add `allow_uncalibrated=True` (calibrator present but `real_tiny_config` is `max_logit`).
- `TestCalibratorBundling.test_load_rejects_tampered_calibrator` — add `allow_uncalibrated=True` (same reason).
- `TestProvenance.test_model_version_recorded_when_provided` — add `allow_uncalibrated=True`.
- `TestProvenance.test_provenance_train_git_sha_recorded` — add `allow_uncalibrated=True`.

Do **not** touch `TestBuildMissingWeightsRaises` — those pass a missing path, so `FileNotFoundError` is raised before the calibration gate.

- [ ] **Step 6: Run the whole build test file to verify green**

Run: `cd core && uv run pytest tests/test_package.py -v`
Expected: PASS — `TestBuildArchive`, `TestManifest`, `TestConfigRoundTrip`, `TestBuildMissingWeightsRaises`, `TestProvenance`, `TestBuildCalibrationGate` all green. (`TestLoadRoundtrip` / `TestCalibratorBundling` still green too — the load gate is not added yet.)

- [ ] **Step 7: Commit**

```bash
git add core/src/temporal_model/core/package.py core/tests/test_package.py
git commit -m "feat(core): gate build_model_package on calibration (opt-in override)"
```

---

## Task 3: Gate `load_model_package` + thread `from_package`/`from_archive`

**Files:**
- Modify: `core/src/temporal_model/core/package.py` (`load_model_package`)
- Modify: `core/src/temporal_model/core/model.py` (`from_package`, `from_archive`)
- Test: `core/tests/test_package.py`

- [ ] **Step 1: Write the failing test**

Append to `core/tests/test_package.py`:

```python
class TestLoadCalibrationGate:
    @patch("temporal_model.core.package.load_yolo")
    def test_load_rejects_uncalibrated_by_default(
        self, mock_yolo: MagicMock, real_tiny_archive: Path, tmp_path: Path
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        # real_tiny_archive is max_logit, no calibrator -> uncalibrated.
        with pytest.raises(UncalibratedModelError, match="not calibrated"):
            load_model_package(real_tiny_archive, extract_dir=tmp_path / "ext")

    @patch("temporal_model.core.package.load_yolo")
    def test_load_allows_uncalibrated_when_opted_in(
        self, mock_yolo: MagicMock, real_tiny_archive: Path, tmp_path: Path
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        pkg = load_model_package(
            real_tiny_archive, extract_dir=tmp_path / "ext", allow_uncalibrated=True
        )
        assert pkg.calibrator is None

    @patch("temporal_model.core.package.load_yolo")
    def test_load_allows_calibrated_by_default(
        self,
        mock_yolo: MagicMock,
        tmp_path: Path,
        dummy_yolo_weights: Path,
        real_tiny_classifier_ckpt: Path,
        real_tiny_config: dict,
    ) -> None:
        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        cfg = dict(real_tiny_config)
        cfg["decision"] = dict(cfg["decision"])
        cfg["decision"]["aggregation"] = "logistic"
        out = tmp_path / "cal.zip"
        build_model_package(
            yolo_weights_path=dummy_yolo_weights,
            classifier_ckpt_path=real_tiny_classifier_ckpt,
            config=cfg,
            variant="tiny",
            output_path=out,
            calibrator=_make_calibrator(),
        )
        pkg = load_model_package(out, extract_dir=tmp_path / "ext")
        assert pkg.calibrator is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd core && uv run pytest tests/test_package.py::TestLoadCalibrationGate -v`
Expected: FAIL — `load_model_package() got an unexpected keyword argument 'allow_uncalibrated'`.

- [ ] **Step 3: Write minimal implementation (load_model_package)**

In `load_model_package`, add the parameter (keyword-only):

```python
def load_model_package(
    package_path: Path,
    extract_dir: Path = DEFAULT_EXTRACT_DIR,
    *,
    allow_uncalibrated: bool = False,
) -> ModelPackage:
```

After the `with zipfile.ZipFile(...)` block closes (i.e. right before `yolo_model = load_yolo(extract_dir / yolo_name)`, where `config` and `calibrator` are in scope), insert:

```python
    if not allow_uncalibrated:
        aggregation = config.get("decision", {}).get("aggregation", "max_logit")
        require_calibrated(calibrator, aggregation, context="load_model_package")
```

This sits **after** `calibrator.verify_sanity_checks()`, so a tampered-calibrator archive still raises the sanity-check `ValueError` first.

- [ ] **Step 4: Thread the flag through `from_package`/`from_archive`**

In `core/src/temporal_model/core/model.py`, update `from_package`:

```python
    @classmethod
    def from_package(
        cls,
        package_path: Path,
        *,
        device: str | torch.device | None = None,
        allow_uncalibrated: bool = False,
    ) -> Self:
        pkg: ModelPackage = load_model_package(
            package_path, allow_uncalibrated=allow_uncalibrated
        )
        return cls(
            yolo_model=pkg.yolo_model,
            classifier=pkg.classifier,
            config=pkg.config,
            device=device,
            calibrator=pkg.calibrator,
        )
```

And `from_archive`:

```python
    @classmethod
    def from_archive(
        cls,
        archive_path: Path,
        *,
        device: str | torch.device | None = None,
        allow_uncalibrated: bool = False,
    ) -> Self:
        """Alias for :meth:`from_package`."""
        return cls.from_package(
            archive_path, device=device, allow_uncalibrated=allow_uncalibrated
        )
```

- [ ] **Step 5: Fix existing load calls broken by the gate**

In `core/tests/test_package.py`, add `allow_uncalibrated=True` to these `load_model_package(...)` calls (they load the uncalibrated `real_tiny_archive` or a calibrator+`max_logit` archive):

- `TestLoadRoundtrip.test_config_passthrough`
- `TestLoadRoundtrip.test_yolo_returned`
- `TestLoadRoundtrip.test_classifier_forward_runs`
- `TestCalibratorBundling.test_package_without_calibrator_has_no_entry`
- `TestCalibratorBundling.test_package_with_calibrator_round_trips` (the `load_model_package(out, ...)` call)

Do **not** change `TestLoadRejectsBadArchive` (KeyError / format_version raise before the gate) nor `test_load_rejects_tampered_calibrator` (sanity-check ValueError raises before the gate).

- [ ] **Step 6: Run the whole core package test file**

Run: `cd core && uv run pytest tests/test_package.py -v`
Expected: PASS (all classes green, including `TestLoadCalibrationGate`).

- [ ] **Step 7: Run the full core suite to catch fallout**

Run: `cd core && uv run pytest tests/ -q`
Expected: PASS — no other core test loads/builds an uncalibrated package without the flag. (If `test_model_parity.py` / `test_smoke.py` load a real package, confirm that package is calibrated; if a fixture is uncalibrated, add `allow_uncalibrated=True` at its `from_package`/`from_archive` call.)

- [ ] **Step 8: Commit**

```bash
git add core/src/temporal_model/core/package.py core/src/temporal_model/core/model.py core/tests/test_package.py
git commit -m "feat(core): gate load_model_package + from_package/from_archive on calibration"
```

---

## Task 4: Benchmark opt-in (`--allow-uncalibrated`)

**Files:**
- Modify: `benchmark/src/temporal_model/benchmark/run_core.py`
- Modify: `benchmark/src/temporal_model/benchmark/cli.py`
- Test: `benchmark/tests/test_run_core_gate.py` (create)

- [ ] **Step 1: Write the failing test**

Create `benchmark/tests/test_run_core_gate.py`:

```python
"""run_core threads allow_uncalibrated into BboxTubeTemporalModel.from_package."""

from pathlib import Path
from unittest.mock import patch

from temporal_model.benchmark import run_core as rc


@patch.object(rc.BboxTubeTemporalModel, "from_package")
def test_run_core_defaults_to_strict(mock_from_package, tmp_path: Path) -> None:
    # iter_sequences returns nothing -> run_core raises SystemExit after load,
    # but we only care that from_package was called strict (allow_uncalibrated=False).
    with patch.object(rc, "iter_sequences", return_value=iter(())):
        try:
            rc.run_core(tmp_path, tmp_path / "m.zip", device="cpu")
        except SystemExit:
            pass
    _, kwargs = mock_from_package.call_args
    assert kwargs.get("allow_uncalibrated", False) is False


@patch.object(rc.BboxTubeTemporalModel, "from_package")
def test_run_core_forwards_opt_in(mock_from_package, tmp_path: Path) -> None:
    with patch.object(rc, "iter_sequences", return_value=iter(())):
        try:
            rc.run_core(
                tmp_path, tmp_path / "m.zip", device="cpu", allow_uncalibrated=True
            )
        except SystemExit:
            pass
    _, kwargs = mock_from_package.call_args
    assert kwargs.get("allow_uncalibrated") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmark && uv run pytest tests/test_run_core_gate.py -v`
Expected: FAIL — `run_core() got an unexpected keyword argument 'allow_uncalibrated'`.

- [ ] **Step 3: Write minimal implementation (run_core)**

In `benchmark/src/temporal_model/benchmark/run_core.py`, add the parameter to `run_core` (keyword-only, alongside the other kw-only params) and pass it to `from_package`:

```python
def run_core(
    store_dir: Path,
    model_path: Path,
    *,
    device: str = "auto",
    reps: int = 5,
    warmup: int = 3,
    limit: int | None = None,
    allow_uncalibrated: bool = False,
) -> pd.DataFrame:
    """Benchmark predict() over every sequence; one row per (sequence, rep)."""
    device = resolve_device(device)
    model = BboxTubeTemporalModel.from_package(
        model_path, device=device, allow_uncalibrated=allow_uncalibrated
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmark && uv run pytest tests/test_run_core_gate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Add the CLI flag**

In `benchmark/src/temporal_model/benchmark/cli.py`, in the `core` subparser argument block (near the other `core.add_argument(...)` lines), add:

```python
    core.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="Permit benchmarking an uncalibrated model package (default: refuse).",
    )
```

In `_run_core_cmd`, thread it into the `run_core(...)` call:

```python
        df = run_core(
            args.store,
            args.model,
            device=device,
            reps=args.reps,
            warmup=args.warmup,
            limit=args.limit,
            allow_uncalibrated=args.allow_uncalibrated,
        )
```

- [ ] **Step 6: Run the full benchmark suite**

Run: `cd benchmark && uv run pytest tests/ -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add benchmark/src/temporal_model/benchmark/run_core.py benchmark/src/temporal_model/benchmark/cli.py benchmark/tests/test_run_core_gate.py
git commit -m "feat(benchmark): add --allow-uncalibrated opt-in for variant comparison"
```

---

## Task 5: Train opts out for intentional `max_logit` variants

**Files:**
- Modify: `train/src/temporal_model/train/package.py` (the `build_model_package(...)` call)
- Test: `train/tests/test_package_builders.py`

**Context:** `package.py` computes `aggregation` (default `"max_logit"`) and a `calibrator` (only fit when `aggregation == "logistic"`). The `build_model_package(...)` call must opt out only when the variant is intentionally uncalibrated. A `logistic` variant whose calibrator fit returned `None` must still FAIL the build.

- [ ] **Step 1: Write the failing test**

Append to `train/tests/test_package_builders.py` a test asserting the opt-out expression is exactly `aggregation != "logistic"`. This is a pure-logic guard (no heavy build):

```python
import pytest

from temporal_model.core.package import require_calibrated, UncalibratedModelError


@pytest.mark.parametrize(
    "aggregation,calibrator_present,should_raise",
    [
        ("logistic", True, False),     # calibrated -> build allowed
        ("logistic", False, True),     # bug: logistic intended but no calibrator -> FAIL
        ("max_logit", False, False),   # intentional uncalibrated -> opted out
        ("max_logit", True, False),    # opted out regardless
    ],
)
def test_train_optout_expression(aggregation, calibrator_present, should_raise) -> None:
    calibrator = object() if calibrator_present else None
    allow_uncalibrated = aggregation != "logistic"
    if should_raise and not allow_uncalibrated:
        with pytest.raises(UncalibratedModelError):
            require_calibrated(calibrator, aggregation, context="build_model_package")
    elif not allow_uncalibrated:
        require_calibrated(calibrator, aggregation, context="build_model_package")
    # allow_uncalibrated True -> build_model_package would skip the check; nothing to assert
```

- [ ] **Step 2: Run test to verify it imports/runs**

Run: `cd train && uv run pytest tests/test_package_builders.py::test_train_optout_expression -v`
Expected: PASS — tasks execute in order, so Task 1's `require_calibrated`/`UncalibratedModelError` are present. This test locks the opt-out invariant (logistic-without-calibrator must raise; `max_logit` opts out) that Step 3 wires into the build call.

- [ ] **Step 3: Write minimal implementation**

In `train/src/temporal_model/train/package.py`, change the `build_model_package(...)` call (currently passing `calibrator=calibrator, train_git_sha=_git_sha()`) to add the opt-out:

```python
    build_model_package(
        yolo_weights_path=yolo_weights,
        classifier_ckpt_path=checkpoint,
        config=config,
        variant=args.variant,
        output_path=args.output,
        calibrator=calibrator,
        train_git_sha=_git_sha(),
        allow_uncalibrated=(aggregation != "logistic"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd train && uv run pytest tests/test_package_builders.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train/src/temporal_model/train/package.py train/tests/test_package_builders.py
git commit -m "feat(train): build uncalibrated only for intentional max_logit variants"
```

---

## Task 6: API boundary — uncalibrated load degrades to unavailable

**Files:**
- Test: `api/tests/test_app.py`

**Context:** No api source change — `ModelRunner.load` → `_load_core_model` → `from_package` is strict by default, so an uncalibrated package raises `UncalibratedModelError`. The existing `lifespan` handler catches it (`except Exception`) and sets `app.state.runner = None`, so `/health` reports `unavailable`. This test locks that behavior in. Pattern mirrors the existing `test_lifespan_passes_calibrator_threshold`.

- [ ] **Step 1: Write the failing test**

Append to `api/tests/test_app.py`:

```python
def test_lifespan_uncalibrated_model_degrades_to_unavailable(monkeypatch):
    from temporal_model.core.package import UncalibratedModelError

    def fake_load(*args, **kwargs):
        raise UncalibratedModelError("load_model_package: model is not calibrated")

    monkeypatch.setattr(settings, "s3_bucket", BUCKET)
    monkeypatch.setattr(ModelRunner, "load", fake_load)
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        with TestClient(app) as c:
            r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "unavailable"
    assert r.json()["model_loaded"] is False
```

(If `test_lifespan_passes_calibrator_threshold` shows a simpler mock-aws setup, match it. The key asserts are `status == "unavailable"` and `model_loaded is False`.)

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `cd api && uv run pytest tests/test_app.py::test_lifespan_uncalibrated_model_degrades_to_unavailable -v`
Expected: PASS once Task 1's `UncalibratedModelError` is importable. (This test documents/locks the inherited boundary behavior; it should pass given the existing lifespan handler.) If it errors on `mock_aws`/bucket setup, copy the exact harness from `test_lifespan_passes_calibrator_threshold`.

- [ ] **Step 3: Run the full api suite**

Run: `cd api && uv run pytest tests/ -q`
Expected: PASS — no api test loads a real uncalibrated package without mocking.

- [ ] **Step 4: Commit**

```bash
git add api/tests/test_app.py
git commit -m "test(api): uncalibrated model load degrades /health to unavailable"
```

---

## Final verification

- [ ] **Run each affected package suite:**

```bash
cd core && uv run pytest tests/ -q
cd ../benchmark && uv run pytest tests/ -q
cd ../train && uv run pytest tests/ -q
cd ../api && uv run pytest tests/ -q
cd ../eval && uv run pytest tests/ -q   # unchanged source; from_archive is mocked in its tests
```

Expected: all green.

- [ ] **Lint/format** (match project tooling):

```bash
cd core && uv run ruff check src tests && uv run ruff format --check src tests
```

(Repeat per package you touched: benchmark, train, api.)

- [ ] **Sanity grep** — every non-test loader is either strict or an explicit opt-in:

```bash
grep -rn "from_package\|from_archive\|load_model_package\|build_model_package" \
  --include='*.py' api benchmark eval train core | grep -v tests
```

Expected: `api` (strict), `eval` (strict), `benchmark` (flag-threaded), `train` (`allow_uncalibrated=(aggregation != "logistic")`), `core` (definitions).

## Notes / out of scope

- `max_logit` aggregation code path is **kept** (train's calibrator-fit scoring pass uses it; research variant comparison needs it).
- The api `calibrated` display flag is **kept** (now always `True` post-gate; informational).
- No change to the response schema or `details` shape.
