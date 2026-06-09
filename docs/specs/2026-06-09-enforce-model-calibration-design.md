# Enforce Model Calibration (Logistic Regressor)

- **Issue:** [#27 — Make the model always calibrated (logistic regressor)](https://github.com/pyronear/temporal-model/issues/27)
- **Date:** 2026-06-09
- **Status:** Approved design, pending implementation

## Problem

The logistic calibrator (`core/.../logistic_calibrator.py`) maps per-tube
features `[logit, log_len, mean_conf, n_tubes]` to a probability in `[0, 1]`.
A model only produces calibrated probabilities — and decides on them — when
**both** of these hold:

1. The package bundles `logistic_calibrator.json`
   (`ModelPackage.calibrator` is non-`None`).
2. `decision.aggregation == "logistic"`, so the decision thresholds the
   calibrated probability. The alternative, `max_logit`, thresholds the raw
   logit and ignores the calibrator (per-tube `probability` comes back `null`).

Today calibration is optional at every layer:

- **core**: `calibrator: LogisticCalibrator | None = None` throughout;
  `DEFAULT_AGGREGATION = "max_logit"`. The only guard is in
  `make_decision_fn`, which raises *only* when `aggregation="logistic"` but the
  calibrator is missing. An uncalibrated `max_logit` package loads and runs
  fine.
- **api**: `ModelRunner.load` loads any package and exposes a `calibrated: bool`
  flag for display; it never refuses an uncalibrated model.
- **eval**: `evaluate.py` calls `from_archive` on any package with no
  calibration check.
- **train**: `package.py` defaults variant aggregation to `max_logit` and
  legitimately builds uncalibrated packages (it is also where the calibrator is
  fit).

The goal is to make calibration a **core contract**: by default, you cannot
load or build an uncalibrated package. `api` and `eval` inherit a hard guarantee.
Research/experimentation paths (`benchmark`, `train`'s intentional `max_logit`
variants) opt out explicitly.

## Decisions

| Question | Decision |
|---|---|
| Enforcement layer | **Contract-level** — gates live in `core` (`package.py`); `api`/`eval` inherit them. |
| Uncalibrated loads | **Strict-by-default + explicit opt-in** (`allow_uncalibrated=True`). |
| Gate condition | **Calibrator present AND `aggregation == "logistic"`.** A calibrator-present-but-`max_logit` package is rejected. |
| Build path | **Mirror the loader**: `build_model_package` is strict-by-default + opt-in. |
| eval escape hatch | **None.** eval is always strict; no CLI flag, no opt-in. |
| `calibrated` api flag | **Kept** (now always `True` post-gate; left as informational). |

## Design

### 1. The rule — single source of truth (`core/.../package.py`)

```python
class UncalibratedModelError(ValueError):
    """Raised when a model package is not calibrated and the caller did not opt out."""


def is_calibrated(calibrator: LogisticCalibrator | None, aggregation: str) -> bool:
    """A package is calibrated iff a calibrator is bundled AND the decision is logistic."""
    return calibrator is not None and aggregation == "logistic"


def require_calibrated(
    calibrator: LogisticCalibrator | None,
    aggregation: str,
    *,
    context: str,
) -> None:
    if not is_calibrated(calibrator, aggregation):
        raise UncalibratedModelError(
            f"{context}: model is not calibrated "
            f"(calibrator={'present' if calibrator is not None else 'missing'}, "
            f"aggregation={aggregation!r}); pass allow_uncalibrated=True to override"
        )
```

- Subclassing `ValueError` preserves any existing `except ValueError` handlers.
- Lives in `package.py`: no import cycle (`model.py` imports from `package.py`,
  not the reverse), and both gated functions live there.
- `aggregation` is read literally; the gate compares against `"logistic"`, so a
  missing/other value is treated as uncalibrated without coupling to
  `DEFAULT_AGGREGATION` in `model.py`.

### 2. Strict-by-default gates (both paths, with opt-in)

- **`load_model_package(package_path, extract_dir=..., *, allow_uncalibrated: bool = False)`**
  After parsing manifest/config/calibrator, read
  `aggregation = config.get("decision", {}).get("aggregation", "max_logit")`
  and call `require_calibrated(calibrator, aggregation, context="load_model_package")`
  unless `allow_uncalibrated`.

- **`build_model_package(..., *, allow_uncalibrated: bool = False)`**
  Before writing the zip, call
  `require_calibrated(calibrator, config["decision"]["aggregation"], context="build_model_package")`
  unless `allow_uncalibrated`.

- **`BboxTubeTemporalModel.from_package` / `from_archive`** gain
  `allow_uncalibrated: bool = False`, threaded into `load_model_package`.

### 3. Consumer wiring

| Consumer | Change |
|---|---|
| **api** (`ModelRunner.load` → `_load_core_model` → `from_package`) | None — inherits the strict default. An uncalibrated model now raises at load; the existing `lifespan` handler logs `model load failed` and `/health` reports `unavailable`. The `calibrated` flag is kept (always `True` post-gate). |
| **eval** (`evaluate.py` → `from_archive`) | None — inherits the strict default. No CLI flag, no opt-in. |
| **benchmark** (`run_core.py` → `from_package`) | Add `allow_uncalibrated: bool = False` parameter + CLI flag, threaded into `from_package`, so variant comparison can opt in. |
| **train** (`package.py` → `build_model_package`) | Pass `allow_uncalibrated=(aggregation != "logistic")`. Intentional `max_logit` variant builds opt in; a `logistic` variant whose calibrator fit silently returned `None` now **fails the build** (the real bug this catches). |

## Testing

- **`core/tests/test_package.py`**
  - `is_calibrated` truth table (calibrator × aggregation).
  - `load_model_package` raises `UncalibratedModelError` for (no calibrator) and
    (calibrator + `max_logit`); succeeds with `allow_uncalibrated=True`; succeeds
    on a calibrated package by default.
  - `build_model_package` raises by default for (no calibrator) and
    (calibrator + `max_logit`); succeeds with `allow_uncalibrated=True`.
- **`api/tests`** — loading an uncalibrated package surfaces as
  `/health` `unavailable` / runner not set.
- **`benchmark` tests** — gate fires by default; `--allow-uncalibrated` bypasses.

## Out of scope

- Not removing the `max_logit` aggregation path — still needed for train's
  calibrator-fitting scoring pass and for research variant comparison.
- No change to the response schema, the `details` shape, or the `calibrated`
  display flag's API surface.
