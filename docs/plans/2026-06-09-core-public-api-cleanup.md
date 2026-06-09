# core/ Public-API Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten `temporal_model.core`'s public boundary, kill duplicated/drifting contracts, and split two muddled modules — without changing model behavior.

**Architecture:** Curated-submodules public API (Option 1). Submodule paths stay the stable surface; every module declares `__all__`; true internals are `_`-prefixed. `data.py` splits into `sequences.py`+`labels.py`; `model_input.py` splits into `crop.py` (pure geometry, stays in core) plus `process_tube`/`save_patch`/`LABEL_TO_INT` (move to `train/`). Parity tests are the guardrail that outputs do not move.

**Tech Stack:** Python 3.11, pytest, ruff, pydantic, numpy, torch/timm/ultralytics. Each package has its own `uv` venv and `make test` / `make lint`.

**Spec:** `docs/specs/2026-06-09-core-public-api-cleanup-design.md`

---

## Conventions for every task

- Run **core** tests with: `cd core && uv run pytest -q`
- Run **train** tests with: `cd train && uv run pytest -q`
- Run **eval** tests with: `cd eval && uv run pytest -q`
- Lint a package with: `cd <pkg> && uv run ruff check . && uv run ruff format --check .`
- Most tasks are behavior-preserving refactors: the **existing suite passing is the test**. Two tasks add genuinely new behavior (Task 3, Task 5) and get a written failing test first.
- Commit messages use Conventional Commits; **do not** add any Claude/AI co-author trailer.

---

## File map (end state)

**core/src/temporal_model/core/**
- `__init__.py` — re-export light common names + `__all__`
- `protocol.py` — adds public `parse_timestamp`
- `types.py` — `SequenceFeatures` deleted; `__all__` added
- `tubes.py` — `__all__` added (internals already `_`)
- `package.py` — `_load_yolo` → `load_yolo`; `__all__`
- `logistic_calibrator.py` — adds `tube_feature_dict`; `__all__`
- `details_schema.py` — `TubeEntry` → `KeptTubeEntry`
- `model.py` — module-level decision-default constants; uses `tube_feature_dict`; `KeptTubeEntry`
- `inference.py` — imports from `crop`; uses `tube_feature_dict`
- `crop.py` — **NEW** pure geometry (from `model_input.py`)
- `sequences.py` — **NEW** sequence discovery (from `data.py`)
- `labels.py` — **NEW** label/record loading (from `data.py`)
- `model_input.py`, `data.py` — **DELETED**
- `temporal_classifier.py`, `detector.py`, `fetch_detector.py`, `stabilize.py`, `stage_timer.py` — `__all__` added

**train/src/temporal_model/train/**
- `crop_patches.py` — **NEW** `process_tube`, `save_patch`, `LABEL_TO_INT`
- `package.py`, `build_model_input.py`, `build_tubes.py`, `package_predict.py` — import updates

**eval/src/temporal_model/eval/**
- `evaluate.py` — import update

**tests**
- `core/tests/test_crop.py` — **NEW** geometry tests (from `test_model_input.py`)
- `core/tests/test_model_input.py` — **DELETED**
- `core/tests/test_data.py` — repointed to `labels`
- `core/tests/test_stabilize_crop_parity.py` — **MOVED** to `train/tests/`
- `train/tests/test_crop_patches.py` — **NEW** (`process_tube`/`save_patch` tests)
- `core/tests/test_package.py`, `test_model_parity.py`, `test_smoke.py` — import updates

---

## Task 1: Remove dead code (`SequenceFeatures`)

**Files:**
- Modify: `core/src/temporal_model/core/types.py:61-72`

- [ ] **Step 1: Confirm zero references**

Run: `cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/temporal-model && grep -rn "SequenceFeatures" --include="*.py" . | grep -v "/.venv/"`
Expected: only the definition line in `types.py` (no other hits).

- [ ] **Step 2: Delete the dataclass**

In `core/src/temporal_model/core/types.py`, delete the entire `SequenceFeatures` block (lines 61-72) and the now-unused `Path` import if nothing else uses it. After deleting, check: `grep -n "Path" core/src/temporal_model/core/types.py` — if no other use, remove `from pathlib import Path`.

- [ ] **Step 3: Run core tests**

Run: `cd core && uv run pytest -q`
Expected: PASS (same count minus nothing; `SequenceFeatures` had no tests).

- [ ] **Step 4: Lint**

Run: `cd core && uv run ruff check . && uv run ruff format --check .`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add core/src/temporal_model/core/types.py
git commit -m "refactor(core): remove unused SequenceFeatures dataclass"
```

---

## Task 2: Promote `_load_yolo` → public `load_yolo`

**Files:**
- Modify: `core/src/temporal_model/core/package.py:143-153,264`
- Modify: `core/tests/test_package.py` (8 patch decorators)
- Modify: `train/src/temporal_model/train/package.py:31,184`

- [ ] **Step 1: Rename in `package.py`**

In `core/src/temporal_model/core/package.py`, rename the function `_load_yolo` to `load_yolo` (definition at line 143 and the call at line 264). Update its docstring's first line to drop the "import-inside-function" privacy note's reference to the underscore — keep the body and the `noqa: PLC0415` exactly:

```python
def load_yolo(weights_path: Path) -> Any:
    """Load an ultralytics YOLO model from a weights file.

    The ``ultralytics`` import is deliberately inside the function body so
    tests can patch ``load_yolo`` without triggering the heavy import chain.
    This is the one and only sanctioned import-inside-function in this
    project (carries a PLC0415 noqa).
    """
    from ultralytics import YOLO  # noqa: PLC0415

    return YOLO(str(weights_path))
```

And at the call site (was line 264):

```python
    yolo_model = load_yolo(extract_dir / yolo_name)
```

- [ ] **Step 2: Update the 8 test patch targets**

In `core/tests/test_package.py`, replace every `@patch("temporal_model.core.package._load_yolo")` (lines 227, 239, 248, 266, 274, 311, 327, 362) with `@patch("temporal_model.core.package.load_yolo")`.

Run: `cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/temporal-model && grep -n "_load_yolo" core/tests/test_package.py`
Expected: no hits remaining.

- [ ] **Step 3: Update train consumer**

In `train/src/temporal_model/train/package.py`:
- Line 31: `from temporal_model.core.package import _load_yolo, build_model_package` → `from temporal_model.core.package import build_model_package, load_yolo`
- Line 184: `yolo_model=_load_yolo(yolo_weights_path),` → `yolo_model=load_yolo(yolo_weights_path),`

- [ ] **Step 4: Run core + train tests**

Run: `cd core && uv run pytest -q && cd ../train && uv run pytest -q`
Expected: PASS in both.

- [ ] **Step 5: Lint both**

Run: `cd core && uv run ruff check . && cd ../train && uv run ruff check .`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add core/src/temporal_model/core/package.py core/tests/test_package.py train/src/temporal_model/train/package.py
git commit -m "refactor(core): make load_yolo public (was _load_yolo)"
```

---

## Task 3: Unify the timestamp parser (the one latent-bug fix)

**Files:**
- Modify: `core/src/temporal_model/core/protocol.py:57-75,118-124`
- Modify: `core/src/temporal_model/core/data.py:10,57-68` (callers within data)
- Test: `core/tests/test_protocol_timestamp.py` (**NEW**)

Context: `protocol._try_parse_timestamp` uses the anchored regex `...SS)$`; `data.parse_timestamp` uses the unanchored variant. We keep the **anchored** behavior (stricter, correct) as the single public function `protocol.parse_timestamp`, and make `data` use it.

- [ ] **Step 1: Write the failing test pinning the anchored contract**

Create `core/tests/test_protocol_timestamp.py`:

```python
"""Pins the unified, anchored timestamp parser."""

from datetime import datetime

from temporal_model.core.protocol import parse_timestamp


def test_parses_pyronear_suffix():
    ts = parse_timestamp("adf_site_999_2023-05-23T17-18-31")
    assert ts == datetime(2023, 5, 23, 17, 18, 31)


def test_returns_none_when_no_timestamp():
    assert parse_timestamp("no_timestamp_here") is None


def test_anchored_rejects_trailing_suffix_after_timestamp():
    # A timestamp NOT at the end of the id must not match (anchored `$`).
    assert parse_timestamp("2023-05-23T17-18-31_extra") is None


def test_returns_none_on_invalid_calendar_value():
    assert parse_timestamp("seq_2023-13-45T99-99-99") is None
```

- [ ] **Step 2: Run it — expect failure**

Run: `cd core && uv run pytest tests/test_protocol_timestamp.py -q`
Expected: FAIL with `ImportError: cannot import name 'parse_timestamp'`.

- [ ] **Step 3: Promote the parser in `protocol.py`**

In `core/src/temporal_model/core/protocol.py`, rename `_try_parse_timestamp` to `parse_timestamp` (public), keeping the anchored regex `_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$")`. Update the internal caller in `load_sequence` (was line 122) to call `parse_timestamp(p.stem)`.

- [ ] **Step 4: Point `data.py` at the unified parser**

In `core/src/temporal_model/core/data.py`:
- Remove the local `_TIMESTAMP_RE` (line 10) and the `parse_timestamp` function (lines 57-68).
- Add import: `from .protocol import parse_timestamp`
- The existing call in `load_frame_detections` (`parse_timestamp(fpath.stem)`) now resolves to the imported one. Remove the now-unused `import re` and `from datetime import datetime` if nothing else in `data.py` uses them (check with `grep -n "re\.\|datetime" core/src/temporal_model/core/data.py`).

- [ ] **Step 5: Run the new test + full core suite**

Run: `cd core && uv run pytest tests/test_protocol_timestamp.py -q && uv run pytest -q`
Expected: PASS. (Parity tests confirm current filenames are unaffected.)

- [ ] **Step 6: Lint + commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/temporal-model
cd core && uv run ruff check . && uv run ruff format --check . && cd ..
git add core/src/temporal_model/core/protocol.py core/src/temporal_model/core/data.py core/tests/test_protocol_timestamp.py
git commit -m "refactor(core): unify timestamp parsing on protocol.parse_timestamp (anchored)"
```

---

## Task 4: Disambiguate the two `TubeEntry` types

**Files:**
- Modify: `core/src/temporal_model/core/details_schema.py:20-34`
- Modify: `core/src/temporal_model/core/model.py:12-19,309`
- Modify: `core/tests/test_details_schema.py` (if it references `TubeEntry`)

- [ ] **Step 1: Rename in `details_schema.py`**

In `core/src/temporal_model/core/details_schema.py`, rename class `TubeEntry` (line 20) to `KeptTubeEntry`. Update the field type in `KeptTube` (line 34): `entries: list[TubeEntry]` → `entries: list[KeptTubeEntry]`.

- [ ] **Step 2: Update `model.py` import + usage**

In `core/src/temporal_model/core/model.py`:
- In the `from .details_schema import (...)` block (lines 12-19), change `TubeEntry` to `KeptTubeEntry`.
- At the entries construction (was line 309), change `TubeEntry(` to `KeptTubeEntry(`.

- [ ] **Step 3: Update the schema test if needed**

Run: `cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/temporal-model && grep -n "TubeEntry" core/tests/test_details_schema.py`
If any hit refers to the details-schema class (not `types.TubeEntry`), rename it to `KeptTubeEntry` and update the import.

- [ ] **Step 4: Run core tests**

Run: `cd core && uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
cd core && uv run ruff check . && cd ..
git add core/src/temporal_model/core/details_schema.py core/src/temporal_model/core/model.py core/tests/test_details_schema.py
git commit -m "refactor(core): rename details_schema.TubeEntry to KeptTubeEntry"
```

---

## Task 5: Single calibrator-feature dict via `tube_feature_dict`

**Files:**
- Modify: `core/src/temporal_model/core/logistic_calibrator.py` (add function)
- Modify: `core/src/temporal_model/core/model.py:287-305`
- Modify: `core/src/temporal_model/core/inference.py:354-369`
- Test: `core/tests/test_logistic_calibrator.py` (add a case)

Context: the dict `{logit, start_frame, end_frame, entries:[{confidence}]}` consumed by `extract_features` is hand-built in `model._probability_for` and in `inference`'s logistic branch. Centralize it. `extract_features(dict, n_tubes)` keeps its dict signature (the on-disk path in train/scripts is untouched).

- [ ] **Step 1: Write the failing test**

Add to `core/tests/test_logistic_calibrator.py`:

```python
from temporal_model.core.logistic_calibrator import (
    extract_features,
    tube_feature_dict,
)
from temporal_model.core.types import Detection, Tube, TubeEntry


def test_tube_feature_dict_matches_inline_shape():
    det = Detection(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.8)
    tube = Tube(
        tube_id=3,
        entries=[
            TubeEntry(frame_idx=0, detection=det),
            TubeEntry(frame_idx=1, detection=None, is_gap=True),
        ],
        start_frame=0,
        end_frame=1,
    )
    d = tube_feature_dict(tube, logit=1.25)
    assert d == {
        "logit": 1.25,
        "start_frame": 0,
        "end_frame": 1,
        "entries": [{"confidence": 0.8}, {"confidence": None}],
    }
    # And it feeds extract_features without error.
    feats = extract_features(d, n_tubes=2)
    assert feats.shape == (4,)
```

- [ ] **Step 2: Run it — expect failure**

Run: `cd core && uv run pytest tests/test_logistic_calibrator.py::test_tube_feature_dict_matches_inline_shape -q`
Expected: FAIL with `ImportError: cannot import name 'tube_feature_dict'`.

- [ ] **Step 3: Add `tube_feature_dict` to `logistic_calibrator.py`**

At the top of `core/src/temporal_model/core/logistic_calibrator.py`, add the import (guard against cycles — `types` has no torch deps):

```python
from .types import Tube
```

Then add, just above `extract_features`:

```python
def tube_feature_dict(tube: Tube, logit: float) -> dict:
    """Build the loose dict consumed by :func:`extract_features` from a Tube.

    Single source of the calibrator's in-memory feature shape; used by both
    the inference logistic branch and the per-tube probability annotation in
    ``BboxTubeTemporalModel.predict``.
    """
    return {
        "logit": logit,
        "start_frame": tube.start_frame,
        "end_frame": tube.end_frame,
        "entries": [
            {
                "confidence": (
                    e.detection.confidence if e.detection is not None else None
                )
            }
            for e in tube.entries
        ],
    }
```

- [ ] **Step 4: Use it in `model.py`**

In `core/src/temporal_model/core/model.py`, update the import (line 29) to include the helper:

```python
from .logistic_calibrator import LogisticCalibrator, extract_features, tube_feature_dict
```

Replace the body of `_probability_for` (lines 287-305) so the dict is built by the helper:

```python
        def _probability_for(tube_idx: int, raw_logit: float) -> float | None:
            if self._calibrator is None:
                return None
            tube_dict = tube_feature_dict(kept[tube_idx], raw_logit)
            features = extract_features(tube_dict, n_tubes=len(kept))
            return float(self._calibrator.predict_proba(features))
```

- [ ] **Step 5: Use it in `inference.py`**

In `core/src/temporal_model/core/inference.py`, update the import (line 15):

```python
from .logistic_calibrator import LogisticCalibrator, extract_features, tube_feature_dict
```

Replace the inline dict in the `logistic` branch of `find_first_crossing_trigger` (lines 354-368) so `decides_positive` uses the helper:

```python
        def decides_positive(logit: float, tube_prefix: Tube, n_tubes: int) -> bool:
            features = extract_features(
                tube_feature_dict(tube_prefix, logit), n_tubes=n_tubes
            )
            return bool(calibrator.predict_proba(features) >= logistic_threshold)
```

- [ ] **Step 6: Run the new test + full core suite (parity!)**

Run: `cd core && uv run pytest tests/test_logistic_calibrator.py -q && uv run pytest -q`
Expected: PASS — including `test_model_parity` (logistic path must be bit-for-bit unchanged).

- [ ] **Step 7: Lint + commit**

```bash
cd core && uv run ruff check . && uv run ruff format --check . && cd ..
git add core/src/temporal_model/core/logistic_calibrator.py core/src/temporal_model/core/model.py core/src/temporal_model/core/inference.py core/tests/test_logistic_calibrator.py
git commit -m "refactor(core): centralize calibrator feature dict in tube_feature_dict"
```

---

## Task 6: Define decision defaults once

**Files:**
- Modify: `core/src/temporal_model/core/model.py:35-38 (new consts),84,89,137,280`

Context: `"max_logit"` and `0.5` are duplicated across the `aggregation`/`logistic_threshold` properties and the `predict()` body.

- [ ] **Step 1: Add module constants**

In `core/src/temporal_model/core/model.py`, just below the existing `_PAD_STRATEGIES` dict (line 38), add:

```python
DEFAULT_AGGREGATION = "max_logit"
DEFAULT_LOGISTIC_THRESHOLD = 0.5
```

- [ ] **Step 2: Reference them everywhere the literals appear**

Replace each literal with the constant:
- Line 84: `return self._cfg["decision"].get("aggregation", DEFAULT_AGGREGATION)`
- Line 89: `return float(self._cfg["decision"].get("logistic_threshold", DEFAULT_LOGISTIC_THRESHOLD))`
- Line 137: `aggregation = dec.get("aggregation", DEFAULT_AGGREGATION)`
- Line 280: `logistic_threshold=float(dec.get("logistic_threshold", DEFAULT_LOGISTIC_THRESHOLD)),`

- [ ] **Step 3: Run core tests**

Run: `cd core && uv run pytest -q`
Expected: PASS.

- [ ] **Step 4: Lint + commit**

```bash
cd core && uv run ruff check . && cd ..
git add core/src/temporal_model/core/model.py
git commit -m "refactor(core): define decision defaults as named constants"
```

---

## Task 7: Split `data.py` → `sequences.py` + `labels.py`

**Files:**
- Create: `core/src/temporal_model/core/sequences.py`
- Create: `core/src/temporal_model/core/labels.py`
- Delete: `core/src/temporal_model/core/data.py`
- Modify: `core/tests/test_data.py` → repoint to `labels`
- Modify: consumers: `eval/.../evaluate.py:19-23`, `train/.../package_predict.py:14-18`, `train/.../build_tubes.py:21-25`

Context (post-Task-3): `data.py` already imports `parse_timestamp` from `protocol`. Discovery functions have no torch deps; label/record loaders depend on `types` + `protocol`.

- [ ] **Step 1: Create `sequences.py`**

Create `core/src/temporal_model/core/sequences.py` with the discovery functions moved verbatim from `data.py` (`list_sequences`, `find_sequence_dir`, `is_wf_sequence`, `get_sorted_frames`):

```python
"""Sequence-directory discovery for the nested pyro-dataset layout."""

from pathlib import Path

__all__ = [
    "list_sequences",
    "find_sequence_dir",
    "is_wf_sequence",
    "get_sorted_frames",
]


def list_sequences(split_dir: Path) -> list[Path]:
    """List all sequence directories in a split, sorted by name.

    Supports the nested pyro-dataset v3.0.0 layout::

        split_dir/{wildfire,fp}/<sequence>/

    Returns:
        Sorted list of sequence directory paths.
    """
    sequences: list[Path] = []
    for category in ("wildfire", "fp"):
        cat_dir = split_dir / category
        if cat_dir.is_dir():
            sequences.extend(d for d in sorted(cat_dir.iterdir()) if d.is_dir())
    sequences.sort(key=lambda p: p.name)
    return sequences


def find_sequence_dir(data_dir: Path, seq_id: str) -> Path | None:
    """Find a sequence directory by ID within the nested layout."""
    for category in ("wildfire", "fp"):
        candidate = data_dir / category / seq_id
        if candidate.is_dir():
            return candidate
    return None


def is_wf_sequence(sequence_dir: Path) -> bool:
    """Determine if a sequence is wildfire based on parent directory name."""
    return sequence_dir.parent.name == "wildfire"


def get_sorted_frames(sequence_dir: Path) -> list[Path]:
    """Return image paths from a sequence directory sorted by timestamp.

    Looks for ``*.jpg`` files in ``sequence_dir/images/``.
    """
    images_dir = sequence_dir / "images"
    if not images_dir.is_dir():
        return []
    return sorted(images_dir.glob("*.jpg"), key=lambda p: p.stem)
```

- [ ] **Step 2: Create `labels.py`**

Create `core/src/temporal_model/core/labels.py` with the label/record loaders moved verbatim (`load_detections`, `load_frame_detections`, `load_tube_record`), importing discovery + parser from their new homes:

```python
"""Label-file and tube-record loading for the pyro-dataset layout."""

import json
from pathlib import Path

from .protocol import parse_timestamp
from .sequences import get_sorted_frames
from .types import Detection, FrameDetections

__all__ = [
    "load_detections",
    "load_frame_detections",
    "load_tube_record",
]


def load_detections(sequence_dir: Path, frame_id: str) -> list[Detection]:
    """Read a YOLO-format label file as :class:`Detection` objects.

    Supports both formats found in the Pyronear dataset:

    * **5-col** ``class cx cy w h`` -- wildfire ground-truth annotations.
      ``confidence`` is set to ``1.0``.
    * **6-col** ``class cx cy w h conf`` -- false-positive YOLO predictions.
      ``confidence`` is read from the last column.

    Malformed lines (wrong column count, non-numeric values) are silently
    skipped.

    Args:
        sequence_dir: Path to the sequence directory (contains ``labels/``).
        frame_id: Frame filename stem.

    Returns:
        List of detections in file order. Empty list if the file is missing
        or empty.
    """
    label_path = sequence_dir / "labels" / f"{frame_id}.txt"
    if not label_path.is_file():
        return []
    content = label_path.read_text().strip()
    if not content:
        return []
    dets: list[Detection] = []
    for line in content.split("\n"):
        parts = line.strip().split()
        try:
            if len(parts) == 5:
                class_id = int(parts[0])
                cx, cy, w, h = (float(p) for p in parts[1:5])
                confidence = 1.0
            elif len(parts) == 6:
                class_id = int(parts[0])
                cx, cy, w, h, confidence = (float(p) for p in parts[1:6])
            else:
                continue
        except ValueError:
            continue
        dets.append(
            Detection(
                class_id=class_id,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                confidence=confidence,
            )
        )
    return dets


def load_frame_detections(sequence_dir: Path) -> list[FrameDetections]:
    """Load all per-frame detections for a sequence in temporal order.

    Iterates frames returned by :func:`~temporal_model.core.sequences.get_sorted_frames`
    and reads the corresponding label file via :func:`load_detections`.

    Args:
        sequence_dir: Path to the sequence directory.

    Returns:
        Ordered list of :class:`FrameDetections`, one per image. Frames
        with no labels yield an entry with an empty ``detections`` list.
    """
    frame_paths = get_sorted_frames(sequence_dir)
    return [
        FrameDetections(
            frame_idx=idx,
            frame_id=fpath.stem,
            timestamp=parse_timestamp(fpath.stem),
            detections=load_detections(sequence_dir, fpath.stem),
        )
        for idx, fpath in enumerate(frame_paths)
    ]


def load_tube_record(path: Path) -> dict:
    """Read+parse a tube JSON file.

    Trivial wrapper around :func:`json.loads`; exists so callers
    (scripts, notebooks) have a single named entry point for tube I/O.

    Args:
        path: Path to a tube ``.json`` file produced by
            ``scripts/build_tubes.py``.

    Returns:
        The parsed record as a plain dict.
    """
    return json.loads(path.read_text())
```

- [ ] **Step 3: Delete `data.py`**

```bash
rm core/src/temporal_model/core/data.py
```

- [ ] **Step 4: Repoint the core test**

In `core/tests/test_data.py`, line 8: `from temporal_model.core.data import load_detections, load_tube_record` → `from temporal_model.core.labels import load_detections, load_tube_record`. (Optionally rename the file to `test_labels.py` with `git mv`; keep it simple and just repoint the import.)

Also repoint `core/tests/test_model_parity.py` line 25: `from temporal_model.core.data import load_frame_detections` → `from temporal_model.core.labels import load_frame_detections`.

- [ ] **Step 5: Repoint consumers**

`eval/src/temporal_model/eval/evaluate.py` (lines 19-23):
```python
from temporal_model.core.sequences import (
    get_sorted_frames,
    is_wf_sequence,
    list_sequences,
)
```

`train/src/temporal_model/train/package_predict.py` (lines 14-18):
```python
from temporal_model.core.sequences import (
    get_sorted_frames,
    is_wf_sequence,
    list_sequences,
)
```

`train/src/temporal_model/train/build_tubes.py` (lines 21-25):
```python
from temporal_model.core.sequences import is_wf_sequence, list_sequences
from temporal_model.core.labels import load_frame_detections
```

- [ ] **Step 6: Verify no stale `core.data` references**

Run: `cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/temporal-model && grep -rn "core\.data\|core import data" --include="*.py" . | grep -v "/.venv/"`
Expected: no hits.

- [ ] **Step 7: Run core + eval + train suites**

Run: `cd core && uv run pytest -q && cd ../eval && uv run pytest -q && cd ../train && uv run pytest -q`
Expected: PASS in all three.

- [ ] **Step 8: Lint + commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/temporal-model
cd core && uv run ruff check . && uv run ruff format --check . && cd ..
cd eval && uv run ruff check . && cd ..
cd train && uv run ruff check . && cd ..
git add -A
git commit -m "refactor(core): split data.py into sequences.py and labels.py"
```

---

## Task 8: Split `model_input.py` → core `crop.py` + train `crop_patches.py`

**Files:**
- Create: `core/src/temporal_model/core/crop.py`
- Create: `train/src/temporal_model/train/crop_patches.py`
- Delete: `core/src/temporal_model/core/model_input.py`
- Modify: `core/src/temporal_model/core/inference.py:16`
- Modify: `train/src/temporal_model/train/build_model_input.py:18`
- Create: `core/tests/test_crop.py` (geometry tests)
- Delete: `core/tests/test_model_input.py`
- Create: `train/tests/test_crop_patches.py` (process_tube/save_patch tests)
- Move: `core/tests/test_stabilize_crop_parity.py` → `train/tests/test_stabilize_crop_parity.py`
- Modify: `core/tests/test_model_parity.py:27-31`, `core/tests/test_smoke.py:5,13`

- [ ] **Step 1: Create core `crop.py` (pure geometry)**

Create `core/src/temporal_model/core/crop.py`:

```python
"""Pure bbox geometry for patch cropping (no I/O).

Used by the inference crop path (``inference.crop_tube_patches``) and the
offline training-data crop (``train.crop_patches.process_tube``); centralized
here so train/inference cropping cannot drift apart.
"""

import numpy as np
from PIL import Image

__all__ = [
    "expand_bbox",
    "norm_bbox_to_pixel_square",
    "crop_and_resize",
]


def expand_bbox(
    cx: float, cy: float, w: float, h: float, factor: float
) -> tuple[float, float, float, float]:
    return cx, cy, w * factor, h * factor


def norm_bbox_to_pixel_square(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    side_px = max(w * img_w, h * img_h)
    half = side_px / 2.0
    cx_px = cx * img_w
    cy_px = cy * img_h
    x0 = int(round(cx_px - half))
    y0 = int(round(cy_px - half))
    x1 = int(round(cx_px + half))
    y1 = int(round(cy_px + half))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_w, x1)
    y1 = min(img_h, y1)
    return x0, y0, x1, y1


def crop_and_resize(
    image: np.ndarray, box: tuple[int, int, int, int], patch_size: int
) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1, :]
    h, w, _ = crop.shape
    side = max(h, w)
    if h != w:
        square = np.zeros((side, side, 3), dtype=np.uint8)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        square[y_off : y_off + h, x_off : x_off + w, :] = crop
        crop = square
    pil = Image.fromarray(crop)
    pil = pil.resize((patch_size, patch_size), Image.BILINEAR)
    return np.array(pil)
```

- [ ] **Step 2: Create train `crop_patches.py` (offline orchestration)**

Create `train/src/temporal_model/train/crop_patches.py` with `save_patch`, `process_tube`, `LABEL_TO_INT` moved from core, repointing imports to core's new homes:

```python
"""Offline training-data crop: tube record -> 224x224 PNG patches + meta.

Moved out of core because this is training-data prep, not a runtime building
block. Shares the crop geometry with the inference path via ``core.crop`` and
the stabilize window policy via ``core.stabilize`` so the two cannot drift.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from temporal_model.core.crop import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)
from temporal_model.core.sequences import find_sequence_dir
from temporal_model.core.stabilize import tube_window

__all__ = ["LABEL_TO_INT", "save_patch", "process_tube"]

LABEL_TO_INT = {"fp": 0, "smoke": 1}


def save_patch(patch: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(patch).save(path, format="PNG", optimize=True)


def process_tube(
    tube_path: Path,
    raw_dir: Path,
    out_dir: Path,
    context_factor: float,
    patch_size: int,
    stabilize: bool = True,
) -> None:
    record = json.loads(tube_path.read_text())
    sequence_id = record["sequence_id"]
    label = record["label"]
    seq_dir = find_sequence_dir(raw_dir, sequence_id)
    if seq_dir is None:
        raise FileNotFoundError(f"raw sequence dir not found for {sequence_id}")

    images_dir = seq_dir / "images"
    seq_out = out_dir / sequence_id
    seq_out.mkdir(parents=True, exist_ok=True)

    entries = record["tube"]["entries"]
    window = None
    if stabilize:
        window = tube_window([(tuple(e["bbox"]), e["is_gap"]) for e in entries])

    frame_meta: list[dict] = []
    for entry in entries:
        frame_id = entry["frame_id"]
        frame_idx = entry["frame_idx"]
        bbox = window if stabilize else entry["bbox"]
        is_gap = entry["is_gap"]

        img_path = images_dir / f"{frame_id}.jpg"
        image = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w, _ = image.shape

        cx, cy, w, h = expand_bbox(bbox[0], bbox[1], bbox[2], bbox[3], context_factor)
        crop_box = norm_bbox_to_pixel_square(cx, cy, w, h, img_w, img_h)
        patch = crop_and_resize(image, crop_box, patch_size)

        filename = f"frame_{frame_idx:02d}.png"
        save_patch(patch, seq_out / filename)

        frame_meta.append(
            {
                "frame_idx": frame_idx,
                "frame_id": frame_id,
                "is_gap": is_gap,
                "orig_bbox": list(entry["bbox"]),
                "crop_bbox_pixels": list(crop_box),
                "filename": filename,
            }
        )

    meta = {
        "sequence_id": sequence_id,
        "split": record["split"],
        "label": label,
        "label_int": LABEL_TO_INT[label],
        "num_frames": record["num_frames"],
        "context_factor": context_factor,
        "patch_size": patch_size,
        "stabilize": stabilize,
        "frames": frame_meta,
    }
    (seq_out / "meta.json").write_text(json.dumps(meta, indent=2))
```

- [ ] **Step 3: Delete core `model_input.py`**

```bash
rm core/src/temporal_model/core/model_input.py
```

- [ ] **Step 4: Repoint core `inference.py`**

In `core/src/temporal_model/core/inference.py`, line 16: `from .model_input import crop_and_resize, expand_bbox, norm_bbox_to_pixel_square` → `from .crop import crop_and_resize, expand_bbox, norm_bbox_to_pixel_square`.

- [ ] **Step 5: Repoint train `build_model_input.py`**

In `train/src/temporal_model/train/build_model_input.py`, line 18: `from temporal_model.core.model_input import LABEL_TO_INT, process_tube` → `from temporal_model.train.crop_patches import LABEL_TO_INT, process_tube`.

- [ ] **Step 6: Create `core/tests/test_crop.py` (geometry tests)**

`git mv core/tests/test_model_input.py core/tests/test_crop.py`, then in `test_crop.py`:
- Change the import block to import only geometry from `core.crop`:
```python
from temporal_model.core.crop import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)
```
- Delete the `save_patch` test (`test_save_patch_writes_png_at_target_size`, ~lines 79-83) and **all** `process_tube` tests (every `def test_process_tube_*`, ~lines 122 to end of file). These move to train in the next step.

- [ ] **Step 7: Create `train/tests/test_crop_patches.py`**

Create `train/tests/test_crop_patches.py` containing the `save_patch` + `process_tube` tests removed in Step 6, with imports repointed:

```python
"""Tests for offline crop-patch generation (moved from core.model_input)."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from temporal_model.train.crop_patches import process_tube, save_patch
```

Paste the `test_save_patch_*` and `test_process_tube_*` function bodies verbatim from the old `test_model_input.py`. (They use only `tmp_path`, `np`, `Image`, `json`, `process_tube`, `save_patch` — no core-internal symbols.)

- [ ] **Step 8: Move the stabilize-crop parity test to train**

`git mv core/tests/test_stabilize_crop_parity.py train/tests/test_stabilize_crop_parity.py`, then fix its imports (was line 28): `from temporal_model.core.model_input import process_tube` → `from temporal_model.train.crop_patches import process_tube`. Leave `from temporal_model.core.inference import crop_tube_patches` and the `core.protocol`/`core.types` imports unchanged.

- [ ] **Step 9: Repoint remaining core tests**

`core/tests/test_model_parity.py` (lines 27-31): `from temporal_model.core.model_input import (...)` → `from temporal_model.core.crop import (crop_and_resize, expand_bbox, norm_bbox_to_pixel_square)`.

`core/tests/test_smoke.py` (lines 5, 13): replace the `model_input` module reference with `crop` (update both the import-list entry and the `(types, tubes, model_input, inference, model)` tuple to `(types, tubes, crop, inference, model)`; add the `crop` import).

- [ ] **Step 10: Verify no stale `model_input` references**

Run: `cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/temporal-model && grep -rn "model_input" --include="*.py" . | grep -v "/.venv/"`
Expected: no hits.

- [ ] **Step 11: Run core + train suites (parity is the guardrail)**

Run: `cd core && uv run pytest -q && cd ../train && uv run pytest -q`
Expected: PASS — `test_stabilize_crop_parity` (now in train) and `test_model_parity` (core) both green.

- [ ] **Step 12: Lint + commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/temporal-model
cd core && uv run ruff check . && uv run ruff format --check . && cd ..
cd train && uv run ruff check . && uv run ruff format --check . && cd ..
git add -A
git commit -m "refactor(core): split model_input into core.crop + train.crop_patches"
```

---

## Task 9: Declare explicit `__all__` and curate the top-level API

**Files:**
- Modify: `core/src/temporal_model/core/__init__.py`
- Modify (add `__all__` where missing): `protocol.py`, `types.py`, `tubes.py`, `package.py`, `temporal_classifier.py`, `inference.py`, `model.py`, `details_schema.py`, `detector.py`, `stabilize.py`, `stage_timer.py`, `fetch_detector.py`

Context: `crop.py`, `sequences.py`, `labels.py`, `logistic_calibrator.py` already got `__all__` in earlier tasks. This task adds it to the rest and finalizes the package root. Do **not** rename any currently-public symbol consumers/tests import (e.g. `compute_iou`, `match_detections`) — only declare surfaces.

- [ ] **Step 1: Add `__all__` to each remaining module**

For each module, add an `__all__` listing its public (non-`_`) names, placed right after the imports. Use these lists:

```python
# protocol.py
__all__ = ["Frame", "TemporalModelOutput", "TemporalModel", "parse_timestamp"]

# types.py
__all__ = ["Detection", "FrameDetections", "TubeEntry", "Tube"]

# tubes.py
__all__ = [
    "compute_iou", "match_detections", "build_tubes", "interpolate_gaps",
    "select_longest_tube", "tube_from_record", "merge_colocated_tubes",
]

# package.py
__all__ = ["ModelPackage", "build_model_package", "load_model_package", "load_yolo"]

# temporal_classifier.py
__all__ = ["TimmBackbone", "TransformerHead", "TemporalSmokeClassifier"]

# inference.py
__all__ = [
    "pad_frames_symmetrically", "pad_frames_uniform", "run_yolo_on_frames",
    "filter_and_interpolate_tubes", "crop_tube_patches", "score_tubes",
    "find_first_crossing_trigger", "build_tubes_for_inference",
]

# model.py
__all__ = ["BboxTubeTemporalModel", "DEFAULT_AGGREGATION", "DEFAULT_LOGISTIC_THRESHOLD"]

# details_schema.py  (TubeEntry was renamed to KeptTubeEntry in Task 4)
__all__ = [
    "KeptTubeEntry", "KeptTube", "Preprocessing", "Tubes", "Decision",
    "BboxTubeDetails",
]

# detector.py
__all__ = ["Detector", "load_detector", "DETECTOR_WEIGHTS_FILENAME"]

# stabilize.py
__all__ = ["union_window", "tube_window"]

# stage_timer.py
__all__ = ["StageTimer", "stage_ctx"]

# fetch_detector.py
__all__ = ["fetch_detector", "main"]
```

- [ ] **Step 2: Finalize the package root**

Replace `core/src/temporal_model/core/__init__.py` with:

```python
"""Core model, tube-building, and inference for the temporal smoke classifier.

Public API. Import the common entry points from here, or reach into the
submodule that owns a symbol (e.g. ``temporal_model.core.tubes``,
``temporal_model.core.model``). The concrete model
(``BboxTubeTemporalModel``) is intentionally NOT re-exported here so that
``import temporal_model.core`` stays light (no torch/timm import); import it
from ``temporal_model.core.model``. Anything prefixed with ``_`` is internal.
"""

from .protocol import Frame, TemporalModel, TemporalModelOutput
from .tubes import build_tubes, merge_colocated_tubes
from .types import Detection, FrameDetections, Tube, TubeEntry

__all__ = [
    "Frame",
    "TemporalModel",
    "TemporalModelOutput",
    "build_tubes",
    "merge_colocated_tubes",
    "Detection",
    "FrameDetections",
    "Tube",
    "TubeEntry",
]
```

- [ ] **Step 3: Verify the package still imports light (no torch)**

Run: `cd core && uv run python -c "import sys, temporal_model.core; assert 'torch' not in sys.modules, 'core root must not import torch'; print('OK light import')"`
Expected: `OK light import`.

- [ ] **Step 4: Run the full core suite**

Run: `cd core && uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
cd core && uv run ruff check . && uv run ruff format --check . && cd ..
git add core/src/temporal_model/core/
git commit -m "refactor(core): declare explicit __all__ and curate package root"
```

---

## Task 10: Update `core/README.md` module map

**Files:**
- Modify: `core/README.md:8-20`

- [ ] **Step 1: Rewrite the module list**

In `core/README.md`, replace the `## Modules` bullet list (lines 8-20) so it matches the new layout. Use:

```markdown
- `protocol.py` — the `TemporalModel` contract (`Frame`, `TemporalModelOutput`,
  the `TemporalModel` ABC) and `parse_timestamp`. Vendored so the repo is
  self-contained.
- `types.py` — `Detection`, `FrameDetections`, `Tube`, `TubeEntry`.
- `tubes.py` — greedy-IoU tube linking, gap interpolation, colocated-tube merge.
- `crop.py` — pure bbox geometry (expand → pixel-square → crop/resize), shared by
  the inference and offline-training crop paths.
- `stabilize.py` — per-tube fixed crop window (union of observed boxes).
- `temporal_classifier.py` — `TimmBackbone` (ViT) + `TransformerHead` +
  `TemporalSmokeClassifier` (one logit per tube).
- `inference.py` — the per-stage inference pipeline (pad → YOLO → tubes → crop →
  score → first-crossing trigger).
- `model.py` — `BboxTubeTemporalModel`, the `TemporalModel` implementation.
- `package.py` — `model.zip` build/load (YOLO + classifier + calibrator + config)
  and `load_yolo`.
- `logistic_calibrator.py` — runtime logistic calibrator (pure numpy) and
  `tube_feature_dict`.
- `details_schema.py` — pydantic schema for `predict()` output details.
- `sequences.py`, `labels.py` — sequence discovery and detection/label/record loading.
- `detector.py`, `fetch_detector.py` — companion-detector identity + verified fetch.
- `stage_timer.py` — optional per-stage wall-clock profiling.
```

- [ ] **Step 2: Commit**

```bash
git add core/README.md
git commit -m "docs(core): update module map for crop/sequences/labels split"
```

---

## Task 11: Full-repo verification

**Files:** none (verification only)

- [ ] **Step 1: Run every package's test suite**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/temporal-model
for p in core api eval train benchmark; do echo "== $p =="; (cd $p && uv run pytest -q) || break; done
```
Expected: PASS in all five.

- [ ] **Step 2: Lint every package**

```bash
for p in core api eval train benchmark; do echo "== $p =="; (cd $p && uv run ruff check . && uv run ruff format --check .) || break; done
```
Expected: clean in all five.

- [ ] **Step 3: Confirm no stale references anywhere**

Run: `grep -rn "core\.data\|core\.model_input\|_load_yolo\|SequenceFeatures" --include="*.py" . | grep -v "/.venv/"`
Expected: no hits.

- [ ] **Step 4: Final review of the diff against the spec**

Run: `git log --oneline main..HEAD` and confirm each spec section maps to a commit. No code commit should change model behavior (parity tests passing is the evidence).
