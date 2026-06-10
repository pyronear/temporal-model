# Eval Qualitative Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local, read-only Streamlit viewer to the `eval/` package that shows the packaged model's per-sequence behaviour (frames with bboxes, stabilized tube crops, keep/discard decision), with pyro-annotator as a first-class re-scored eval source.

**Architecture:** `evaluate.py` gains a unified per-sequence emit (`details/<key>.json`, `sequences/<key>.json`, `results.{parquet,json}`) over two source kinds — directory-convention train/val and a ported meta-store (pyro-annotator). A small `core` change persists the already-computed stabilized crop window into `output.details`. The viewer reads only the reporting tree + frame files; it never runs the model. The emitted artifacts are a stable, documented, frontend-agnostic contract for a later React rewrite.

**Tech Stack:** Python 3.11+, pydantic (details schema), pandas + pyarrow (results table), Pillow (frame drawing), Streamlit + Altair (viewer), pytest, uv, DVC.

**Reference source (verbatim-copy origin):** the explorer lives at
`../vision-rd/experiments/temporal-models/temporal-model-explorer/` (referred to below
as `EXPLORER`). Its `app.py`, `store.py`, `outcomes.py`, and `tests/test_app_helpers.py`
are the port sources.

---

## File Structure

**Core (small, contained change):**
- Modify `core/src/temporal_model/core/stabilize.py` — add `tube_stabilized_window` helper.
- Modify `core/src/temporal_model/core/details_schema.py` — add `stabilized_window` to `KeptTube`.
- Modify `core/src/temporal_model/core/model.py` — emit `stabilized_window` in the kept-tube loop.
- Modify `core/src/temporal_model/core/inference.py` — reuse the helper in `crop_tube_patches` (DRY).
- Create `core/tests/test_stabilized_window.py` — unit tests for the helper + emit.

**Eval (new + modified):**
- Create `eval/src/temporal_model/eval/store.py` — meta-store reader (ported).
- Create `eval/src/temporal_model/eval/outcomes.py` — pure decision/outcome helpers (ported).
- Create `eval/src/temporal_model/eval/view_store.py` — normalized per-sequence view meta + emit.
- Modify `eval/src/temporal_model/eval/evaluate.py` — per-sequence emit + meta-store source.
- Create `eval/src/temporal_model/eval/render.py` — pure render helpers (ported from `EXPLORER/app.py`).
- Create `eval/src/temporal_model/eval/app.py` — Streamlit UI (ported + adapted).
- Create `eval/tests/test_store.py`, `eval/tests/test_outcomes.py`, `eval/tests/test_render.py`, `eval/tests/test_view_store.py`.
- Modify `eval/tests/test_evaluate_driver.py` — assert new emitted artifacts.
- Modify `eval/pyproject.toml` — add pandas, pyarrow, pillow, streamlit, altair; add `temporal-eval-viewer` console hint.
- Modify `eval/Makefile` — add `app` target.
- Modify `eval/dvc.yaml` — add `pyro-annotator` to the `evaluate` foreach.
- Modify `eval/README.md` — document the viewer + data contract.
- Create `eval/scripts/copy_pyro_annotator.py` — one-time copy of explorer sequences into eval.

---

## Task 1: `tube_stabilized_window` helper in core

The stabilized crop window (union of a tube's observed boxes) is computed inside
`crop_tube_patches` (`inference.py:244-258`) but never persisted. Extract it to a pure,
testable helper so both the crop path and the details-emit path use one implementation.

**Files:**
- Modify: `core/src/temporal_model/core/stabilize.py`
- Test: `core/tests/test_stabilized_window.py`

- [ ] **Step 1: Write the failing test**

Create `core/tests/test_stabilized_window.py`:

```python
"""Unit tests for tube_stabilized_window (the persisted stabilized crop window)."""

from dataclasses import dataclass

from temporal_model.core.stabilize import tube_stabilized_window


@dataclass
class _Det:
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class _Entry:
    detection: _Det | None
    is_gap: bool = False


def test_window_is_union_of_observed_boxes():
    # Two boxes; the union/enclosing box centers between them and spans both.
    entries = [
        _Entry(_Det(0.2, 0.2, 0.1, 0.1)),
        _Entry(_Det(0.4, 0.4, 0.1, 0.1)),
    ]
    cx, cy, w, h = tube_stabilized_window(entries)
    # x spans 0.15..0.45 -> center 0.30, width 0.30; same for y.
    assert round(cx, 4) == 0.3
    assert round(cy, 4) == 0.3
    assert round(w, 4) == 0.3
    assert round(h, 4) == 0.3


def test_window_ignores_gap_only_entries_without_detection():
    entries = [
        _Entry(None, is_gap=True),
        _Entry(_Det(0.5, 0.5, 0.2, 0.2)),
    ]
    cx, cy, w, h = tube_stabilized_window(entries)
    assert (round(cx, 4), round(cy, 4), round(w, 4), round(h, 4)) == (0.5, 0.5, 0.2, 0.2)


def test_window_none_when_no_usable_detection():
    entries = [_Entry(None, is_gap=True), _Entry(None, is_gap=True)]
    assert tube_stabilized_window(entries) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd core && uv run pytest tests/test_stabilized_window.py -v`
Expected: FAIL — `ImportError: cannot import name 'tube_stabilized_window'`.

- [ ] **Step 3: Write minimal implementation**

In `core/src/temporal_model/core/stabilize.py`, add to `__all__` and append the helper.
Change the `__all__` line to:

```python
__all__ = ["union_window", "tube_window", "tube_stabilized_window"]
```

Append at end of file:

```python
def tube_stabilized_window(entries):
    """Fixed crop window (union of a tube's observed boxes), or None.

    Mirrors the window ``crop_tube_patches`` uses when ``stabilize=True``: the
    union (enclosing) box of the tube's observed detections, applied to every
    frame. ``entries`` are tube entries exposing ``.detection`` (with
    ``cx/cy/w/h`` or ``None``) and ``.is_gap``. Returns ``None`` when the tube
    has no usable detection (every entry is a gap).
    """
    boxes = [
        (
            (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
            if e.detection is not None
            else None,
            e.is_gap,
        )
        for e in entries
    ]
    if any(box is not None for box, _ in boxes):
        return tube_window(boxes)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd core && uv run pytest tests/test_stabilized_window.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add core/src/temporal_model/core/stabilize.py core/tests/test_stabilized_window.py
git commit -m "feat(core): add tube_stabilized_window helper"
```

---

## Task 2: Persist `stabilized_window` in details + DRY the crop path

**Files:**
- Modify: `core/src/temporal_model/core/details_schema.py`
- Modify: `core/src/temporal_model/core/model.py:380-410`
- Modify: `core/src/temporal_model/core/inference.py:244-258`
- Test: `core/tests/test_stabilized_window.py` (extend)

- [ ] **Step 1: Add the schema field**

In `core/src/temporal_model/core/details_schema.py`, in `class KeptTube`, add the field
after `first_crossing_frame` and before `entries`:

```python
class KeptTube(_Frozen):
    tube_id: int
    start_frame: int
    end_frame: int
    logit: float
    probability: float | None
    first_crossing_frame: int | None
    stabilized_window: tuple[float, float, float, float] | None = None
    entries: list[KeptTubeEntry]
```

Note: pydantic requires non-default fields after defaulted ones to be keyword-only or
reordered. `KeptTube` is always built with keyword args (see `model.py`), so move
`stabilized_window` to the END to avoid the "non-default after default" error:

```python
class KeptTube(_Frozen):
    tube_id: int
    start_frame: int
    end_frame: int
    logit: float
    probability: float | None
    first_crossing_frame: int | None
    entries: list[KeptTubeEntry]
    stabilized_window: tuple[float, float, float, float] | None = None
```

- [ ] **Step 2: Write the failing test (emit + DRY)**

Append to `core/tests/test_stabilized_window.py`:

```python
def test_kept_tube_schema_accepts_window():
    from temporal_model.core.details_schema import KeptTube

    t = KeptTube(
        tube_id=0,
        start_frame=0,
        end_frame=2,
        logit=1.0,
        probability=None,
        first_crossing_frame=None,
        entries=[],
        stabilized_window=(0.3, 0.3, 0.3, 0.3),
    )
    assert t.stabilized_window == (0.3, 0.3, 0.3, 0.3)


def test_kept_tube_window_defaults_none():
    from temporal_model.core.details_schema import KeptTube

    t = KeptTube(
        tube_id=0,
        start_frame=0,
        end_frame=2,
        logit=1.0,
        probability=None,
        first_crossing_frame=None,
        entries=[],
    )
    assert t.stabilized_window is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd core && uv run pytest tests/test_stabilized_window.py -v`
Expected: the two new tests FAIL until the schema field exists (if Step 1 already applied,
they pass — that's fine; the point is the schema field is present).

- [ ] **Step 4: Emit the window in `model.py`**

In `core/src/temporal_model/core/model.py`, add the import near the other `core` imports
(there is already `from .inference import ...`; add a dedicated import):

```python
from .stabilize import tube_stabilized_window
```

In the kept-tube loop (around line 400-410), pass the window. The loop variable is `tube`
and the stabilize flag is `mi.get("stabilize", True)`:

```python
            kept_models.append(
                KeptTube(
                    tube_id=tube.tube_id,
                    start_frame=tube.start_frame,
                    end_frame=tube.end_frame,
                    logit=logits_list[tube_idx],
                    probability=_probability_for(tube_idx, logits_list[tube_idx]),
                    first_crossing_frame=first_crossing,
                    entries=entries_models,
                    stabilized_window=(
                        tube_stabilized_window(tube.entries)
                        if mi.get("stabilize", True)
                        else None
                    ),
                )
            )
```

- [ ] **Step 5: DRY `crop_tube_patches` to use the helper**

In `core/src/temporal_model/core/inference.py`, replace the inline window block
(lines 244-258, the `window = None` / `if stabilize:` / `boxes = [...]` /
`if any(...)` block) with:

```python
    window = tube_stabilized_window(tube.entries) if stabilize else None
```

Add the import at the top of `inference.py` alongside `from .stabilize import tube_window`:

```python
from .stabilize import tube_stabilized_window, tube_window
```

(Leave `tube_window` imported — it is still referenced indirectly only via the helper now;
if ruff flags it as unused, drop `tube_window` from this import.)

- [ ] **Step 6: Run the full core suite (parity is the guard)**

Run: `cd core && uv run pytest -q`
Expected: PASS. If `tests/test_model_parity.py` or `tests/test_package.py` assert an
**exact** kept-tube dict, they will now show the extra `stabilized_window` key — update
those expected dicts to include `"stabilized_window": <value-or-None>`. The crop refactor
must not change any numeric parity assertion (identical window computation).

- [ ] **Step 7: Commit**

```bash
git add core/src/temporal_model/core/details_schema.py \
        core/src/temporal_model/core/model.py \
        core/src/temporal_model/core/inference.py \
        core/tests/test_stabilized_window.py
git commit -m "feat(core): persist stabilized_window in kept-tube details"
```

---

## Task 3: Port the meta-store reader into eval

**Files:**
- Create: `eval/src/temporal_model/eval/store.py`
- Test: `eval/tests/test_store.py`

- [ ] **Step 1: Write the failing test**

Create `eval/tests/test_store.py`:

```python
import json
from pathlib import Path

from temporal_model.eval.store import (
    SequenceMeta,
    build_frames,
    iter_sequence_dirs,
    normalize_label,
    read_meta,
)


def _write_store_seq(root: Path, key: str, label: str) -> Path:
    seq = root / "org-a" / "cam-1" / key
    (seq / "images").mkdir(parents=True)
    (seq / "images" / "f0.jpg").write_bytes(b"\xff")
    (seq / "images" / "f1.jpg").write_bytes(b"\xff")
    meta = {
        "key": key,
        "sequence_id": key,
        "source": "pyro-annotator",
        "label": label,
        "label_detail": None,
        "label_source": "pyro_annotator_folder",
        "frames": [
            {"file": "images/f0.jpg", "detection_id": None, "created_at": None},
            {"file": "images/f1.jpg", "detection_id": None, "created_at": None},
        ],
        "camera_id": 1,
        "camera_name": "cam-1",
        "organization_id": 7,
        "organization_name": "org-a",
        "started_at": "2026-05-19T14:10:01",
    }
    (seq / "meta.json").write_text(json.dumps(meta))
    return seq


def test_iter_and_read_meta(tmp_path):
    _write_store_seq(tmp_path, "seq-1", "smoke")
    dirs = list(iter_sequence_dirs(tmp_path))
    assert len(dirs) == 1
    meta = read_meta(dirs[0])
    assert isinstance(meta, SequenceMeta)
    assert meta.key == "seq-1"
    assert meta.label == "smoke"
    assert meta.organization_name == "org-a"
    assert len(meta.frames) == 2


def test_build_frames_orders_by_meta(tmp_path):
    seq = _write_store_seq(tmp_path, "seq-1", "fp")
    meta = read_meta(seq)
    frames = build_frames(seq, meta)
    assert [f.image_path.name for f in frames] == ["f0.jpg", "f1.jpg"]


def test_normalize_label():
    assert normalize_label("wildfire", ["wildfire"], ["false_positive"]) == "smoke"
    assert normalize_label("false_positive", ["wildfire"], ["false_positive"]) == "fp"
    assert normalize_label(None, [], []) == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd eval && uv run pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: temporal_model.eval.store`.

- [ ] **Step 3: Port the implementation**

Create `eval/src/temporal_model/eval/store.py` by copying
`EXPLORER/src/temporal_model_explorer/store.py` **verbatim**, with one change: replace

```python
from pyrocore import Frame
```

with

```python
from temporal_model.core.protocol import Frame
```

(Everything else — `slug`, `FrameRef`, `SequenceMeta`, `write_meta`, `read_meta`,
`iter_sequence_dirs`, `normalize_label`, `build_frames` — is copied unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd eval && uv run pytest tests/test_store.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add eval/src/temporal_model/eval/store.py eval/tests/test_store.py
git commit -m "feat(eval): port meta-store reader"
```

---

## Task 4: Port the pure outcome helpers into eval

**Files:**
- Create: `eval/src/temporal_model/eval/outcomes.py`
- Test: `eval/tests/test_outcomes.py`

- [ ] **Step 1: Write the failing test**

Create `eval/tests/test_outcomes.py`:

```python
import pandas as pd

from temporal_model.eval.outcomes import (
    compute_outcome,
    decision_from_output,
    filter_results,
    max_probability,
    performance_summary,
)


def test_decision_from_output():
    assert decision_from_output(True) == "keep"
    assert decision_from_output(False) == "discard"


def test_compute_outcome_matrix():
    assert compute_outcome("keep", "smoke") == "kept-smoke"
    assert compute_outcome("discard", "smoke") == "discarded-smoke"
    assert compute_outcome("keep", "fp") == "kept-fp"
    assert compute_outcome("discard", "fp") == "discarded-fp"
    assert compute_outcome("keep", "unknown") == "n/a"


def test_max_probability():
    details = {"tubes": {"kept": [{"probability": 0.2}, {"probability": 0.8}]}}
    assert max_probability(details) == 0.8
    assert max_probability({"tubes": {"kept": []}}) is None
    assert max_probability(None) is None


def test_filter_results_errors_only():
    df = pd.DataFrame(
        {"outcome": ["kept-smoke", "kept-fp", "discarded-smoke", "discarded-fp"]}
    )
    out = filter_results(df, errors_only=True)
    assert set(out["outcome"]) == {"kept-fp", "discarded-smoke"}


def test_performance_summary_counts_and_rates():
    df = pd.DataFrame(
        {
            "label": ["smoke", "smoke", "fp", "fp"],
            "outcome": ["kept-smoke", "discarded-smoke", "discarded-fp", "kept-fp"],
        }
    )
    s = performance_summary(df)
    assert s["n_smoke"] == 2 and s["n_fp"] == 2
    assert s["recall"] == 0.5
    assert s["specificity"] == 0.5
    assert s["precision"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd eval && uv run pytest tests/test_outcomes.py -v`
Expected: FAIL — `ModuleNotFoundError: temporal_model.eval.outcomes`.

- [ ] **Step 3: Port the implementation**

Create `eval/src/temporal_model/eval/outcomes.py` by copying
`EXPLORER/src/temporal_model_explorer/outcomes.py` **verbatim** (no import changes needed —
it only depends on `pandas`).

- [ ] **Step 4: Add `pandas` + `pyarrow` to eval deps**

In `eval/pyproject.toml`, add to `[project].dependencies`:

```python
    "pandas>=2.0",
    "pyarrow>=15",
```

Run: `cd eval && uv sync`

- [ ] **Step 5: Run test to verify it passes**

Run: `cd eval && uv run pytest tests/test_outcomes.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add eval/src/temporal_model/eval/outcomes.py eval/tests/test_outcomes.py \
        eval/pyproject.toml eval/uv.lock
git commit -m "feat(eval): port pure outcome helpers"
```

---

## Task 5: Normalized view-store emit (`view_store.py`)

A single per-sequence record the viewer reads for BOTH source kinds: key, source, label,
metadata, and ordered frame paths (relative to the eval package dir, where both `dvc repro`
and the Streamlit app run).

**Files:**
- Create: `eval/src/temporal_model/eval/view_store.py`
- Test: `eval/tests/test_view_store.py`

- [ ] **Step 1: Write the failing test**

Create `eval/tests/test_view_store.py`:

```python
import json
from pathlib import Path

from temporal_model.eval.view_store import SequenceView, write_sequence_view


def test_write_sequence_view_roundtrips(tmp_path):
    view = SequenceView(
        key="wf_seq_a",
        source="train",
        label="smoke",
        organization_name=None,
        camera_name=None,
        started_at=None,
        frames=["data/01_raw/datasets/train/wildfire/wf_seq_a/images/f0.jpg"],
    )
    out_dir = tmp_path / "sequences"
    write_sequence_view(out_dir, view)
    payload = json.loads((out_dir / "wf_seq_a.json").read_text())
    assert payload["key"] == "wf_seq_a"
    assert payload["source"] == "train"
    assert payload["label"] == "smoke"
    assert payload["frames"] == [
        "data/01_raw/datasets/train/wildfire/wf_seq_a/images/f0.jpg"
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd eval && uv run pytest tests/test_view_store.py -v`
Expected: FAIL — `ModuleNotFoundError: temporal_model.eval.view_store`.

- [ ] **Step 3: Write the implementation**

Create `eval/src/temporal_model/eval/view_store.py`:

```python
"""Normalized per-sequence record the viewer reads (both source kinds).

One ``sequences/<key>.json`` per scored sequence: identity + metadata + ordered
frame paths (relative to the eval package dir, where dvc repro and the Streamlit
app both run). The viewer joins these to ``results.json`` (scalar verdicts) and
``details/<key>.json`` (tubes) on ``key``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SequenceView:
    key: str
    source: str
    label: str  # "smoke" | "fp" | "unknown"
    organization_name: str | None
    camera_name: str | None
    started_at: str | None
    frames: list[str] = field(default_factory=list)  # paths relative to eval dir


def write_sequence_view(sequences_dir: Path, view: SequenceView) -> None:
    """Write ``sequences/<key>.json`` for one sequence."""
    sequences_dir.mkdir(parents=True, exist_ok=True)
    (sequences_dir / f"{view.key}.json").write_text(json.dumps(asdict(view), indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd eval && uv run pytest tests/test_view_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/src/temporal_model/eval/view_store.py eval/tests/test_view_store.py
git commit -m "feat(eval): add normalized view-store record"
```

---

## Task 6: Emit per-sequence details + results table from `evaluate.py`

Extend `evaluate.py` so every scored directory-convention sequence also writes
`details/<key>.json`, `sequences/<key>.json`, and rows into `results.{parquet,json}`.
Metrics/predictions behaviour is unchanged.

**Files:**
- Modify: `eval/src/temporal_model/eval/evaluate.py`
- Test: `eval/tests/test_evaluate_driver.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `eval/tests/test_evaluate_driver.py` (inside the existing
`test_evaluate_packaged_writes_expected_outputs`, after the existing assertions add a new
block; or add a new test reusing the same setup). Add this new test function:

```python
def test_evaluate_packaged_writes_viewer_artifacts(tmp_path, monkeypatch):
    import pandas as pd

    sequences_dir = tmp_path / "sequences"
    output_dir = tmp_path / "out"
    _make_sequence(sequences_dir, "wildfire", "wf_seq_a", n_frames=4)  # TP
    _make_sequence(sequences_dir, "fp", "fp_seq_c", n_frames=4)  # FP

    monkeypatch.setattr(
        model_module.BboxTubeTemporalModel,
        "from_archive",
        classmethod(lambda cls, path, device=None: _FakeModel()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_packaged.py",
            "--model-zip",
            str(tmp_path / "placeholder.zip"),
            "--sequences-dir",
            str(sequences_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "vit_dinov2_finetune-train",
            "--source",
            "train",
        ],
    )
    evaluate_packaged.main()

    # per-sequence details + view records
    assert (output_dir / "details" / "wf_seq_a.json").is_file()
    assert (output_dir / "sequences" / "wf_seq_a.json").is_file()
    view = json.loads((output_dir / "sequences" / "wf_seq_a.json").read_text())
    assert view["source"] == "train"
    assert view["label"] == "smoke"
    assert len(view["frames"]) == 4

    # results table (json + parquet), one row per sequence
    assert (output_dir / "results.parquet").is_file()
    rows = json.loads((output_dir / "results.json").read_text())
    by_key = {r["key"]: r for r in rows}
    assert set(by_key) == {"wf_seq_a", "fp_seq_c"}
    assert by_key["wf_seq_a"]["decision"] == "keep"
    assert by_key["wf_seq_a"]["outcome"] == "kept-smoke"
    assert by_key["wf_seq_a"]["source"] == "train"
    assert by_key["fp_seq_c"]["outcome"] == "kept-fp"  # FakeModel keeps 4-frame seqs
    df = pd.read_parquet(output_dir / "results.parquet")
    assert len(df) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd eval && uv run pytest tests/test_evaluate_driver.py::test_evaluate_packaged_writes_viewer_artifacts -v`
Expected: FAIL — `--source` unknown arg / missing `details/` dir.

- [ ] **Step 3: Implement the emit**

In `eval/src/temporal_model/eval/evaluate.py`:

(a) Add imports near the top:

```python
import pandas as pd

from temporal_model.eval.outcomes import compute_outcome, decision_from_output, max_probability
from temporal_model.eval.view_store import SequenceView, write_sequence_view
```

(b) Add the `--source` arg in `_parse_args` (after `--model-name`):

```python
    parser.add_argument(
        "--source",
        default=None,
        help="Source label for the results table (e.g. 'train', 'val', "
        "'pyro-annotator'). Defaults to the sequences-dir name.",
    )
```

(c) In `main()`, derive the source and collect viewer rows. After
`args.output_dir.mkdir(...)` add:

```python
    source = args.source or args.sequences_dir.name
    details_dir = args.output_dir / "details"
    sequences_dir = args.output_dir / "sequences"
    result_rows: list[dict] = []
```

(d) Inside the per-sequence loop, after `output = model.predict(...)` and the
`records.append(build_record(...))` call, add the per-sequence emit. The directory-convention
key is the sequence dir name; frame paths are stored as posix strings relative to cwd
(dvc passes a relative `--sequences-dir`, so `frame_paths` are already relative):

```python
        key = seq_dir.name
        decision = decision_from_output(output.is_positive)
        outcome = compute_outcome(decision, label)
        details_dir.mkdir(parents=True, exist_ok=True)
        (details_dir / f"{key}.json").write_text(
            json.dumps(output.details, indent=2, default=str)
        )
        write_sequence_view(
            sequences_dir,
            SequenceView(
                key=key,
                source=source,
                label=label,
                organization_name=None,
                camera_name=None,
                started_at=None,
                frames=[p.as_posix() for p in frame_paths],
            ),
        )
        kept = output.details.get("tubes", {}).get("kept", [])
        result_rows.append(
            {
                "key": key,
                "source": source,
                "label": label,
                "decision": decision,
                "outcome": outcome,
                "score": max(t["logit"] for t in kept) if kept else None,
                "probability": max_probability(output.details),
                "trigger_frame_index": output.trigger_frame_index,
                "organization_name": None,
                "camera_name": None,
                "started_at": None,
            }
        )
```

(e) After the loop and the existing metrics/predictions/plots writes, write the results
table:

```python
    results_df = pd.DataFrame(result_rows)
    results_df.to_parquet(args.output_dir / "results.parquet")
    (args.output_dir / "results.json").write_text(
        json.dumps(result_rows, indent=2)
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd eval && uv run pytest tests/test_evaluate_driver.py -v`
Expected: PASS — including the 3 pre-existing driver tests (unchanged behaviour) and the new one.

- [ ] **Step 5: Commit**

```bash
git add eval/src/temporal_model/eval/evaluate.py eval/tests/test_evaluate_driver.py
git commit -m "feat(eval): emit per-sequence details + results table"
```

---

## Task 7: Meta-store source kind in `evaluate.py` (pyro-annotator)

Let `evaluate.py` score a meta-store source (frames + `meta.json`), taking labels from
meta, excluding `unknown` from metrics, and emitting the same viewer artifacts (with
org/camera/started_at populated).

**Files:**
- Modify: `eval/src/temporal_model/eval/evaluate.py`
- Test: `eval/tests/test_evaluate_driver.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `eval/tests/test_evaluate_driver.py`:

```python
def _write_store_seq(root, key, label, n_frames):
    import json as _json

    seq = root / "org-a" / "cam-1" / key
    (seq / "images").mkdir(parents=True)
    frames = []
    for i in range(n_frames):
        (seq / "images" / f"f{i}.jpg").write_bytes(b"\xff")
        frames.append({"file": f"images/f{i}.jpg", "detection_id": None, "created_at": None})
    meta = {
        "key": key,
        "sequence_id": key,
        "source": "pyro-annotator",
        "label": label,
        "label_detail": None,
        "label_source": "pyro_annotator_folder",
        "frames": frames,
        "camera_id": 1,
        "camera_name": "cam-1",
        "organization_id": 7,
        "organization_name": "org-a",
        "started_at": "2026-05-19T14:10:01",
    }
    (seq / "meta.json").write_text(_json.dumps(meta))
    return seq


def test_evaluate_store_source_excludes_unknown_from_metrics(tmp_path, monkeypatch):
    store_dir = tmp_path / "pyro"
    output_dir = tmp_path / "out"
    _write_store_seq(store_dir, "seq_smoke", "smoke", n_frames=4)  # TP
    _write_store_seq(store_dir, "seq_unknown", "unknown", n_frames=4)  # not labeled

    monkeypatch.setattr(
        model_module.BboxTubeTemporalModel,
        "from_archive",
        classmethod(lambda cls, path, device=None: _FakeModel()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_packaged.py",
            "--model-zip",
            str(tmp_path / "placeholder.zip"),
            "--sequences-dir",
            str(store_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "vit_dinov2_finetune-pyro-annotator",
            "--source",
            "pyro-annotator",
            "--store",
        ],
    )
    evaluate_packaged.main()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["num_sequences"] == 1  # unknown excluded from metrics

    rows = json.loads((output_dir / "results.json").read_text())
    by_key = {r["key"]: r for r in rows}
    assert set(by_key) == {"seq_smoke", "seq_unknown"}  # both viewable
    assert by_key["seq_unknown"]["outcome"] == "n/a"
    assert by_key["seq_smoke"]["organization_name"] == "org-a"
    assert by_key["seq_smoke"]["camera_name"] == "cam-1"

    view = json.loads((output_dir / "sequences" / "seq_unknown.json").read_text())
    assert view["organization_name"] == "org-a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd eval && uv run pytest tests/test_evaluate_driver.py::test_evaluate_store_source_excludes_unknown_from_metrics -v`
Expected: FAIL — `--store` unknown.

- [ ] **Step 3: Implement the store source path**

In `eval/src/temporal_model/eval/evaluate.py`:

(a) Add imports:

```python
from temporal_model.eval.store import build_frames, iter_sequence_dirs, read_meta
```

(b) Add the `--store` flag in `_parse_args`:

```python
    parser.add_argument(
        "--store",
        action="store_true",
        help="Treat --sequences-dir as a meta.json store (pyro-annotator) "
        "instead of the {fp,wildfire}/<seq>/images directory convention.",
    )
```

(c) Refactor the loop body into a small iterator that yields a uniform tuple
`(key, frames, label, meta_or_none, frame_paths)` for both source kinds. Replace the
existing `for seq_dir in tqdm(sequences, ...)` setup. Define before the loop:

```python
    def _iter_dir_convention():
        for seq_dir in list_sequences(args.sequences_dir):
            frame_paths = get_sorted_frames(seq_dir)
            if not frame_paths:
                dropped.append({"sequence_id": seq_dir.name, "reason": "no_images"})
                continue
            label = "smoke" if is_wf_sequence(seq_dir) else "fp"
            frames = model.load_sequence(frame_paths)
            yield seq_dir.name, frames, label, None, frame_paths

    def _iter_store():
        for seq_dir in iter_sequence_dirs(args.sequences_dir):
            meta = read_meta(seq_dir)
            frame_paths = [seq_dir / f.file for f in meta.frames]
            if not frame_paths:
                dropped.append({"sequence_id": meta.key, "reason": "no_images"})
                continue
            frames = build_frames(seq_dir, meta)
            yield meta.key, frames, meta.label, meta, frame_paths

    iterator = _iter_store() if args.store else _iter_dir_convention()
```

(d) Replace the per-sequence loop with one that consumes the uniform tuple. Labeled
records (`smoke`/`fp`) feed metrics; ALL sequences get viewer artifacts:

```python
    for key, frames, label, meta, frame_paths in tqdm(
        iterator, desc=args.model_name, unit="seq"
    ):
        output = model.predict(frames, compute_trigger=True)
        if label in ("smoke", "fp"):
            records.append(
                build_record(
                    sequence_dir=Path(key),
                    label=label,
                    frames=frames,
                    output=output,
                )
            )
        decision = decision_from_output(output.is_positive)
        outcome = compute_outcome(decision, label)
        details_dir.mkdir(parents=True, exist_ok=True)
        (details_dir / f"{key}.json").write_text(
            json.dumps(output.details, indent=2, default=str)
        )
        org = meta.organization_name if meta else None
        cam = meta.camera_name if meta else None
        started = meta.started_at if meta else None
        write_sequence_view(
            sequences_dir,
            SequenceView(
                key=key,
                source=source,
                label=label,
                organization_name=org,
                camera_name=cam,
                started_at=started,
                frames=[p.as_posix() for p in frame_paths],
            ),
        )
        kept = output.details.get("tubes", {}).get("kept", [])
        result_rows.append(
            {
                "key": key,
                "source": source,
                "label": label,
                "decision": decision,
                "outcome": outcome,
                "score": max(t["logit"] for t in kept) if kept else None,
                "probability": max_probability(output.details),
                "trigger_frame_index": output.trigger_frame_index,
                "organization_name": org,
                "camera_name": cam,
                "started_at": started,
            }
        )
```

Notes for the engineer:
- `build_record` only reads `sequence_dir.name`, so `Path(key)` is sufficient and avoids a
  bogus path. (Confirm by reading `protocol_eval.build_record`.)
- Remove the now-duplicated emit block added in Task 6 — this loop supersedes it. Keep the
  Task 6 emit logic ONLY here (single loop). The `_make_sequence` dir-convention tests from
  Task 6 still pass because `_iter_dir_convention` reproduces the old behaviour.

- [ ] **Step 4: Run the full eval driver suite**

Run: `cd eval && uv run pytest tests/test_evaluate_driver.py -v`
Expected: PASS — all driver tests (3 original + Task 6 + Task 7).

- [ ] **Step 5: Commit**

```bash
git add eval/src/temporal_model/eval/evaluate.py eval/tests/test_evaluate_driver.py
git commit -m "feat(eval): score meta-store (pyro-annotator) source"
```

---

## Task 8: Port the pure render helpers (`render.py`)

Port every pure (non-Streamlit) helper from `EXPLORER/src/temporal_model_explorer/app.py`
into `eval/src/temporal_model/eval/render.py`, repointed to `core.crop`.

**Files:**
- Create: `eval/src/temporal_model/eval/render.py`
- Test: `eval/tests/test_render.py`

- [ ] **Step 1: Write the failing test**

Create `eval/tests/test_render.py` by copying `EXPLORER/tests/test_app_helpers.py`
**verbatim**, then change the import block at the top from
`from temporal_model_explorer.app import (...)` to
`from temporal_model.eval.render import (...)` — keeping the same imported names:
`_lightning_polygon, correctness_label, crop_around_bbox, day_of, draw_bboxes,
frame_bboxes_by_input_index, legend_html, processed_to_input_index, row_background,
triggering_tube_ids, tube_color, tube_input_boxes, tube_timeline_df`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd eval && uv run pytest tests/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: temporal_model.eval.render`.

- [ ] **Step 3: Port the implementation**

Create `eval/src/temporal_model/eval/render.py`. Copy these definitions **verbatim** from
`EXPLORER/src/temporal_model_explorer/app.py` (the pure helpers and their module-level
constants — do NOT copy `_drilldown`, `render_performance`, `main`, or any `st.*` code):

- module constants: `CROP_CONTEXT`, `CROP_SIZE`, `_BBOX_FONT` (the try/except block),
  `CORRECTNESS`, `ROW_BG`, `KEEP_BG`, `DISCARD_BG`, `TUBE_PALETTE`
- functions: `day_of`, `correctness_label`, `row_background`, `tube_color`, `legend_html`,
  `processed_to_input_index`, `frame_bboxes_by_input_index`, `tube_input_boxes`,
  `triggering_tube_ids`, `_lightning_polygon`, `draw_bboxes`, `crop_around_bbox`,
  `tube_timeline_df`

Change the crop-helper import. Replace:

```python
from bbox_tube_temporal.model_input import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)
```

with:

```python
from temporal_model.core.crop import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)
```

Keep `import numpy as np`, `import pandas as pd`, and
`from PIL import Image, ImageDraw, ImageFont` at the top.

- [ ] **Step 4: Add `pillow` to eval deps**

In `eval/pyproject.toml` `[project].dependencies`, add:

```python
    "pillow>=10",
```

Run: `cd eval && uv sync`

- [ ] **Step 5: Run test to verify it passes**

Run: `cd eval && uv run pytest tests/test_render.py -v`
Expected: PASS (all ported helper tests).

- [ ] **Step 6: Commit**

```bash
git add eval/src/temporal_model/eval/render.py eval/tests/test_render.py \
        eval/pyproject.toml eval/uv.lock
git commit -m "feat(eval): port pure render helpers"
```

---

## Task 9: Streamlit viewer (`app.py`) + Makefile target

Port the explorer's Streamlit UI, adapted to: read eval's reporting tree (`results.json` +
`details/` + `sequences/`), adaptive sidebar (source selector; org/camera filters only when
present), and single-mode stabilized tube crops.

**Files:**
- Create: `eval/src/temporal_model/eval/app.py`
- Modify: `eval/Makefile`
- Modify: `eval/pyproject.toml`
- Test: `eval/tests/test_render.py` (add an import-smoke test) — see Step 4

- [ ] **Step 1: Add Streamlit + Altair deps**

In `eval/pyproject.toml` `[project].dependencies`, add:

```python
    "streamlit>=1.40",
    "altair>=5",
```

Run: `cd eval && uv sync`

- [ ] **Step 2: Write `app.py`**

Create `eval/src/temporal_model/eval/app.py`. Port `EXPLORER/src/temporal_model_explorer/app.py`
with these specific changes (everything not listed is structurally the same, importing pure
helpers from `render.py` and outcome helpers from `outcomes.py`):

(a) Imports — replace the explorer's imports with:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from temporal_model.eval.render import (
    correctness_label,
    crop_around_bbox,
    draw_bboxes,
    frame_bboxes_by_input_index,
    legend_html,
    processed_to_input_index,
    row_background,
    tube_color,
    tube_input_boxes,
    triggering_tube_ids,
    tube_timeline_df,
    CORRECTNESS,
)
```

(b) Path constants — the viewer scans every per-source reporting dir. Replace the
explorer's `RESULTS/DETAILS/STORE/PARAMS` constants with:

```python
REPORTING = Path("data/08_reporting")
MODEL_NAME = "vit_dinov2_finetune"
PLAY_FPS = 1


def reporting_dirs() -> list[Path]:
    """Every <source>/vit_dinov2_finetune reporting dir that has a results.json."""
    if not REPORTING.exists():
        return []
    return sorted(
        p.parent
        for p in REPORTING.glob(f"*/{MODEL_NAME}/results.json")
    )


def load_results() -> pd.DataFrame:
    """Concatenate results.json across all sources (adds nothing if none)."""
    frames = []
    for d in reporting_dirs():
        rows = json.loads((d / "results.json").read_text())
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_details(source: str, key: str) -> dict:
    path = REPORTING / source / MODEL_NAME / "details" / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_sequence_view(source: str, key: str) -> dict:
    path = REPORTING / source / MODEL_NAME / "sequences" / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else {}
```

(c) Frame access — the explorer used a `store` to resolve frames. Replace its
`_find_seq_dir`/`meta.frames` usage in the drill-down with the view record's `frames`
list (paths relative to the eval dir). In `_drilldown`, resolve frames as:

```python
    view = load_sequence_view(source, key)
    frame_files = [Path(p) for p in view.get("frames", [])]
    n = len(frame_files)
```

and where the explorer drew `seq_dir / ref.file`, use `frame_files[i]` directly.

(d) Stabilized crop — single mode. In the tube-crops column, replace the explorer's
`stab_window = tube.get("stabilized_window")` dual-mode block (the `if stab_window else
at_frame[i]` and the `🔒 stabilized` badge) with always cropping the tube's
`stabilized_window`:

```python
        window = tube.get("stabilized_window")
        crop_box = tuple(window) if window else at_frame.get(i)
        if crop_box is not None:
            tubes_col.image(crop_around_bbox(frame_files[i], crop_box), width=220)
        else:
            tubes_col.caption("inactive at this frame")
```

(no `stab_note`, no 🔒 badge).

(e) Sidebar — adaptive. Replace the explorer's source→org→camera→model sidebar with a
source selector over `df["source"].unique()`, and show org/camera filters only when the
selected source has non-null values:

```python
    sources = sorted(df["source"].dropna().unique())
    source = st.sidebar.selectbox("source", sources, key="source")
    view = df[df["source"] == source].reset_index(drop=True)
    has_org = view["organization_name"].notna().any()
    has_cam = view["camera_name"].notna().any()
```

In the filter popover, gate the org/camera selectboxes on `has_org`/`has_cam`; always show
the ground-truth / model-verdict / correctness selectboxes (as in the explorer).

(f) Keep verbatim from the explorer (they are UI-agnostic to the above): the
correctness-coloured table styling (`_style_row`, `legend_box`, `st.dataframe` with
`on_select`), the metric cards (`render_performance`, fed `performance_summary` from
`outcomes`), the autoplay fragment mechanics (`@st.fragment(run_every=...)`, the
`frame_key` session-state advance), the tube-timeline Altair chart (`_tube_timeline_chart`),
and the bbox/trigger overlay logic. Import `performance_summary` from
`temporal_model.eval.outcomes`.

(g) Mark all Streamlit functions with `# pragma: no cover` (as the explorer does:
`_drilldown`, `render_performance`, `main`, `_tube_timeline_chart`).

- [ ] **Step 3: Add the `app` Makefile target**

In `eval/Makefile`, add to `.PHONY` and append:

```makefile
.PHONY: install lint format test update-model app

app: ## launch the Streamlit viewer
	uv run streamlit run src/temporal_model/eval/app.py
```

- [ ] **Step 4: Write an import-smoke test**

Append to `eval/tests/test_render.py`:

```python
def test_app_module_imports():
    # App must import cleanly (no Streamlit calls at import time).
    import temporal_model.eval.app as app  # noqa: F401

    assert hasattr(app, "main")
    assert hasattr(app, "reporting_dirs")
```

- [ ] **Step 5: Run the test + lint**

Run: `cd eval && uv run pytest tests/test_render.py::test_app_module_imports -v`
Expected: PASS.
Run: `cd eval && uv run ruff check .`
Expected: clean (fix any unused-import / ordering issues).

- [ ] **Step 6: Commit**

```bash
git add eval/src/temporal_model/eval/app.py eval/Makefile eval/pyproject.toml \
        eval/uv.lock eval/tests/test_render.py
git commit -m "feat(eval): add Streamlit qualitative viewer"
```

---

## Task 10: Wire pyro-annotator into the DVC pipeline + copy helper

**Files:**
- Modify: `eval/dvc.yaml`
- Create: `eval/scripts/copy_pyro_annotator.py`
- Modify: `eval/README.md`

- [ ] **Step 1: Add the copy helper**

Create `eval/scripts/copy_pyro_annotator.py`:

```python
"""One-time copy of the explorer's enriched pyro-annotator sequences into eval.

Copies frames + meta.json from the temporal-model-explorer store into eval's
data/01_raw/pyro-annotator/. Run once, then `dvc add` the result so it travels
via the eval DVC remote. Requires the explorer checkout to be present locally
with its data pulled (`dvc pull` in the explorer).
"""

import argparse
import shutil
from pathlib import Path

DEFAULT_SRC = Path(
    "../../vision-rd/experiments/temporal-models/temporal-model-explorer/"
    "data/03_primary/sequences/pyro-annotator"
)
DEFAULT_DST = Path("data/01_raw/pyro-annotator")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = parser.parse_args()
    if not args.src.exists():
        raise SystemExit(f"source not found: {args.src} (is the explorer data pulled?)")
    args.dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for meta in args.src.rglob("meta.json"):
        rel = meta.parent.relative_to(args.src)
        shutil.copytree(meta.parent, args.dst / rel, dirs_exist_ok=True)
        n += 1
    print(f"copied {n} sequences -> {args.dst}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the copy (one-time data op) + dvc add**

This is a data operation, not a code change — run it once locally:

```bash
cd eval
uv run python scripts/copy_pyro_annotator.py
uv run dvc add data/01_raw/pyro-annotator
```

Expected: `copied <N> sequences -> data/01_raw/pyro-annotator` and a new
`data/01_raw/pyro-annotator.dvc` file.

If the explorer data is not available on this machine, skip the run and note it — the
`dvc.yaml` wiring (next step) and `dvc repro` for this source will simply have no input
until the copy is done.

- [ ] **Step 3: Add `pyro-annotator` to the evaluate foreach**

`eval/dvc.yaml`'s `evaluate` stage uses one `cmd` template for all foreach items, but
pyro-annotator needs the `--store` flag and a different `--sequences-dir`. Since the
directory-convention items (`train`, `val`) and the store item differ in flags, split into
two foreach blocks is overkill; instead make the command source-aware via a wrapper is also
overkill. The simplest correct change: keep `train`/`val` in the existing block and add a
SECOND stage for the store source. Add this stage alongside `evaluate`:

```yaml
  evaluate_pyro_annotator:
    cmd: >-
      uv run python -m temporal_model.eval.evaluate
      --model-zip data/06_models/vit_dinov2_finetune/model.zip
      --sequences-dir data/01_raw/pyro-annotator
      --output-dir data/08_reporting/pyro-annotator/vit_dinov2_finetune
      --model-name vit_dinov2_finetune-pyro-annotator
      --source pyro-annotator
      --store
    deps:
      - src/temporal_model/eval/evaluate.py
      - src/temporal_model/eval/protocol_eval.py
      - src/temporal_model/eval/eval_plots.py
      - src/temporal_model/eval/store.py
      - src/temporal_model/eval/outcomes.py
      - src/temporal_model/eval/view_store.py
      - ../core/src/temporal_model/core/model.py
      - ../core/src/temporal_model/core/inference.py
      - ../core/src/temporal_model/core/stabilize.py
      - ../core/src/temporal_model/core/crop.py
      - ../core/src/temporal_model/core/protocol.py
      - ../core/src/temporal_model/core/details_schema.py
      - data/06_models/vit_dinov2_finetune/model.zip
      - data/01_raw/pyro-annotator
    outs:
      - data/08_reporting/pyro-annotator/vit_dinov2_finetune/predictions.json:
          cache: false
      - data/08_reporting/pyro-annotator/vit_dinov2_finetune/dropped.json:
          cache: false
      - data/08_reporting/pyro-annotator/vit_dinov2_finetune/results.json:
          cache: false
      - data/08_reporting/pyro-annotator/vit_dinov2_finetune/results.parquet:
          cache: false
      - data/08_reporting/pyro-annotator/vit_dinov2_finetune/details:
          cache: false
      - data/08_reporting/pyro-annotator/vit_dinov2_finetune/sequences:
          cache: false
    metrics:
      - data/08_reporting/pyro-annotator/vit_dinov2_finetune/metrics.json:
          cache: false
```

Also add the new emitted outputs (`results.json`, `results.parquet`, `details`,
`sequences`) to the existing `evaluate` foreach `outs:` block, so train/val emit them too:

```yaml
      outs:
        - data/08_reporting/${item}/vit_dinov2_finetune/predictions.json:
            cache: false
        - data/08_reporting/${item}/vit_dinov2_finetune/dropped.json:
            cache: false
        - data/08_reporting/${item}/vit_dinov2_finetune/results.json:
            cache: false
        - data/08_reporting/${item}/vit_dinov2_finetune/results.parquet:
            cache: false
        - data/08_reporting/${item}/vit_dinov2_finetune/details:
            cache: false
        - data/08_reporting/${item}/vit_dinov2_finetune/sequences:
            cache: false
```

And add the new eval module deps to the existing `evaluate` foreach `deps:` block:
`src/temporal_model/eval/store.py`, `src/temporal_model/eval/outcomes.py`,
`src/temporal_model/eval/view_store.py`.

- [ ] **Step 4: Validate the DVC graph parses**

Run: `cd eval && uv run dvc status` (or `uv run dvc dag`)
Expected: no YAML/parse error; the two stages (`evaluate@train`, `evaluate@val`,
`evaluate_pyro_annotator`) appear. (Stages may show as "changed"/"not in cache" — that is
expected until `dvc repro` runs with real inputs.)

- [ ] **Step 5: Update the README**

In `eval/README.md`, add a section documenting:
- the viewer (`make app`), what it shows, and that it is read-only;
- the new emitted artifacts and the **frontend-agnostic data contract**:
  `results.{json,parquet}` (per-sequence rows), `details/<key>.json` (`BboxTubeDetails`
  incl. `stabilized_window`), `sequences/<key>.json` (key, source, label, metadata, frame
  paths) — the stable interface a future React frontend consumes;
- the pyro-annotator source: copied via `scripts/copy_pyro_annotator.py`, then re-scored by
  eval's own `model.zip` (`evaluate_pyro_annotator` stage).

- [ ] **Step 6: Commit**

```bash
git add eval/dvc.yaml eval/scripts/copy_pyro_annotator.py eval/README.md eval/dvc.lock
git commit -m "feat(eval): wire pyro-annotator source + document viewer contract"
```

---

## Task 11: Final verification

- [ ] **Step 1: Full core suite**

Run: `cd core && uv run pytest -q`
Expected: PASS (incl. new `test_stabilized_window.py`, unchanged parity tests).

- [ ] **Step 2: Full eval suite + lint**

Run: `cd eval && uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all PASS, lint/format clean.

- [ ] **Step 3: Manual viewer smoke (if data available)**

If `dvc pull`/copy made data available, run `cd eval && uv run dvc repro` then `make app`
and confirm: source selector lists train/val/pyro-annotator; a sequence drill-down autoplays
frames with bboxes and shows stabilized tube crops + the decision. (Skip if no data on this
machine; the import-smoke test already guards the app module.)

- [ ] **Step 4: Final commit (if any fixups)**

```bash
git add -A && git commit -m "chore(eval): viewer verification fixups"
```
```

(no-op if nothing changed)
