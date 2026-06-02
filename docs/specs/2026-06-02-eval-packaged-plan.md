# eval/ Packaged-Model Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `eval/` scaffold stub with a DVC-driven pipeline that runs a packaged `model.zip` end-to-end over raw image sequences and emits leaderboard-schema metrics + PR/ROC/confusion plots.

**Architecture:** A faithful, ViT-only, core-only port of the `evaluate_packaged` path from `vision-rd/experiments/temporal-models/bbox-tube-temporal`. Three modules move into `src/temporal_model/eval/` with mechanical import rewrites (`pyrocore` → `temporal_model.core.protocol`, `bbox_tube_temporal.*` → `temporal_model.core.*`, `bbox_tube_temporal_exp.*` → `temporal_model.eval.*`). Invoked via `python -m temporal_model.eval.evaluate`, mirroring the merged `train` package convention. No logic changes — only imports.

**Tech Stack:** Python 3.11, uv, pytest, ruff, DVC, numpy, scikit-learn, matplotlib, tqdm, `temporal-model-core` (path dep).

**Reference spec:** `docs/specs/2026-06-02-eval-packaged-design.md`

---

## Source files (read these to copy from)

All under `/mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/temporal-models/bbox-tube-temporal/`:

- `scripts/evaluate_packaged.py` → `eval/src/temporal_model/eval/evaluate.py`
- `src/bbox_tube_temporal_exp/protocol_eval.py` → `eval/src/temporal_model/eval/protocol_eval.py`
- `src/bbox_tube_temporal_exp/eval_plots.py` → `eval/src/temporal_model/eval/eval_plots.py`
- `tests/test_protocol_eval.py` → `eval/tests/test_protocol_eval.py`
- `tests/test_eval_plots.py` → `eval/tests/test_eval_plots.py`
- `tests/test_evaluate_packaged_driver.py` → `eval/tests/test_evaluate_driver.py`

## File Structure (created / modified)

| Action | Path | Responsibility |
|---|---|---|
| Create | `eval/src/temporal_model/eval/protocol_eval.py` | `SequenceRecord`, `build_record`, `compute_metrics` |
| Create | `eval/src/temporal_model/eval/eval_plots.py` | matplotlib PR/ROC/confusion helpers |
| Modify | `eval/src/temporal_model/eval/evaluate.py` | replace stub with packaged-model evaluator (`main()`) |
| Modify | `eval/src/temporal_model/eval/__init__.py` | drop "scaffold stub" docstring |
| Create | `eval/tests/test_protocol_eval.py` | metrics/record parity tests |
| Create | `eval/tests/test_eval_plots.py` | plot-helper smoke tests |
| Create | `eval/tests/test_evaluate_driver.py` | end-to-end driver test (mocked model) |
| Delete | `eval/tests/test_smoke.py` | superseded by ported tests (keep one import check, see Task 6) |
| Modify | `eval/pyproject.toml` | add numpy/scikit-learn/matplotlib/tqdm; drop pyyaml |
| Modify | `eval/dvc.yaml` | replace `noop` with the `evaluate` stage |
| Delete | `eval/params.yaml` | unused placeholder |
| Modify | `eval/README.md` | describe the real pipeline |
| Modify | `README.md` (root) | flip `eval` status to `implemented` |

All commands assume the working directory is the repo root: `/mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/temporal-model`. The branch `arthur/feat-eval-packaged` is already checked out.

---

### Task 1: Add eval runtime dependencies

**Files:**
- Modify: `eval/pyproject.toml`

- [ ] **Step 1: Replace the `dependencies` array**

In `eval/pyproject.toml`, replace:

```toml
dependencies = [
    "temporal-model-core",
    "pyyaml>=6.0",
]
```

with:

```toml
dependencies = [
    "temporal-model-core",
    "numpy>=1.26,<2",
    "scikit-learn>=1.4",
    "matplotlib>=3.8",
    "tqdm>=4.66",
]
```

Leave everything else in the file (the `temporal-eval = "temporal_model.eval.evaluate:main"` script entry, `[tool.uv.sources]`, ruff config) unchanged.

- [ ] **Step 2: Sync the environment**

Run: `cd eval && uv sync && cd ..`
Expected: resolves and installs numpy, scikit-learn, matplotlib, tqdm without error.

- [ ] **Step 3: Verify the new deps import**

Run: `cd eval && uv run python -c "import numpy, sklearn, matplotlib, tqdm; print('ok')" && cd ..`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add eval/pyproject.toml eval/uv.lock
git commit -m "build(eval): add numpy/scikit-learn/matplotlib/tqdm; drop unused pyyaml"
```

---

### Task 2: Port `protocol_eval.py` (test-first)

**Files:**
- Create: `eval/tests/test_protocol_eval.py`
- Create: `eval/src/temporal_model/eval/protocol_eval.py`

- [ ] **Step 1: Create the test by copying the source and rewriting imports**

Copy `…/bbox-tube-temporal/tests/test_protocol_eval.py` to `eval/tests/test_protocol_eval.py` verbatim, then apply exactly these two line replacements:

Replace:
```python
from pyrocore import Frame, TemporalModelOutput

from bbox_tube_temporal_exp.protocol_eval import (
```
with:
```python
from temporal_model.core.protocol import Frame, TemporalModelOutput

from temporal_model.eval.protocol_eval import (
```

No other lines change.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd eval && uv run pytest tests/test_protocol_eval.py -q; cd ..`
Expected: collection/import error — `ModuleNotFoundError: No module named 'temporal_model.eval.protocol_eval'`.

- [ ] **Step 3: Create the module by copying the source and rewriting one import**

Copy `…/bbox-tube-temporal/src/bbox_tube_temporal_exp/protocol_eval.py` to `eval/src/temporal_model/eval/protocol_eval.py` verbatim, then apply exactly this replacement:

Replace:
```python
from pyrocore import Frame, TemporalModelOutput
```
with:
```python
from temporal_model.core.protocol import Frame, TemporalModelOutput
```

No other lines change. (`Frame` is imported only for the `build_record` type hint; keep it.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd eval && uv run pytest tests/test_protocol_eval.py -q; cd ..`
Expected: all tests pass (12 tests).

- [ ] **Step 5: Commit**

```bash
git add eval/src/temporal_model/eval/protocol_eval.py eval/tests/test_protocol_eval.py
git commit -m "feat(eval): port protocol_eval (SequenceRecord, build_record, compute_metrics)"
```

---

### Task 3: Port `eval_plots.py` (test-first)

**Files:**
- Create: `eval/tests/test_eval_plots.py`
- Create: `eval/src/temporal_model/eval/eval_plots.py`

- [ ] **Step 1: Create the test by copying the source and rewriting one import**

Copy `…/bbox-tube-temporal/tests/test_eval_plots.py` to `eval/tests/test_eval_plots.py` verbatim, then apply exactly this replacement:

Replace:
```python
from bbox_tube_temporal_exp.eval_plots import (
```
with:
```python
from temporal_model.eval.eval_plots import (
```

No other lines change.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd eval && uv run pytest tests/test_eval_plots.py -q; cd ..`
Expected: import error — `ModuleNotFoundError: No module named 'temporal_model.eval.eval_plots'`.

- [ ] **Step 3: Create the module by copying the source verbatim (no rewrites)**

Copy `…/bbox-tube-temporal/src/bbox_tube_temporal_exp/eval_plots.py` to `eval/src/temporal_model/eval/eval_plots.py` with **no changes** — it imports only stdlib, numpy, matplotlib, and sklearn.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd eval && uv run pytest tests/test_eval_plots.py -q; cd ..`
Expected: all tests pass (7 tests). Matplotlib uses a non-interactive backend by default under pytest; if a display error appears, set `MPLBACKEND=Agg` for the run.

- [ ] **Step 5: Commit**

```bash
git add eval/src/temporal_model/eval/eval_plots.py eval/tests/test_eval_plots.py
git commit -m "feat(eval): port eval_plots (PR/ROC/confusion matplotlib helpers)"
```

---

### Task 4: Port the `evaluate.py` driver (test-first)

**Files:**
- Create: `eval/tests/test_evaluate_driver.py`
- Modify: `eval/src/temporal_model/eval/evaluate.py` (currently the stub)

- [ ] **Step 1: Create the driver test by copying the source and rewriting imports**

Copy `…/bbox-tube-temporal/tests/test_evaluate_packaged_driver.py` to `eval/tests/test_evaluate_driver.py` verbatim, then apply exactly these three replacements:

Replace:
```python
from bbox_tube_temporal import model as model_module
from pyrocore import Frame, TemporalModelOutput

from scripts import evaluate_packaged
```
with:
```python
from temporal_model.core import model as model_module
from temporal_model.core.protocol import Frame, TemporalModelOutput

from temporal_model.eval import evaluate as evaluate_packaged
```

No other lines change. The local alias `evaluate_packaged` keeps every later `evaluate_packaged.main()` reference valid. The monkeypatch target `model_module.BboxTubeTemporalModel.from_archive` patches the class object itself, so it takes effect even though `evaluate.py` binds the class via `from … import BboxTubeTemporalModel`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd eval && uv run pytest tests/test_evaluate_driver.py -q; cd ..`
Expected: fails — the current `evaluate.py` stub has no packaged-eval logic (`main()` raises `SystemExit("temporal-eval: not implemented yet (scaffold stub)")`, so assertions on output files fail / it exits non-zero).

- [ ] **Step 3: Replace `evaluate.py` with the ported evaluator**

Overwrite `eval/src/temporal_model/eval/evaluate.py` entirely with the contents of `…/bbox-tube-temporal/scripts/evaluate_packaged.py`, applying exactly these import replacements:

Replace:
```python
from bbox_tube_temporal.data import get_sorted_frames, is_wf_sequence, list_sequences
from bbox_tube_temporal.model import BboxTubeTemporalModel
from tqdm import tqdm

from bbox_tube_temporal_exp.eval_plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from bbox_tube_temporal_exp.protocol_eval import (
    SequenceRecord,
    build_record,
    compute_metrics,
)
```
with:
```python
from temporal_model.core.data import (
    get_sorted_frames,
    is_wf_sequence,
    list_sequences,
)
from temporal_model.core.model import BboxTubeTemporalModel
from tqdm import tqdm

from temporal_model.eval.eval_plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from temporal_model.eval.protocol_eval import (
    SequenceRecord,
    build_record,
    compute_metrics,
)
```

Everything else (the module docstring, `_parse_args`, `_record_to_json`, `main`, the `if __name__ == "__main__":` guard) is copied unchanged.

- [ ] **Step 4: Run the driver test to verify it passes**

Run: `cd eval && uv run pytest tests/test_evaluate_driver.py -q; cd ..`
Expected: all 3 tests pass (`…writes_expected_outputs`, `…strict_errors_abort`, `…skips_sequences_without_images`).

- [ ] **Step 5: Verify the module entry point runs**

Run: `cd eval && uv run python -m temporal_model.eval.evaluate --help; cd ..`
Expected: argparse usage text listing `--model-zip`, `--sequences-dir`, `--output-dir`, `--model-name`, `--device`.

- [ ] **Step 6: Commit**

```bash
git add eval/src/temporal_model/eval/evaluate.py eval/tests/test_evaluate_driver.py
git commit -m "feat(eval): port packaged-model evaluator (evaluate_packaged -> evaluate.py)"
```

---

### Task 5: Clean up package metadata and the stub test

**Files:**
- Modify: `eval/src/temporal_model/eval/__init__.py`
- Delete: `eval/tests/test_smoke.py`

- [ ] **Step 1: Update the package docstring**

Replace the entire contents of `eval/src/temporal_model/eval/__init__.py`:

```python
"""Evaluation pipeline for the temporal smoke classifier."""
```

- [ ] **Step 2: Remove the superseded scaffold smoke test**

The two assertions in `eval/tests/test_smoke.py` (`evaluate.main` is callable; `temporal_model.core` imports) are now covered by the driver test and the core dependency itself. Delete it:

Run: `git rm eval/tests/test_smoke.py`

- [ ] **Step 3: Run the full eval suite**

Run: `cd eval && uv run pytest -q; cd ..`
Expected: 22 tests pass (12 protocol_eval + 7 eval_plots + 3 driver), no errors.

- [ ] **Step 4: Commit**

```bash
git add eval/src/temporal_model/eval/__init__.py
git commit -m "chore(eval): drop scaffold-stub docstring and superseded smoke test"
```

---

### Task 6: Wire the DVC pipeline and remove the params placeholder

**Files:**
- Modify: `eval/dvc.yaml`
- Delete: `eval/params.yaml`

- [ ] **Step 1: Replace `eval/dvc.yaml` entirely**

```yaml
stages:
  evaluate:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python -m temporal_model.eval.evaluate
        --model-zip data/06_models/vit_dinov2_finetune/model.zip
        --sequences-dir data/01_raw/datasets/${item}
        --output-dir data/08_reporting/${item}/vit_dinov2_finetune
        --model-name vit_dinov2_finetune-${item}
      deps:
        - src/temporal_model/eval/evaluate.py
        - src/temporal_model/eval/protocol_eval.py
        - src/temporal_model/eval/eval_plots.py
        - ../core/src/temporal_model/core/data.py
        - ../core/src/temporal_model/core/model.py
        - ../core/src/temporal_model/core/inference.py
        - ../core/src/temporal_model/core/tubes.py
        - ../core/src/temporal_model/core/model_input.py
        - ../core/src/temporal_model/core/protocol.py
        - data/06_models/vit_dinov2_finetune/model.zip
        - data/01_raw/datasets/${item}
      outs:
        - data/08_reporting/${item}/vit_dinov2_finetune/predictions.json:
            cache: false
        - data/08_reporting/${item}/vit_dinov2_finetune/dropped.json:
            cache: false
      metrics:
        - data/08_reporting/${item}/vit_dinov2_finetune/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item}/vit_dinov2_finetune/pr_curve.png
        - data/08_reporting/${item}/vit_dinov2_finetune/roc_curve.png
        - data/08_reporting/${item}/vit_dinov2_finetune/confusion_matrix.png
        - data/08_reporting/${item}/vit_dinov2_finetune/confusion_matrix_normalized.png
```

- [ ] **Step 2: Remove the unused params placeholder**

Run: `git rm eval/params.yaml`

- [ ] **Step 3: Validate the DVC stage definition parses**

Run: `cd eval && uv run dvc stage list 2>&1 | head; cd ..`
Expected: lists `evaluate@train` and `evaluate@val` (DVC expands the `foreach`). It is expected that `dvc repro` would fail without real data/`model.zip` — we only validate that the YAML parses and the stages are recognized, not that they run.

- [ ] **Step 4: Commit**

```bash
git add eval/dvc.yaml
git commit -m "feat(eval): add DVC evaluate stage (foreach train/val); drop params placeholder"
```

---

### Task 7: Documentation and final verification

**Files:**
- Modify: `eval/README.md`
- Modify: `README.md` (root)

- [ ] **Step 1: Rewrite `eval/README.md`**

```markdown
# temporal-model-eval

DVC evaluation pipeline for the temporal smoke classifier: runs a packaged
`model.zip` end-to-end over raw image sequences and reports protocol-level
metrics plus PR/ROC/confusion-matrix plots.

Import as `temporal_model.eval`; CLI entry point `temporal-eval` (also runnable
as `python -m temporal_model.eval.evaluate`). Depends on `temporal-model-core`.

## Modules

- `evaluate.py` — the packaged-model evaluator. Loads `model.zip` via
  `core.model.BboxTubeTemporalModel.from_archive`, iterates the sequences in a
  split, calls `load_sequence` + `predict` per sequence, and writes metrics,
  per-sequence predictions, and plots.
- `protocol_eval.py` — `SequenceRecord` + `build_record` + `compute_metrics`
  (precision/recall/F1/FPR, mean/median TTD in frames, PR/ROC AUC). Field names
  and rounding match the leaderboard schema.
- `eval_plots.py` — matplotlib PR/ROC/confusion-matrix helpers.

## Pipeline

`dvc.yaml` defines one `evaluate` stage, run `foreach` train/val. It consumes a
packaged model at `data/06_models/vit_dinov2_finetune/model.zip` (an **external
input** — nothing in this repo builds it; supply it or pull via DVC) and raw
sequences under `data/01_raw/datasets/{train,val}/{fp,wildfire}/<seq>/images/`,
writing `metrics.json`, `predictions.json`, `dropped.json`, and PR/ROC/confusion
PNGs under `data/08_reporting/{split}/vit_dinov2_finetune/`.

Ground truth comes from the directory convention (`wildfire/` → smoke, else fp).
Error policy is strict: any per-sequence inference exception aborts the run;
sequences with no images are recorded in `dropped.json` and skipped.

## Run

```bash
make install
make test
uv run dvc repro            # needs model.zip + raw sequences in place
```
```

- [ ] **Step 2: Flip the eval status in the root `README.md`**

In `README.md`, replace:
```
> Status: `core`, `train`, and `api` are implemented (migrated from the
> `vision-rd` `bbox-tube-temporal` work). `eval` is still a scaffold stub.
```
with:
```
> Status: all four packages — `core`, `train`, `api`, and `eval` — are
> implemented (migrated from the `vision-rd` `bbox-tube-temporal` work).
```

And in the packages table, replace:
```
| `eval/`  | `temporal-model-eval`  | `temporal_model.eval`  | DVC evaluation pipeline. Depends on `core`. | scaffold |
```
with:
```
| `eval/`  | `temporal-model-eval`  | `temporal_model.eval`  | DVC evaluation pipeline (packaged-model protocol metrics). Depends on `core`. | implemented |
```

- [ ] **Step 3: Run lint across the eval package**

Run: `cd eval && uv run ruff check . && uv run ruff format --check . && cd ..`
Expected: no lint errors; formatting clean. If `ruff format --check` reports diffs, run `uv run ruff format .` and re-stage.

- [ ] **Step 4: Run the full eval test suite once more**

Run: `cd eval && uv run pytest -q; cd ..`
Expected: 22 passed.

- [ ] **Step 5: Confirm nothing else broke (core still green, since eval imports it)**

Run: `cd core && uv run pytest -q; cd ..`
Expected: 163 passed (unchanged — eval only consumes core, no core edits).

- [ ] **Step 6: Commit**

```bash
git add eval/README.md README.md
git commit -m "docs(eval): document packaged-eval pipeline; mark eval implemented"
```

---

## Self-Review notes

- **Spec coverage:** modules (Tasks 2–4), tests (Tasks 2–4), dvc.yaml train+val + model.zip-as-external-input + params removal (Task 6), pyproject deps incl. pyyaml removal (Task 1), `__init__` cleanup + test_smoke removal (Task 5), CI unchanged + READMEs (Task 7). Non-goals (checkpoint eval, analyze/compare, multi-variant, leaderboard registry, FiftyOne) are not touched.
- **Import-rewrite consistency:** `pyrocore` → `temporal_model.core.protocol`; `bbox_tube_temporal.data`/`.model` → `temporal_model.core.data`/`.model`; `bbox_tube_temporal_exp.{eval_plots,protocol_eval}` → `temporal_model.eval.{eval_plots,protocol_eval}`. Applied identically in modules and their tests.
- **Test count:** 12 (protocol_eval) + 7 (eval_plots) + 3 (driver) = 22.
- **Symbols verified present in core:** `core.data.{list_sequences,is_wf_sequence,get_sorted_frames}`, `core.model.BboxTubeTemporalModel.{from_archive,predict}`, `core.protocol.{Frame,TemporalModelOutput,TemporalModel.load_sequence}`.
