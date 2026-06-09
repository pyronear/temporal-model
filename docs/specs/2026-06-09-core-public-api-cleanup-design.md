# core/ cleanup: boundaries, dedup, and targeted reorg

**Date:** 2026-06-09
**Status:** Approved (design)
**Scope:** `core/` + all four consumers (`api`, `eval`, `train`, `benchmark`)

## Motivation

`temporal_model.core` is the shared library every other package depends on. Its
docstrings, pure-function isolation, and parity discipline are strong, but its
**boundaries and internal consistency** have drifted:

- The public surface is fiction: `__init__.__all__` lists 5 names while consumers
  import ~25 from submodules — and `train/package.py` imports the *private*
  `_load_yolo`. Nothing protects internal structure from becoming a de-facto API.
- Parallel contracts can silently diverge: two timestamp parsers (different
  regexes), two classes named `TubeEntry`, and the calibrator's "tube dict" shape
  hand-built in three places.
- Two modules mix unrelated responsibilities (`data.py`, `model_input.py`).
- Dead code: `types.SequenceFeatures` is referenced nowhere.

This work tightens boundaries, kills the duplication, and does a *targeted* reorg
of the two muddled modules — without repackaging everything into subpackages and
without becoming a standalone published library.

## Constraints

- **Behavior-preserving by default; fix latent bugs deliberately.** Consolidations
  keep current outputs identical. Where a duplicate is a genuine latent bug, fix it
  to the correct behavior and add a test pinning the new contract.
- **Parity is the guardrail.** `test_model_parity` and `test_stabilize_crop_parity`
  must stay green; train/inference bit-for-bit parity is the property the whole
  feature relies on.
- **Cheap import preserved.** `import temporal_model.core` must not eagerly pull
  torch/timm/ultralytics.

## Chosen approach: curated submodules (Option 1)

Submodule paths remain the stable public API. Only the two genuinely-muddled files
are split. Every module declares an explicit `__all__`; true internals get a `_`
prefix. (Options considered and rejected: a single flat facade — high churn + eager
heavy imports; grouped subpackages — largest diff, deferred until core grows.)

## Design

### 1. Public API surface

- Every module declares an explicit `__all__`. True internals get a `_` prefix.
- **Top-level `__init__.py` re-exports only the light common names:** `Frame`,
  `TemporalModel`, `TemporalModelOutput`, `Detection`, `FrameDetections`, `Tube`,
  `TubeEntry`, `build_tubes`, `merge_colocated_tubes`.
- **`BboxTubeTemporalModel` stays out of the top-level re-export** so
  `import temporal_model.core` does not pull torch/timm. Consumers keep importing
  it from `core.model` (as they already do).
- Stable public submodule paths: `core.model`, `core.tubes`, `core.types`,
  `core.protocol`, `core.package`, `core.temporal_classifier`, `core.inference`,
  `core.logistic_calibrator`, `core.details_schema`, `core.detector`,
  `core.stabilize`, `core.stage_timer`, plus new `core.sequences`, `core.labels`,
  `core.crop`.

### 2. Boundary fix

- `package._load_yolo` → public **`load_yolo`**. Update `train/package.py` (import
  + call site) and the 8 `@patch("temporal_model.core.package._load_yolo")` targets
  in `core/tests/test_package.py`.

### 3. Dedup / drift kills (preserve current outputs)

- **Timestamp parser (the one deliberate latent-bug fix):** remove
  `data.parse_timestamp`; promote `protocol._try_parse_timestamp` to public
  `protocol.parse_timestamp`, keeping the **anchored `$`** regex (the stricter,
  correct one). `labels.py` and all callers use that single function. The unanchored
  variant could match a wrong timestamp-like substring mid-id; the anchored behavior
  is pinned by a new test. Current Pyronear filenames are unaffected (parity tests
  confirm).
- **Two `TubeEntry`:** rename the pydantic one in `details_schema.py` →
  **`KeptTubeEntry`** (it nests under `KeptTube`). `types.TubeEntry` keeps its name.
  `model.py` no longer imports two identically-named symbols.
- **Calibrator dict built in 3 places:** add
  **`tube_feature_dict(tube: Tube, logit: float) -> dict`** in
  `logistic_calibrator.py`. `model._probability_for` and `inference.py`'s logistic
  branch both call it. `extract_features(dict, n_tubes)` keeps its dict-accepting
  signature — the on-disk path in `train`/`scripts` is untouched. Removes 2 of the 3
  hand-built copies.
- **Decision defaults:** `aggregation="max_logit"` and `logistic_threshold=0.5` are
  duplicated between `model.py`'s properties and the `predict()` body. Define each
  default **once** (module-level constants) and reference from both.

### 4. Targeted module reorg

- `data.py` → **`sequences.py`** (`list_sequences`, `find_sequence_dir`,
  `get_sorted_frames`, `is_wf_sequence`) + **`labels.py`** (`load_detections`,
  `load_frame_detections`, `load_tube_record`).
- `model_input.py` → **`crop.py`** (pure geometry: `expand_bbox`,
  `norm_bbox_to_pixel_square`, `crop_and_resize`).
- **`process_tube`, `save_patch`, `LABEL_TO_INT` move out to `train/`**
  (training-data-prep; `train/build_model_input.py` is their only consumer). Keeps
  core to runtime building blocks.

### 5. Dead code

- Delete `types.SequenceFeatures` (zero references).

### 6. Consumer migration (all 4 packages)

Rewrite the affected imports:

- `_load_yolo` → `load_yolo` (`train/package.py`).
- `core.data` → `core.sequences` / `core.labels` (`eval/evaluate.py`,
  `train/build_tubes.py`, `train/package_predict.py`, and any other caller).
- `core.model_input` → `core.crop`; the `process_tube`/`save_patch`/`LABEL_TO_INT`
  move lands in `train` (`train/build_model_input.py`).

Estimated ~4–6 import lines plus test patch targets.

### 7. Verification (success criteria)

- `make test` green in **every** package (`core`, `api`, `eval`, `train`,
  `benchmark`). Parity tests are the guardrail that outputs did not move.
- `ruff check` clean across all packages.
- New tests: (a) `parse_timestamp` anchored-contract test; (b) `tube_feature_dict`
  output-shape test asserting parity with the previous inline dict.
- `core/README.md` module list updated to the new layout.

## Out of scope

- Single flat facade or grouped-subpackage repackaging.
- Treating core as a standalone/versioned published library.
- Typed (pydantic) schema for the full package config — only the decision defaults
  are de-duplicated here.
- Any change to model behavior, weights, or the on-disk package/tube/record formats.
