# Trigger search is eval-only: gate it behind `compute_trigger`

Date: 2026-06-09
Status: Approved (brainstorm)
Refs: issue #23, PR #22 (benchmark), `docs/specs/2026-06-09-benchmark-package-design.md`

## Problem

The benchmark (issue #23) showed the `trigger_search` stage —
`find_first_crossing_trigger` in `core/inference.py` — is **34% of CPU
`predict()` latency** (~5.0 s of 14.8 s mean), the biggest lever after the
detector. It re-scores growing tube prefixes `L = min_prefix_length ..
len(tube.entries)`, running the classifier once per prefix, serially.

Issue #23 proposed *batching* those prefix forwards (correctness-preserving).
But investigation shows a bigger lever: **the entire stage produces only
time-to-detection data, which production never consumes.**

## Key finding: what trigger_search actually produces

The stage computes three outputs. The binary alarm decision is **not** one of
them:

| Output | Consumer | Affects the alarm? |
|---|---|---|
| `is_positive` (`is_smoke`) | the alarm | **No — already free.** Bit-for-bit `any(full_logit >= threshold)` (or the logistic equivalent). The prefix loop does not change it. |
| `trigger_frame_index` + per-tube `first_crossing_frame` | API response / eval | No — retrospective time-to-detection (TTD) metric |
| `trigger_tube_id` | `_decision_probability` picks which tube's prob is the top-level `probability` when smoke | No (only which prob is reported) |

`is_positive` is already determined by the full-length per-tube logits/probs
that `predict()` computes in the `classifier` stage. The 5 s buys only **"at
which frame *would* we have fired"** (TTD) plus a tube-selection key for
probability reporting.

### Decisions confirmed during brainstorming

1. **Prod does not act on `trigger_frame_index`** — it is an eval/benchmark
   metric. (Confirmed by user.)
2. **Eval consumes trigger_search for exactly one metric: TTD**
   (`mean_ttd` / `median_ttd` in `eval/protocol_eval.py`, fed from
   `trigger_frame_index`). All classification metrics (precision, recall, F1,
   FPR, PR/ROC-AUC) run off `is_positive` and `score` (full-length tube
   logits) — never the prefix search. `trigger_tube_id` is passed through to
   `predictions.json` but no metric reads it; per-tube `first_crossing_frame`
   is not read at all.
3. **TTD is worth keeping** for the eval leaderboard, so the search stays —
   but only behind an explicit flag.
4. **`trigger_tube_id` is dropped from the prod path.** The top-level
   `probability` becomes `max(kept-tube prob)` unconditionally (accepted
   prod-facing change).
5. **Do not batch** the prefix loop now (issue #23's proposal). Unnecessary
   once prod skips the stage; revisit only if eval runs get slow.

## Design

### 1. `predict()` gains `compute_trigger: bool = False`

`core/model.py::BboxTubeTemporalModel.predict(self, frames, *, timer=None,
compute_trigger=False)`.

- **`compute_trigger=False` (prod default):** skip
  `find_first_crossing_trigger`. Derive `is_positive` directly from the
  full-length per-tube logits/probs `predict()` already computes:
  - `max_logit`: `any(logit >= threshold)`
  - `logistic`: `any(prob >= logistic_threshold)`, where `prob` is the same
    `predict_proba(extract_features(tube_feature_dict(tube, logit)))` the
    method already computes per tube.

  Set `trigger_frame_index = None`, `trigger_tube_id = None`, and every kept
  tube's `first_crossing_frame = None`.
- **`compute_trigger=True` (eval):** call `find_first_crossing_trigger`
  exactly as today. Bit-identical output; the function itself is unchanged, so
  its parity/edge tests stand.

### 2. Share the decision predicate (avoid drift)

Extract the `decides_positive` closure currently built inside
`find_first_crossing_trigger` into a shared helper in `core/inference.py`
(e.g. `make_decision_fn(aggregation, *, threshold, calibrator,
logistic_threshold)` returning `decides_positive(logit, tube, n_tubes)`).
Both the prod inline decision and the eval search use it, so the two paths
cannot diverge on the rule.

### 3. Probability contract — `_decision_probability` (`api/schemas.py`)

Collapse to `max(kept-tube prob)` **unconditionally**:
- `None` if uncalibrated.
- otherwise `max(probs)` over kept tubes, `0.0` if no kept tubes.

Removes the `trigger_tube_id` branch. The one observable prod change: on a
positive, reported `probability` is the max tube's prob rather than the
trigger tube's. This is monotonic with the decision (a fire means some tube
cleared threshold, so the max is ≥ that) and matches the existing no-smoke
rule.

### 4. Call sites

| Call site | Change | Why |
|---|---|---|
| `api/model_runner.py` → `predict_sequence` | none (inherits `False`) | prod stays fast |
| `train/package_predict.py` → `predict_sequence` | none (inherits `False`) | reads only `details.tubes.kept`; **gets faster for free** |
| `eval/evaluate.py` → `model.predict(frames)` | pass `compute_trigger=True` | TTD preserved |
| `benchmark/run_core.py` → `model.predict(...)` | none (default `False`) | benchmark profiles the prod target; `trigger_search` correctly drops to ~0 — this *is* the resolution of #23 |

`predict_sequence` / the `TemporalModel` protocol are **not** threaded with
the flag (YAGNI: no caller needs trigger via `predict_sequence`; eval calls
`predict()` directly).

### 5. Stage timer

Keep the `stage_ctx(timer, "trigger_search")` wrapping the decision logic in
**both** modes (full-logit decision when off, full search when on). The stage
list stays intact and benchmarks show the stage dropping to ~0 rather than a
missing stage.

## Out of scope

- Batching the prefix forwards (issue #23 as written).
- Threading `compute_trigger` through `predict_sequence` / the protocol.
- Removing TTD from the eval leaderboard.

## Testing

`find_first_crossing_trigger`'s own parity/edge tests are unchanged — they
exercise the `compute_trigger=True` path and must stay green.

Update for the new default-path behavior (`trigger_*` = `None`, `probability`
= max):

- `core/tests/test_model_edge_cases.py`
- `core/tests/test_inference_units.py`
- `core/tests/test_details_schema.py`
- `api/tests/test_app.py`
- `api/tests/test_schemas.py`
- `eval/tests/test_evaluate_driver.py`
- `eval/tests/test_protocol_eval.py`

Add: a test asserting `predict(frames)` (default) returns
`trigger_frame_index is None` while `predict(frames, compute_trigger=True)`
returns the same trigger output as before for the same input (lock the gate).

## Acceptance

- `predict(frames)` (default) produces identical `is_positive` to today for
  all parity-test inputs, with `trigger_frame_index`/`trigger_tube_id`/
  per-tube `first_crossing_frame` all `None`.
- `predict(frames, compute_trigger=True)` returns the exact tuple
  `find_first_crossing_trigger` returns today.
- Top-level API `probability` = `max(kept-tube prob)` on both smoke and
  no-smoke.
- Eval TTD metrics (`mean_ttd`/`median_ttd`) are unchanged (eval passes
  `compute_trigger=True`).
- Re-running `temporal-benchmark core` (default path) shows `trigger_search`
  share drop to ~0.
