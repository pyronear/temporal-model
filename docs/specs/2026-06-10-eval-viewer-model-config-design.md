# Eval viewer — model-config panel

**Date:** 2026-06-10
**Status:** Approved design, pending implementation plan.

## Goal

Add a model-config panel to the eval Streamlit viewer (bottom of the left
sidebar) so a reviewer can see, at a glance, which model produced the predictions
they're looking at: detector origin, decision thresholds, padding, stabilization,
etc. A headline set is always visible; the full config is available in an expander.

## Background

The viewer (`eval/src/temporal_model/eval/app.py`) is **read-only over the
reporting tree** — it reads `results.json` / `details/` / `sequences/` per source
and never opens `model.zip`. The packaged model carries everything we want to show:

- `manifest.yaml` — `detector.source` (e.g. `hf:pyronear/yolo11s_nimble-narwhal_v6.0.0`),
  `detector.type`, `variant`, `train_git_sha`.
- `config.yaml` — `decision` (`aggregation`, `threshold`, `logistic_threshold`,
  `target_recall`, `trigger_rule`), `infer` (`confidence_threshold`, `image_size`,
  `iou_nms`, `pad_strategy`, `pad_to_min_frames`), `model_input` (`stabilize`,
  `context_factor`, `patch_size`, `normalization`), `tubes`, `classifier`.
- `logistic_calibrator.json` — `features`, `coefficients`, `intercept`.

`core`'s `ModelPackage` exposes `config` (the parsed `config.yaml`) but **not**
`manifest.yaml` (detector source / variant / train_git_sha).

## Key decisions

- **Emit during eval, not load-in-app.** `evaluate.py` (which already loads the
  model) writes a `model_config.json` into each source's reporting dir; the app
  reads it. Keeps the app read-only, makes the config travel per-source with the
  run, and fits the frontend-agnostic data contract.
- **Headline + expandable full.** A curated headline set is always visible; an
  `st.expander` holds the complete merged config for when more is needed.
- **No `core` change.** A small eval helper reads the three files from the
  `model.zip` directly. The package's internal filenames (`manifest.yaml`,
  `config.yaml`, `logistic_calibrator.json`) are stable; reading them in eval
  avoids a `core` API change for a viewer-only feature.

## Non-goals

- No editing of config from the UI (read-only display).
- No `core` `ModelPackage` change to expose the manifest.
- No multi-model comparison (one config per source, matching eval's single-variant
  scope).

## Architecture

### Data flow

```
evaluate.py ── read_model_config(model.zip) ──▶ write model_config.json
   (manifest.yaml + config.yaml + logistic_calibrator.json merged)
                                                        │
   data/08_reporting/<source>/vit_dinov2_finetune/model_config.json
                                                        │
                                    app.py: load_model_config(source)
                                                        │
                                    render_model_config(cfg) → st.sidebar (bottom)
```

### Components

**1. `eval/src/temporal_model/eval/model_config.py` (new) — pure, testable.**

```python
def read_model_config(model_zip: Path) -> dict:
    """Merge manifest.yaml + config.yaml + logistic_calibrator.json from a
    packaged model.zip into one plain dict (JSON-serializable)."""
```

Returns a dict shaped:

```json
{
  "detector": {"source": "...", "type": "yolo"},
  "variant": "vit_dinov2_finetune",
  "train_git_sha": "4b4d43a...",
  "decision": { ... },        // from config.yaml
  "infer": { ... },
  "model_input": { ... },
  "tubes": { ... },
  "classifier": { ... },
  "calibrator": { "features": [...], "coefficients": [...], "intercept": ... }
}
```

Reads the small text members via `zipfile` + `yaml.safe_load` / `json.loads`.
Adds `pyyaml` to eval's dependencies.

**2. `evaluate.py` — emit once per run.** After `BboxTubeTemporalModel.from_archive`,
write the config:

```python
(args.output_dir / "model_config.json").write_text(
    json.dumps(read_model_config(args.model_zip), indent=2, default=str)
)
```

Applies to both the dir-convention and `--store` paths (the line runs once in
`main()`, not per sequence).

**3. `app.py` — load + render.**

```python
def load_model_config(source: str) -> dict:
    path = REPORTING / source / MODEL_NAME / "model_config.json"
    return json.loads(path.read_text()) if path.exists() else {}
```

`render_model_config(cfg)` (called at the bottom of the sidebar in `main()`,
after the source selectbox) renders, when `cfg` is non-empty:

- **Headline** (st.sidebar markdown / small metrics):
  - detector source (`cfg["detector"]["source"]`)
  - variant
  - train_git_sha (first 8 chars)
  - aggregation (`decision.aggregation`)
  - decision threshold (`decision.threshold`)
  - logistic_threshold (`decision.logistic_threshold`)
  - stabilize (`model_input.stabilize`)
  - context_factor (`model_input.context_factor`)
  - max_frames (`classifier.max_frames`)
  - pad (`infer.pad_strategy` / `infer.pad_to_min_frames`)
- **`st.expander("full config")`** → `st.json(cfg)`.

When `cfg` is empty (older runs without the file), render a muted caption
"model config unavailable" — no crash. `render_model_config` is `# pragma: no cover`.

**4. `eval/dvc.yaml`** — add `model_config.json` (`cache: false`) to the `outs` of
both the `evaluate` foreach stage and the `evaluate_pyro_annotator` stage.

**5. `eval/README.md`** — add `model_config.json` to the data-contract list with a
one-line description (merged manifest + config + calibrator for the scored model).

## Error handling

- `read_model_config`: **tolerant**. A missing or unreadable `model_zip` returns
  `{}`; a missing member inside the zip (e.g. no `logistic_calibrator.json` for an
  uncalibrated package) yields that key as `None`/omitted rather than raising. This
  matters for the existing driver tests, which monkeypatch `from_archive` and pass a
  non-existent `placeholder.zip` — the emit must still succeed (writing `{}`). In
  production the zip is always valid (the model was just loaded from it), so `{}` only
  occurs in tests.
- App: missing or empty `model_config.json` → muted caption, no crash.

## Testing

- **`read_model_config`** (unit): build a tiny in-memory zip with a `manifest.yaml`,
  `config.yaml`, and `logistic_calibrator.json` and assert the merged dict resolves
  the headline fields (`detector.source`, `variant`, `decision.threshold`,
  `infer.pad_to_min_frames`, `model_input.stabilize`, `classifier.max_frames`).
  Also assert an absent `logistic_calibrator.json` yields `calibrator = None`
  without raising.
- **`evaluate.py`** (driver): extend an existing driver test to assert
  `model_config.json` is written. The `_FakeModel` path passes a non-existent
  `placeholder.zip`, so `read_model_config` returns `{}` and the file is written as
  `{}` — the test asserts the file exists. (The real merge logic is covered by the
  `read_model_config` unit test above.) The 5 existing driver tests must still pass
  unchanged.
- Streamlit render: `# pragma: no cover`.

## Open implementation details (for the plan)

- Whether the driver test builds a fixture zip or reuses the real `model.zip`
  (prefer a tiny synthetic zip to keep the test hermetic).
- Exact headline formatting (markdown table vs `st.metric` stack) — cosmetic.
