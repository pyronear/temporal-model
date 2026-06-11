# Eval Viewer Model-Config Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the scored model's config (detector origin, decision thresholds, padding, stabilization, …) at the bottom of the eval viewer's left sidebar.

**Architecture:** `evaluate.py` reads `manifest.yaml` + `config.yaml` + `logistic_calibrator.json` from the packaged `model.zip` and writes a merged `model_config.json` into each source's reporting dir; the Streamlit app reads it and renders a headline panel + a full-config expander. No `core` change; the app stays read-only over the reporting tree.

**Tech Stack:** Python 3.11+, zipfile + PyYAML (read the zip members), Streamlit (sidebar render), pytest, DVC.

---

## File Structure

- Create: `eval/src/temporal_model/eval/model_config.py` — `read_model_config(model_zip) -> dict` (pure).
- Modify: `eval/src/temporal_model/eval/evaluate.py` — emit `model_config.json` once per run.
- Modify: `eval/src/temporal_model/eval/app.py` — `load_model_config` + `render_model_config`, wired into the sidebar.
- Modify: `eval/pyproject.toml` — add `pyyaml` dependency.
- Modify: `eval/dvc.yaml` — declare `model_config.json` as an out of both stages.
- Modify: `eval/README.md` — add `model_config.json` to the data contract.
- Create: `eval/tests/test_model_config.py` — unit tests for `read_model_config`.
- Modify: `eval/tests/test_evaluate_driver.py` — assert `model_config.json` is written.

All commands run from `eval/` unless stated otherwise.

---

## Task 1: `read_model_config` helper

**Files:**
- Create: `eval/src/temporal_model/eval/model_config.py`
- Test: `eval/tests/test_model_config.py`

- [ ] **Step 1: Write the failing test**

Create `eval/tests/test_model_config.py`:

```python
import io
import json
import zipfile
from pathlib import Path

from temporal_model.eval.model_config import read_model_config

MANIFEST = """\
detector:
  source: hf:pyronear/yolo11s_nimble-narwhal_v6.0.0
  type: yolo
train_git_sha: 4b4d43ad77c401bab6d01d561b0aa2337f7ee031
variant: vit_dinov2_finetune
yolo_weights: yolo_weights.pt
"""

CONFIG = """\
decision:
  aggregation: logistic
  logistic_threshold: 0.4736
  threshold: 0.8699
  trigger_rule: end_of_winner
infer:
  pad_strategy: symmetric
  pad_to_min_frames: 6
model_input:
  context_factor: 1.5
  stabilize: true
  patch_size: 224
classifier:
  backbone: vit_small_patch14_dinov2.lvd142m
  max_frames: 20
tubes:
  min_tube_length: 4
"""

CALIBRATOR = '{"features": ["logit"], "coefficients": [0.5], "intercept": -5.1}'


def _make_zip(tmp_path: Path, *, with_calibrator: bool = True) -> Path:
    zp = tmp_path / "model.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("manifest.yaml", MANIFEST)
        z.writestr("config.yaml", CONFIG)
        if with_calibrator:
            z.writestr("logistic_calibrator.json", CALIBRATOR)
    return zp


def test_read_model_config_merges_members(tmp_path):
    cfg = read_model_config(_make_zip(tmp_path))
    assert cfg["detector"]["source"] == "hf:pyronear/yolo11s_nimble-narwhal_v6.0.0"
    assert cfg["variant"] == "vit_dinov2_finetune"
    assert cfg["train_git_sha"].startswith("4b4d43a")
    assert cfg["decision"]["aggregation"] == "logistic"
    assert cfg["decision"]["threshold"] == 0.8699
    assert cfg["decision"]["logistic_threshold"] == 0.4736
    assert cfg["infer"]["pad_strategy"] == "symmetric"
    assert cfg["infer"]["pad_to_min_frames"] == 6
    assert cfg["model_input"]["stabilize"] is True
    assert cfg["model_input"]["context_factor"] == 1.5
    assert cfg["classifier"]["max_frames"] == 20
    assert cfg["calibrator"]["features"] == ["logit"]


def test_read_model_config_missing_calibrator_is_none(tmp_path):
    cfg = read_model_config(_make_zip(tmp_path, with_calibrator=False))
    assert cfg["calibrator"] is None
    assert cfg["variant"] == "vit_dinov2_finetune"


def test_read_model_config_missing_zip_returns_empty(tmp_path):
    assert read_model_config(tmp_path / "does_not_exist.zip") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_config.py -v`
Expected: FAIL — `ModuleNotFoundError: temporal_model.eval.model_config`.

- [ ] **Step 3: Write the implementation**

Create `eval/src/temporal_model/eval/model_config.py`:

```python
"""Read a packaged model.zip's metadata into one plain dict for the viewer.

Merges manifest.yaml (detector provenance), config.yaml (decision/infer/
model_input/tubes/classifier), and logistic_calibrator.json. Tolerant: a missing
or unreadable zip returns {} (e.g. test runs that monkeypatch model loading); a
missing member is omitted or set to None.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import yaml


def _read_member(z: zipfile.ZipFile, name: str):
    """Parse a zip member by extension (yaml/json), or None if absent."""
    if name not in z.namelist():
        return None
    raw = z.read(name).decode()
    if name.endswith((".yaml", ".yml")):
        return yaml.safe_load(raw)
    return json.loads(raw)


def read_model_config(model_zip: Path) -> dict:
    """Merged, JSON-serializable view of a packaged model's config.

    Returns {} when the zip is missing/unreadable. Keys: detector, variant,
    train_git_sha (from manifest.yaml); decision, infer, model_input, tubes,
    classifier (from config.yaml); calibrator (logistic_calibrator.json or None).
    """
    model_zip = Path(model_zip)
    if not model_zip.exists():
        return {}
    try:
        with zipfile.ZipFile(model_zip) as z:
            manifest = _read_member(z, "manifest.yaml") or {}
            config = _read_member(z, "config.yaml") or {}
            calibrator = _read_member(z, "logistic_calibrator.json")
    except (zipfile.BadZipFile, OSError):
        return {}
    return {
        "detector": manifest.get("detector"),
        "variant": manifest.get("variant"),
        "train_git_sha": manifest.get("train_git_sha"),
        "decision": config.get("decision"),
        "infer": config.get("infer"),
        "model_input": config.get("model_input"),
        "tubes": config.get("tubes"),
        "classifier": config.get("classifier"),
        "calibrator": calibrator,
    }
```

- [ ] **Step 4: Add `pyyaml` to eval deps**

In `eval/pyproject.toml`, under `[project].dependencies`, add:

```python
    "pyyaml>=6",
```

Run: `uv sync`

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_model_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add eval/src/temporal_model/eval/model_config.py eval/tests/test_model_config.py \
        eval/pyproject.toml eval/uv.lock
git commit -m "feat(eval): read merged model config from model.zip"
```

---

## Task 2: Emit `model_config.json` from `evaluate.py`

**Files:**
- Modify: `eval/src/temporal_model/eval/evaluate.py`
- Test: `eval/tests/test_evaluate_driver.py`

- [ ] **Step 1: Add the assertion to an existing driver test**

In `eval/tests/test_evaluate_driver.py`, inside
`test_evaluate_packaged_writes_viewer_artifacts`, after the existing
`assert (output_dir / "results.parquet").is_file()` line, add:

```python
    # model_config.json is always emitted (here {} since the fake zip is absent)
    assert (output_dir / "model_config.json").is_file()
    import json as _json

    assert _json.loads((output_dir / "model_config.json").read_text()) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate_driver.py::test_evaluate_packaged_writes_viewer_artifacts -v`
Expected: FAIL — `model_config.json` does not exist.

- [ ] **Step 3: Implement the emit**

In `eval/src/temporal_model/eval/evaluate.py`:

(a) Add the import alongside the other eval imports (near `from temporal_model.eval.view_store import ...`):

```python
from temporal_model.eval.model_config import read_model_config
```

(b) In `main()`, immediately after the model is loaded
(`model = BboxTubeTemporalModel.from_archive(args.model_zip, device=args.device)`,
line ~109), add:

```python
    (args.output_dir / "model_config.json").write_text(
        json.dumps(read_model_config(args.model_zip), indent=2, default=str)
    )
```

(`json` and `args.output_dir.mkdir(...)` are already in scope above this point.)

- [ ] **Step 4: Run the full driver suite**

Run: `uv run pytest tests/test_evaluate_driver.py -v`
Expected: PASS — all 5 existing driver tests plus the new assertion (the fake
`placeholder.zip` is absent, so `read_model_config` returns `{}` and the file is
written as `{}`).

- [ ] **Step 5: Commit**

```bash
git add eval/src/temporal_model/eval/evaluate.py eval/tests/test_evaluate_driver.py
git commit -m "feat(eval): emit model_config.json per run"
```

---

## Task 3: Render the model-config panel in the viewer

**Files:**
- Modify: `eval/src/temporal_model/eval/app.py`
- Test: `eval/tests/test_render.py` (import-smoke already covers app import)

- [ ] **Step 1: Add the loader + renderer**

In `eval/src/temporal_model/eval/app.py`, add a loader next to the other
`load_*` helpers (after `load_sequence_view`, ~line 65):

```python
def load_model_config(source: str) -> dict:
    path = REPORTING / source / MODEL_NAME / "model_config.json"
    return json.loads(path.read_text()) if path.exists() else {}
```

Add the renderer near `render_performance` (it is Streamlit UI, so mark it
`# pragma: no cover`):

```python
def render_model_config(source: str) -> None:  # pragma: no cover - Streamlit UI
    """Sidebar panel: headline model fields + a full-config expander."""
    cfg = load_model_config(source)
    st.sidebar.divider()
    st.sidebar.caption("Model config")
    if not cfg:
        st.sidebar.caption("model config unavailable")
        return
    detector = (cfg.get("detector") or {}).get("source", "—")
    decision = cfg.get("decision") or {}
    model_input = cfg.get("model_input") or {}
    infer = cfg.get("infer") or {}
    classifier = cfg.get("classifier") or {}
    sha = (cfg.get("train_git_sha") or "")[:8] or "—"
    lines = [
        f"**detector** `{detector}`",
        f"**variant** {cfg.get('variant', '—')}",
        f"**train sha** `{sha}`",
        f"**aggregation** {decision.get('aggregation', '—')}",
        f"**threshold** {decision.get('threshold', '—')}",
        f"**logistic threshold** {decision.get('logistic_threshold', '—')}",
        f"**stabilize** {model_input.get('stabilize', '—')}",
        f"**context factor** {model_input.get('context_factor', '—')}",
        f"**max frames** {classifier.get('max_frames', '—')}",
        f"**pad** {infer.get('pad_strategy', '—')} / min "
        f"{infer.get('pad_to_min_frames', '—')}",
    ]
    st.sidebar.markdown("  \n".join(lines))
    with st.sidebar.expander("full config"):
        st.json(cfg)
```

- [ ] **Step 2: Wire it into the sidebar**

In `main()`, immediately after the source selectbox line
(`source = st.sidebar.selectbox("source", sources, key="source")`, ~line 304),
add:

```python
    render_model_config(source)
```

This places the panel at the bottom of the left pane (it is the last sidebar
content; the table/cards render in the main pane).

- [ ] **Step 3: Verify import + lint**

Run: `uv run pytest tests/test_render.py::test_app_module_imports -v`
Expected: PASS (app imports cleanly; `load_model_config`/`render_model_config`
defined).
Run: `uv run ruff check src/temporal_model/eval/app.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add eval/src/temporal_model/eval/app.py
git commit -m "feat(eval): show model config panel in the viewer sidebar"
```

---

## Task 4: Track `model_config.json` in DVC + document it

**Files:**
- Modify: `eval/dvc.yaml`
- Modify: `eval/README.md`

- [ ] **Step 1: Declare the new out in both stages**

In `eval/dvc.yaml`, in the `evaluate` foreach stage's `outs:` block, add (next to
the other `cache: false` reporting outs, e.g. after the `results.json` entry):

```yaml
        - data/08_reporting/${item}/vit_dinov2_finetune/model_config.json:
            cache: false
```

And in the `evaluate_pyro_annotator` stage's `outs:` block, add:

```yaml
      - data/08_reporting/pyro-annotator/vit_dinov2_finetune/model_config.json:
          cache: false
```

- [ ] **Step 2: Validate the DVC graph parses**

Run: `uv run dvc dag`
Expected: no YAML/parse error; the three stages still render. (Stages will show as
"changed" until re-run — expected.)

- [ ] **Step 3: Document the artifact**

In `eval/README.md`, in the "Data contract (frontend-agnostic)" bullet list, add:

```markdown
- `model_config.json` — the scored model's merged metadata (detector source +
  variant + train_git_sha from the package manifest; decision/infer/model_input/
  tubes/classifier config; logistic calibrator). Drives the viewer's sidebar
  model-config panel.
```

- [ ] **Step 4: Commit**

```bash
git add eval/dvc.yaml eval/README.md
git commit -m "chore(eval): track model_config.json + document it"
```

---

## Task 5: Final verification

- [ ] **Step 1: Full eval suite + lint + format**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all PASS, lint/format clean.

- [ ] **Step 2: Manual viewer smoke (if data available)**

If a reporting tree with a real run exists, regenerate one source's
`model_config.json` (`uv run python -m temporal_model.eval.evaluate ... --source val`
against a real `model.zip`), launch `make app`, and confirm the sidebar shows the
detector source, thresholds, stabilize, and pad, with a working "full config"
expander. (Skip if no model/data on this machine — the unit + import-smoke tests
guard the logic.)
