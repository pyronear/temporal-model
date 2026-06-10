# Eval Viewer Logistic-Threshold Slider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A slider below the performance cards that re-decides keep/discard at a chosen logistic threshold and updates the cards + table live.

**Architecture:** A pure `apply_threshold(df, thr)` helper recomputes `decision`/`outcome` from the existing `probability` column; `app.py` reads the slider value, applies it to the source's rows before rendering the cards + table, and keeps the drill-down on the model's original decision. App-only; no eval/core changes.

**Tech Stack:** pandas (vectorised re-decide), Streamlit (slider + reset), pytest.

All commands run from `eval/`.

---

## File Structure

- Modify: `eval/src/temporal_model/eval/outcomes.py` — add pure `apply_threshold`.
- Modify: `eval/src/temporal_model/eval/app.py` — slider wiring in `main()`.
- Modify: `eval/tests/test_outcomes.py` — unit test for `apply_threshold`.

---

## Task 1: `apply_threshold` helper

**Files:**
- Modify: `eval/src/temporal_model/eval/outcomes.py`
- Test: `eval/tests/test_outcomes.py`

- [ ] **Step 1: Write the failing test**

In `eval/tests/test_outcomes.py`, add `apply_threshold` to the existing import line
(`from temporal_model.eval.outcomes import (...)`) and append:

```python
def test_apply_threshold_redecides_and_recomputes_outcome():
    df = pd.DataFrame(
        {
            "label": ["smoke", "smoke", "fp", "fp", "smoke"],
            "probability": [0.9, 0.2, 0.8, 0.1, None],
            "decision": ["keep", "keep", "keep", "discard", "keep"],
            "outcome": ["kept-smoke", "kept-smoke", "kept-fp", "discarded-fp", "kept-smoke"],
            "score": [5.0, 1.0, 4.0, 0.5, 2.0],
        }
    )
    out = apply_threshold(df, 0.5)
    assert list(out["decision"]) == ["keep", "discard", "keep", "discard", "discard"]
    assert list(out["outcome"]) == [
        "kept-smoke",
        "discarded-smoke",
        "kept-fp",
        "discarded-fp",
        "discarded-smoke",  # probability None -> discard
    ]
    # raising the threshold flips the 0.8 fp from kept-fp to discarded-fp
    out2 = apply_threshold(df, 0.85)
    assert out2.loc[2, "decision"] == "discard"
    assert out2.loc[2, "outcome"] == "discarded-fp"
    # untouched columns preserved; input not mutated
    assert list(out["score"]) == [5.0, 1.0, 4.0, 0.5, 2.0]
    assert list(df["decision"]) == ["keep", "keep", "keep", "discard", "keep"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_outcomes.py::test_apply_threshold_redecides_and_recomputes_outcome -v`
Expected: FAIL — `ImportError: cannot import name 'apply_threshold'`.

- [ ] **Step 3: Implement the helper**

In `eval/src/temporal_model/eval/outcomes.py`, append:

```python
def apply_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Re-decide keep/discard at ``threshold`` from the ``probability`` column.

    Returns a copy: ``decision`` = keep iff ``probability`` is non-null and
    ``>= threshold`` (a sequence with no kept tubes has null probability ->
    discard), and ``outcome`` recomputed via :func:`compute_outcome`. All other
    columns (``score``, ``probability``, ``label``, metadata) are unchanged.
    """
    out = df.copy()
    keep = out["probability"].notna() & (out["probability"] >= threshold)
    out["decision"] = keep.map({True: "keep", False: "discard"})
    out["outcome"] = [
        compute_outcome(d, lbl)
        for d, lbl in zip(out["decision"], out["label"], strict=True)
    ]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_outcomes.py -v`
Expected: PASS (existing outcome tests + the new one).

- [ ] **Step 5: Commit**

```bash
git add eval/src/temporal_model/eval/outcomes.py eval/tests/test_outcomes.py
git commit -m "feat(eval): add apply_threshold to re-decide at a logistic threshold"
```

---

## Task 2: Slider wiring in the viewer

**Files:**
- Modify: `eval/src/temporal_model/eval/app.py`
- Test: `eval/tests/test_render.py` (existing import-smoke covers it)

- [ ] **Step 1: Import the helper**

In `eval/src/temporal_model/eval/app.py`, change the outcomes import:

```python
from temporal_model.eval.outcomes import apply_threshold, performance_summary
```

- [ ] **Step 2: Capture the original rows + apply the threshold before the cards**

In `main()`, replace this block:

```python
    view = df[df["source"] == source].reset_index(drop=True)
    has_org = view["organization_name"].notna().any()
    has_cam = view["camera_name"].notna().any()

    render_performance(view)
```

with:

```python
    view = df[df["source"] == source].reset_index(drop=True)
    original = view  # pre-threshold rows: the drill-down shows the model's real run
    has_org = view["organization_name"].notna().any()
    has_cam = view["camera_name"].notna().any()

    # Logistic-threshold explorer. Read the current slider value from session_state
    # BEFORE the cards/table (Streamlit widgets persist their value by key), so they
    # reflect it; the slider widget itself renders just below the cards. Only shown
    # for calibrated sources (some non-null probability).
    has_prob = bool(view["probability"].notna().any())
    default_thr = float(
        (load_model_config(source).get("decision") or {}).get(
            "logistic_threshold", 0.5
        )
    )
    thr_key = f"thr_{source}"
    thr = float(st.session_state.get(thr_key, default_thr)) if has_prob else default_thr
    if has_prob:
        view = apply_threshold(view, thr)

    render_performance(view)

    if has_prob:
        scol, rcol = st.columns([5, 1], vertical_alignment="bottom")
        scol.slider(
            "logistic threshold",
            0.0,
            1.0,
            value=default_thr,
            step=0.01,
            key=thr_key,
            help="Re-decides keep/discard live (cards + table). "
            "Drill-down shows the model's actual run.",
        )
        if rcol.button("↺ reset", help=f"model default: {default_thr:.3f}"):
            st.session_state.pop(thr_key, None)
            st.rerun()
        st.caption(f"model default logistic threshold: {default_thr:.3f}")
```

- [ ] **Step 3: Point the drill-down at the original (pre-threshold) row**

In `main()`, change the final drill-down call:

```python
    _drilldown(source, selected, view[view["key"] == selected].iloc[0])
```

to:

```python
    _drilldown(source, selected, original[original["key"] == selected].iloc[0])
```

(`selected` is chosen from the filtered `view`, which shares all keys with `original`
since `apply_threshold` and the filters never drop the selected row before this point;
`original` carries the model's real `decision`/`trigger_frame_index`.)

- [ ] **Step 4: Verify import + lint + format**

Run: `uv run pytest tests/test_render.py::test_app_module_imports -v`
Expected: PASS.
Run: `uv run ruff check src/temporal_model/eval/app.py`
Expected: clean.
Run: `uv run ruff format --check src/temporal_model/eval/app.py`
Expected: already formatted (run `uv run ruff format src/temporal_model/eval/app.py` if not).

- [ ] **Step 5: Commit**

```bash
git add eval/src/temporal_model/eval/app.py
git commit -m "feat(eval): interactive logistic-threshold slider (cards + table)"
```

---

## Task 3: Final verification

- [ ] **Step 1: Full eval suite + lint + format**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all PASS, lint/format clean.

- [ ] **Step 2: Manual viewer smoke (if a reporting tree + server are available)**

Launch `make app`, pick the `pyro-annotator` source, and confirm:
- the slider appears below the three performance cards, defaulting to ~0.474;
- dragging it lower increases kept sequences (recall up, precision down) and the table
  rows recolour live (more false alarms / fewer missed smoke);
- "↺ reset" returns the slider to the default and the cards/table snap back to the
  model's numbers;
- opening a sequence whose verdict flipped shows the model's original decision +
  trigger frame in the drill-down (unchanged by the slider).

(Skip if no data/server on this machine — the `apply_threshold` unit test + import-smoke
guard the logic.)
