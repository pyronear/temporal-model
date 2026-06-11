# Eval viewer — interactive logistic-threshold slider

**Date:** 2026-06-10
**Status:** Approved design, pending implementation plan.

## Goal

Let a reviewer explore the keep/discard threshold live: a slider (below the model
performance cards) re-decides every sequence at the chosen logistic threshold and
updates the performance cards and the sequence table (decision, correctness, row
colour) as it moves — so they can pick the threshold that gives the recall/precision
trade-off they want before committing it in the model config.

## Background

The viewer is read-only over the reporting tree. Each `results.json` row already
carries `probability` = the max calibrated probability across the sequence's kept
tubes (`max_probability(details)`), which is exactly the quantity the model's
**logistic** decision thresholds on: a sequence is kept iff any kept tube's
probability ≥ `logistic_threshold`, i.e. `max-prob ≥ threshold`.

**Verified:** re-deciding `keep = (probability is not None and probability ≥ thr)`
reproduces the model's stored `decision` with **0 mismatches across all 650
sequences** (val + pyro-annotator) at the model's default `logistic_threshold`
(0.4736). `probability is None` (no kept tubes) → always discard, which also matches.

So the slider is a **pure app-side recompute** — no model re-run, no eval/core
changes. The per-source default threshold comes from `model_config.json`
(`decision.logistic_threshold`), which the viewer already loads.

## Key decisions

- **App-side recompute from `probability`.** At the default threshold it reproduces
  the model exactly; moving it explores other operating points.
- **Updates the performance cards + table only** (decision/correctness/row colour +
  the correctness filter). The **drill-down keeps the model's actual decision +
  trigger frame** — the trigger frame can't be recomputed app-side (it needs
  per-frame prefix probabilities we don't store). A caption notes this.
- **Default + reset.** Slider default = the source's `decision.logistic_threshold`
  (fallback 0.5). A "↺ reset" button returns it to that default. Range 0.0–1.0,
  step 0.01.
- **Gated on calibration.** The slider shows only when the source has ≥1 non-null
  `probability`. For uncalibrated sources (all null) it's hidden and stored
  decisions show as-is.

## Non-goals

- No re-running the model, no new emitted artifacts, no eval/core changes.
- Drill-down trigger frame is not recomputed for the slider threshold.
- The slider does not write the chosen threshold back to any config (it's an
  exploration tool; committing the value stays a manual model-config edit).

## Architecture

### Pure helper (`eval/.../outcomes.py`)

```python
def apply_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return a copy of df with decision/outcome re-decided at `threshold`.

    keep iff probability is not null and probability >= threshold; outcome via
    compute_outcome(decision, label). score/probability/label/metadata unchanged.
    """
```

Implementation: vectorised — `keep = df["probability"].notna() & (df["probability"] >= threshold)`,
`decision = where(keep, "keep", "discard")`, `outcome = [compute_outcome(d, l) for ...]`.
Pure pandas, no Streamlit, unit-tested.

### Wiring (`app.py main()`)

Order becomes: source select → `render_model_config` (sidebar) → load `view` for the
source → **render_performance + slider + table all over the re-decided view**.

1. `default_thr = (load_model_config(source).get("decision") or {}).get("logistic_threshold", 0.5)`.
2. `has_prob = view["probability"].notna().any()`.
3. If `has_prob`: render the slider + reset (main pane, just under the cards) and read
   `thr`; else `thr = default_thr` and no slider. Then `view = apply_threshold(view, thr)`.
   **Apply before** `render_performance(view)` and before building the table / filters /
   colours, so all of them reflect `thr`.
4. **Drill-down stays original**: select the drill-down row from the source's
   *pre-threshold* frame (`df[df.key == selected]`), not the re-decided `view`, so the
   panel shows the model's real verdict + trigger frame.

Reset pattern (Streamlit): the slider uses `key=f"thr_{source}"`. The reset button is
checked *before* the slider is instantiated; when clicked, delete the key from
`st.session_state` (so the slider re-initialises to `value=default_thr`) and
`st.rerun()`. The per-source key also resets the slider naturally when the source
changes.

Caption under the slider: "Re-decides keep/discard live (table + metrics). Drill-down
shows the model's actual run."

### Layout note

`render_performance` already renders cards; today it's called with `view`. With the
slider, it is called with `apply_threshold(view, thr)`. No change to
`render_performance` itself.

## Error handling

- `probability` null → discard (handled by `notna()` mask).
- Source with no probabilities → slider hidden, stored decisions shown.
- Threshold at default → reproduces stored decisions (verified).

## Testing

- **`apply_threshold`** (unit, `tests/test_outcomes.py`): a small df with mixed
  `(label, probability)` →
  - at a mid threshold, a high-prob smoke stays `kept-smoke`, a low-prob smoke becomes
    `discarded-smoke`, a high-prob fp becomes `kept-fp`;
  - `probability=None` → `discard` / outcome per label;
  - raising the threshold flips a borderline row from keep to discard (and its outcome
    updates);
  - `score`/`probability` columns are preserved.
- Streamlit slider/reset wiring stays `# pragma: no cover` (covered by the app
  import-smoke test).

## Open implementation details (for the plan)

- Whether the reset control is a `st.button` next to the slider or an inline column —
  cosmetic; a small button labelled "↺ reset".
- Exact caption wording.
