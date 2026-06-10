# Eval viewer — React/Next.js + Tailwind redesign

**Date:** 2026-06-10
**Status:** Approved design, pending implementation plan.
**Stacked on:** PR #39 (`arthur/feat-eval-qualitative-viewer`) — this is the Phase-2
React rewrite anticipated by that PR's frontend-agnostic data contract.

## Goal

Replace the Streamlit eval viewer with a polished local Next.js + React + Tailwind
app, at **feature parity** with the Streamlit version but a cleaner, more beautiful
UI: a master–detail two-pane layout, a refined light colour scheme, and crisp
client-side rendering of frames / bboxes / stabilized tube crops.

It is a **local developer tool** (like `make app` today): run with `npm run dev`
against the local eval reporting tree. No hosting, no auth, no DB, no new data.

## Background

PR #39 established a frontend-agnostic data contract under
`eval/data/08_reporting/<source>/vit_dinov2_finetune/`:

- `results.json` — one row per sequence: `key, source, label, decision, outcome,
  score, probability, trigger_frame_index, organization_name, camera_name,
  started_at`.
- `details/<key>.json` — `BboxTubeDetails`: `preprocessing.padded_frame_indices`,
  `tubes.kept[]` (`tube_id, logit, probability, first_crossing_frame,
  stabilized_window, entries[]{frame_idx, bbox, confidence, is_gap}`),
  `decision{aggregation, threshold, logistic_threshold, trigger_tube_id}`.
- `sequences/<key>.json` — `SequenceView`: `key, source, label,
  organization_name, camera_name, started_at, frames[]` (frame paths relative to
  the `eval/` dir).
- `model_config.json` — detector/variant/train_git_sha + decision/infer/
  model_input/tubes/classifier + calibrator.

The Streamlit app (`eval/src/temporal_model/eval/app.py`) reads these directly and
does image work in Python (PIL bbox overlay, `core.crop` stabilized crops). The
React app consumes the *same files* and re-implements only the presentation +
small pure helpers in TypeScript. The Streamlit app stays as-is (this is additive).

## Key decisions (from brainstorming)

- **Local dev tool**, Next.js App Router + TypeScript + Tailwind, `npm run dev`.
- **Feature parity + polished redesign** (not an MVP subset, not a UX rethink).
- **Master–detail two-pane** layout (left control rail · center table · sticky
  right detail), with keyboard ↑/↓ row navigation.
- **Light theme**, palette A (soft 50-tint row + saturated accent dot + bold label).
- **Client-side image rendering**: API serves raw jpgs; bboxes as SVG overlay;
  stabilized crops via CSS. No server-side image library.
- **Stacked PR** off `arthur/feat-eval-qualitative-viewer`.
- **Location:** new top-level `viewer/` (keeps the JS toolchain separate from the
  Python packages).

## Non-goals

- No hosting / auth / multi-user / object storage. Local files only.
- No change to the Streamlit app, the eval pipeline, `core`, or the data contract.
- No re-running the model; the app only reads emitted artifacts (same as Streamlit).
- No static export (frames are too large to bundle; served on demand).

## Architecture

### Layout

```
┌────────────┬───────────────────────────┬──────────────────────────┐
│ control    │  sequence table           │  detail (sticky)         │
│ rail       │  (filter · sort · ↑/↓)    │  decision header         │
│ source ▾   │  ── error-coloured rows ──│  frame viewer + bbox SVG │
│ perf cards │  day·camera·GT·verdict·   │  tube timeline (SVG)     │
│ threshold  │  correctness·score·prob   │  stabilized tube crops   │
│ + reset    │                           │  (synced to frame)       │
│ model cfg  │                           │                          │
└────────────┴───────────────────────────┴──────────────────────────┘
```

### Stack & dependencies (lean)

- Next.js (App Router) + TypeScript + Tailwind CSS.
- TanStack Table for the grid (sort/filter/keyboard selection).
- Custom SVG for the tube timeline (no charting lib).
- Vitest + React Testing Library for tests.
- No server-side image library (rendering is client-side).

### Data access — Next.js route handlers

`DATA_ROOT` env var → the `eval/` directory (default `../eval`). Derived:
`REPORTING = $DATA_ROOT/data/08_reporting`, frame paths joined onto `$DATA_ROOT`.

- `GET /api/sources` → source names (dirs under REPORTING that have `results.json`).
- `GET /api/results` → all sources' `results.json` concatenated (array of rows).
- `GET /api/sequence/[source]/[key]` → `{ details, view }` (the two JSONs).
- `GET /api/model-config/[source]` → `model_config.json` (or `{}`).
- `GET /api/frame?path=<relpath>` → streams the jpg. **Path validation:** resolve
  against `$DATA_ROOT`, reject anything escaping it (no `..` traversal).

All handlers read the local filesystem; responses cached per-path in the client.

### TypeScript types

Mirror the contract: `ResultRow`, `BboxTubeDetails` (`Preprocessing`, `KeptTube`,
`KeptTubeEntry`, `Decision`), `SequenceView`, `ModelConfig`. One `types.ts` module,
the single source of truth the API + components share.

### Ported pure logic (TS, unit-tested for parity with Python)

- `applyThreshold(rows, thr)` → re-decide `keep = probability != null && probability
  >= thr`, recompute `outcome` (mirrors `outcomes.apply_threshold`).
- `computeOutcome(decision, label)`, `correctnessLabel(outcome)`, correctness →
  colour token map (mirrors `outcomes`/`render`).
- `performanceSummary(rows)` → recall / specificity / precision (mirrors
  `performance_summary`).
- `processedToInputIndex(frameIdx, padded)`, `frameBboxesByInputIndex(details)`,
  `tubeInputBoxes(tube, padded)`, `triggeringTubeIds(details)` (mirror `render`).

### Components

- `ControlRail` (source select, `PerfCards`, `ThresholdSlider`, `ModelConfigPanel`).
- `SequenceTable` (TanStack Table; row colour from correctness token; keyboard nav;
  filter popover).
- `DetailPanel` → `FrameViewer` (autoplay; `<img>` from `/api/frame` + `BboxOverlay`
  SVG), `TubeTimeline` (SVG bars + trigger/current rules), `TubeCrops` (CSS crop of
  the stabilized window synced to the current frame), decision header.
- Threshold slider re-decides client-side; cards + table reflect it live; the detail
  panel shows the model's original run (mirrors the Streamlit behaviour).

### Image rendering (client-side)

- **Frame:** `<img src="/api/frame?path=...">`.
- **Bboxes:** an absolutely-positioned `<svg viewBox="0 0 1 1" preserveAspectRatio>`
  over the image; rects from normalized `(cx,cy,w,h)`; decisive/would-trigger styled
  per `triggeringTubeIds`/`trigger_tube_id`.
- **Stabilized crop:** a fixed-size box (`overflow:hidden`) containing the frame
  `<img>` scaled so the tube's `stabilized_window` (normalized, expanded by the
  display context factor) fills the box — pure CSS transform; falls back to the
  per-frame bbox when `stabilized_window` is null.

## Visual system (palette A, light)

Tailwind theme tokens for correctness (row tint / accent / text):

| outcome | row bg | accent dot | text |
|---|---|---|---|
| kept-smoke (smoke kept) | emerald-50 `#ecfdf5` | emerald-600 `#059669` | emerald-800 `#065f46` |
| discarded-fp (fp filtered) | teal-50 `#f0fdfa` | teal-600 `#0d9488` | teal-800 `#115e59` |
| kept-fp (false alarm) | amber-50 `#fffbeb` | amber-500 `#f59e0b` | amber-800 `#92400e` |
| discarded-smoke (missed smoke) | rose-50 `#fff1f2` | rose-600 `#e11d48` | rose-800 `#9f1239` |
| flagged · GT unknown | blue-50 `#eff6ff` | blue-500 `#3b82f6` | blue-800 `#1e40af` |
| discarded · GT unknown | slate-50 `#f8fafc` | slate-400 `#94a3b8` | slate-600 `#475569` |

Missed smoke (the worst error) gets the strongest hue. One neutral accent for chrome
(slate/indigo). Soft tints keep the table calm while errors still read instantly.

## Error handling

- Missing artifacts (no `results.json` for a source, missing `model_config.json`,
  missing `details`/`sequence`) → empty/graceful states, never a crash (mirrors the
  Streamlit defensive reads).
- `/api/frame` rejects paths outside `DATA_ROOT` (400) and missing files (404).
- Threshold slider shown only when the source's model aggregation is `logistic` and
  probabilities are present (mirrors the Streamlit gate).
- `probability == null` → discard (no kept tubes).

## Testing

- **Unit (Vitest):** the ported pure helpers — `applyThreshold`, `computeOutcome`,
  `performanceSummary`, `processedToInputIndex`, `triggeringTubeIds` — asserted
  against the same cases the Python tests use (parity).
- **Component (RTL):** `SequenceTable` colours/sorts/filters a fixture; `BboxOverlay`
  positions a known bbox; `ThresholdSlider` re-decides + resets; `DetailPanel`
  renders a fixture sequence.
- **Optional e2e (Playwright):** load against a small fixture reporting tree, select a
  row, drag the slider — deferred unless cheap.
- No Python tests change.

## Branching / PR strategy

- New branch `arthur/feat-eval-viewer-nextjs` created **from
  `arthur/feat-eval-qualitative-viewer`** (not `main`), in its own worktree.
- PR base = `arthur/feat-eval-qualitative-viewer` → GitHub shows only the redesign
  diff, stacked on #39. When #39 merges, GitHub auto-retargets the base to `main`.
- The redesign depends on #39's data contract, so stacking is correct.

## Open implementation details (for the plan)

- Exact `package.json` scripts / Node version pin (Node 22 available locally).
- Whether selection/threshold/source state lives in the URL (shareable) or React
  state — lean React state; URL sync is a nice-to-have.
- TanStack Table vs a hand-rolled table if the dep feels heavy for the column set.
