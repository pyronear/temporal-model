# Eval Qualitative Viewer (React / Next.js)

A local, read-only web viewer over the temporal-model **eval reporting tree** — the
React/Next.js + Tailwind port of the Streamlit viewer (`eval/src/temporal_model/eval/app.py`).
Same data, polished UI: a master–detail layout, light palette, and client-side rendering of
frames, bounding boxes, and stabilized tube crops.

It reads only the artifacts eval emits (see
`docs/specs/2026-06-10-eval-viewer-nextjs-redesign-design.md`): `results.json`,
`details/<key>.json`, `sequences/<key>.json`, `model_config.json`, plus the frame images — it
never runs the model.

## Prerequisites

- Node 22+, npm.
- A populated eval reporting tree (`<DATA_ROOT>/data/08_reporting/...`) and the frame images it
  points at. Generate with `uv run dvc repro` in `eval/`, or `dvc pull`.

## Run

```bash
cd viewer
cp .env.local.example .env.local   # set DATA_ROOT to your eval/ dir (default ../eval)
npm install
npm run dev                        # http://localhost:3000
```

`DATA_ROOT` points at the `eval/` package directory; the app derives the reporting tree
(`$DATA_ROOT/data/08_reporting`) and resolves frame paths (relative to `$DATA_ROOT`). The frame
route refuses any path that escapes `DATA_ROOT`.

## What it shows

- **Left rail:** source selector, performance cards, the live **logistic-threshold slider**
  (+ reset), and the model-config panel (hover a field for its description).
- **Center:** an error-coloured, keyboard-navigable (↑/↓) sequence table.
- **Right:** the selected sequence's autoplaying frame viewer with bbox overlay, the per-tube
  timeline, and the stabilized tube crops. The threshold slider re-decides the cards + table
  live; the drill-down shows the model's actual run.

## Develop

```bash
npm test            # vitest (pure-logic parity + component tests)
npm run lint        # eslint
npm run build       # production build
```

The pure helpers in `lib/` (`outcomes`, `details`, `crop`, `correctness`) mirror the Python
viewer and are unit-tested against the same cases — the Python is the source of truth if a
number ever disagrees.

## Scope

Feature parity with the Streamlit viewer. The table filter popover + column sort are a
documented fast-follow (they layer onto the row list without touching data or detail code).
