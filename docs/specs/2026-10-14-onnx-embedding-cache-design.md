# Split ONNX Graphs and Per-Frame Embedding Cache

- **Date:** 2026-10-14
- **Status:** Implemented
- **Builds on:** [`2026-09-09-onnx-export-design.md`](2026-09-09-onnx-export-design.md)

## Problem

On the engine box the temporal service re-scores a sliding window every
round: the last N frames of a camera pose, one frame newer than the round
before. With the format 1 package (`classifier.onnx`, backbone + head fused)
each call runs the ViT on `n_tubes × max_frames` patches: the padded slots
(a 10-frame window in a 20-slot graph is 50 % zeros) and the nine frames the
previous round already embedded are all recomputed. The backbone is ~all of
the cost (3.4 s per 20-slot tube on a Pi 5); the head is a 2-layer
transformer over 21 tokens, negligible.

## Decisions

### Two graphs instead of one (format 2)

`temporal-export-onnx` now exports `classifier.backbone` and
`classifier.head` separately:

- `backbone.onnx`: `patches[M, 3, P, P]` → `features[M, D]`
- `head.onnx`: `features[N, max_frames, D]` + `mask[N, max_frames]` → `logit[N]`

Both batch axes are dynamic (`M` = slots to embed, `N` = tubes). The manifest
carries `format_version: 2`, `backbone_onnx`, `head_onnx` and the I/O spec
of each graph. `load_onnx_package` still accepts format 1 packages (every
`model_onnx.zip` published up to v0.6.0), which run as before: no slot
skipping, no cache.

### Only real slots go through the backbone

`score_split` embeds `patches[mask]` and leaves masked slots as zero
features. The head masks those keys out of attention, so their content never
reaches the logit; `verify_export` measures this against the torch
reference (which still embeds every slot) over the usual four mask patterns,
`atol=1e-4`.

### Embedding cache keyed on the crop, not on the frame

A slot's embedding is fully determined by `(frame_id, pixel crop box)`. The
crop box is the stabilized window of the tube (union of its observed boxes),
so it moves whenever a new box extends the union or the frame holding an
extremum leaves the window; a cached embedding is only reused for the exact
crop it was cut from. The geometry lives in `inference.tube_crop_boxes`, the
single source of the crop policy for `crop_tube_patches` (training parity
tests) and the cached path. The image is only *opened* for its size at key
time; it is decoded at most once per call, and only on a miss.

`OnnxTemporalModel(embedding_cache_size=N)` keeps an LRU of N embeddings
(one entry is `D` float32, 1.5 KB for ViT-S); it is off by default, so the
evaluation and benchmark paths are unchanged. Frame ids must be unique per
image across calls (the engine prefixes them with the camera pose and
timestamp). The model is not thread-safe: score from one thread at a time.

### Where it plugs into the pipeline

`TubePipelineModel._crop_and_score` is the new backend hook covering the
`crop` and `classifier` stages. The default implementation is the former
inline code; the ONNX backend overrides it to crop/embed slot by slot through
the cache. The trigger search (`compute_trigger=True`) needs every patch
back, so it keeps the default path, as does a format 1 package.

## Alternatives considered

- **Cache per frame regardless of the crop box** — rejected: the embedding of
  a frame depends on where it is cropped; reusing it across a moved window
  would silently change the logits.
- **Dynamic time axis to drop the padding** — rejected (again): skipping
  masked slots before the backbone removes the padding cost without touching
  the head's fixed-length contract.
- **Keeping `classifier.onnx` in format 2 as well** — rejected: it doubles
  the artifact for a path the runtime no longer takes.
