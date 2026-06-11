# API: Caller-supplied detection bbox on `/predict` (`bbox_xyxyn`)

**Date:** 2026-06-11
**Status:** Approved

## Motivation

The platform calls `/predict` to confirm an alert its own detector already
localized: it holds a bounding box for the smoke. Today the API re-runs the
bundled YOLO over every frame to rediscover what the caller already knows —
the dominant cost of a cold request. Letting the caller supply that box
directly skips detection entirely and answers faster, while still running the
full temporal pipeline (tube → stabilized crops → ViT → calibrator).

## Decisions

1. **The box is the detection, not a filter.** Unlike `roi_xyxyn` (which
   filters tubes built from YOLO detections), `bbox_xyxyn` replaces detection:
   the same box becomes the single detection on every frame. The tube builder
   then yields exactly one full-length tube (IoU = 1.0 across frames, no
   gaps), and crop/classify/calibrate run unchanged.
2. **Ride the existing injection seam.** The core model already accepts
   pre-supplied detections via `predict(frame_detections=...)` (the detection
   cache uses it). A new pure helper,
   `make_forced_detections(frames, bbox_xyxyn, confidence)` in
   `core/inference.py`, builds the synthetic `FrameDetections`; there is no
   second inference path to maintain.
3. **Same convention as the ROI.** `bbox_xyxyn` is
   `[x_min, y_min, x_max, y_max]` normalized (ultralytics `xyxyn`), validated
   by the shared `validate_roi` core rules. Internally it is converted once to
   the center-based `Detection` format.
4. **Mutually exclusive with `roi_xyxyn`** (400). The bbox already pins where
   the smoke is; an ROI filter on top of it is meaningless and likely a
   client bug.
5. **Optional `bbox_confidence`, default 1.0.** Stamped on every synthetic
   detection. It gates nothing downstream (tube building has no confidence
   threshold) but feeds the calibrator's mean-confidence feature, so callers
   passing their upstream detector's score get a probability consistent with
   it. Must be in (0, 1]; rejected (400) when sent without `bbox_xyxyn`.
6. **Detection cache bypassed entirely.** The forced run neither reads nor
   writes the cache — synthetic detections must never masquerade as real YOLO
   output for later requests, and the cache invariant stays "full-frame YOLO
   detections only".

## API contract

```json
{
  "frames": ["..."],
  "bucket": "...",
  "bbox_xyxyn": [0.30, 0.35, 0.50, 0.55],
  "bbox_confidence": 0.82
}
```

Validation (400 `invalid_request`): geometry rules shared with `roi_xyxyn`
(4 values in [0, 1], `x_min < x_max`, `y_min < y_max`); `bbox_confidence` in
(0, 1] and only alongside `bbox_xyxyn`; `bbox_xyxyn` + `roi_xyxyn` rejected.

Response shape is unchanged. With profiling on, the server-side `profiling`
block reports `forced_bbox: true` instead of cache counters, and the
`detector` stage is absent (never ran).

## Implementation map

- `core/inference.py` — `make_forced_detections()` (pure, unit-tested).
- `core/tubes.py` — `validate_roi(..., name=...)` so error messages name the
  field being validated.
- `api/schemas.py` — `bbox_xyxyn` / `bbox_confidence` fields + cross-field
  validation.
- `api/model_runner.py` — `predict(bbox=..., bbox_confidence=...)` short
  circuit: build forced detections, skip cache and `detect()`.
- `api/app.py` — threads the fields from the request body to the runner.

## Limits

- One box per request. Multiple simultaneous alerts mean multiple requests
  (or a future `bboxes_xyxyn` extension if the need appears).
- The box is static across frames. Smoke drifting out of the (context-
  expanded) crop window over a long sequence is on the caller; the stabilized
  crop already tolerates in-window motion by design.
