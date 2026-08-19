# API: Caller-Supplied Detections on `/predict` (Detector Bypass)

**Date:** 2026-06-11
**Status:** Approved

## Motivation

The Pyronear edge devices (RPis running pyro-engine) already run a YOLO
detector on every frame and ship the resulting bboxes with their alerts. When
the alert-api calls `/predict` to get a temporal verdict on an alert sequence,
the API re-runs its own bundled YOLO on the same frames — a redundant GPU pass
that adds latency and compute. This feature lets the caller supply the
per-frame bboxes it already holds; the API skips the detector stage and feeds
the supplied boxes straight into tube building. Everything downstream —
tube building, ROI filtering, cropping, classifier scoring, calibration,
decision — runs unchanged.

## Decisions (agreed in brainstorming)

1. **All-or-nothing bypass with explicit empties.** If the request carries
   `detections`, the bundled detector never runs — there is no hybrid mode
   that detects "uncovered" frames. A frame with no detections is expressed
   as an explicit empty list, meaning "a detector ran and saw nothing"; such
   frames participate in tube gap handling exactly as if the bundled detector
   had returned nothing.
2. **Index-aligned list shape — every frame must be covered.** `detections`
   is a list with exactly one entry per frame, in the same order as
   `frames`. Missing detections for any frame is an error: a shorter (or
   longer) list fails the length check, and `null` entries are rejected by
   the schema — the *only* way to express "no detections for frame i" is an
   explicit empty list. No partial coverage, no omitted-key rules a
   dict-keyed shape would need.
3. **`xyxyn` + `confidence` inner objects.** Boxes arrive as normalized
   corners — the convention pyro-engine produces and the alert-api stores, and
   the same ultralytics vocabulary as the existing `roi_xyxyn` field. The API
   converts to the internal center-based `xywhn` (`Detection` dataclass) at
   the boundary. `class_id` is not exposed; supplied boxes are smoke
   (`class_id=0`) by definition.
4. **Detection cache fully bypassed — no read, no write.** Supplied
   detections are fresher truth than the cache, and writing them would poison
   the shared LRU with a foreign detector's outputs for subsequent
   detector-path requests. This extends the ROI spec's cache invariant:
   only full-frame detections **produced by the bundled detector** ever enter
   the cache.
5. **Confidences are trusted, not re-thresholded.** The edge detector already
   applied its own confidence threshold; the API validates ranges only and
   does not apply the packaged `infer.confidence_threshold` to supplied
   boxes.
6. **Provenance in verbose details.** `details.preprocessing` gains
   `detections_source: "request" | "detector"` so a logged response shows
   which path produced the tubes.

## API contract

### Request

New optional field on `PredictRequest`:

```json
{
  "frames": ["org/img_t0.jpg", "org/img_t1.jpg", "org/img_t2.jpg"],
  "detections": [
    [ {"xyxyn": [0.41, 0.30, 0.47, 0.36], "confidence": 0.62} ],
    [],
    [ {"xyxyn": [0.42, 0.29, 0.49, 0.36], "confidence": 0.71},
      {"xyxyn": [0.10, 0.50, 0.15, 0.55], "confidence": 0.33} ]
  ]
}
```

Composes with `bucket`, `roi_xyxyn`, and `?verbose=true` unchanged.

Validation (Pydantic, fails as `400 invalid_request` via the existing
`RequestValidationError` handler):

- `len(detections) == len(frames)` (model-level validator, checked after
  field validation) — every frame must have an entry; partial coverage is
  rejected.
- Inner entries must be lists: `null` is rejected by the
  `list[list[SuppliedDetection]]` type. "No detections" is only expressible
  as an explicit `[]`.
- Each box: `xyxyn` is exactly 4 floats, each in `[0, 1]` inclusive, with
  `x_min < x_max` and `y_min < y_max` — same rules and fail-closed rationale
  as `roi_xyxyn` (zero-area and inverted boxes rejected; catches most
  accidental `xywhn` input).
- `confidence` in `[0, 1]` inclusive.
- `detections` omitted or `null` → exactly today's behavior: cache + bundled
  detector, byte-identical responses.

```python
class SuppliedDetection(BaseModel):
    xyxyn: tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("xyxyn")
    @classmethod
    def _validate_xyxyn(cls, v):
        x_min, y_min, x_max, y_max = v
        if not all(0.0 <= c <= 1.0 for c in v):
            raise ValueError("xyxyn coordinates must be in [0, 1]")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("xyxyn requires x_min < x_max and y_min < y_max")
        return v


class PredictRequest(BaseModel):
    frames: list[str]
    bucket: str | None = None
    roi_xyxyn: tuple[float, float, float, float] | None = None
    detections: list[list[SuppliedDetection]] | None = None

    @model_validator(mode="after")
    def _detections_match_frames(self):
        if self.detections is not None and len(self.detections) != len(self.frames):
            raise ValueError(
                "detections must have exactly one entry per frame "
                f"(got {len(self.detections)} entries for {len(self.frames)} frames)"
            )
        return self
```

Pydantic's type system handles the rest of the well-formedness for free:
a detection that is not an object, lacks `xyxyn`/`confidence`, has a
non-numeric value, or an `xyxyn` that is not exactly 4 numbers all fail type
validation before the custom validators run.

### Response

- Top-level shape unchanged (`is_smoke`, `probability`, `model`).
- Verbose `details.preprocessing` gains
  `detections_source: Literal["request", "detector"]`. Populated at the API
  layer (the app knows whether the request carried `detections`); core
  details are untouched.
- Verbose `details.profiling`: the `detector` stage is absent from stage
  timings (it never ran); `cache_hits`/`cache_misses` are reported as `0`/`0`
  (the cache was not consulted).

### Conversion at the boundary

Each supplied box maps to the internal `Detection`:

```
cx = (x_min + x_max) / 2;  cy = (y_min + y_max) / 2
w  = x_max - x_min;        h  = y_max - y_min
class_id = 0;              confidence = confidence
```

## API plumbing

All changes live in the API layer; **core is untouched** —
`BboxTubeTemporalModel.predict()` already accepts a complete
`frame_detections` dict and `_resolve_frame_detections()` only detects
frames missing from it (`run_yolo_on_frames([])` early-returns).

- `app.predict()` passes `body.detections` to
  `ModelRunner.predict(detections=...)`, and threads
  `detections_source` into `to_response()` → `_to_details()`.
- `ModelRunner._predict_sync()` branches after `load_sequence()`:
  - `detections is None` → existing path (cache lookup, `detect()` on
    misses, cache write-back).
  - `detections` supplied → build
    `resolved = {frames[i].frame_id: FrameDetections(frame_idx=i,
    frame_id=frames[i].frame_id, timestamp=frames[i].timestamp,
    detections=[converted boxes])}` and call
    `self._model.predict(frames, frame_detections=resolved, roi=roi,
    timer=timer)` directly. No cache read, no cache write, no `detect()`
    call, no `detector` stage timing.
- Frames are still fetched from S3 — the classifier needs the image crops;
  only the detector pass is skipped.
- Index alignment holds end to end: request `frames[i]` →
  `frame_paths[i]` → `load_sequence` `frames[i]` (order-preserving) →
  `detections[i]`.

### Edge cases

- **Truncation/padding:** core truncates to `max_frames` and may pad;
  both look detections up by `frame_id` in the resolved dict, so supplied
  entries for truncated-away frames are simply unused and padded duplicate
  frames resolve to their original frame's detections — identical to the
  cache path today.
- **Duplicate frame basenames** collapse in the `frame_id`-keyed dict
  (last wins). This is pre-existing behavior on the cache path, not new.

## Documented risk (out of scope to validate)

The temporal classifier and calibrator were trained on tubes built from the
bundled detector's boxes (`yolo11s_nimble-narwhal`). Edge-detector boxes may
differ in tightness, confidence distribution, and threshold; tubes built from
them may shift crop geometry and calibration. The classifier scores image
crops, not box metadata, so the mechanism is expected to work — but
calibration on real RPi boxes is unvalidated. Validation happens at
alert-api-integration time, not in this work.

## Testing

**API** (`api/tests/`):

- Validation matrix: length mismatch with `frames` (both shorter and
  longer); `null` inner entry; coordinate out of `[0, 1]`;
  `x_min >= x_max`; `y_min >= y_max`; wrong tuple length; confidence out of
  `[0, 1]` → 400 with message.
- Bypass proof: with `detections` supplied, the model's `detect()` is never
  called (mock/monkeypatch assertion) and the cache is neither read nor
  written.
- Equivalence: supplying the exact boxes the bundled detector would produce
  (xywhn → xyxyn round-trip) yields the same verdict and tubes as the
  detector path.
- Explicit empties: all-empty `detections` → no tubes →
  `is_smoke: false`, `probability: 0.0` (calibrated).
- Conversion: a known `xyxyn` box arrives at the core model as the expected
  `(cx, cy, w, h)` `Detection` with `class_id=0`.
- Composition: `detections` + `roi_xyxyn` filters tubes as usual.
- Verbose: `detections_source` is `"request"` when supplied, `"detector"`
  otherwise; profiling shows no `detector` stage when bypassed.
- `detections: null` / omitted → identical behavior to today (regression).

**Core**: no changes, no new tests.

## Out of scope

- Hybrid mode (detect only uncovered frames) — mixes two detectors' outputs
  in one tube; rejected in brainstorming.
- Calibration/accuracy validation of foreign detector boxes (see Documented
  risk).
- Exposing `class_id` or other detector metadata in the request.
- Caching supplied detections (would poison the bundled-detector cache).
