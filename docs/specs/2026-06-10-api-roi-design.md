# API: Optional Region-of-Interest (ROI) on `/predict`

**Date:** 2026-06-10
**Status:** Approved

## Motivation

The platform calls `/predict` to confirm a specific alert: it already knows
*where* in the frame the alert fired and wants the temporal verdict scoped to
that region. Today the verdict (`is_smoke`, `probability`) is the max over all
kept tubes anywhere in the frame, so unrelated activity elsewhere can drive
the answer. Example: sequence 9711 builds two tubes (clusters near x≈0.14 and
x≈0.38); a client confirming the x≈0.38 alert gets a verdict polluted by the
other tube.

## Decisions (agreed in brainstorming)

1. **Tube-level intersection.** Tubes are built from all detections as today;
   the ROI then drops whole tubes that do not intersect it. Tube building and
   merging are unaffected, and a tube that starts in the ROI and drifts out
   still counts as one in-ROI tube.
2. **Filter before scoring.** Out-of-ROI tubes are dropped right after
   `build_tubes_for_inference()`, before crop/classify/calibrate. The model
   genuinely "only looks at" the ROI: no GPU spent on irrelevant tubes, and
   the calibrator's `n_tubes` feature counts only in-ROI tubes — as if the
   rest of the frame were empty.
3. **Named-field object** in the request: most self-documenting and
   mistake-proof for clients.
4. **Details addition: one count only.** `num_outside_roi` in the tubes
   block. No ROI echo (client knows what it sent; API echoes no other
   inputs), no dropped-tube geometry (dropped tubes are unscored by decision
   2; re-running without `roi` reproduces the full picture — detections are
   cached).

## API contract

### Request

New optional field on `PredictRequest`:

```json
{
  "frames": ["..."],
  "bucket": "...",
  "roi": {"x_min": 0.30, "y_min": 0.35, "x_max": 0.50, "y_max": 0.55}
}
```

Validation (Pydantic, fails as `400 invalid_request` via the existing
`RequestValidationError` handler):

- All four fields required when `roi` is present; extra fields rejected.
- Each coordinate in `[0, 1]` inclusive (`Field(ge=0.0, le=1.0)`).
  `{0, 0, 1, 1}` is a valid whole-frame ROI.
- `x_min < x_max` and `y_min < y_max` (model validator) — zero-area and
  inverted rectangles rejected.
- `roi` omitted or `null` → exactly today's behavior, byte-identical
  responses.

```python
class Roi(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_extent(self) -> "Roi":
        if self.x_min >= self.x_max:
            raise ValueError("roi requires x_min < x_max")
        if self.y_min >= self.y_max:
            raise ValueError("roi requires y_min < y_max")
        return self
```

### Response

- `is_smoke` / `probability` computed only over ROI-intersecting tubes.
- ROI filters out every tube → existing "no tubes kept" shape:
  `is_smoke: false`, `probability: 0.0` (calibrated).
- Verbose `details`:
  - `tubes.kept` contains only in-ROI tubes (falls out naturally).
  - New `num_tubes_outside_roi: int` on the API `Preprocessing` DTO
    (mapped from the core tubes block, following the `num_tube_candidates`
    precedent). `0` when no ROI was given. Resolves the ambiguity between
    "frame was quiet" and "activity existed but outside the ROI".

### Tube-intersects-ROI rule

A tube is kept iff **any of its real (non-gap) detection bboxes** intersects
the ROI rectangle — standard axis-aligned overlap test on the detection's
`(cx, cy, w, h)` box vs the ROI corners; touching edges counts. Gap-interpolated
entries are synthetic and do not count.

## Core changes

- `BboxTubeTemporalModel.predict()` gains
  `roi: tuple[float, float, float, float] | None = None`, interpreted as
  `(x_min, y_min, x_max, y_max)` normalized. Core takes a plain tuple — it
  stays Pydantic-light at its boundary; the API converts the named object.
- Core defensively validates the tuple (coords in `[0, 1]`, `x_min < x_max`,
  `y_min < y_max`) and raises `ValueError` — `predict()` is a public API
  callable without the HTTP layer.
- After `build_tubes_for_inference()` produces `kept`:
  `kept = [t for t in kept if tube_intersects_roi(t, roi)]` when `roi` is not
  `None`. The helper lives in `tubes.py` next to the other tube geometry code.
- `details_schema.Tubes` gains `num_outside_roi: int = 0`. The default keeps
  previously serialized details parseable by
  `BboxTubeDetails.model_validate`.
- `num_candidates` keeps its current meaning (pre-merge, pre-ROI): it
  describes raw detector activity.

## API plumbing

- `PredictRequest.roi: Roi | None = None`.
- `app.predict()` → `ModelRunner.predict(roi=...)` → `_predict_sync` →
  `self._model.predict(frames, frame_detections=resolved, roi=roi,
  timer=timer)` — a pass-through tuple param at each hop.
- **Detection cache unaffected:** keyed by `frame_id`, stores raw full-frame
  detections; ROI filtering happens downstream at the tube level, so the same
  cached detections serve requests with different ROIs correctly.
- API `Preprocessing` response DTO gains `num_tubes_outside_roi: int = 0`,
  populated from the core tubes block.

## Testing

**Core** (`core/tests/`, new `test_roi.py` or extend `test_tubes.py`):

- Intersection helper: overlap, touching-edge (counts), fully outside,
  tube whose only ROI-overlapping entries are gaps (dropped).
- `predict()` with an ROI around one of two synthetic tube clusters keeps
  only that tube; verdict computed from it alone.
- ROI excluding everything → negative verdict, `num_outside_roi` correct.
- `roi=None` → output identical to today.
- Invalid tuples (out of range, inverted) → `ValueError`.

**API** (`api/tests/`):

- Validation matrix: coordinate out of `[0, 1]`, `x_min >= x_max`,
  `y_min >= y_max`, missing field, extra field → 400 with message.
- Happy path: stubbed runner receives the `(x_min, y_min, x_max, y_max)`
  tuple; omitted `roi` → runner receives `None`.
- `verbose=true` response includes `num_tubes_outside_roi`.

**Manual sanity** (not committed — scratch data is not in the repo):
`scratch/annot_seq_9711` with an ROI around the x≈0.38 cluster vs the x≈0.14
cluster should keep different tubes and change the verdict accordingly.

## Out of scope

- Multiple ROIs per request / exclusion masks (different use case).
- Cropping images to the ROI before detection (would break the detection
  cache and change detector behavior).
- Exposing dropped-tube geometry in details.
