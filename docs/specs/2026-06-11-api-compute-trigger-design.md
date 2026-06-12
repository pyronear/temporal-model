# API: Explicit `?compute_trigger=true` Flag for Time-to-Detection

**Date:** 2026-06-11
**Status:** Approved
**Refs:** issue #26, PR #25, issue #23

## Motivation

PR #25 made the trigger / first-crossing search eval-only: the serving path
runs `predict(..., compute_trigger=False)` and the always-null trigger fields
(`trigger_frame_index`, `trigger_tube_id`, per-tube `first_crossing_frame`)
were dropped from the `/predict` response. The API therefore never computes
time-to-detection (TTD).

We deliberately did not couple TTD to `?verbose=true`: `verbose` only exposes
already-computed details and must not silently run the O(tube_length)
classifier loop (~34% of CPU latency). When an HTTP consumer needs TTD, it
opts in explicitly — the param name signals "do extra work".

## Decisions (agreed in brainstorming)

1. **Explicit query param.** `POST /predict?compute_trigger=true`, default
   `false`. Same FastAPI bool-param pattern as `verbose`, orthogonal to it.
2. **Plumbing only — core already supports it.**
   `BboxTubeTemporalModel.predict` already takes `compute_trigger` and
   populates `TemporalModelOutput.trigger_frame_index`,
   `decision.trigger_tube_id` and per-tube `first_crossing_frame` in the core
   details. The API threads the flag through
   `ModelRunner.predict` → `_predict_sync` → `model.predict`. (The ticket's
   `predict_sequence` step is obsolete; that function was refactored away.)
3. **Omit-when-unset serialization.** The route already serializes with
   `response_model_exclude_unset=True`. Trigger fields are added back to the
   DTOs as optional fields but only *set* when `compute_trigger=true`, so the
   default response stays byte-identical. (Always-present nullable fields
   were rejected: they change the default response, violating the
   acceptance criterion.)
4. **Explicit `null` means "searched, no crossing".** When
   `compute_trigger=true` and the search finds no crossing (e.g. a negative
   decision), `trigger_frame_index` serializes as an explicit `null` — same
   convention the response already uses for `probability` (null carries
   meaning, absence means "not computed").

## API contract

| Query params | Response |
|---|---|
| (default) | Unchanged fast path; no trigger fields anywhere. |
| `compute_trigger=true` | Top-level `trigger_frame_index` (int or null). |
| `compute_trigger=true&verbose=true` | Also `details.decision.trigger_tube_id` and per-tube `details.tubes[].first_crossing_frame`. |
| `verbose=true` only | `details` as today, still no trigger fields. |

DTO changes (`api/src/temporal_model/api/schemas.py`):

- `PredictResponse.trigger_frame_index: int | None` — unset unless the flag
  is on.
- `Decision.trigger_tube_id: int | None` and
  `Tube.first_crossing_frame: int | None` — unset unless the flag is on.
  The core details dict already carries these keys (pydantic's default
  extra-key handling currently ignores them); `_to_details` gains a
  `compute_trigger` flag to forward them.

## Components

- `api/src/temporal_model/api/app.py` — `compute_trigger: bool = False`
  route param; forwarded to `runner.predict(...)` and `to_response(...)`.
- `api/src/temporal_model/api/model_runner.py` — `compute_trigger`
  keyword threaded through `predict` and `_predict_sync` to
  `self._model.predict`.
- `api/src/temporal_model/api/schemas.py` — DTO fields above;
  `to_response`/`_to_details` gate the fields on the flag.
- `api/README.md` — document the param in the `/predict` section.

No core changes.

## Testing

`api/tests/test_app.py`, existing `FakeRunner` pattern:

- Default response contains no trigger keys (byte-compatible with today).
- The flag is forwarded to `runner.predict` (FakeRunner records kwargs).
- `compute_trigger=true` → `trigger_frame_index` present and equal to the
  core output's value; explicit `null` when the core reports none.
- `compute_trigger=true&verbose=true` → `decision.trigger_tube_id` and
  `tubes[].first_crossing_frame` populated from the core details.
- `verbose=true` alone → details present but without trigger fields.

## Acceptance (from issue #26)

- Default response unchanged (no trigger fields, fast path).
- `?compute_trigger=true` returns a populated `trigger_frame_index` and
  verbose trigger fields, matching `find_first_crossing_trigger`.
