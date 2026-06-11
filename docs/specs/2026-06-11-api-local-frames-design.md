# API: Local Frame Files on `/predict`

**Date:** 2026-06-11
**Status:** Approved

## Motivation

Today `/predict` only accepts frames as S3 keys: the API downloads each key
from a bucket before inference. On an edge deployment (e.g. a Pyronear
station), the API runs next to the capture process and the frames already sit
on a disk both processes share — S3 may not exist there at all. The contract
needs a way to reference frames on the local filesystem.

The edge caller is a different codebase from the cloud caller (the engine vs
alert-api), so the two worlds do not need an identical request shape — but the
contract should not foreclose a single instance serving both sources, since
that requirement is still open.

## Decisions (agreed in brainstorming)

1. **Per-request `source` field, defaulting to server config.** A new
   optional `source: "s3" | "local"` on `PredictRequest`; omitted means the
   server's configured default. Backward compatible (today's alert-api
   requests are byte-for-byte unchanged in behavior), self-describing, and
   absorbs per-request source mixing later without a breaking change.
   (Alternatives considered: a config-only mode switch — zero contract change
   but another contract change later if mixing is ever needed; URI schemes in
   `frames` (`s3://`, `file://`) — per-frame mixing nobody needs, reintroduces
   bucket parsing inside strings, and `file://` naturally expresses absolute
   paths, which fights decision 2.)
2. **No absolute paths from the network.** Request frames in local mode are
   relative identifiers resolved under a server-configured root
   (`TEMPORAL_API_FRAMES_ROOT`), with a post-resolution containment check.
   A request-supplied filesystem root would be an arbitrary-file-read
   primitive.
3. **`bucket` and `frames_root` stay separate concepts.** Both are "the
   container frames are relative to", but they differ in who may choose them:
   `bucket` is safely request-suppliable because the server's IAM credentials
   bound what any bucket name can reach; a filesystem root has no such
   per-container ACL, so it is settings-only. A unified root URI in settings
   (`s3://...` vs `/path`) was rejected: it reintroduces URI parsing, overlaps
   confusingly with per-request `bucket`/`source`, and `bucket` would survive
   anyway for per-org dynamic bucket names. `frames_root` as an s3 key prefix
   was also rejected: alert-api sends full keys; nobody needs it.
4. **Hard-reject `bucket` in local mode** (400), not silently ignore — a
   request that names a bucket while asking for local frames is confused and
   should hear about it.
5. **Same error contract for both sources.** A missing local file raises the
   same `404 frame_not_found` as a missing S3 key; callers handle one shape.

## API contract

### Request

New optional field on `PredictRequest`:

```json
{
  "source": "local",
  "frames": [
    "station-3/2026-06-11/0001.jpg",
    "station-3/2026-06-11/0002.jpg"
  ]
}
```

The effective source is `body.source or settings.frame_source`. Validation by
effective source:

|  | `s3` (effective) | `local` (effective) |
|---|---|---|
| `frames` strings | bare S3 keys, no `://` (unchanged) | relative paths: no absolute paths, no `..` segments, no `://` |
| `bucket` | allowed, falls back to `settings.s3_bucket` (unchanged) | present → 400 `invalid_request` |
| server prerequisite | bucket resolvable, else 400 (unchanged) | `frames_root` configured, else 400 |

Source-independent checks (non-empty `frames`, no `://`) stay as Pydantic
validators in `schemas.py`; source-dependent checks live in the route and in
`local.py` — the schema layer cannot see settings, and the route is where the
existing "no S3 bucket" 400 already lives.

`source` omitted or `null` → exactly the server default. On an s3-default
server, explicit `source: "s3"` behaves identically to omitting it.

### Response

Unchanged. Where frames come from is invisible to the response: `is_smoke`,
`probability`, `model`, and verbose `details` are identical for both sources.
The only observable difference is the profiling stage name (`local_resolve`
instead of `s3_fetch`), visible only with `TEMPORAL_API_PROFILE` on and
`?verbose=true`.

### Errors

All errors reuse the existing `ApiError` machinery (`{detail, code}`):

| Condition | Status / code |
|---|---|
| `source: "local"` but `frames_root` unset | 400 `invalid_request` (message names `TEMPORAL_API_FRAMES_ROOT`) |
| `bucket` present with effective local source | 400 `invalid_request` |
| frame absolute, contains `..`, or resolves outside root | 400 `invalid_request` (message echoes the request string, never the resolved server path) |
| frame file missing | 404 `frame_not_found` |
| frame unreadable (permissions, IO) | 500 `inference_error` (existing catch-all) |
| `source` not `"s3"`/`"local"` | 400 `invalid_request` (Pydantic `Literal`) |

## Settings

Two new fields, mirroring the existing S3 pair:

```python
frame_source: Literal["s3", "local"] = "s3"   # TEMPORAL_API_FRAME_SOURCE
frames_root: str = ""                          # TEMPORAL_API_FRAMES_ROOT
```

The `"s3"` default makes the change invisible to every existing deployment.
An edge box opts in with `TEMPORAL_API_FRAME_SOURCE=local` and
`TEMPORAL_API_FRAMES_ROOT=/data/frames`.

## API plumbing

- New module `local.py`, a deliberate sibling of `s3.py`:

  ```python
  def resolve_frames(root: Path, frames: list[str]) -> list[Path]
  ```

  For each frame string: resolve under `root`, require the resolved path to
  stay inside `root.resolve()` (the real traversal guard — `Path.resolve()`
  follows symlinks, so a symlink inside the root pointing outside also fails
  containment), require the file to exist (missing → `FrameNotFound`), return
  paths in request order. **No copying, no temp dir** — `ModelRunner` never
  sorts paths; sequence order is the returned list order, so real file paths
  pass straight through. On an edge box this removes the whole fetch stage
  from the latency budget.
- Route flow in `app.py`:

  ```
  effective_source = body.source or settings.frame_source
  s3:    unchanged — tempdir, threadpool fetch_frames, "s3_fetch" stage
  local: prerequisite checks (root set, no bucket) → resolve_frames
         → paths straight to runner, profiled as "local_resolve"
  ```
- The S3 client keeps being constructed at startup even on a local-only box:
  boto3 does not touch the network or validate credentials at construction,
  so it is harmless and avoids a conditional in `lifespan`.
- Runner, ROI handling, detection cache, and response mapping are untouched:
  from `runner.predict` onward, a path is a path.

## Testing

**API** (`api/tests/`):

- `test_local.py` (new, mirrors `test_s3.py`): resolution preserves request
  order; nested relative paths work; absolute path, `..` segment, and a
  symlink escaping the root each → `InvalidRequest`; missing file →
  `FrameNotFound`; returned paths are the real files (no copy).
- `test_schemas.py`: `source` accepts `"s3"`, `"local"`, omitted; rejects
  anything else.
- `test_app.py` (TestClient, `tmp_path` as `frames_root`): local happy path
  end-to-end with the stub runner; `bucket` + local → 400; local without
  root → 400; `source` omitted follows the `frame_source` setting both ways;
  explicit `source: "s3"` identical to omitted on an s3-default server.
- `test_settings.py`: env overrides for the two new fields.
- **Back-compat proof:** the entire existing suite passes unmodified — no
  existing test needs editing.

## Out of scope

- URI schemes in `frames` (`s3://`, `file://`).
- A unified root URI in settings replacing `frame_source`/`bucket`.
- `frames_root` as a key prefix in s3 mode.
- Client-uploaded frame bytes (multipart/base64) — a different use case
  ("local to the caller", not "local to the server").
- Per-frame source mixing within one request (one request is one camera
  sequence).
