"""Optional per-stage wall-clock profiling for BboxTubeTemporalModel.predict().

A StageTimer is threaded into ``predict()`` only when profiling is requested.
When no timer is passed the prediction path uses ``nullcontext`` and is
bit-for-bit identical to the unprofiled path — no timing, no CUDA syncs.

On a CUDA device the timer synchronises at each stage boundary so GPU stage
times reflect real kernel completion rather than launch latency. These syncs
run only while profiling is active.
"""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

__all__ = ["STAGES", "StageTimer", "stage_ctx"]

# The stages of BboxTubeTemporalModel.predict(), in order. Single source of
# truth: predict() names its stage_ctx blocks with these, and the benchmark
# reads them back per-stage.
STAGES = ("pad", "detector", "tubes", "crop", "classifier", "trigger_search")


def _sync_fn_for(device: Any) -> Callable[[], None] | None:
    """Return the device-synchronise callable, or None for sync devices (CPU).

    Async accelerators must be synchronised at stage boundaries for honest
    timing — CUDA and MPS both qualify; CPU needs nothing. ``device`` is a
    device string or a ``torch.device``. When torch is installed it normalizes
    and validates the value (a typo like ``"gpu"`` raises here instead of
    silently producing unsynchronized timings); in torch-free runtimes there
    is no accelerator to sync, so any device is a no-op.
    """
    if device is None:
        return None
    try:
        import torch  # noqa: PLC0415  # keep the timer usable in torch-free runtimes
    except ModuleNotFoundError:
        return None  # no torch, no async accelerator to sync
    dev_type = torch.device(device).type
    if dev_type == "cuda":
        return torch.cuda.synchronize
    if dev_type == "mps":
        return torch.mps.synchronize
    return None


class StageTimer:
    """Accumulates per-stage wall-clock durations in milliseconds."""

    def __init__(self, device: Any = None) -> None:
        self._sync = _sync_fn_for(device)
        self._timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self._sync is not None:
            self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._sync is not None:
                self._sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._timings[name] = self._timings.get(name, 0.0) + elapsed_ms

    def as_dict(self) -> dict[str, float]:
        return dict(self._timings)


def stage_ctx(timer: StageTimer | None, name: str):
    """Return ``timer.stage(name)`` or a no-op context when ``timer`` is None."""
    return timer.stage(name) if timer is not None else nullcontext()
