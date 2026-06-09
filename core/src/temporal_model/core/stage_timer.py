"""Optional per-stage wall-clock profiling for BboxTubeTemporalModel.predict().

A StageTimer is threaded into ``predict()`` only when profiling is requested.
When no timer is passed the prediction path uses ``nullcontext`` and is
bit-for-bit identical to the unprofiled path — no timing, no CUDA syncs.

On a CUDA device the timer synchronises at each stage boundary so GPU stage
times reflect real kernel completion rather than launch latency. These syncs
run only while profiling is active.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

import torch


class StageTimer:
    """Accumulates per-stage wall-clock durations in milliseconds."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        dev = torch.device(device) if device is not None else None
        self._cuda = dev is not None and dev.type == "cuda"
        self._timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self._cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._cuda:
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._timings[name] = self._timings.get(name, 0.0) + elapsed_ms

    def as_dict(self) -> dict[str, float]:
        return dict(self._timings)


def stage_ctx(timer: StageTimer | None, name: str):
    """Return ``timer.stage(name)`` or a no-op context when ``timer`` is None."""
    return timer.stage(name) if timer is not None else nullcontext()
