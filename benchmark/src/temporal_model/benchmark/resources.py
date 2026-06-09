"""Background sampler for CPU/RAM (always) and GPU (when NVML is available).

Runs a daemon thread that snapshots utilisation every ``interval`` seconds
between ``__enter__`` and ``__exit__``. GPU metrics are best-effort: if the
``pynvml`` bindings or an NVIDIA device are absent, the sampler silently omits
them (CPU-only VMs are fully supported).
"""

import threading
import time

import psutil

try:  # best-effort GPU support
    import pynvml

    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # noqa: BLE001 — any failure means "no GPU metrics"
    pynvml = None
    _GPU_HANDLE = None


class ResourceSampler:
    """Context manager that records a utilisation timeline."""

    def __init__(self, interval: float = 0.1) -> None:
        self._interval = interval
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def __enter__(self) -> "ResourceSampler":
        psutil.cpu_percent(None)  # prime the interval baseline
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _sample(self) -> dict:
        row = {
            "t": time.perf_counter() - self._t0,
            "cpu_pct": psutil.cpu_percent(None),
            "ram_gb": psutil.virtual_memory().used / 1e9,
        }
        if _GPU_HANDLE is not None:
            util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
            row["gpu_util"] = float(util.gpu)
            row["gpu_mem_gb"] = mem.used / 1e9
        return row

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._samples.append(self._sample())
            self._stop.wait(self._interval)

    def timeline(self) -> list[dict]:
        return list(self._samples)

    def peaks(self) -> dict:
        if not self._samples:
            return {}
        keys = {k for s in self._samples for k in s if k != "t"}
        return {k: max(s.get(k, 0.0) for s in self._samples) for k in keys}
