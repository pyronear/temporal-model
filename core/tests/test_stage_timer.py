"""Unit tests for the StageTimer profiling helper."""

import time
from contextlib import nullcontext

from temporal_model.core.stage_timer import StageTimer, stage_ctx


def test_records_stage_duration_in_ms():
    timer = StageTimer()
    with timer.stage("detector"):
        time.sleep(0.01)
    timings = timer.as_dict()
    assert set(timings) == {"detector"}
    assert timings["detector"] >= 9.0  # ~10ms, allow scheduling slack


def test_accumulates_repeated_stage():
    timer = StageTimer()
    for _ in range(3):
        with timer.stage("classifier"):
            time.sleep(0.005)
    assert timer.as_dict()["classifier"] >= 12.0  # 3 * ~5ms


def test_as_dict_returns_a_copy():
    timer = StageTimer()
    with timer.stage("crop"):
        pass
    snapshot = timer.as_dict()
    snapshot["crop"] = -1.0
    assert timer.as_dict()["crop"] != -1.0


def test_cpu_timer_has_no_sync():
    timer = StageTimer(device="cpu")
    assert timer._sync is None


def test_stage_ctx_is_noop_without_timer():
    ctx = stage_ctx(None, "yolo")
    assert isinstance(ctx, nullcontext)


def test_stage_ctx_delegates_to_timer():
    timer = StageTimer()
    with stage_ctx(timer, "tubes"):
        pass
    assert "tubes" in timer.as_dict()
