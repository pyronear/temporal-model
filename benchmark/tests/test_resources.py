"""Tests for the background resource sampler."""

import time

from temporal_model.benchmark.resources import ResourceSampler


def test_collects_samples_while_active():
    with ResourceSampler(interval=0.02) as sampler:
        time.sleep(0.1)
    timeline = sampler.timeline()
    assert len(timeline) >= 2
    sample = timeline[0]
    assert {"t", "cpu_pct", "ram_gb"} <= set(sample)


def test_peaks_present_for_cpu_and_ram():
    with ResourceSampler(interval=0.02) as sampler:
        time.sleep(0.06)
    peaks = sampler.peaks()
    assert "cpu_pct" in peaks
    assert "ram_gb" in peaks


def test_no_samples_before_start():
    sampler = ResourceSampler(interval=0.02)
    assert sampler.timeline() == []
