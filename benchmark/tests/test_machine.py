"""Tests for machine metadata capture."""

import torch

from temporal_model.benchmark import machine

REQUIRED_KEYS = {
    "hostname",
    "platform",
    "cpu_model",
    "cpu_count_physical",
    "cpu_count_logical",
    "ram_total_gb",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "python_version",
    "device",
    "torch_num_threads",
}


def test_machine_info_has_required_keys():
    info = machine.machine_info(device="cpu")
    assert set(info) >= REQUIRED_KEYS


def test_machine_info_reports_requested_device():
    info = machine.machine_info(device="cpu")
    assert info["device"] == "cpu"


def test_gpu_name_none_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = machine.machine_info(device="cpu")
    assert info["gpu_name"] is None
    assert info["cuda_version"] is None
