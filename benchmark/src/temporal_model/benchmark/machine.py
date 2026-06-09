"""Capture host + runtime metadata so every result dir is self-describing."""

import platform
import socket
import sys

import psutil
import torch


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def machine_info(*, device: str) -> dict:
    """Return a flat dict of host/CPU/GPU/runtime facts for this run."""
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda else None,
        "python_version": sys.version.split()[0],
        "device": device,
        "torch_num_threads": torch.get_num_threads(),
    }
