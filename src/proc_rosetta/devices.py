from __future__ import annotations

import os

import torch


def default_device() -> str:
    """Return the deliberately conservative default execution device."""

    return "cpu"


def default_training_device() -> str:
    """Prefer an available accelerator for model training."""

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def available_devices() -> list[str]:
    """List selectable devices with the default CPU first."""

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def configure_cpu_worker() -> None:
    """Keep each process single-threaded when process-level parallelism is active."""

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        return torch.device(default_device())
    return torch.device(device)
