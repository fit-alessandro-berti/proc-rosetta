from __future__ import annotations

import torch


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        return torch.device(default_device())
    return torch.device(device)
