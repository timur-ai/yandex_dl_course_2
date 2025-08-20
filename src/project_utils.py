"""Project utilities for PyTorch runtime setup.

This module centralizes common runtime helpers used across notebooks/scripts:
- Device detection (CUDA/MPS/CPU)
- Conservative CPU thread configuration on CPU-only hosts
- Deterministic seeding for NumPy, Python, and PyTorch
- AMP helpers (autocast and GradScaler) gated by CUDA availability
- Environment summary printing

All functions avoid side effects at import time. Use ``initialize_runtime()``
to perform setup and retrieve the context you need. Keeping these operations in
one place simplifies experiments and ensures reproducibility.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import numpy as np
import torch


def detect_device() -> torch.device:
    """Return the best available `torch.device` among CUDA, MPS, then CPU.

    - Prefers CUDA if available
    - Falls back to Apple Metal (MPS) if available
    - Otherwise uses CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_cpu_threads_for_device(
    device: torch.device, max_threads: int = 8
) -> None:
    """Conservatively limit torch CPU worker threads when on CPU-only.

    This can improve interactivity in notebook environments on Windows/macOS
    by avoiding over-parallelization.
    """
    if device.type != "cpu":
        return
    import os as os_mod
    cpu_count = os_mod.cpu_count() or 1
    torch.set_num_threads(min(max_threads, cpu_count))


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch; favor determinism when possible.

    - Sets cudnn.deterministic=True and cudnn.benchmark=False when available
    - Calls torch.cuda.manual_seed_all when CUDA is available
    """
    import random as random_mod

    random_mod.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(slots=True)
class RuntimeContext:
    """Holds handles returned by `initialize_runtime`.

    - device: The selected torch.device
    - autocast: CUDA AMP autocast function or a nullcontext for non-CUDA
    - scaler: CUDA AMP GradScaler (enabled on CUDA, disabled otherwise)
    """

    device: torch.device
    autocast: Callable[..., AbstractContextManager[object]]
    scaler: object


def amp_setup_for_device(
    device: torch.device,
) -> tuple[bool, Callable[..., AbstractContextManager[object]], object]:
    """Return (enabled, autocast, scaler) for the given device.

    AMP is enabled only for CUDA devices; on other backends autocast is a
    `nullcontext()` and the scaler is constructed with `enabled=False`.
    """
    amp_enabled = device.type == "cuda"
    autocast_func: Callable[..., AbstractContextManager[object]]
    autocast_func = torch.cuda.amp.autocast if amp_enabled else nullcontext
    from torch.cuda.amp import GradScaler as CudaGradScaler  # type: ignore[attr-defined]
    scaler_obj: object = CudaGradScaler(enabled=amp_enabled)
    return amp_enabled, autocast_func, scaler_obj


def print_environment(device: torch.device) -> None:
    """Print a concise summary of the runtime environment."""
    import sys as sys_mod
    logger = logging.getLogger(__name__)

    logger.info("Python: %s", sys_mod.version.split()[0])
    logger.info("PyTorch: %s", torch.__version__)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
        logger.info("Capability: %s", torch.cuda.get_device_capability(0))
        logger.info("AMP: enabled")
    else:
        logger.info("AMP: disabled")
    logger.info("Torch threads: %s", torch.get_num_threads())


def initialize_runtime(
    seed: int = 42, limit_cpu_threads: bool = True
) -> RuntimeContext:
    """Perform standard runtime setup and return a `RuntimeContext`.

    Steps performed:
    - Detect device
    - Optionally limit CPU threads when on CPU
    - Seed Python/NumPy/PyTorch deterministically
    - Configure AMP helpers (autocast + scaler)
    """
    device = detect_device()
    if limit_cpu_threads:
        configure_cpu_threads_for_device(device)
    seed_everything(seed)
    _enabled, autocast, scaler = amp_setup_for_device(device)
    return RuntimeContext(device=device, autocast=autocast, scaler=scaler)


__all__ = [
    "RuntimeContext",
    "detect_device",
    "configure_cpu_threads_for_device",
    "seed_everything",
    "amp_setup_for_device",
    "print_environment",
    "initialize_runtime",
]
