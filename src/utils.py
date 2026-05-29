"""Reproducibility and small shared utilities for FreqLite.

Seeds, deterministic flags, device selection, and a peak-GPU-memory helper.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

# Must be set BEFORE the first CUDA context / cuBLAS call for deterministic
# GEMMs (our linear heads). Setting at import time guarantees it is in place
# before any model touches the GPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch (CPU+CUDA) and set deterministic flags.

    Determinism matters for the reproducibility bar: every reported number must
    be regenerable. We enable cudnn.deterministic and disable benchmark. We also
    request torch.use_deterministic_algorithms; CUDA needs CUBLAS_WORKSPACE_CONFIG
    for deterministic GEMMs, set here before any CUDA context heavy use.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic cuBLAS GEMMs (matmul in our linear heads).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # warn_only keeps us going if a kernel lacks a deterministic impl
            pass


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


@dataclass
class RunStats:
    """Container for per-run efficiency metrics (efficiency table)."""

    params: int = 0
    flops: int = 0
    sec_per_epoch: float = 0.0
    peak_gpu_mem_mb: float = 0.0
