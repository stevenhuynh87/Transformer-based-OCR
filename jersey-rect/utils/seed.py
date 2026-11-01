"""Reproducibility helpers: fix every RNG and switch PyTorch to deterministic kernels.

Usage::

    from utils.seed import seed_everything, seed_worker, make_generator

    seed_everything(42)
    loader = DataLoader(ds, worker_init_fn=seed_worker, generator=make_generator(42))
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
_CUBLAS_WORKSPACE = ":4096:8"


def seed_everything(seed: int = 42, deterministic: bool = True, warn_only: bool = True) -> int:
    """Seed Python, NumPy and PyTorch (CPU + all GPUs).

    Args:
        seed: The seed to use everywhere.
        deterministic: If True, force deterministic cuDNN/cuBLAS kernels and disable
            cuDNN autotuning. Slower, but runs are repeatable.
        warn_only: Passed to ``torch.use_deterministic_algorithms``. Keep True: some ops we
            rely on (e.g. ``torchvision.ops.deform_conv2d`` backward) have no deterministic
            CUDA implementation and would otherwise raise.

    Returns:
        The seed that was applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Must be set before the first cuBLAS call to take effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

    return seed


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoader: give each worker a distinct, reproducible seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """A seeded ``torch.Generator`` for DataLoader shuffling / random_split."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
