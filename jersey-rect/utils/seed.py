from __future__ import annotations

import os
import random

import numpy as np
import torch

# Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
_CUBLAS_WORKSPACE = ":4096:8"


def seed_everything(seed: int = 42, deterministic: bool = True, warn_only: bool = True) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
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
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
