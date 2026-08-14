"""Utility helpers: seeding, device selection, logging, and dataset statistics."""
from __future__ import annotations

import logging
import os
import random
from typing import Any, Tuple

import numpy as np


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (incl. CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinism vs. speed trade-off: we favour speed on Colab but keep
        # cudnn benchmark on for a throughput win. Flip these for exact repro.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:  # torch not installed yet
        logger.warning("torch not available while seeding.")


def get_device() -> "Any":
    """Return the best available torch device."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def count_trainable_params(model: "Any") -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: "Any") -> float:
    """Approximate model size in MB from parameter + buffer bytes."""
    n_bytes = 0
    for p in model.parameters():
        n_bytes += p.numel() * p.element_size()
    for buf in model.buffers():
        n_bytes += buf.numel() * buf.element_size()
    return n_bytes / (1024 ** 2)


def calculate_mean_std(dataset_class: Any, root: str = "./data") -> Tuple["Any", "Any"]:
    """Compute per-channel mean/std over the training split.

    Kept compatible with the starter template. Uses a subset for speed on large
    datasets to avoid a full pass just for normalisation statistics.
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    ds = dataset_class(
        root=root, split="train", download=True, transform=transforms.ToTensor()
    )
    # Subsample up to 2000 images for a fast, stable estimate.
    n = len(ds)
    if n > 2000:
        idx = np.random.RandomState(0).choice(n, size=2000, replace=False).tolist()
        ds = Subset(ds, idx)

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    mean = 0.0
    std = 0.0
    total = 0
    for images, _ in loader:
        b = images.size(0)
        images = images.view(b, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total += b
    mean /= total
    std /= total
    return mean, std
