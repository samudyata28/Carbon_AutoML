"""Data preparation, stratified splits, caching and transforms.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def stratified_indices(
    labels: List[int], val_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Return (train_idx, val_idx) with per-class stratification."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_transforms(
    resolution: int,
    channels: int,
    mean: "Any",
    std: "Any",
    aug_strength: str = "none",
    train: bool = True,
):
    """Compose torchvision transforms for a given resolution and aug strength."""
    from torchvision import transforms

    tfs: List[Any] = [transforms.Resize((resolution, resolution))]

    if train and aug_strength != "none":
        tfs.append(transforms.RandomHorizontalFlip())
        if aug_strength in ("medium", "strong"):
            tfs.append(transforms.RandomRotation(15))
            tfs.append(transforms.ColorJitter(0.2, 0.2, 0.2) if channels == 3
                       else transforms.RandomAffine(0, translate=(0.1, 0.1)))
        if aug_strength == "strong":
            try:
                tfs.append(transforms.RandAugment())
            except Exception:
                pass  

    tfs.append(transforms.ToTensor())
    tfs.append(transforms.Normalize(mean, std))
    return transforms.Compose(tfs)


class SubsetWithTransform:
    """Wrap a base dataset + index list + transform, applied lazily.

    We keep the base dataset transform-free and apply the (resolution-dependent)
    transform here so the *same* base dataset object can be reused across trials
    and fidelities without re-instantiating it.
    """

    def __init__(self, base, indices: List[int], transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, label = self.base[self.indices[i]]  # base returns a PIL image
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def labels(self) -> List[int]:
        base_labels = self.base._labels
        return [base_labels[i] for i in self.indices]


def balanced_subset_indices(
    labels: List[int], per_class: int, seed: int
) -> List[int]:
    """Sample up to `per_class` indices per class → class-balanced proxy subset.

    Used at low-fidelity rungs so cheap evaluations rank configs by
    minority-sensitive performance rather than majority-class bias (novelty 4.2).
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    chosen: List[int] = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        chosen.extend(idx[:per_class].tolist())
    rng.shuffle(chosen)
    return chosen


def make_weighted_sampler(labels: List[int]):
    """A WeightedRandomSampler that oversamples minority classes."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = np.asarray(labels)
    class_count = np.bincount(labels, minlength=int(labels.max() + 1)).astype(float)
    class_count = np.maximum(class_count, 1.0)
    weights = 1.0 / class_count[labels]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
