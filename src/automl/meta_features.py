"""Dataset Analysis & Meta-Feature Extraction
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetaFeatures:
    name: str
    num_classes: int
    num_train: int
    height: int
    width: int
    channels: int
    imbalance_ratio: float          # max class count / min class count
    minority_fraction: float        # smallest class share of the data

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def vector(self) -> np.ndarray:
        """Normalised vector for nearest-dataset lookup (log-scaled counts)."""
        return np.array([
            np.log1p(self.num_classes),
            np.log1p(self.num_train),
            np.log1p(self.height * self.width),
            self.channels,
            np.log1p(self.imbalance_ratio),
        ], dtype=float)


def extract_meta_features(dataset_class: Any, root: str = "./data") -> MetaFeatures:
    """Read labels from the train split and compute meta-features."""
    ds = dataset_class(root=root, split="train", download=False)
    labels = list(ds._labels)  # provided by BaseVisionDataset
    counts = Counter(labels)
    values = np.array(list(counts.values()), dtype=float)
    imbalance = float(values.max() / max(values.min(), 1.0))
    minority_fraction = float(values.min() / values.sum())

    mf = MetaFeatures(
        name=getattr(dataset_class, "_dataset_name", dataset_class.__name__),
        num_classes=int(dataset_class.num_classes),
        num_train=int(len(labels)),
        height=int(dataset_class.height),
        width=int(dataset_class.width),
        channels=int(dataset_class.channels),
        imbalance_ratio=imbalance,
        minority_fraction=minority_fraction,
    )
    logger.info("Meta-features for %s: %s", mf.name, mf.to_dict())
    return mf


def class_weights(dataset_class: Any, scheme: str, root: str = "./data") -> "Any":
    """Return per-class weights for a weighted loss.

    scheme: 'none' | 'inverse_freq' | 'effective_number'
    """
    import torch

    ds = dataset_class(root=root, split="train", download=False)
    counts = Counter(ds._labels)
    n_classes = int(dataset_class.num_classes)
    freq = np.array([counts.get(c, 0) for c in range(n_classes)], dtype=float)
    freq = np.maximum(freq, 1.0)

    if scheme == "none":
        w = np.ones(n_classes, dtype=float)
    elif scheme == "inverse_freq":
        w = freq.sum() / (n_classes * freq)
    elif scheme == "effective_number":
  
        beta = 0.999
        eff = 1.0 - np.power(beta, freq)
        w = (1.0 - beta) / eff
        w = w / w.sum() * n_classes
    else:
        raise ValueError(f"Unknown class-weight scheme: {scheme}")

    return torch.tensor(w, dtype=torch.float32)
