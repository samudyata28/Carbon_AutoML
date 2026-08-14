"""GreenVision — Resource-Aware Multi-Fidelity AutoML for Image Classification."""
from __future__ import annotations

__version__ = "0.1.0"

from .automl import AutoML  # noqa: F401
from .datasets import (  # noqa: F401
    FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset,
)

__all__ = [
    "AutoML",
    "FashionDataset",
    "FlowersDataset",
    "EmotionsDataset",
    "SkinCancerDataset",
]
