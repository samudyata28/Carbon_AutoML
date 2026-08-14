""" Carbon-Aware Fidelity Allocation (CAFA).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CAFAConfig:
    enabled: bool = True
    lambda0: float = 0.5           # initial cost aversion
    anneal_fraction: float = 0.6   # fraction of budget over which lambda -> 0
    cooling: str = "cosine"        # 'cosine' | 'linear'


def lambda_at(progress: float, cfg: CAFAConfig) -> float:
    """Annealed cost weight given search progress in [0,1]."""
    if not cfg.enabled:
        return 0.0
    if progress >= cfg.anneal_fraction:
        return 0.0
    x = progress / max(cfg.anneal_fraction, 1e-9)  # 0 -> 1 across the anneal window
    if cfg.cooling == "linear":
        factor = 1.0 - x
    else:  # cosine
        factor = 0.5 * (1.0 + math.cos(math.pi * x))
    return cfg.lambda0 * factor


def cost_cooled_utility(
    accuracy: float, normalized_cost: float, progress: float, cfg: CAFAConfig
) -> float:
    """The value reported to the pruner instead of raw accuracy."""
    lam = lambda_at(progress, cfg)
    return accuracy - lam * normalized_cost
