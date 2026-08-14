"""Novelty 4.3 — Portfolio-to-Prior Meta Warm-Start (Weeks 3 + 6).

A portfolio is a JSON file mapping practice-dataset name -> {meta_features, best
config}. Offline you populate it by running the search on fashion/flowers/emotions
and calling `save_portfolio_entry`. At search time, `nearest_configs` finds the
closest practice dataset by meta-feature distance and returns its config so the
Optuna study can `enqueue_trial` it as a prior.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .meta_features import MetaFeatures

logger = logging.getLogger(__name__)


def load_portfolio(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def save_portfolio_entry(path: str, mf: MetaFeatures, best_config: Dict[str, Any]) -> None:
    p = Path(path)
    portfolio = load_portfolio(path)
    portfolio[mf.name] = {
        "meta_features": mf.to_dict(),
        "vector": mf.vector().tolist(),
        "config": best_config,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(portfolio, f, indent=2)
    logger.info("Saved portfolio entry for %s -> %s", mf.name, path)


def nearest_configs(
    path: str, mf: MetaFeatures, k: int = 1, exclude_self: bool = True
) -> List[Dict[str, Any]]:
    """Return the config(s) of the k nearest practice datasets by meta-features."""
    portfolio = load_portfolio(path)
    if not portfolio:
        return []
    target = mf.vector()
    scored = []
    for name, entry in portfolio.items():
        if exclude_self and name == mf.name:
            continue
        vec = np.asarray(entry.get("vector", []), dtype=float)
        if vec.shape != target.shape:
            continue
        dist = float(np.linalg.norm(vec - target))
        scored.append((dist, name, entry["config"]))
    scored.sort(key=lambda x: x[0])
    if scored:
        logger.info("Warm-start neighbours: %s", [(n, round(d, 3)) for d, n, _ in scored[:k]])
    return [cfg for _, _, cfg in scored[:k]]
