"""Search-space construction 

"""
from __future__ import annotations

from typing import Any, Dict

BACKBONES = {
    "resnet18": 1.0,
    "efficientnet_b0": 1.3,
    "mobilenetv3": 0.7,
}

RESOLUTION_CHOICES = [128, 160, 224]


def suggest_config(trial: "Any") -> Dict[str, Any]:
    """Sample a full fine-tuning configuration from an Optuna trial."""
    cfg: Dict[str, Any] = {}
    cfg["backbone"] = trial.suggest_categorical("backbone", list(BACKBONES.keys()))
    cfg["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    cfg["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    cfg["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    cfg["scheduler"] = trial.suggest_categorical("scheduler", ["cosine", "step", "none"])
    # 0 = fine-tune everything; higher = freeze more of the backbone stem.
    cfg["frozen_stage"] = trial.suggest_int("frozen_stage", 0, 3)
    cfg["aug_strength"] = trial.suggest_categorical(
        "aug_strength", ["light", "medium", "strong"]
    )
    cfg["class_weighting"] = trial.suggest_categorical(
        "class_weighting", ["none", "inverse_freq", "effective_number"]
    )
    
    cfg["max_resolution"] = trial.suggest_categorical("max_resolution", RESOLUTION_CHOICES)
    return cfg


def fixed_config_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct a config dict from a stored Optuna params dict (for warm-start
    and for the final retrain)."""
    return dict(params)
