"""Training / evaluation for a single configuration, with multi-fidelity support.

"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .cafa import CAFAConfig, cost_cooled_utility
from .carbon import EnergyTracker, CostModel
from .data import (
    SubsetWithTransform, build_transforms, balanced_subset_indices,
    make_weighted_sampler,
)
from .meta_features import class_weights
from .models import build_model
from .utils import get_device, model_size_mb

logger = logging.getLogger(__name__)


def _make_optimizer(name: str, params, lr: float, weight_decay: float):
    import torch.optim as optim
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(name)


def _make_scheduler(name: str, optimizer, max_epochs: int):
    import torch.optim.lr_scheduler as sched
    if name == "cosine":
        return sched.CosineAnnealingLR(optimizer, T_max=max(max_epochs, 1))
    if name == "step":
        return sched.StepLR(optimizer, step_size=max(max_epochs // 3, 1), gamma=0.3)
    return None


def resolution_for_epoch_budget(max_resolution: int, max_epochs: int, budget_ceiling: int) -> int:
    """Secondary fidelity: cheaper (fewer-epoch) rungs train at lower resolution."""
    if budget_ceiling <= 0:
        return max_resolution
    frac = min(max_epochs / budget_ceiling, 1.0)
    if frac <= 0.34:
        return min(128, max_resolution)
    if frac <= 0.67:
        return min(160, max_resolution)
    return max_resolution


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0))


def evaluate(model, loader, device, num_classes: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
    import torch
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            preds.append(out.argmax(1).cpu().numpy())
            labels.append(np.asarray(y))
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc = float((preds == labels).mean())
    f1 = macro_f1(labels, preds, num_classes)
    return acc, f1, preds, labels


def train_config(
    *,
    cfg: Dict[str, Any],
    dataset_class: Any,
    base_train,           # transform-free base dataset (train split)
    train_idx,
    val_idx,
    mean,
    std,
    max_epochs: int,
    budget_ceiling: int,
    cafa: CAFAConfig,
    cost_model: CostModel,
    progress: float,
    root: str = "./data",
    report_cb: Optional[Callable[[int, float], bool]] = None,
    low_fidelity: bool = False,
    track_energy: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train one configuration for `max_epochs`; return metrics dict.

    report_cb(epoch, utility) -> should_prune. If it returns True we stop early.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = get_device()
    num_classes = int(dataset_class.num_classes)
    channels = int(dataset_class.channels)

    resolution = resolution_for_epoch_budget(cfg["max_resolution"], max_epochs, budget_ceiling)

    train_tf = build_transforms(resolution, channels, mean, std, cfg["aug_strength"], train=True)
    val_tf = build_transforms(resolution, channels, mean, std, "none", train=False)

    labels_all = [base_train._labels[i] for i in train_idx]
    if low_fidelity:
        sub = balanced_subset_indices(labels_all, per_class=200, seed=seed)
        used_train_idx = [train_idx[i] for i in sub]
    else:
        used_train_idx = train_idx

    train_ds = SubsetWithTransform(base_train, used_train_idx, train_tf)
    val_ds = SubsetWithTransform(base_train, val_idx, val_tf)

    # sampler for imbalance handling on the (full-fidelity) training set
    sampler = None
    shuffle = True
    if cfg["class_weighting"] != "none" and not low_fidelity:
        sampler = make_weighted_sampler(train_ds.labels)
        shuffle = False

    batch_size = 64 if resolution <= 160 else 32  # fixed-ish to fit T4 memory
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=2, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 64), shuffle=False,
                            num_workers=2, pin_memory=True)

    model = build_model(cfg["backbone"], num_classes, channels,
                        dropout=cfg["dropout"], frozen_stage=cfg["frozen_stage"]).to(device)

    weight = None
    if cfg["class_weighting"] != "none":
        weight = class_weights(dataset_class, cfg["class_weighting"], root=root).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = _make_optimizer(cfg["optimizer"], trainable, cfg["lr"], cfg["weight_decay"])
    scheduler = _make_scheduler(cfg["scheduler"], optimizer, max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc, best_f1 = 0.0, 0.0
    pruned = False

    tracker = EnergyTracker(enabled=track_energy)
    with tracker:
        for epoch in range(max_epochs):
            model.train()
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if scheduler is not None:
                scheduler.step()

            acc, f1, _, _ = evaluate(model, val_loader, device, num_classes)
            best_acc = max(best_acc, acc)
            best_f1 = max(best_f1, f1)

            # CAFA: report cost-cooled utility to the pruner.
            ncost = cost_model.normalized_cost(cfg["backbone"], resolution, max_epochs)
            utility = cost_cooled_utility(acc, ncost, progress, cafa)
            if report_cb is not None and report_cb(epoch, utility):
                pruned = True
                break

    cost_model.observe(cfg["backbone"], resolution, max_epochs, tracker.energy_kwh)
    size_mb = model_size_mb(model)

    return {
        "val_accuracy": best_acc,
        "val_macro_f1": best_f1,
        "energy_kwh": tracker.energy_kwh,
        "energy_measured": tracker.measured,
        "co2_kg": tracker.co2_kg,
        "seconds": tracker.seconds,
        "model_size_mb": size_mb,
        "resolution": resolution,
        "epochs": max_epochs,
        "pruned": pruned,
        "_model": model,  
    }
