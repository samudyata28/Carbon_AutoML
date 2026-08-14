""" Carbon-Aware Multi-Fidelity Bayesian Optimization.

"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cafa import CAFAConfig
from .carbon import CostModel
from .data import stratified_indices
from .meta_features import MetaFeatures, extract_meta_features
from .search_space import BACKBONES, suggest_config
from .train import train_config
from .utils import calculate_mean_std
from .warm_start import nearest_configs

logger = logging.getLogger(__name__)


def _rung_is_low_fidelity(max_epochs: int, budget_ceiling: int) -> bool:
    return max_epochs < budget_ceiling  # any promoted-below-max rung is "cheap"


def run_search(
    *,
    dataset_class: Any,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the HPO study and return best config + full trial log."""
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    root = cfg["paths"]["data_root"]
    budget_ceiling = int(cfg["search"]["max_epochs"])          # full fidelity
    min_epochs = int(cfg["search"]["min_epochs"])
    reduction = int(cfg["search"]["reduction_factor"])
    n_trials = int(cfg["search"]["n_trials"])
    timeout_h = float(cfg["search"]["budget_hours"])
    val_fraction = float(cfg["data"]["val_fraction"])

    cafa = CAFAConfig(
        enabled=bool(cfg["cafa"]["enabled"]),
        lambda0=float(cfg["cafa"]["lambda0"]),
        anneal_fraction=float(cfg["cafa"]["anneal_fraction"]),
        cooling=str(cfg["cafa"]["cooling"]),
    )
    cost_model = CostModel(BACKBONES)

    # ---- Stage 1: meta-features + normalisation stats ----
    mf = extract_meta_features(dataset_class, root=root)
    mean, std = calculate_mean_std(dataset_class, root=root)


    base_train = dataset_class(root=root, split="train", download=False)
    train_idx, val_idx = stratified_indices(base_train._labels, val_fraction, seed)
    logger.info("Split: %d train / %d val", len(train_idx), len(val_idx))


    storage = cfg["paths"].get("optuna_storage")  
    study_name = f"greenvision_{mf.name}"
    sampler = TPESampler(seed=seed, multivariate=True, group=True)
    pruner = HyperbandPruner(
        min_resource=min_epochs, max_resource=budget_ceiling, reduction_factor=reduction
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    # ---- Stage 4: warm-start priors ----
    if cfg["warm_start"]["enabled"]:
        for wc in nearest_configs(cfg["paths"]["portfolio"], mf,
                                  k=int(cfg["warm_start"]["k"])):
            try:
               
                try:
                    study.enqueue_trial(wc, skip_if_exists=True)
                except TypeError:
                    study.enqueue_trial(wc)
                logger.info("Enqueued warm-start config: %s", wc.get("backbone"))
            except Exception as e:
                logger.warning("Could not enqueue warm-start config: %s", e)

    t_start = time.time()
    deadline = t_start + timeout_h * 3600.0

    def objective(trial: "Any") -> float:
        # progress in [0,1] by wall-clock, used by CAFA cooling
        progress = min((time.time() - t_start) / max(timeout_h * 3600.0, 1e-9), 1.0)
        config = suggest_config(trial)

        def report_cb(epoch: int, utility: float) -> bool:
            trial.report(utility, step=epoch)

            if time.time() >= deadline:
                logger.info("Deadline hit mid-trial %d at epoch %d; cutting.",
                            trial.number, epoch)
                return True
            return trial.should_prune()

        low_fi = False  
        result = train_config(
            cfg=config,
            dataset_class=dataset_class,
            base_train=base_train,
            train_idx=train_idx,
            val_idx=val_idx,
            mean=mean, std=std,
            max_epochs=budget_ceiling,
            budget_ceiling=budget_ceiling,
            cafa=cafa,
            cost_model=cost_model,
            progress=progress,
            root=root,
            report_cb=report_cb,
            low_fidelity=low_fi,
            track_energy=bool(cfg["carbon"]["enabled"]),
            seed=seed,
        )
        # log everything for post-hoc Pareto + reporting
        trial.set_user_attr("val_accuracy", result["val_accuracy"])
        trial.set_user_attr("val_macro_f1", result["val_macro_f1"])
        trial.set_user_attr("energy_kwh", result["energy_kwh"])
        trial.set_user_attr("energy_measured", result.get("energy_measured", False))
        trial.set_user_attr("co2_kg", result["co2_kg"])
        trial.set_user_attr("seconds", result["seconds"])
        trial.set_user_attr("model_size_mb", result["model_size_mb"])
        trial.set_user_attr("resolution", result["resolution"])
        trial.set_user_attr("config", config)

        if result["pruned"]:
            raise optuna.TrialPruned()
        return result["val_accuracy"]

    def stop_on_deadline(study, trial):
        if time.time() >= deadline:
            logger.info("Budget reached (%.2f h); stopping study.", timeout_h)
            study.stop()


    study.set_user_attr("run_start_ts", t_start)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=max(deadline - time.time(), 1),
        callbacks=[stop_on_deadline],
        gc_after_trial=True,
    )

    best = study.best_trial
    best_config = best.user_attrs.get("config", best.params)
    logger.info("Best val accuracy: %.4f  (macro-F1 %.4f)",
                best.value, best.user_attrs.get("val_macro_f1", float("nan")))

    return {
        "meta_features": mf,
        "mean": mean, "std": std,
        "best_config": best_config,
        "best_value": best.value,
        "study": study,
        "elapsed_h": (time.time() - t_start) / 3600.0,
    }


def pareto_front(study) -> List[Dict[str, Any]]:
    """Reconstruct an (accuracy, macro-F1) Pareto front from completed trials."""
    pts = []
    for tr in study.trials:
        a = tr.user_attrs.get("val_accuracy")
        f = tr.user_attrs.get("val_macro_f1")
        if a is None or f is None:
            continue
        pts.append({"trial": tr.number, "accuracy": a, "macro_f1": f,
                    "energy_kwh": tr.user_attrs.get("energy_kwh", 0.0),
                    "config": tr.user_attrs.get("config", tr.params)})
    front = []
    for p in pts:
        dominated = any(
            (q["accuracy"] >= p["accuracy"] and q["macro_f1"] >= p["macro_f1"] and
             (q["accuracy"] > p["accuracy"] or q["macro_f1"] > p["macro_f1"]))
            for q in pts
        )
        if not dominated:
            front.append(p)
    front.sort(key=lambda d: d["accuracy"], reverse=True)
    return front
