"""GreenVision entry point.

Modes
-----
search : Stage 1-7, runs the carbon-aware multi-fidelity search within the budget,
         writes <output_dir>/best_config.json and interpretability artifacts.
final  : loads a config, retrains at full fidelity (excluded from budget), predicts
         on the test split, writes predictions .npy (and data/exam_dataset/predictions.npy
         for skin_cancer autograding).

Examples
--------
python run.py --dataset skin_cancer --mode search --seed 42
python run.py --dataset skin_cancer --mode final  --config outputs/best_config.json \
              --output-path final_test_preds.npy
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from automl.automl import AutoML
from automl.datasets import (
    FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset,
)
from automl.utils import setup_logging

logger = logging.getLogger("greenvision")

DATASETS = {
    "fashion": FashionDataset,
    "flowers": FlowersDataset,
    "emotions": EmotionsDataset,
    "skin_cancer": SkinCancerDataset,
}


def load_config(path: str, overrides: dict) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # apply CLI overrides (flat keys mapped into nested cfg)
    for k, v in overrides.items():
        if v is None:
            continue
        if k == "budget_hours":
            cfg["search"]["budget_hours"] = v
        elif k == "n_trials":
            cfg["search"]["n_trials"] = v
        elif k == "data_root":
            cfg["paths"]["data_root"] = v
        elif k == "output_dir":
            cfg["paths"]["output_dir"] = v
        elif k == "optuna_storage":
            cfg["paths"]["optuna_storage"] = v
        elif k == "portfolio":
            cfg["paths"]["portfolio"] = v
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="GreenVision AutoML")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--mode", default="search", choices=["search", "final"])
    parser.add_argument("--config", default="configs/default.yaml",
                        help="YAML config (search) or best_config.json (final).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=Path, default=Path("final_test_preds.npy"))
    # convenience overrides
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--optuna-storage", type=str, default=None)
    parser.add_argument("--portfolio", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.WARNING if args.quiet else logging.INFO)
    dataset_class = DATASETS[args.dataset]

    if args.mode == "search":
        cfg = load_config(args.config, {
            "budget_hours": args.budget_hours,
            "n_trials": args.n_trials,
            "data_root": args.data_root,
            "output_dir": args.output_dir,
            "optuna_storage": args.optuna_storage,
            "portfolio": args.portfolio,
        })
        automl = AutoML(seed=args.seed, config=cfg)
        automl.fit(dataset_class)
        out = Path(cfg["paths"]["output_dir"]) / "best_config.json"
        logger.info("Search complete. Best config at %s", out)

    else:  # final

        base_cfg = load_config("configs/default.yaml", {
            "data_root": args.data_root,
            "output_dir": args.output_dir,
        })

        cfg_path = Path(args.config)
        if cfg_path.suffix != ".json":
            cfg_path = Path(base_cfg["paths"]["output_dir"]) / "best_config.json"
            logger.info("Resolved --config -> %s", cfg_path)
        if not cfg_path.exists():
            raise SystemExit(
                f"No trained config at {cfg_path}. Run --mode search first, "
                f"or pass --config <path to best_config.json>.")

        automl = AutoML(seed=args.seed, config=base_cfg)
        automl.load_best_config(cfg_path)
        automl.final_train(dataset_class)
        preds, labels = automl.predict(dataset_class)

        with args.output_path.open("wb") as f:
            np.save(f, preds)
        logger.info("Wrote predictions -> %s", args.output_path)

        if args.dataset == "skin_cancer":
            tp = Path("data/exam_dataset/predictions.npy")
            tp.parent.mkdir(parents=True, exist_ok=True)
            with tp.open("wb") as f:
                np.save(f, preds)
            logger.info("Wrote autograder predictions -> %s", tp)

        if not np.isnan(np.asarray(labels, dtype=float)).any():
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            logger.info("Test accuracy %.4f | macro-F1 %.4f", acc, f1)
        else:
            logger.info("No labels for '%s' (exam test split).", args.dataset)


if __name__ == "__main__":
    main()
