"""AutoML orchestrator.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .cafa import CAFAConfig
from .carbon import CostModel
from .data import SubsetWithTransform, build_transforms, stratified_indices
from .interpret import run_all
from .meta_features import extract_meta_features
from .search import run_search, pareto_front
from .search_space import BACKBONES
from .train import train_config, evaluate
from .utils import calculate_mean_std, get_device, set_seed
from .warm_start import save_portfolio_entry

logger = logging.getLogger(__name__)


class AutoML:
    def __init__(self, seed: int, config: Dict[str, Any]):
        self.seed = seed
        self.cfg = config
        self._best_config: Optional[Dict[str, Any]] = None
        self._mean = None
        self._std = None
        self._model = None
        self._search_result: Optional[Dict[str, Any]] = None
        set_seed(seed)

    # ------------------------------------------------------------------ search
    def fit(self, dataset_class: Any) -> "AutoML":
        """Stage 1-7: run the carbon-aware multi-fidelity search."""
        result = run_search(dataset_class=dataset_class, cfg=self.cfg, seed=self.seed)
        self._search_result = result
        self._best_config = result["best_config"]
        self._mean, self._std = result["mean"], result["std"]

        out = self.cfg["paths"]["output_dir"]
        Path(out).mkdir(parents=True, exist_ok=True)
        self.save_best_config(Path(out) / "best_config.json")

        front = pareto_front(result["study"])
        all_pts = [
            {"accuracy": t.user_attrs.get("val_accuracy", 0.0),
             "macro_f1": t.user_attrs.get("val_macro_f1", 0.0)}
            for t in result["study"].trials
            if t.user_attrs.get("val_accuracy") is not None
        ]
        run_all(result["study"], front, all_pts, result["elapsed_h"], out)

        if self.cfg["warm_start"].get("update_portfolio", False):
            save_portfolio_entry(self.cfg["paths"]["portfolio"],
                                 result["meta_features"], self._best_config)
        return self

    # ------------------------------------------------------------- final train
    def final_train(self, dataset_class: Any) -> "AutoML":
        """Retrain the selected config at full fidelity on the full train split.

        Excluded from the 24h budget per exam rules.
        """
        assert self._best_config is not None, "No config: run fit() or load_best_config()."
        root = self.cfg["paths"]["data_root"]
        if self._mean is None:
            self._mean, self._std = calculate_mean_std(dataset_class, root=root)

        base_train = dataset_class(root=root, split="train", download=False)

        train_idx, val_idx = stratified_indices(
            base_train._labels, self.cfg["data"]["val_fraction"], self.seed
        )
        budget = int(self.cfg["final"]["epochs"])

        result = train_config(
            cfg=self._best_config,
            dataset_class=dataset_class,
            base_train=base_train,
            train_idx=train_idx + val_idx,  
            val_idx=val_idx,
            mean=self._mean, std=self._std,
            max_epochs=budget,
            budget_ceiling=budget,
            cafa=CAFAConfig(enabled=False),  # no cost-cooling at final training
            cost_model=CostModel(BACKBONES),
            progress=1.0,
            root=root,
            report_cb=None,
            low_fidelity=False,
            track_energy=bool(self.cfg["carbon"]["enabled"]),
            seed=self.seed,
        )
        self._model = result["_model"]

        logger.info("Final model trained. IN-SAMPLE acc %.4f | F1 %.4f "
                    "(train+val seen; not a generalisation estimate)",
                    result["val_accuracy"], result["val_macro_f1"])
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, dataset_class: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on the test split. Returns (predictions, labels)."""
        assert self._model is not None, "No model: run final_train() first."
        import torch
        from torch.utils.data import DataLoader

        root = self.cfg["paths"]["data_root"]
        device = get_device()
        num_classes = int(dataset_class.num_classes)
        channels = int(dataset_class.channels)
        res = int(self._best_config["max_resolution"])
        tf = build_transforms(res, channels, self._mean, self._std, "none", train=False)

        base_test = dataset_class(root=root, split="test", download=False)
        test_ds = SubsetWithTransform(base_test, list(range(len(base_test))), tf)
        loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

        self._model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                out = self._model(x)
                preds.append(out.argmax(1).cpu().numpy())
                labels.append(np.asarray(y))
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        return preds, labels

    # -------------------------------------------------------------- (de)serialise
    def save_best_config(self, path) -> None:
        payload = {
            "best_config": self._best_config,
            "mean": [float(x) for x in (self._mean.tolist() if hasattr(self._mean, "tolist") else self._mean)],
            "std": [float(x) for x in (self._std.tolist() if hasattr(self._std, "tolist") else self._std)],
            "seed": self.seed,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved best config -> %s", path)

    def load_best_config(self, path) -> "AutoML":
        import torch
        with open(path) as f:
            payload = json.load(f)
        self._best_config = payload["best_config"]
        self._mean = torch.tensor(payload["mean"])
        self._std = torch.tensor(payload["std"])
        self.seed = payload.get("seed", self.seed)
        logger.info("Loaded best config <- %s", path)
        return self
