"""Interpretability, ablations and the carbon report 
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def hyperparameter_importance(study, out_dir: str) -> Optional[Dict[str, float]]:
    try:
        import optuna
        from optuna.importance import FanovaImportanceEvaluator
        complete = [t for t in study.trials if t.state.name == "COMPLETE"]
        if len(complete) < 30:
            logger.warning("HP importance SKIPPED: only %d COMPLETE trials "
                           "(need >=30 before fANOVA means anything).",
                           len(complete))
            return None
        imp = optuna.importance.get_param_importances(
            study, evaluator=FanovaImportanceEvaluator(seed=0))
        with (Path(out_dir) / "hyperparameter_importance.json").open("w") as f:
            json.dump(imp, f, indent=2)
        logger.info("HP importance: %s", imp)
        _bar(list(imp.keys()), list(imp.values()),
             "Hyperparameter importance (fANOVA)", out_dir, "hp_importance.png")
        return imp
    except Exception as e:
        logger.warning("HP importance skipped: %s", e)
        return None


def plot_pareto(front: List[Dict[str, Any]], all_trials: List[Dict[str, Any]],
                out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        if all_trials:
            ax.scatter([t["accuracy"] for t in all_trials],
                       [t["macro_f1"] for t in all_trials],
                       c="lightgray", s=18, label="all trials")
        ax.plot([p["accuracy"] for p in front], [p["macro_f1"] for p in front],
                "-o", color="#1B7F5C", label="Pareto front")
        ax.set_xlabel("Validation accuracy"); ax.set_ylabel("Macro-F1")
        ax.set_title("Accuracy vs. Macro-F1 Pareto front"); ax.legend()
        fig.tight_layout(); fig.savefig(Path(out_dir) / "pareto_front.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        logger.warning("Pareto plot skipped: %s", e)


def plot_accuracy_vs_energy(study, out_dir: str) -> None:
    """Headline panel: cumulative best accuracy vs cumulative energy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = [(t.user_attrs.get("energy_kwh", 0.0), t.user_attrs.get("val_accuracy", 0.0))
                for t in study.trials if t.user_attrs.get("val_accuracy") is not None]
        if not rows:
            return
        cum_e, best_a, ce, ba = [], [], 0.0, 0.0
        for e, a in rows:
            ce += e; ba = max(ba, a); cum_e.append(ce); best_a.append(ba)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(cum_e, best_a, "-o", color="#1B7F5C")
        ax.set_xlabel("Cumulative energy (kWh)")
        ax.set_ylabel("Best validation accuracy so far")
        ax.set_title("GreenVision: accuracy vs. energy")
        fig.tight_layout(); fig.savefig(Path(out_dir) / "accuracy_vs_energy.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        logger.warning("Accuracy-vs-energy plot skipped: %s", e)


def confusion_and_report(y_true, y_pred, num_classes, out_dir: str) -> None:
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        np.save(Path(out_dir) / "confusion_matrix.npy", cm)
        rep = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        with (Path(out_dir) / "classification_report.json").open("w") as f:
            json.dump(rep, f, indent=2)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(cm, cmap="Greens")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
            fig.tight_layout(); fig.savefig(Path(out_dir) / "confusion_matrix.png", dpi=130)
            plt.close(fig)
        except Exception:
            pass
    except Exception as e:
        logger.warning("Confusion/report skipped: %s", e)


def grad_cam(model, sample_images, out_dir: str) -> None:
    """Best-effort Grad-CAM; requires pytorch-grad-cam. Silently skips if absent."""
    try:
        import torch
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        # Target layer selection is model-specific; we try a reasonable default.
        target_layers = [list(model.modules())[-3]]
        cam = GradCAM(model=model, target_layers=target_layers)
        for i, img in enumerate(sample_images[:4]):
            grayscale = cam(input_tensor=img.unsqueeze(0))[0]
            np.save(Path(out_dir) / f"gradcam_{i}.npy", grayscale)
        logger.info("Grad-CAM maps saved.")
    except Exception as e:
        logger.warning("Grad-CAM skipped (optional): %s", e)


def carbon_report(study, elapsed_h: float, out_dir: str) -> Dict[str, Any]:
    
    run_start = study.user_attrs.get("run_start_ts")
    trials = study.trials
    if run_start is not None:
        trials = [t for t in trials
                  if t.datetime_start is not None
                  and t.datetime_start.timestamp() >= run_start - 5.0]

    total_e = sum(t.user_attrs.get("energy_kwh", 0.0) for t in trials)
    total_c = sum(t.user_attrs.get("co2_kg", 0.0) for t in trials)
    total_s = sum(t.user_attrs.get("seconds", 0.0) for t in trials)
    done = [t for t in trials if t.user_attrs.get("val_accuracy") is not None]
    measured = [t for t in done if t.user_attrs.get("energy_measured")]

    trial_h = total_s / 3600.0
    if trial_h > elapsed_h + 1e-6:
        logger.error("SCOPING BUG: trial time %.3f h exceeds wall clock %.3f h",
                     trial_h, elapsed_h)

    states: Dict[str, int] = {}
    for t in study.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1

    report = {
        "search_wall_clock_hours": round(elapsed_h, 3),
        "trial_wall_hours": round(trial_h, 3),
        "search_overhead_hours": round(elapsed_h - trial_h, 3),
        "total_energy_kwh": round(total_e, 5),
        "total_co2_kg": round(total_c, 5),
        "completed_trials": len(done),
        "energy_measured_fraction": round(len(measured) / max(len(done), 1), 3),
        "study_cumulative_trials": len(study.trials),
        "trial_states": states,
    }
    with (Path(out_dir) / "carbon_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    logger.info("Carbon report: %s", report)
    return report


def _bar(names, values, title, out_dir, fname):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(names[::-1], values[::-1], color="#1B7F5C")
        ax.set_title(title)
        fig.tight_layout(); fig.savefig(Path(out_dir) / fname, dpi=130)
        plt.close(fig)
    except Exception:
        pass


def run_all(study, front, all_trials, elapsed_h, out_dir: str,
            y_true=None, y_pred=None, num_classes=None, model=None, sample_images=None):
    _ensure_dir(out_dir)
    hyperparameter_importance(study, out_dir)
    plot_pareto(front, all_trials, out_dir)
    plot_accuracy_vs_energy(study, out_dir)
    if y_true is not None and y_pred is not None and num_classes is not None:
        confusion_and_report(y_true, y_pred, num_classes, out_dir)
    if model is not None and sample_images is not None:
        grad_cam(model, sample_images, out_dir)
    return carbon_report(study, elapsed_h, out_dir)
