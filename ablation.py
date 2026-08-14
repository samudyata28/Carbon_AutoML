#!/usr/bin/env python3
"""
    random       RandomSampler        + NopPruner        (full fidelity)
    tpe          TPESampler           + NopPruner        (full fidelity)
    greenvision  TPESampler           + HyperbandPruner  + CAFA cost-cooling

"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from automl.cafa import CAFAConfig
from automl.carbon import CostModel
from automl.data import stratified_indices
from automl.datasets import (
    FashionDataset, FlowersDataset, EmotionsDataset, SkinCancerDataset,
)
from automl.meta_features import extract_meta_features
from automl.search_space import BACKBONES, suggest_config
from automl.train import train_config
from automl.utils import calculate_mean_std, set_seed, setup_logging

logger = logging.getLogger("ablation")

DATASETS = {
    "fashion": FashionDataset,
    "flowers": FlowersDataset,
    "emotions": EmotionsDataset,
    "skin_cancer": SkinCancerDataset,
}

# sampler / pruner / cost-cooling per arm
METHODS: Dict[str, Dict[str, Any]] = {
    "random":      {"sampler": "random", "pruner": "nop",       "cafa": False,
                    "label": "Random search (full fidelity)"},
    "tpe":         {"sampler": "tpe",    "pruner": "nop",       "cafa": False,
                    "label": "TPE BO (full fidelity)"},
    "tpe_hb":      {"sampler": "tpe",    "pruner": "hyperband", "cafa": False,
                    "label": "TPE + Hyperband (no cost-cooling)"},
    "greenvision": {"sampler": "tpe",    "pruner": "hyperband", "cafa": True,
                    "label": "GreenVision (CAFA multi-fidelity)"},
}

COLORS = {"random": "#B0413E", "tpe": "#3B6EA5", "tpe_hb": "#C77B30", "greenvision": "#1B7F5C"}



def subsample_stratified(indices, labels_all, n_total: int, seed: int):
    """Class-balanced subsample of `indices`, capped at ~n_total items.

    The ablation compares SEARCH STRATEGIES, not achievable accuracy, so a
    smaller training proxy is legitimate -- it buys the trial counts needed for
    the comparison to mean anything. Document this on the poster.
    """
    labels = np.array([labels_all[i] for i in indices])
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, n_total // len(classes))
    keep: List[int] = []
    for c in classes:
        pos = np.where(labels == c)[0]
        rng.shuffle(pos)
        keep.extend(pos[:per_class].tolist())
    keep.sort()
    return [indices[i] for i in keep]


def build_shared(dataset_class, cfg: Dict[str, Any], split_seed: int,
                 train_subset: Optional[int] = None,
                 val_subset: Optional[int] = None) -> Dict[str, Any]:
    """Meta-features, normalisation stats and the FIXED split.

    Computed once and reused by every arm. The split uses a single fixed seed
    across all methods and all repetitions so that arms are never advantaged by
    an easier validation set -- only the search strategy varies.
    """
    root = cfg["paths"]["data_root"]
    mf = extract_meta_features(dataset_class, root=root)
    mean, std = calculate_mean_std(dataset_class, root=root)
    base_train = dataset_class(root=root, split="train", download=False)
    train_idx, val_idx = stratified_indices(
        base_train._labels, float(cfg["data"]["val_fraction"]), split_seed
    )
    if train_subset:
        before = len(train_idx)
        train_idx = subsample_stratified(train_idx, base_train._labels,
                                         train_subset, split_seed)
        logger.info("Training proxy: %d -> %d images (class-balanced)",
                    before, len(train_idx))
    if val_subset:
        before = len(val_idx)
        val_idx = subsample_stratified(val_idx, base_train._labels,
                                       val_subset, split_seed)
        logger.info("Validation proxy: %d -> %d images (class-balanced)",
                    before, len(val_idx))
    logger.info("Shared split: %d train / %d val (%d classes)",
                len(train_idx), len(val_idx), int(dataset_class.num_classes))
    return {"mf": mf, "mean": mean, "std": std, "base_train": base_train,
            "train_idx": train_idx, "val_idx": val_idx, "root": root}


def make_study(method: str, seed: int, cfg: Dict[str, Any], tpe_startup: int):
    import optuna
    from optuna.pruners import HyperbandPruner, NopPruner
    from optuna.samplers import RandomSampler, TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec = METHODS[method]

    if spec["sampler"] == "random":
        sampler = RandomSampler(seed=seed)
    else:
        # n_startup_trials matters enormously here: with Optuna's default of 10,
        # TPE draws its first 10 configs at random from the same seed, so a
        # short run makes the TPE arms literally identical to random search.
        sampler = TPESampler(seed=seed, multivariate=True, group=True,
                             n_startup_trials=tpe_startup)

    if spec["pruner"] == "hyperband":
        min_r = int(cfg["search"]["min_epochs"])
        max_r = int(cfg["search"]["max_epochs"])
        eta = int(cfg["search"]["reduction_factor"])
        if max_r < min_r * eta * eta:
            logger.warning(
                "Hyperband is DEGENERATE: max_epochs=%d < min_epochs=%d * eta^2=%d. "
                "Fewer than 3 rungs means almost nothing gets pruned and the "
                "multi-fidelity arm collapses toward plain TPE.",
                max_r, min_r, min_r * eta * eta)
        pruner = HyperbandPruner(min_resource=min_r, max_resource=max_r,
                                 reduction_factor=eta)
    else:
        pruner = NopPruner()

    # In-memory: each (method, seed) is an independent replication.
    return optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)


# ------------------------------------------------------------------- one arm
def run_arm(method: str, seed: int, shared: Dict[str, Any], cfg: Dict[str, Any],
            budget_h: float, n_trials: int, tpe_startup: int,
            on_record) -> List[Dict[str, Any]]:
    """Run one (method, seed) replication under a wall-clock budget."""
    import optuna
    import torch

    spec = METHODS[method]
    set_seed(seed)

    budget_ceiling = int(cfg["search"]["max_epochs"])
    cafa_cfg = CAFAConfig(
        enabled=bool(spec["cafa"]),
        lambda0=float(cfg["cafa"]["lambda0"]),
        anneal_fraction=float(cfg["cafa"]["anneal_fraction"]),
        cooling=str(cfg["cafa"]["cooling"]),
    )
    cost_model = CostModel(BACKBONES)
    study = make_study(method, seed, cfg, tpe_startup)

    t0 = time.time()
    deadline = t0 + budget_h * 3600.0
    records: List[Dict[str, Any]] = []
    cum_e, cum_s = 0.0, 0.0
    best = 0.0

    def objective(trial) -> float:
        nonlocal cum_e, cum_s, best
        progress = min((time.time() - t0) / max(budget_h * 3600.0, 1e-9), 1.0)
        config = suggest_config(trial)


        cut = {"deadline": False}

        def report_cb(epoch: int, utility: float) -> bool:

            if spec["pruner"] != "nop":
                trial.report(utility, step=epoch)
                if trial.should_prune():
                    return True          # (a) genuine Hyperband pruning
            if time.time() >= deadline:
                cut["deadline"] = True   # (b) budget exhausted
                return True
            return False

        result = train_config(
            cfg=config,
            dataset_class=shared["dataset_class"],
            base_train=shared["base_train"],
            train_idx=shared["train_idx"],
            val_idx=shared["val_idx"],
            mean=shared["mean"], std=shared["std"],
            max_epochs=budget_ceiling,
            budget_ceiling=budget_ceiling,
            cafa=cafa_cfg,
            cost_model=cost_model,
            progress=progress,
            root=shared["root"],
            report_cb=report_cb,
            low_fidelity=False,
            track_energy=bool(cfg["carbon"]["enabled"]),
            seed=seed,
        )


        result.pop("_model", None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cum_e += float(result["energy_kwh"])
        cum_s += float(result["seconds"])
        best = max(best, float(result["val_accuracy"]))

        rec = {
            "method": method, "seed": seed, "trial": trial.number,
            "accuracy": float(result["val_accuracy"]),
            "macro_f1": float(result["val_macro_f1"]),
            "energy_kwh": float(result["energy_kwh"]),
            "energy_measured": bool(result.get("energy_measured", False)),
            "co2_kg": float(result["co2_kg"]),
            "seconds": float(result["seconds"]),
            "pruned": bool(result["pruned"]),
            "deadline_cut": bool(cut["deadline"]),
            "hb_pruned": bool(result["pruned"]) and not cut["deadline"],
            "resolution": int(result["resolution"]),
            "backbone": config["backbone"],
            "cum_energy_kwh": cum_e,
            "cum_hours": cum_s / 3600.0,
            "best_so_far": best,
        }
        records.append(rec)
        on_record(rec)   # checkpoint after EVERY trial

        tag = ("(hyperband-pruned)" if rec["hb_pruned"]
               else "(deadline-cut)" if rec["deadline_cut"] else "")
        logger.info("[%s seed=%d] trial %d acc=%.4f %s | cum %.4f kWh / %.2f h",
                    method, seed, trial.number, rec["accuracy"],
                    tag, cum_e, cum_s / 3600.0)

        if result["pruned"]:
            raise optuna.TrialPruned()
        return result["val_accuracy"]

    def stop_cb(study_, trial_):
        if time.time() >= deadline:
            study_.stop()

    try:
        study.optimize(objective, n_trials=n_trials,
                       timeout=max(deadline - time.time(), 1),
                       callbacks=[stop_cb], gc_after_trial=True)
    except KeyboardInterrupt:
        logger.warning("Interrupted; keeping %d completed trials.", len(records))

    logger.info("[%s seed=%d] DONE %d trials | best %.4f | %.4f kWh | %.2f h",
                method, seed, len(records), best, cum_e, cum_s / 3600.0)
    return records


# ----------------------------------------------------------------- aggregate
def step_curve(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Right-continuous step interpolation of a best-so-far curve onto `grid`."""
    idx = np.searchsorted(xs, grid, side="right") - 1
    out = np.full(grid.shape, np.nan)
    ok = idx >= 0
    out[ok] = ys[np.clip(idx[ok], 0, len(ys) - 1)]
    return out


def curves_for(records: List[Dict[str, Any]], method: str, xkey: str):
    """Per-seed (x, best_so_far) arrays for one method."""
    seeds = sorted({r["seed"] for r in records if r["method"] == method})
    out = []
    for s in seeds:
        rs = sorted([r for r in records if r["method"] == method and r["seed"] == s],
                    key=lambda r: r["trial"])
        if not rs:
            continue
        out.append((np.array([r[xkey] for r in rs], dtype=float),
                    np.array([r["best_so_far"] for r in rs], dtype=float)))
    return out


def energy_to_target(records: List[Dict[str, Any]], method: str,
                     target: float) -> List[float]:
    """Per-seed cumulative kWh at which best-so-far first reaches `target`."""
    vals = []
    for xs, ys in curves_for(records, method, "cum_energy_kwh"):
        hit = np.where(ys >= target)[0]
        vals.append(float(xs[hit[0]]) if len(hit) else float("nan"))
    return vals


def make_plot(records, xkey, xlabel, title, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plotted = False
    for method in METHODS:
        cs = curves_for(records, method, xkey)
        if not cs:
            continue
        # only interpolate over the span where EVERY seed has data, so the
        # mean/band are never computed from a partially-populated column
        xmin = max(float(xs[0]) for xs, _ in cs)
        xmax = min(float(xs[-1]) for xs, _ in cs)
        if xmax <= xmin:
            continue
        grid = np.linspace(xmin, xmax, 200)
        mat = np.vstack([step_curve(xs, ys, grid) for xs, ys in cs])
        mean = np.nanmean(mat, axis=0)
        lo, hi = np.nanmin(mat, axis=0), np.nanmax(mat, axis=0)
        c = COLORS[method]
        ax.plot(grid, mean, color=c, lw=2, label=METHODS[method]["label"])
        if mat.shape[0] > 1:
            ax.fill_between(grid, lo, hi, color=c, alpha=0.16, linewidth=0)
        plotted = True

    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Best validation accuracy so far")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", path)


def summarise(records, out_dir: Path) -> Dict[str, Any]:
    present = [m for m in METHODS if any(r["method"] == m for r in records)]
    per: Dict[str, Any] = {}
    for m in present:
        cs = curves_for(records, m, "cum_energy_kwh")
        finals = [float(ys[-1]) for _, ys in cs]
        tot_e = [float(xs[-1]) for xs, _ in cs]
        rs = [r for r in records if r["method"] == m]
        per[m] = {
            "label": METHODS[m]["label"],
            "n_seeds": len(cs),
            "trials_per_seed": [sum(1 for r in rs if r["seed"] == s)
                                for s in sorted({r["seed"] for r in rs})],
            "hyperband_pruned": sum(1 for r in rs if r.get("hb_pruned")),
            "deadline_cuts": sum(1 for r in rs if r.get("deadline_cut")),
            "final_accuracy_mean": float(np.mean(finals)) if finals else None,
            "final_accuracy_std": float(np.std(finals)) if finals else None,
            "total_energy_kwh_mean": float(np.mean(tot_e)) if tot_e else None,
            "energy_measured_fraction": round(
                sum(1 for r in rs if r["energy_measured"]) / max(len(rs), 1), 3),
        }

    all_finals = []
    for m in present:
        all_finals += [float(ys[-1])
                       for _, ys in curves_for(records, m, "cum_energy_kwh")]
    target = float(min(all_finals)) if all_finals else None
    if target is not None:
        for m in present:
            e = energy_to_target(records, m, target)
            e = [x for x in e if not np.isnan(x)]
            per[m]["energy_to_target_kwh_mean"] = float(np.mean(e)) if e else None

    summary = {"target_accuracy": target, "methods": per,
               "note": ("energy_to_target = cumulative kWh at which an arm first "
                        "reached the accuracy that ALL arms reached. Lower is better.")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["| Method | Trials/seed | HB-pruned | Deadline cuts "
             "| Final acc (mean±std) | Total kWh | kWh to reach target |",
             "|---|---|---|---|---|---|---|"]
    for m in present:
        p = per[m]
        ett = p.get("energy_to_target_kwh_mean")
        lines.append(
            f"| {p['label']} | {p['trials_per_seed']} | {p['hyperband_pruned']} "
            f"| {p['deadline_cuts']} "
            f"| {p['final_accuracy_mean']:.4f} ± {p['final_accuracy_std']:.4f} "
            f"| {p['total_energy_kwh_mean']:.5f} "
            f"| {'n/a' if ett is None else f'{ett:.5f}'} |")
    if target is not None:
        lines += ["", f"Target accuracy (reached by all arms): **{target:.4f}**"]
    md = "\n".join(lines)
    (out_dir / "summary.md").write_text(md + "\n")
    print("\n" + md + "\n")
    return summary


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="GreenVision ablation driver")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output-dir", default="outputs/ablation")
    ap.add_argument("--budget-hours", type=float, default=0.5,
                    help="wall-clock budget PER (method, seed) run")
    ap.add_argument("--n-trials", type=int, default=200,
                    help="trial cap per run; the budget normally binds first")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                    choices=list(METHODS.keys()))
    ap.add_argument("--val-subset", type=int, default=None,
                    help="cap validation images; once train is subsampled the "
                         "val pass dominates per-trial cost, so set this too")
    ap.add_argument("--train-subset", type=int, default=None,
                    help="cap training images (class-balanced) to buy trial count")
    ap.add_argument("--tpe-startup", type=int, default=5,
                    help="TPE random startup trials; Optuna's default of 10 makes "
                         "short TPE runs identical to random search")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override search.max_epochs (lower => more trials/arm)")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--split-seed", type=int, default=42,
                    help="fixed across all arms so the split never varies")
    ap.add_argument("--plots-only", action="store_true",
                    help="regenerate plots/summary from an existing records.json")
    args = ap.parse_args()

    setup_logging(logging.INFO)
    cfg = yaml.safe_load(open(args.config))
    if args.data_root:
        cfg["paths"]["data_root"] = args.data_root
    if args.max_epochs:
        cfg["search"]["max_epochs"] = args.max_epochs

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records.json"

    records: List[Dict[str, Any]] = []
    if rec_path.exists():
        records = json.loads(rec_path.read_text())
        logger.info("Resuming: %d existing trial records.", len(records))

    if not args.plots_only:
        dataset_class = DATASETS[args.dataset]
        shared = build_shared(dataset_class, cfg, args.split_seed,
                              args.train_subset, args.val_subset)
        shared["dataset_class"] = dataset_class

        runs = [(m, s) for s in args.seeds for m in args.methods]

 
        done_path = out_dir / "completed_runs.json"
        done = set()
        if done_path.exists():
            done = {(m, s) for m, s in json.loads(done_path.read_text())}
        n_before_filter = len(records)
        records[:] = [r for r in records if (r["method"], r["seed"]) in done]
        if len(records) != n_before_filter:
            logger.warning("Discarded %d trial(s) from interrupted run(s); "
                           "those arms will be re-run in full.",
                           n_before_filter - len(records))
            rec_path.write_text(json.dumps(records, indent=1))
        todo = [r for r in runs if r not in done]
        est = len(todo) * args.budget_hours
        logger.info("%d/%d runs remaining -- approx %.1f h of GPU time.",
                    len(todo), len(runs), est)
        if est > 11:
            logger.warning("Estimated %.1f h exceeds a typical Colab session. "
                           "This script resumes, so re-run after a disconnect.", est)

        def checkpoint(rec):
            records.append(rec)
            rec_path.write_text(json.dumps(records, indent=1))

        for method, seed in todo:
            logger.info("=" * 68)
            logger.info("RUN  method=%s  seed=%d  budget=%.2f h",
                        method, seed, args.budget_hours)
            n_before = len(records)
            try:
                run_arm(method, seed, shared, cfg, args.budget_hours,
                        args.n_trials, args.tpe_startup, checkpoint)
                done.add((method, seed))
                done_path.write_text(json.dumps(sorted(done)))
            except Exception as e:
                logger.exception("Run %s/seed%d failed: %s", method, seed, e)
                # roll back a partial run so resume retries it cleanly
                del records[n_before:]
                rec_path.write_text(json.dumps(records, indent=1))

    if not records:
        logger.error("No records to plot.")
        return

    make_plot(records, "cum_energy_kwh", "Cumulative energy (kWh)",
              "Search efficiency: accuracy per unit of energy",
              out_dir / "accuracy_vs_energy.png")
    make_plot(records, "cum_hours", "Cumulative GPU wall-clock (hours)",
              "Search efficiency: accuracy per unit of time",
              out_dir / "accuracy_vs_time.png")
    summarise(records, out_dir)
    logger.info("Ablation artifacts in %s", out_dir)


if __name__ == "__main__":
    main()
