# GreenVision
### Carbon-Aware Fidelity Allocation in Multi-Fidelity Bayesian Optimisation
 
A one-click AutoML pipeline for image classification that treats **energy consumption as a first-class objective, not an afterthought.** Given a vision dataset and a compute budget, it searches for a hyperparameter configuration that maximises accuracy while actively steering search effort away from carbon-expensive dead ends.

```
python run.py --mode search   # discovers the best configuration
python run.py --mode final    # trains the best config and produces test predictions
```

---
 
## The Idea
 
Standard AutoML treats every candidate configuration equally, regardless of how expensive it is to evaluate. A large, power-hungry architecture and a small, efficient one get the same shot at the search budget — even if the expensive one was never going to win.
 
**CAFA (Carbon-Aware Fidelity Allocation)** fixes this by changing what Optuna's Hyperband pruner actually prunes on. Instead of comparing trials on raw accuracy, it compares them on a cost-cooled utility:
 
```
utility(t) = accuracy − λ(t) × normalised_cost
 
λ(t) = λ₀ × 0.5 × (1 + cos(π × t / 0.6))   for t < 0.6
λ(t) = 0                                     for t ≥ 0.6
      (λ₀ = 0.5)
```

**★ Three novel contributions** — CAFA cost-cooled fidelity allocation (`cafa.py`),
imbalance-aware low-fidelity proxies (`data.py`), portfolio-to-prior meta warm-start
(`warm_start.py`).

## Repository layout
```
greenvision/     
├── run.py                      # CLI: --mode search | final
├── configs/default.yaml        # all budget/search/CAFA/warm-start knobs
├── portfolios/portfolio.json   # meta warm-start portfolio (populated offline)
├── download_datasets.py        # dataset downloader
└── src/automl/
    ├── automl.py               # orchestrator: fit / final_train / predict
    ├── datasets.py             # dataset classes
    ├── meta_features.py        # Stage 1
    ├── data.py                 # Stage 2 + imbalance proxy + samplers
    ├── search_space.py         # Stage 3
    ├── models.py               # backbones (resnet18 / efficientnet_b0 / mobilenetv3)
    ├── carbon.py               # CodeCarbon wrapper + CAFA cost model
    ├── cafa.py                 # carbon-aware fidelity allocation
    ├── train.py                # per-config training with epoch/resolution fidelity
    ├── warm_start.py           # portfolio-to-prior warm-start
    ├── search.py               # Stage 5+6 Optuna study
    ├── interpret.py            # Stage 9 plots + carbon report
    └── utils.py
```

## Notes
* Optuna pruners require single-objective studies, so the search optimizes accuracy with
  Hyperband, and the **Pareto front is reconstructed post-hoc** from logged macro-F1/energy.
* CodeCarbon, grad-cam and fanova are optional; the pipeline degrades gracefully without them.
* On Colab install with `--no-deps` so pip never replaces the CUDA torch build.
