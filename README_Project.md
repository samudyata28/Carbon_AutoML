### Carbon-Aware Fidelity Allocation in Multi-Fidelity Bayesian Optimisation
*AutoML Machine Learning Lab — SS2026 · Vision modality*

Carbon-aware multi-fidelity Bayesian Optimization over fine-tuning recipes for image
classification. Optimizes top-1 accuracy on a hidden test set,
and treats energy/CO₂ as a measured, first-class property of the search.

## Install (local / cluster)
```bash
python3 -m venv venv && source venv/bin/activate
pip install -e .
python -c "import automl; print(automl.__version__)"
```
## Data
```bash
python download_datasets.py     
```

## Usage — two commands (submission spec)
```bash
# 1) SEARCH — carbon-aware multi-fidelity BO.
python run.py --dataset skin_cancer --mode search --seed 42 \
    --budget-hours 12 --n-trials 40 \
    --optuna-storage "sqlite:////content/drive/MyDrive/greenvision/study.db"

# 2) FINAL — retrain the best config at full fidelity and predict.
python run.py --dataset skin_cancer --mode final \
    --config outputs/best_config.json --output-path final_test_preds.npy
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
