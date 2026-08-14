# Reproducibility Instructions

These two commands reproduce our final test predictions on the `skin_cancer`
(final-exam) dataset.

## Environment
```bash
python3 -m venv venv && source venv/bin/activate
pip install -e .
python download_datasets.py
```
Python ≥ 3.10. GPU recommended (single T4 sufficient). Exact pinned versions are in
`pyproject.toml`. On Colab, install with `--no-deps` plus `requirements-colab.txt` so the preinstalled CUDA torch build is not replaced by a CPU wheel.

## Command 1 — Search (returns a configuration; respects the ≤24h budget)
```bash
python run.py --dataset skin_cancer --mode search --seed 42 \
    --budget-hours 4 --n-trials 60 \
    --optuna-storage "sqlite:///./outputs/study.db" \
    --output-dir outputs
```
Output: `outputs/best_config.json` (the selected hyperparameter configuration), plus
`outputs/carbon_report.json`, `outputs/pareto_front.png`,
`outputs/accuracy_vs_energy.png`.

The search self-terminates at `--budget-hours`. Because the Optuna study is persisted to
the SQLite file, re-running the same command **resumes** the study rather than restarting.

## Command 2 — Final training + prediction (excluded from the 24h budget)
```bash
python run.py --dataset skin_cancer --mode final \
    --output-dir outputs
```
Output: `final_test_preds.npy` (one predicted class per test input, in the order of
`X_test`) and a copy at `data/exam_dataset/predictions.npy` for the GitHub autograder.

## Budget accounting
* **Counted toward 24h:** Command 1 (search). Its wall-clock is logged in
  `outputs/carbon_report.json → search_wall_clock_hours`.
* **Excluded per exam rules:** the offline meta warm-start portfolio (built on practice
  datasets only) and Command 2 (final model training + evaluation).

## Determinism
Seeds are set for python/numpy/torch. We keep `cudnn.benchmark=True` for throughput; set
`torch.backends.cudnn.deterministic=True` and `benchmark=False` in `utils.set_seed` for
bitwise reproducibility at some speed cost.
