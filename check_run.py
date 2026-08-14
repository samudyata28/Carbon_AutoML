#!/usr/bin/env python3
"""Post-run validation.

"""
import json
import sys
from collections import Counter
from pathlib import Path

import optuna

out_dir, storage, req_budget = sys.argv[1], sys.argv[2], float(sys.argv[3])

report = json.loads((Path(out_dir) / "carbon_report.json").read_text())
names = optuna.get_all_study_names(storage)
study = optuna.load_study(study_name=names[0], storage=storage)
states = Counter(t.state.name for t in study.trials)

passed = []


def gate(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name:26s} {detail}")
    passed.append(bool(cond))


gate("report scoping",
     report["trial_wall_hours"] <= report["search_wall_clock_hours"],
     f'{report["trial_wall_hours"]}h trials / '
     f'{report["search_wall_clock_hours"]}h wall')

gate("budget discipline",
     report["search_wall_clock_hours"] <= 1.10 * req_budget,
     f'{report["search_wall_clock_hours"]}h vs {req_budget}h requested')

gate("fresh study",
     report["study_cumulative_trials"] == report["completed_trials"],
     f'{report["study_cumulative_trials"]} in db / '
     f'{report["completed_trials"]} this run')

gate("pruning active",
     states.get("PRUNED", 0) > 0,
     str(dict(states)))

backbones = Counter(t.params.get("backbone")
                    for t in study.trials if "backbone" in t.params)
gate("backbone diversity", len(backbones) >= 2, str(dict(backbones)))

for artifact in ["best_config.json", "carbon_report.json",
                 "pareto_front.png", "accuracy_vs_energy.png"]:
    gate(f"artifact {artifact[:16]}", (Path(out_dir) / artifact).exists())

print()
print(f"INFO  energy_measured_fraction = {report['energy_measured_fraction']} "
      f"({'MEASURED' if report['energy_measured_fraction'] > 0.5 else 'MODELLED -- must be disclosed on the poster'})")
print(f"INFO  best val accuracy        = {study.best_value:.4f}")
print(f"INFO  search overhead          = {report['search_overhead_hours']}h "
      f"(data loading, meta-features, plotting)")

print()
print("ALL GATES PASS" if all(passed) else "GATES FAILED -- do not proceed")
sys.exit(0 if all(passed) else 1)
