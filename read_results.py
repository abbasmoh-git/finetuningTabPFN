"""
read_results.py
---------------
Quick summary of all experiment results stored under results/.
Prints one row per dataset/method with mean test metrics across folds/repeats.

Usage (from repo root):
    python read_results.py
"""

import pickle
from pathlib import Path

results_root = Path("results/finetuning_experiments")

header = (
    f"{'dataset':<35} {'method':<20} "
    f"{'acc':>6} {'bal_acc':>8} {'roc_auc':>8} {'logloss':>8}"
)
print(header)
print("-" * len(header))

for pkl_file in sorted(results_root.rglob("*.pkl")):
    with open(pkl_file, "rb") as f:
        results = pickle.load(f)

    for dataset_name, entry in results.get("datasets", {}).items():
        method = entry.get("method", "?")
        summary = entry.get("summary", {}).get("test", {})
        if not summary:
            continue
        print(
            f"{dataset_name:<35} {method:<20} "
            f"{summary['accuracy']['mean']:>6.4f} "
            f"{summary['balanced_accuracy']['mean']:>8.4f} "
            f"{summary['roc_auc']['mean']:>8.4f} "
            f"{summary['logloss']['mean']:>8.4f}"
        )
