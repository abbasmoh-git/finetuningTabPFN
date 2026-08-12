"""
compare_to_baseline.py
-----------------------
Proper statistical comparison of each fine-tuning variant against the
no-fine-tuning baseline (TabPFNv3), instead of eyeballing the raw table.

For each experiment (attention_only, mlp_only, layerwise_layer0,
own_finetuning) this script:
  1. Loads all result pickles under results/finetuning_experiments/.
  2. Deduplicates datasets within each experiment/baseline (if a dataset
     appears more than once -- e.g. from a resumed/re-submitted job -- the
     last occurrence found is used; values are expected to be near-identical
     across duplicates, being the same experiment re-run).
  3. Joins each variant to the baseline ONLY on datasets present in both
     (datasets missing from either side, e.g. jobs still running or
     time-limited out, are excluded and reported separately).
  4. Computes per-dataset deltas for accuracy, balanced accuracy, and
     ROC-AUC (higher = better) and log loss (lower = better).
  5. Prints win / loss / tie counts (tie = |delta| < TIE_THRESHOLD, i.e.
     within noise) and the average delta per metric per experiment.

Usage (from repo root, same environment you used for read_results.py):
    python compare_to_baseline.py
"""

import pickle
from pathlib import Path

results_root = Path("results/finetuning_experiments")

# Deltas smaller than this are treated as "no real difference" (noise-level),
# not a genuine win or loss. Adjust if you want a stricter/looser cutoff.
TIE_THRESHOLD = 0.005  # 0.5 percentage points for acc/bal_acc/roc_auc

BASELINE_EXPERIMENT_METHOD = "no_finetuning"


def load_all_runs():
    """Return {experiment_name: {dataset_name: summary_dict}}.

    summary_dict has keys accuracy, balanced_accuracy, roc_auc, logloss,
    each itself a dict with a 'mean' key (matches summarize_results() output
    in main.py). Later files overwrite earlier ones per (experiment,
    dataset) -- i.e. duplicates are resolved by "last one wins", which is
    fine since duplicates come from re-running/resuming the same config.
    """
    runs: dict = {}
    for pkl_file in sorted(results_root.rglob("*.pkl")):
        try:
            experiment = pkl_file.relative_to(results_root).parts[0]
        except ValueError:
            continue
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)
        runs.setdefault(experiment, {})
        for dataset_name, entry in results.get("datasets", {}).items():
            summary = entry.get("summary", {}).get("test", {})
            if not summary:
                continue
            runs[experiment][dataset_name] = {
                "method": entry.get("method", "?"),
                "acc": summary["accuracy"]["mean"],
                "bal_acc": summary["balanced_accuracy"]["mean"],
                "roc_auc": summary["roc_auc"]["mean"],
                "logloss": summary["logloss"]["mean"],
            }
    return runs


def find_baseline_experiment(runs: dict) -> str:
    """Find the experiment folder that actually holds the no_finetuning
    baseline results (its method field is 'no_finetuning'), rather than
    assuming a fixed folder name."""
    for experiment, datasets in runs.items():
        for d in datasets.values():
            if d["method"] == BASELINE_EXPERIMENT_METHOD:
                return experiment
    raise RuntimeError(
        "No experiment with method='no_finetuning' found under "
        f"{results_root} -- did the baseline run actually finish?"
    )


def compare(baseline: dict, variant: dict, metric: str, higher_is_better: bool):
    common = sorted(set(baseline) & set(variant))
    missing_from_variant = sorted(set(baseline) - set(variant))
    wins = losses = ties = 0
    deltas = []
    for name in common:
        b = baseline[name][metric]
        v = variant[name][metric]
        delta = (v - b) if higher_is_better else (b - v)  # positive = variant better
        deltas.append(delta)
        if abs(delta) < TIE_THRESHOLD:
            ties += 1
        elif delta > 0:
            wins += 1
        else:
            losses += 1
    avg_delta = sum(deltas) / len(deltas) if deltas else float("nan")
    return {
        "n_common": len(common),
        "n_missing": len(missing_from_variant),
        "missing": missing_from_variant,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "avg_delta": avg_delta,
    }


def main():
    runs = load_all_runs()
    baseline_experiment = find_baseline_experiment(runs)
    baseline = runs[baseline_experiment]
    print(f"Baseline experiment folder: '{baseline_experiment}' "
          f"({len(baseline)} datasets)\n")

    metrics = [
        ("acc", True),
        ("bal_acc", True),
        ("roc_auc", True),
        ("logloss", False),  # lower is better
    ]

    for experiment in sorted(runs):
        if experiment == baseline_experiment:
            continue
        # Only compare experiments that actually contain fine-tuned runs
        # (skip stray/unrelated folders such as old single-dataset LR sweeps).
        if len(runs[experiment]) < 5:
            continue

        print(f"=== {experiment} vs baseline ===")
        for metric, higher_is_better in metrics:
            r = compare(baseline, runs[experiment], metric, higher_is_better)
            direction = "higher=better" if higher_is_better else "lower=better"
            print(
                f"  {metric:<10} ({direction}): "
                f"wins={r['wins']:>2} losses={r['losses']:>2} ties={r['ties']:>2} "
                f"| avg delta={r['avg_delta']:+.4f} "
                f"| compared on {r['n_common']} datasets "
                f"({r['n_missing']} missing from this experiment)"
            )
        missing = compare(baseline, runs[experiment], "acc", True)["missing"]
        if missing:
            print(f"  [missing datasets, not yet finished/compared]: {', '.join(missing)}")
        print()


if __name__ == "__main__":
    main()
