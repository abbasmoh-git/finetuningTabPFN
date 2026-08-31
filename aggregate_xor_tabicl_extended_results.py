"""
aggregate_xor_tabicl_extended_results.py
--------------------------------------------
Combines the 5 partial per-seed result files produced by the TabICL v2
selective fine-tuning SLURM array
(results/xor_seed_sweep_tabicl_extended_n800/partial/seed{1..5}_results.pkl)
with the EXISTING baseline/full-finetuning results from
results/xor_seed_sweep_tabicl_n800/results.pkl.

The original results.pkl is opened READ-ONLY. Its baseline (no_finetuning)
and full_finetuning entries are copied into the new combined file purely
for comparison -- nothing in results/xor_seed_sweep_tabicl_n800/ is ever
written to or modified.

Output (all NEW, under a NEW directory):

  results/xor_seed_sweep_tabicl_extended_n800/
    results.pkl   -- per seed: config, no_finetuning (copied), full_finetuning
                       (copied), attention_only, mlp_only, layerwise_icl0,
                       layerwise_icl3, layerwise_icl6, layerwise_icl8,
                       layerwise_icl11 (all new, from the partial files)
    summary.md     -- Accuracy / Balanced Accuracy / ROC-AUC / Negative Log
                       Loss + deltas vs. baseline, per seed and strategy,
                       plus mean deltas across the 5 seeds. No scientific
                       interpretation is added.

Usage
-----
    python aggregate_xor_tabicl_extended_results.py
(run only after all 5 SLURM array tasks have finished --
 submit_xor_tabicl_selective_array.sh)
"""

import pickle
from pathlib import Path

import numpy as np

ORIGINAL_RESULTS = Path("results/xor_seed_sweep_tabicl_n800/results.pkl")
PARTIAL_DIR = Path("results/xor_seed_sweep_tabicl_extended_n800/partial")
OUTPUT_DIR = Path("results/xor_seed_sweep_tabicl_extended_n800")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 2, 3, 4, 5]
DATASET_NAMES = [f"xor_seed{s}_n800" for s in SEEDS]

SELECTIVE_STRATEGIES = [
    "attention_only", "mlp_only",
    "layerwise_icl0", "layerwise_icl3", "layerwise_icl6", "layerwise_icl8", "layerwise_icl11",
]


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    if not ORIGINAL_RESULTS.exists():
        raise FileNotFoundError(
            f"{ORIGINAL_RESULTS} not found -- run the original TabICL sweep "
            f"(run_xor_seed_sweep_tabicl.py) first. This script only reads "
            f"that file, never writes to it."
        )
    original = load_pickle(ORIGINAL_RESULTS)  # read-only use throughout

    combined = {}
    for name in DATASET_NAMES:
        if name not in original:
            raise KeyError(f"{name} missing from {ORIGINAL_RESULTS} -- cannot combine.")
        # Copy ONLY the two existing strategies, for comparison. Nothing
        # else from `original` is referenced or modified.
        combined[name] = {
            "config": original[name]["config"],
            "no_finetuning": original[name]["no_finetuning"],
            "full_finetuning": original[name]["full_finetuning"],
        }

    missing_partials = []
    for s in SEEDS:
        partial_path = PARTIAL_DIR / f"seed{s}_results.pkl"
        name = f"xor_seed{s}_n800"
        if not partial_path.exists():
            missing_partials.append(partial_path)
            continue
        partial = load_pickle(partial_path)
        seed_data = partial[name]
        for strategy in SELECTIVE_STRATEGIES:
            combined[name][strategy] = seed_data[strategy]

    if missing_partials:
        print("WARNING: the following partial result files are missing -- "
              "summary.md will be incomplete for the affected seed(s):")
        for p in missing_partials:
            print(f"  {p}")

    results_path = OUTPUT_DIR / "results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(combined, f)
    print(f"Combined results written to: {results_path}")

    write_summary(combined)


def write_summary(combined: dict) -> None:
    lines = []
    lines.append("# XOR TabICL v2 Extended Selective Fine-Tuning -- Seed Sweep (n=800)\n")
    lines.append(
        "`no_finetuning` and `full_finetuning` columns are copied UNCHANGED "
        "from results/xor_seed_sweep_tabicl_n800/results.pkl (read-only, "
        "not recomputed). The 7 selective strategies (attention_only, "
        "mlp_only, layerwise_icl0/3/6/8/11) are new, from the SLURM array "
        "in this extended experiment.\n"
    )
    lines.append(
        "Fine-tuning hyperparameters, unchanged from the existing TabICL "
        "sweep: epochs=50, learning_rate=1e-4, query_ratio=0.3, "
        "weight_decay=0.01, grad_clip=1.0, warmup_proportion=0.1, "
        "patience=20, random_state=42.\n"
    )

    strategy_order = ["full_finetuning"] + SELECTIVE_STRATEGIES
    per_strategy_deltas = {s: {"acc": [], "bal_acc": [], "roc_auc": [], "neg_logloss": []}
                           for s in strategy_order}

    lines.append(
        "| Dataset | Strategy | Accuracy | Bal. Accuracy | ROC-AUC | Neg. Log Loss | "
        "Δ Accuracy | Δ Bal. Accuracy | Δ ROC-AUC | Δ Neg. Log Loss |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for name in DATASET_NAMES:
        res = combined[name]
        base = res["no_finetuning"]["test"]
        base_neg_ll = -base["logloss"]
        lines.append(
            f"| {name} | baseline | {base['accuracy']:.4f} | {base['balanced_accuracy']:.4f} | "
            f"{base['roc_auc']:.4f} | {base_neg_ll:.4f} | - | - | - | - |"
        )
        for strategy in strategy_order:
            if strategy not in res:
                continue
            m = res[strategy]["test"]
            neg_ll = -m["logloss"]
            d_acc = m["accuracy"] - base["accuracy"]
            d_bal = m["balanced_accuracy"] - base["balanced_accuracy"]
            d_roc = m["roc_auc"] - base["roc_auc"]
            d_negll = neg_ll - base_neg_ll
            lines.append(
                f"| {name} | {strategy} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | "
                f"{m['roc_auc']:.4f} | {neg_ll:.4f} | {d_acc:+.4f} | {d_bal:+.4f} | "
                f"{d_roc:+.4f} | {d_negll:+.4f} |"
            )
            per_strategy_deltas[strategy]["acc"].append(d_acc)
            per_strategy_deltas[strategy]["bal_acc"].append(d_bal)
            per_strategy_deltas[strategy]["roc_auc"].append(d_roc)
            per_strategy_deltas[strategy]["neg_logloss"].append(d_negll)

    lines.append("")
    lines.append("## Mean Δ across the 5 seeds\n")
    lines.append(
        "| Strategy | Mean Δ Accuracy | Mean Δ Bal. Accuracy | Mean Δ ROC-AUC | "
        "Mean Δ Neg. Log Loss | n seeds |"
    )
    lines.append("|---|---|---|---|---|---|")
    for strategy in strategy_order:
        d = per_strategy_deltas[strategy]
        n = len(d["acc"])
        if n == 0:
            lines.append(f"| {strategy} | n/a | n/a | n/a | n/a | 0 |")
            continue
        lines.append(
            f"| {strategy} | {np.mean(d['acc']):.4f} | {np.mean(d['bal_acc']):.4f} | "
            f"{np.mean(d['roc_auc']):.4f} | {np.mean(d['neg_logloss']):.4f} | {n} |"
        )
    lines.append("")

    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
