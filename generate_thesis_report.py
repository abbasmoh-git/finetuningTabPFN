"""
generate_thesis_report.py
--------------------------
Generates ready-to-use tables (Markdown, paste into Word) and plots for
Chapter 5 (Experimental Results) of the thesis, following the structure
agreed on with the supervisor / discussed with ChatGPT:

  Table 5.1   Experimental coverage (which configs, how many datasets, LR)
  Table       Full fine-tuning / each selective variant vs baseline
              (per metric: mean baseline, mean variant, mean difference,
              wins/ties/losses)
  Table 5.5   Overall comparison across all strategies (mean + median delta
              per metric)
  Plot 1      Boxplots of per-dataset deltas vs baseline, one panel per
              metric, one box per strategy
  Plot 2      Stacked bar chart of wins/ties/losses (based on accuracy) per
              strategy

Works with however many datasets are currently finished -- rerun any time,
including before all 31 datasets are complete, to sanity-check the output.

Outputs (under results/thesis_report/):
  tables.md               -- all tables, Markdown format (paste into Word/Overleaf)
  deltas_boxplot.png
  wins_ties_losses.png

Usage (same environment as read_results.py / compare_to_baseline.py):
    python generate_thesis_report.py
"""

import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display available on cluster/login node
import matplotlib.pyplot as plt
import numpy as np

results_root = Path("results/finetuning_experiments")
output_dir = Path("results/thesis_report")
output_dir.mkdir(parents=True, exist_ok=True)

TIE_THRESHOLD = 0.005  # |delta| below this counts as a tie, not a win/loss
BASELINE_METHOD = "no_finetuning"

# (internal key, display label, higher_is_better)
METRICS = [
    ("acc", "Accuracy", True),
    ("bal_acc", "Balanced Accuracy", True),
    ("roc_auc", "ROC-AUC", True),
    ("logloss", "Log Loss", False),
]


def load_all_runs():
    """Return (runs, configs).

    runs:    {experiment_name: {dataset_name: {metric_key: mean, ..., 'method': str}}}
    configs: {experiment_name: config_dict}  (as saved inside the pickle, used
              to read back e.g. the learning rate actually used)
    """
    runs: dict = {}
    configs: dict = {}
    for pkl_file in sorted(results_root.rglob("*.pkl")):
        try:
            experiment = pkl_file.relative_to(results_root).parts[0]
        except ValueError:
            continue
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)
        runs.setdefault(experiment, {})
        if "config" in results:
            configs[experiment] = results["config"]
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
    return runs, configs


def find_baseline_experiment(runs: dict) -> str:
    for experiment, datasets in runs.items():
        for d in datasets.values():
            if d["method"] == BASELINE_METHOD:
                return experiment
    raise RuntimeError(
        f"No experiment with method='{BASELINE_METHOD}' found under "
        f"{results_root} -- did the baseline run finish?"
    )


def display_name(experiment: str) -> str:
    """Map a results folder name to a human-readable strategy name."""
    m = re.match(r"layerwise_layer(\d+)", experiment)
    if m:
        return f"Layer-wise (layer {m.group(1)})"
    return {
        "own_finetuning": "Full fine-tuning",
        "attention_only": "Attention-only",
        "mlp_only": "MLP-only",
    }.get(experiment, experiment)


def get_learning_rate(configs: dict, experiment: str):
    cfg = configs.get(experiment, {})
    return cfg.get("finetuning_hyperparams", {}).get("learning_rate", "-")


def per_dataset_deltas(baseline: dict, variant: dict, metric: str, higher_is_better: bool) -> dict:
    """Positive delta = variant is better than baseline on this metric."""
    common = sorted(set(baseline) & set(variant))
    deltas = {}
    for name in common:
        b = baseline[name][metric]
        v = variant[name][metric]
        deltas[name] = (v - b) if higher_is_better else (b - v)
    return deltas


def summarize(deltas: dict) -> dict:
    values = list(deltas.values())
    wins = sum(1 for v in values if v > TIE_THRESHOLD)
    losses = sum(1 for v in values if v < -TIE_THRESHOLD)
    ties = len(values) - wins - losses
    mean = sum(values) / len(values) if values else float("nan")
    median = float(np.median(values)) if values else float("nan")
    return {"wins": wins, "losses": losses, "ties": ties, "mean": mean, "median": median, "n": len(values)}


def fmt(x, digits=4) -> str:
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def main():
    runs, configs = load_all_runs()
    baseline_experiment = find_baseline_experiment(runs)
    baseline = runs[baseline_experiment]

    variant_experiments = [
        e for e in sorted(runs)
        if e != baseline_experiment and len(runs[e]) >= 5  # skip stray/old folders
    ]

    lines = []

    # --- Table 5.1: Experimental coverage ---
    lines.append("## Table 5.1 -- Experimental Coverage\n")
    lines.append("| Configuration | Model | Completed datasets | Learning rate |")
    lines.append("|---|---|---|---|")
    lines.append(f"| No fine-tuning | TabPFN v3 | {len(baseline)} | - |")
    for e in variant_experiments:
        lr = get_learning_rate(configs, e)
        lines.append(f"| {display_name(e)} | TabPFN v3 | {len(runs[e])} | {lr} |")
    lines.append("")

    # --- Per-variant tables ---
    all_deltas: dict = {}
    all_summaries: dict = {}

    for e in variant_experiments:
        all_deltas[e] = {}
        all_summaries[e] = {}
        common = sorted(set(baseline) & set(runs[e]))

        lines.append(f"## Table -- {display_name(e)} vs Baseline\n")
        lines.append("| Metric | Mean baseline | Mean variant | Mean difference | Wins | Ties | Losses |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, label, higher_is_better in METRICS:
            deltas = per_dataset_deltas(baseline, runs[e], key, higher_is_better)
            all_deltas[e][key] = deltas
            s = summarize(deltas)
            all_summaries[e][key] = s
            mean_baseline = sum(baseline[d][key] for d in common) / len(common) if common else float("nan")
            mean_variant = sum(runs[e][d][key] for d in common) / len(common) if common else float("nan")
            lines.append(
                f"| {label} | {fmt(mean_baseline)} | {fmt(mean_variant)} | {fmt(s['mean'], 5)} | "
                f"{s['wins']} | {s['ties']} | {s['losses']} |"
            )
        lines.append(f"\n(compared on {len(common)} datasets present in both baseline and this run)\n")

    # --- Table 5.5: Overall comparison (mean deltas) ---
    lines.append("## Table 5.5 -- Overall Comparison of Fine-Tuning Strategies (mean)\n")
    lines.append("| Strategy | Datasets | Mean Δ Accuracy | Mean Δ Bal. Acc. | Mean Δ ROC-AUC | Mean Δ Log Loss |")
    lines.append("|---|---|---|---|---|---|")
    for e in variant_experiments:
        n = all_summaries[e]["acc"]["n"]
        row = f"| {display_name(e)} | {n} "
        for key, _, _ in METRICS:
            row += f"| {fmt(all_summaries[e][key]['mean'], 5)} "
        row += "|"
        lines.append(row)
    lines.append("")

    # --- Table 5.5b: median deltas ---
    lines.append("## Table 5.5b -- Overall Comparison of Fine-Tuning Strategies (median)\n")
    lines.append("| Strategy | Median Δ Accuracy | Median Δ Bal. Acc. | Median Δ ROC-AUC | Median Δ Log Loss |")
    lines.append("|---|---|---|---|---|")
    for e in variant_experiments:
        row = f"| {display_name(e)} "
        for key, _, _ in METRICS:
            row += f"| {fmt(all_summaries[e][key]['median'], 5)} "
        row += "|"
        lines.append(row)
    lines.append("")

    # --- Table 5.6: Common-subset comparison (fair, same datasets for all) ---
    # Only datasets present in the baseline AND every fine-tuning variant --
    # the strictest possible comparison, so no strategy is compared on an
    # easier or harder subset of datasets than any other.
    common_all = set(baseline)
    for e in variant_experiments:
        common_all &= set(runs[e])
    common_all = sorted(common_all)

    lines.append(
        "## Table 5.6 -- Common-Subset Comparison "
        f"(only datasets present in baseline + all {len(variant_experiments)} strategies, "
        f"n={len(common_all)})\n"
    )
    if len(common_all) < 5:
        lines.append(
            "*(Too few common datasets to report -- fewer than 5 datasets have "
            "results for every strategy simultaneously.)*\n"
        )
    else:
        lines.append("| Strategy | Mean Δ Accuracy | Mean Δ Bal. Acc. | Mean Δ ROC-AUC | Mean Δ Log Loss |")
        lines.append("|---|---|---|---|---|")
        common_summaries: dict = {}
        for e in variant_experiments:
            common_summaries[e] = {}
            row = f"| {display_name(e)} "
            for key, label, higher_is_better in METRICS:
                deltas = {
                    name: all_deltas[e][key][name]
                    for name in common_all
                    if name in all_deltas[e][key]
                }
                s = summarize(deltas)
                common_summaries[e][key] = s
                row += f"| {fmt(s['mean'], 5)} "
            row += "|"
            lines.append(row)
        lines.append("")

        lines.append("| Strategy | Wins | Ties | Losses | (Accuracy, on common subset) |")
        lines.append("|---|---|---|---|---|")
        for e in variant_experiments:
            s = common_summaries[e]["acc"]
            lines.append(f"| {display_name(e)} | {s['wins']} | {s['ties']} | {s['losses']} | n={s['n']} |")
        lines.append("")

        lines.append(f"Datasets included in the common subset: {', '.join(common_all)}\n")

    tables_path = output_dir / "tables.md"
    tables_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Tables written to: {tables_path}")

    # --- Plot 1: Boxplots of per-dataset deltas, one panel per metric ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    labels = [display_name(e) for e in variant_experiments]
    for ax, (key, label, _) in zip(axes, METRICS):
        data = [list(all_deltas[e][key].values()) or [0.0] for e in variant_experiments]
        ax.boxplot(data)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_ylabel(f"{label} difference vs baseline")
    plt.tight_layout()
    plot1_path = output_dir / "deltas_boxplot.png"
    plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Boxplot saved to: {plot1_path}")

    # --- Plot 2: Stacked bar chart of wins/ties/losses (accuracy) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    wins = [all_summaries[e]["acc"]["wins"] for e in variant_experiments]
    ties = [all_summaries[e]["acc"]["ties"] for e in variant_experiments]
    losses = [all_summaries[e]["acc"]["losses"] for e in variant_experiments]
    x = np.arange(len(labels))
    ax.bar(x, wins, label="Wins", color="#4CAF50")
    ax.bar(x, ties, bottom=wins, label="Ties", color="#9E9E9E")
    bottom2 = [w + t for w, t in zip(wins, ties)]
    ax.bar(x, losses, bottom=bottom2, label="Losses", color="#E53935")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Number of datasets (based on accuracy)")
    ax.set_title("Wins / Ties / Losses vs Baseline (Accuracy)")
    ax.legend()
    plt.tight_layout()
    plot2_path = output_dir / "wins_ties_losses.png"
    plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved to: {plot2_path}")

    print("\nDone. Copy results/thesis_report/tables.md into your thesis, "
          "and insert the two PNGs as figures.")


if __name__ == "__main__":
    main()
