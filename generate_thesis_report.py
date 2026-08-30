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
from scipy import stats

results_root = Path("results/finetuning_experiments")
output_dir = Path("results/thesis_report")
output_dir.mkdir(parents=True, exist_ok=True)

TIE_THRESHOLD = 0.005  # |delta| below this counts as a tie, not a win/loss
BASELINE_METHOD = "no_finetuning"

# (internal key, display label, higher_is_better)
#
# Log Loss is reported as NEGATIVE log loss (-log_loss) so that, like the
# other three metrics, "higher is better". This gives every metric in this
# file a single, uniform delta convention:
#     delta = fine_tuned - baseline   (positive delta = fine-tuning helped)
# with no per-metric sign flipping anywhere downstream (deltas, wins/ties/
# losses, CIs, plots). Numerically this is equivalent to the old
# "baseline_logloss - variant_logloss" delta -- only the sign convention
# and the label changed, not the underlying values.
METRICS = [
    ("acc", "Accuracy", True),
    ("bal_acc", "Balanced Accuracy", True),
    ("roc_auc", "ROC-AUC", True),
    ("neg_logloss", "Negative Log Loss", True),
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
                # Stored as NEGATIVE log loss so higher = better, matching
                # every other metric (see METRICS comment above).
                "neg_logloss": -summary["logloss"]["mean"],
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


def confidence_interval_95(deltas: dict):
    """Two-sided 95% CI (t-distribution, df=n-1) over dataset-level deltas.

    Computed directly from the same per-dataset delta values used for the
    mean/median/boxplots -- NOT averaged from per-fold CIs. Returns
    (mean, lo, hi); (mean, nan, nan) if fewer than 2 datasets are available.
    """
    values = list(deltas.values())
    n = len(values)
    mean = float(np.mean(values)) if values else float("nan")
    if n < 2:
        return mean, float("nan"), float("nan")
    sem = np.std(values, ddof=1) / np.sqrt(n)
    if sem == 0:
        return mean, mean, mean
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, mean - t_crit * sem, mean + t_crit * sem


def fmt_mean_ci(deltas: dict, digits=5) -> str:
    """Compact 'Mean Δ [95% CI]' string, e.g. '0.00123 [-0.00050, 0.00296]'."""
    mean, lo, hi = confidence_interval_95(deltas)
    if np.isnan(lo):
        return f"{mean:.{digits}f} [n/a]"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main():
    runs, configs = load_all_runs()
    baseline_experiment = find_baseline_experiment(runs)
    baseline = runs[baseline_experiment]

    variant_experiments = [
        e for e in sorted(runs)
        if e != baseline_experiment and len(runs[e]) >= 5  # skip stray/old folders
    ]

    # "Main" comparison set = the 4 core strategies (Full / Attention-only /
    # MLP-only / Layer-wise at layer 0). This is what the original thesis
    # structure compares head-to-head: the per-variant "vs Baseline" tables,
    # Table 5.5/5.5b/5.6, deltas_boxplot, and wins_ties_losses. The layer
    # depth sweep (layers 6/11/17/23, plus layer 0 again as one of the 5
    # sweep points) is a separate analysis and appears ONLY in Table 5.3 and
    # layerwise_boxplot -- never mixed into the main comparison.
    _extra_layer_re = re.compile(r"layerwise_layer(\d+)$")
    def _is_extra_layer_sweep_point(e: str) -> bool:
        m = _extra_layer_re.match(e)
        return m is not None and m.group(1) != "0"

    main_variants = [e for e in variant_experiments if not _is_extra_layer_sweep_point(e)]

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

    # --- Compute deltas/summaries for EVERY variant (main set + full layer
    #     sweep) -- needed later by Table 5.3 / layerwise_boxplot even though
    #     those extra layers are not written into the tables/plots below. ---
    all_deltas: dict = {}
    all_summaries: dict = {}

    for e in variant_experiments:
        all_deltas[e] = {}
        all_summaries[e] = {}
        for key, label, higher_is_better in METRICS:
            deltas = per_dataset_deltas(baseline, runs[e], key, higher_is_better)
            all_deltas[e][key] = deltas
            all_summaries[e][key] = summarize(deltas)

    # --- Per-variant tables (main comparison set only: Full / Attention-only
    #     / MLP-only / Layer-wise layer 0 -- NOT the layer 6/11/17/23 sweep) ---
    for e in main_variants:
        common = sorted(set(baseline) & set(runs[e]))

        lines.append(f"## Table -- {display_name(e)} vs Baseline\n")
        lines.append("| Metric | Mean baseline | Mean variant | Mean difference | Wins | Ties | Losses |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, label, higher_is_better in METRICS:
            s = all_summaries[e][key]
            mean_baseline = sum(baseline[d][key] for d in common) / len(common) if common else float("nan")
            mean_variant = sum(runs[e][d][key] for d in common) / len(common) if common else float("nan")
            lines.append(
                f"| {label} | {fmt(mean_baseline)} | {fmt(mean_variant)} | {fmt(s['mean'], 5)} | "
                f"{s['wins']} | {s['ties']} | {s['losses']} |"
            )
        lines.append(f"\n(compared on {len(common)} datasets present in both baseline and this run)\n")

    # --- Table 5.5: Overall comparison (mean deltas + 95% CI) ---
    lines.append("## Table 5.5 -- Overall Comparison of Fine-Tuning Strategies (mean Δ [95% CI])\n")
    lines.append(
        "| Strategy | Datasets | Mean Δ Accuracy [95% CI] | Mean Δ Bal. Acc. [95% CI] | "
        "Mean Δ ROC-AUC [95% CI] | Mean Δ Negative Log Loss [95% CI] |"
    )
    lines.append("|---|---|---|---|---|---|")
    for e in main_variants:
        n = all_summaries[e]["acc"]["n"]
        row = f"| {display_name(e)} | {n} "
        for key, _, _ in METRICS:
            row += f"| {fmt_mean_ci(all_deltas[e][key])} "
        row += "|"
        lines.append(row)
    lines.append(
        "\n*95% CI: two-sided t-distribution (df = n_datasets - 1), computed directly "
        "over the dataset-level deltas shown above -- not averaged from per-fold CIs.*\n"
    )

    # --- Table 5.5b: median deltas ---
    lines.append("## Table 5.5b -- Overall Comparison of Fine-Tuning Strategies (median)\n")
    lines.append(
        "| Strategy | Median Δ Accuracy | Median Δ Bal. Acc. | Median Δ ROC-AUC | "
        "Median Δ Negative Log Loss |"
    )
    lines.append("|---|---|---|---|---|")
    for e in main_variants:
        row = f"| {display_name(e)} "
        for key, _, _ in METRICS:
            row += f"| {fmt(all_summaries[e][key]['median'], 5)} "
        row += "|"
        lines.append(row)
    lines.append("")

    # --- Table 5.6: Common-subset comparison (fair, same datasets for all) ---
    # Only datasets present in the baseline AND every fine-tuning variant in
    # the MAIN comparison set (layers 6/11/17/23 excluded, same as Table 5.5)
    # -- the strictest possible comparison, so no strategy is compared on an
    # easier or harder subset of datasets than any other.
    common_all = set(baseline)
    for e in main_variants:
        common_all &= set(runs[e])
    common_all = sorted(common_all)

    lines.append(
        "## Table 5.6 -- Common-Subset Comparison "
        f"(only datasets present in baseline + all {len(main_variants)} strategies, "
        f"n={len(common_all)})\n"
    )
    if len(common_all) < 5:
        lines.append(
            "*(Too few common datasets to report -- fewer than 5 datasets have "
            "results for every strategy simultaneously.)*\n"
        )
    else:
        lines.append(
            "| Strategy | Mean Δ Accuracy [95% CI] | Mean Δ Bal. Acc. [95% CI] | "
            "Mean Δ ROC-AUC [95% CI] | Mean Δ Negative Log Loss [95% CI] |"
        )
        lines.append("|---|---|---|---|---|")
        common_summaries: dict = {}
        common_deltas_subset: dict = {}
        for e in main_variants:
            common_summaries[e] = {}
            common_deltas_subset[e] = {}
            row = f"| {display_name(e)} "
            for key, label, higher_is_better in METRICS:
                deltas = {
                    name: all_deltas[e][key][name]
                    for name in common_all
                    if name in all_deltas[e][key]
                }
                s = summarize(deltas)
                common_summaries[e][key] = s
                common_deltas_subset[e][key] = deltas
                row += f"| {fmt_mean_ci(deltas)} "
            row += "|"
            lines.append(row)
        lines.append(
            "\n*95% CI: two-sided t-distribution (df = n_datasets - 1) over the "
            "common-subset dataset-level deltas.*\n"
        )

        lines.append("| Strategy | Wins | Ties | Losses | (Accuracy, on common subset) |")
        lines.append("|---|---|---|---|---|")
        for e in main_variants:
            s = common_summaries[e]["acc"]
            lines.append(f"| {display_name(e)} | {s['wins']} | {s['ties']} | {s['losses']} | n={s['n']} |")
        lines.append("")

        lines.append(f"Datasets included in the common subset: {', '.join(common_all)}\n")

    # --- Table 5.3: Layer-wise fine-tuning -- Mean Delta per layer ---
    # Identifies every "layerwise_layer<N>" experiment and sorts by the
    # numeric layer index (NOT alphabetically -- "layer11" < "layer6"
    # alphabetically, which would be a wrong, confusing order in a table
    # meant to show a trend across network depth).
    layer_re = re.compile(r"layerwise_layer(\d+)$")
    layer_experiments = []
    for e in variant_experiments:
        m = layer_re.match(e)
        if m:
            layer_experiments.append((int(m.group(1)), e))
    layer_experiments.sort(key=lambda t: t[0])

    if layer_experiments:
        lines.append("## Table 5.3 -- Layer-wise Fine-Tuning: Mean Δ per Layer [95% CI]\n")
        lines.append(
            "| Layer | Datasets | Mean Δ Accuracy [95% CI] | Mean Δ Bal. Acc. [95% CI] | "
            "Mean Δ ROC-AUC [95% CI] | Mean Δ Negative Log Loss [95% CI] |"
        )
        lines.append("|---|---|---|---|---|---|")
        for layer_idx, e in layer_experiments:
            n = all_summaries[e]["acc"]["n"]
            row = f"| {layer_idx} | {n} "
            for key, _, _ in METRICS:
                row += f"| {fmt_mean_ci(all_deltas[e][key])} "
            row += "|"
            lines.append(row)
        lines.append(
            "\n*95% CI: two-sided t-distribution (df = n_datasets - 1) over the "
            "dataset-level deltas for that layer.*\n"
        )

    tables_path = output_dir / "tables.md"
    tables_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Tables written to: {tables_path}")

    # --- Plot 1 (Figure 5.1): Boxplots of per-dataset deltas, one panel per metric ---
    # 2x2 grid: Accuracy / Balanced Accuracy / ROC-AUC / Negative Log Loss.
    # Main comparison set only (Full / Attention-only / MLP-only / Layer-wise
    # layer 0) -- the layer 6/11/17/23 sweep points are NOT included here,
    # they live only in layerwise_boxplot / Table 5.3.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    labels = [display_name(e) for e in main_variants]
    for ax, (key, label, _) in zip(axes_flat, METRICS):
        data = [list(all_deltas[e][key].values()) or [0.0] for e in main_variants]
        ax.boxplot(data)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_ylabel(f"{label} difference vs baseline")
    plt.tight_layout()
    plot1_path = output_dir / "deltas_boxplot.png"
    plot1_pdf_path = output_dir / "deltas_boxplot.pdf"
    plot1_svg_path = output_dir / "deltas_boxplot.svg"
    plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
    plt.savefig(plot1_pdf_path, bbox_inches="tight")
    plt.savefig(plot1_svg_path, bbox_inches="tight")
    plt.close()
    print(f"Boxplot saved to: {plot1_path}, {plot1_pdf_path} and {plot1_svg_path}")

    # --- Plot 2: Stacked bar chart of wins/ties/losses (accuracy) ---
    # This figure ONLY (not Plot 1 / the tables) uses all 8 configurations,
    # in a fixed, explicitly requested x-axis order, with shortened
    # "Layer X" labels instead of "Layer-wise (layer X)" for readability.
    # Values/calculations are unchanged -- only the ordering and labels
    # shown on this one figure.
    wtl_order = [
        ("attention_only",    "Attention-only"),
        ("layerwise_layer0",  "Layer 0"),
        ("layerwise_layer6",  "Layer 6"),
        ("layerwise_layer11", "Layer 11"),
        ("layerwise_layer17", "Layer 17"),
        ("layerwise_layer23", "Layer 23"),
        ("mlp_only",          "MLP-only"),
        ("own_finetuning",    "Full FT"),
    ]
    wtl_experiments = [(e, lbl) for e, lbl in wtl_order if e in all_summaries]
    wtl_labels = [lbl for _, lbl in wtl_experiments]

    fig, ax = plt.subplots(figsize=(8, 5))
    wins = [all_summaries[e]["acc"]["wins"] for e, _ in wtl_experiments]
    ties = [all_summaries[e]["acc"]["ties"] for e, _ in wtl_experiments]
    losses = [all_summaries[e]["acc"]["losses"] for e, _ in wtl_experiments]
    x = np.arange(len(wtl_labels))
    ax.bar(x, wins, label="Wins", color="#4CAF50")
    ax.bar(x, ties, bottom=wins, label="Ties", color="#9E9E9E")
    bottom2 = [w + t for w, t in zip(wins, ties)]
    ax.bar(x, losses, bottom=bottom2, label="Losses", color="#E53935")
    ax.set_xticks(x)
    ax.set_xticklabels(wtl_labels, rotation=30, ha="right")
    ax.set_ylabel("Number of datasets (based on accuracy)")
    ax.set_title("Wins / Ties / Losses vs Baseline (Accuracy)")
    ax.legend()
    plt.tight_layout()
    plot2_path = output_dir / "wins_ties_losses.png"
    plot2_pdf_path = output_dir / "wins_ties_losses.pdf"
    plot2_svg_path = output_dir / "wins_ties_losses.svg"
    plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
    plt.savefig(plot2_pdf_path, bbox_inches="tight")
    plt.savefig(plot2_svg_path, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved to: {plot2_path}, {plot2_pdf_path} and {plot2_svg_path}")

    # --- Plot 3 (Figure 5.3): Boxplots of per-dataset deltas across LAYERS ---
    # Answers a different question than Plot 1: not "which strategy is
    # best" but "does the effect change across network depth". X-axis is
    # the layer index in numeric order (not alphabetical), one panel per
    # metric, same visual style as Plot 1 for consistency. Only individual
    # per-dataset boxplots are shown (no connecting line between layers) so
    # the figure does not imply anything about untested layers in between.
    if layer_experiments:
        # 2x2 grid, same layout as Figure 5.1, for visual consistency.
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()
        layer_labels = [str(layer_idx) for layer_idx, _ in layer_experiments]
        for ax, (key, label, _) in zip(axes_flat, METRICS):
            data = [
                list(all_deltas[e][key].values()) or [0.0]
                for _, e in layer_experiments
            ]
            ax.boxplot(data)
            ax.set_xticklabels(layer_labels)
            ax.axhline(0, color="red", linestyle="--", linewidth=1)
            ax.set_title(label)
            ax.set_xlabel("Transformer block (icl_blocks index)")
            ax.set_ylabel(f"{label} difference vs baseline")
        plt.tight_layout()
        plot3_path = output_dir / "layerwise_boxplot.png"
        plot3_pdf_path = output_dir / "layerwise_boxplot.pdf"
        plot3_svg_path = output_dir / "layerwise_boxplot.svg"
        plt.savefig(plot3_path, dpi=150, bbox_inches="tight")
        plt.savefig(plot3_pdf_path, bbox_inches="tight")
        plt.savefig(plot3_svg_path, bbox_inches="tight")
        plt.close()
        print(f"Layer-wise boxplot saved to: {plot3_path}, {plot3_pdf_path} and {plot3_svg_path}")

    print("\nDone. Copy results/thesis_report/tables.md into your thesis, "
          "and insert the figures (PNG, PDF, or SVG -- SVG usually embeds "
          "most cleanly as a real vector image in Word).")


if __name__ == "__main__":
    main()
