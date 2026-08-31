"""
generate_xor_thesis_figures_tabicl_selective.py
---------------------------------------------------
Two NEW thesis figures for the TabICL v2 selective fine-tuning extended
experiment, built purely from the already-combined result pickle:

  results/xor_seed_sweep_tabicl_extended_n800/results.pkl

This script only reads that pickle (read-only) and plots numbers already
stored in it. It does not fit, train, or evaluate anything, does not touch
any existing figure (fig1/fig2/fig3a/fig3b under results/xor_thesis_figures/,
or anything under results/xor_seed_sweep_tabpfn_n800/,
results/xor_seed_sweep_tabicl_n800/), and does not modify any experimental
code.

Figures produced (under results/xor_thesis_figures/, PNG + SVG each, NEW
filenames, nothing overwritten):

  fig4_tabicl_selective_accuracy_per_seed.{png,svg}
      Test accuracy per seed (1-5), one group of bars per seed, 9 bars per
      group: Baseline, Full FT, Attention-only, MLP-only, and Layer-wise
      ICL blocks 0/3/6/8/11.

  fig5_tabicl_selective_mean_delta_accuracy.{png,svg}
      Mean (strategy - Baseline) test accuracy across the 5 seeds, one bar
      per strategy (the same 8 non-baseline strategies as above), error
      bars = standard deviation across the 5 seeds (same convention as
      fig2_mean_accuracy_improvement.png).

Also APPENDS a new section to the existing
results/xor_thesis_figures/figure_sources.txt (never overwrites its
existing content) listing exactly which pickle keys were read and the
numeric values plotted in each of these two new figures.

Usage
-----
    python generate_xor_thesis_figures_tabicl_selective.py
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("results/xor_seed_sweep_tabicl_extended_n800/results.pkl")
OUTPUT_DIR = Path("results/xor_thesis_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCES_PATH = OUTPUT_DIR / "figure_sources.txt"

SEEDS = [1, 2, 3, 4, 5]
DATASET_NAMES = [f"xor_seed{s}_n800" for s in SEEDS]

# (pickle key, display label) -- order also fixes bar/color order in both figures.
STRATEGIES_FIG1 = [
    ("no_finetuning", "Baseline"),
    ("full_finetuning", "Full FT"),
    ("attention_only", "Attention-only"),
    ("mlp_only", "MLP-only"),
    ("layerwise_icl0", "ICL Block 0"),
    ("layerwise_icl3", "ICL Block 3"),
    ("layerwise_icl6", "ICL Block 6"),
    ("layerwise_icl8", "ICL Block 8"),
    ("layerwise_icl11", "ICL Block 11"),
]
STRATEGIES_FIG2 = STRATEGIES_FIG1[1:]  # everything except Baseline (deltas are relative to it)


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} does not exist -- run "
            f"aggregate_xor_tabicl_extended_results.py first."
        )
    with open(RESULTS_PATH, "rb") as f:
        return pickle.load(f)


def main():
    results = load_results()

    sources_lines = ["", "TabICL v2 selective fine-tuning figures", "=" * 60,
                      f"Source: {RESULTS_PATH}", ""]

    # ------------------------------------------------------------------
    # Gather accuracy per (strategy, seed) once, reused by both figures.
    # ------------------------------------------------------------------
    acc_by_strategy = {key: [] for key, _ in STRATEGIES_FIG1}
    for name in DATASET_NAMES:
        res = results[name]
        for key, _ in STRATEGIES_FIG1:
            acc_by_strategy[key].append(res[key]["test"]["accuracy"])

    # ------------------------------------------------------------------
    # Figure 4: Accuracy per seed, 9 strategies (baseline + 8 variants)
    # ------------------------------------------------------------------
    n_strategies = len(STRATEGIES_FIG1)
    n_seeds = len(SEEDS)
    x = np.arange(n_seeds)
    total_width = 0.82
    bar_width = total_width / n_strategies
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (key, label) in enumerate(STRATEGIES_FIG1):
        offset = (i - (n_strategies - 1) / 2) * bar_width
        ax.bar(x + offset, acc_by_strategy[key], bar_width, label=label, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("TabICL v2 Selective Fine-Tuning: Accuracy per Seed (n=800)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5)
    plt.tight_layout()

    fig4_png = OUTPUT_DIR / "fig4_tabicl_selective_accuracy_per_seed.png"
    fig4_svg = OUTPUT_DIR / "fig4_tabicl_selective_accuracy_per_seed.svg"
    plt.savefig(fig4_png, dpi=150, bbox_inches="tight")
    plt.savefig(fig4_svg, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig4_png}, {fig4_svg}")

    sources_lines.append(f"Figure: {fig4_png.name} / {fig4_svg.name}")
    sources_lines.append("-" * 60)
    for key, label in STRATEGIES_FIG1:
        vals = acc_by_strategy[key]
        vals_str = ", ".join(f"seed{s}={v:.4f}" for s, v in zip(SEEDS, vals))
        sources_lines.append(f"  {label} ({key}): {vals_str}")
    sources_lines.append("")

    # ------------------------------------------------------------------
    # Figure 5: Mean delta accuracy across 5 seeds, 8 strategies
    # (delta = strategy - baseline, per seed; error bars = std across seeds,
    # same convention as fig2_mean_accuracy_improvement.png)
    # ------------------------------------------------------------------
    baseline_acc = acc_by_strategy["no_finetuning"]
    deltas_by_strategy = {
        key: [acc_by_strategy[key][i] - baseline_acc[i] for i in range(n_seeds)]
        for key, _ in STRATEGIES_FIG2
    }
    means = [float(np.mean(deltas_by_strategy[key])) for key, _ in STRATEGIES_FIG2]
    stds = [float(np.std(deltas_by_strategy[key], ddof=1)) for key, _ in STRATEGIES_FIG2]
    labels2 = [label for _, label in STRATEGIES_FIG2]
    colors2 = colors[1:1 + len(STRATEGIES_FIG2)]  # skip index 0 (Baseline's color in fig4)

    fig, ax = plt.subplots(figsize=(10, 6))
    x2 = np.arange(len(labels2))
    ax.bar(x2, means, yerr=stds, capsize=6, color=colors2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x2)
    ax.set_xticklabels(labels2, rotation=30, ha="right")
    ax.set_ylabel("Mean Δ Accuracy (Strategy − Baseline)")
    ax.set_title("TabICL v2 Selective Fine-Tuning: Mean Accuracy Improvement Across 5 Seeds (n=800)")
    plt.tight_layout()

    fig5_png = OUTPUT_DIR / "fig5_tabicl_selective_mean_delta_accuracy.png"
    fig5_svg = OUTPUT_DIR / "fig5_tabicl_selective_mean_delta_accuracy.svg"
    plt.savefig(fig5_png, dpi=150, bbox_inches="tight")
    plt.savefig(fig5_svg, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig5_png}, {fig5_svg}")

    sources_lines.append(f"Figure: {fig5_png.name} / {fig5_svg.name}")
    sources_lines.append("-" * 60)
    sources_lines.append(
        "Delta = strategy['test']['accuracy'] - no_finetuning['test']['accuracy'], per seed. "
        "Error bars = std (ddof=1) across the 5 seeds."
    )
    for (key, label), mean, std in zip(STRATEGIES_FIG2, means, stds):
        per_seed = ", ".join(f"seed{s}={d:+.4f}" for s, d in zip(SEEDS, deltas_by_strategy[key]))
        sources_lines.append(f"  {label} ({key}): mean={mean:+.4f}, std={std:.4f} -- {per_seed}")
    sources_lines.append("")

    if SOURCES_PATH.exists():
        with open(SOURCES_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(sources_lines))
        print(f"Appended sources/values to: {SOURCES_PATH}")
    else:
        SOURCES_PATH.write_text("\n".join(sources_lines), encoding="utf-8")
        print(f"Sources/values written to: {SOURCES_PATH}")


if __name__ == "__main__":
    main()
