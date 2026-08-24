"""
generate_xor_thesis_figures_summary.py
-----------------------------------------
Thesis figures 1 and 2 for the controlled n=800 XOR seed sweep (seeds 1-5),
built purely from the two ALREADY-COMPLETED result pickles:

  results/xor_seed_sweep_tabpfn_n800/results.pkl
  results/xor_seed_sweep_tabicl_n800/results.pkl

This script does not fit, train, or evaluate anything. It only reads the
two pickles above and plots numbers already stored in them. No experiment
is rerun or modified, and no other result directory is read or written.

Figures produced (under results/xor_thesis_figures/, PNG + SVG each):
  fig1_accuracy_per_seed.{png,svg}
      2 panels (TabPFN v3 | TabICL v2), each showing Baseline vs. Full
      Fine-Tuning test accuracy for every seed 1-5 as grouped bars.
  fig2_mean_accuracy_improvement.{png,svg}
      1 panel: mean (Full Fine-Tuning - Baseline) test accuracy across the
      5 seeds, one bar per model, error bars = standard deviation across
      the 5 seeds.

Also writes/creates results/xor_thesis_figures/figure_sources.txt, listing
exactly which pickle keys were read and the numeric values plotted in each
figure (see generate_xor_thesis_figures_boundaries.py, which appends its
own section to the same file for figure 3).

Usage
-----
    python generate_xor_thesis_figures_summary.py
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TABPFN_RESULTS = Path("results/xor_seed_sweep_tabpfn_n800/results.pkl")
TABICL_RESULTS = Path("results/xor_seed_sweep_tabicl_n800/results.pkl")
OUTPUT_DIR = Path("results/xor_thesis_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCES_PATH = OUTPUT_DIR / "figure_sources.txt"

SEEDS = [1, 2, 3, 4, 5]
TABPFN_DATASET_NAMES = [f"xor_seed{s}_n800" for s in SEEDS]
TABICL_DATASET_NAMES = [f"xor_seed{s}_n800" for s in SEEDS]  # identical naming, identical configs


def load_pickle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. This script only reads already-completed "
            f"sweep results -- run the corresponding sweep script first "
            f"(run_xor_seed_sweep_tabpfn.py / run_xor_seed_sweep_tabicl.py)."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def get_accuracies(all_results: dict, dataset_names: list, finetune_key: str):
    """Return (baseline_accs, finetuned_accs) as lists aligned with dataset_names."""
    baseline_accs, finetuned_accs = [], []
    for name in dataset_names:
        res = all_results[name]
        baseline_accs.append(res["no_finetuning"]["test"]["accuracy"])
        finetuned_accs.append(res[finetune_key]["test"]["accuracy"])
    return baseline_accs, finetuned_accs


def main():
    tabpfn_results = load_pickle(TABPFN_RESULTS)
    tabicl_results = load_pickle(TABICL_RESULTS)

    tabpfn_base, tabpfn_ft = get_accuracies(tabpfn_results, TABPFN_DATASET_NAMES, "full_finetuning")
    tabicl_base, tabicl_ft = get_accuracies(tabicl_results, TABICL_DATASET_NAMES, "full_finetuning")

    sources_lines = []
    sources_lines.append("XOR thesis figures -- source files and plotted values")
    sources_lines.append("=" * 60)
    sources_lines.append("")
    sources_lines.append(f"TabPFN v3 source: {TABPFN_RESULTS}")
    sources_lines.append(f"TabICL v2 source: {TABICL_RESULTS}")
    sources_lines.append("")

    # ------------------------------------------------------------------
    # Figure 1: Baseline vs. Full Fine-Tuning accuracy, per seed, 2 panels
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(SEEDS))
    width = 0.35

    axes[0].bar(x - width / 2, tabpfn_base, width, label="Baseline", color="#9E9E9E")
    axes[0].bar(x + width / 2, tabpfn_ft, width, label="Full Fine-Tuning", color="#1f77b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(s) for s in SEEDS])
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("TabPFN v3")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].bar(x - width / 2, tabicl_base, width, label="Baseline", color="#9E9E9E")
    axes[1].bar(x + width / 2, tabicl_ft, width, label="Full Fine-Tuning", color="#d62728")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(s) for s in SEEDS])
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("TabICL v2")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    fig1_png = OUTPUT_DIR / "fig1_accuracy_per_seed.png"
    fig1_svg = OUTPUT_DIR / "fig1_accuracy_per_seed.svg"
    plt.savefig(fig1_png, dpi=150, bbox_inches="tight")
    plt.savefig(fig1_svg, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig1_png}, {fig1_svg}")

    sources_lines.append("Figure 1: fig1_accuracy_per_seed.{png,svg}")
    sources_lines.append("-" * 60)
    sources_lines.append("Left panel -- TabPFN v3 (source: xor_seed_sweep_tabpfn_n800/results.pkl)")
    for s, b, f in zip(SEEDS, tabpfn_base, tabpfn_ft):
        sources_lines.append(
            f"  seed {s}: baseline accuracy = {b:.4f} "
            f"(['xor_seed{s}_n800']['no_finetuning']['test']['accuracy']), "
            f"full_finetuning accuracy = {f:.4f} "
            f"(['xor_seed{s}_n800']['full_finetuning']['test']['accuracy'])"
        )
    sources_lines.append("Right panel -- TabICL v2 (source: xor_seed_sweep_tabicl_n800/results.pkl)")
    for s, b, f in zip(SEEDS, tabicl_base, tabicl_ft):
        sources_lines.append(
            f"  seed {s}: baseline accuracy = {b:.4f} "
            f"(['xor_seed{s}_n800']['no_finetuning']['test']['accuracy']), "
            f"full_finetuning accuracy = {f:.4f} "
            f"(['xor_seed{s}_n800']['full_finetuning']['test']['accuracy'])"
        )
    sources_lines.append("")

    # ------------------------------------------------------------------
    # Figure 2: Mean accuracy improvement (FT - baseline) across 5 seeds,
    # std across seeds as error bars, one bar per model.
    # ------------------------------------------------------------------
    tabpfn_deltas = np.array(tabpfn_ft) - np.array(tabpfn_base)
    tabicl_deltas = np.array(tabicl_ft) - np.array(tabicl_base)

    tabpfn_mean, tabpfn_std = float(tabpfn_deltas.mean()), float(tabpfn_deltas.std(ddof=1))
    tabicl_mean, tabicl_std = float(tabicl_deltas.mean()), float(tabicl_deltas.std(ddof=1))

    fig, ax = plt.subplots(figsize=(5, 5))
    models = ["TabPFN v3", "TabICL v2"]
    means = [tabpfn_mean, tabicl_mean]
    stds = [tabpfn_std, tabicl_std]
    colors = ["#1f77b4", "#d62728"]

    ax.bar(models, means, yerr=stds, capsize=8, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Δ Accuracy (Full Fine-Tuning − Baseline)")
    ax.set_title("Mean Accuracy Improvement Across 5 Seeds (n=800)")

    plt.tight_layout()
    fig2_png = OUTPUT_DIR / "fig2_mean_accuracy_improvement.png"
    fig2_svg = OUTPUT_DIR / "fig2_mean_accuracy_improvement.svg"
    plt.savefig(fig2_png, dpi=150, bbox_inches="tight")
    plt.savefig(fig2_svg, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig2_png}, {fig2_svg}")

    sources_lines.append("Figure 2: fig2_mean_accuracy_improvement.{png,svg}")
    sources_lines.append("-" * 60)
    sources_lines.append(
        "Per-seed delta accuracy = full_finetuning['test']['accuracy'] - "
        "no_finetuning['test']['accuracy'], same source files as Figure 1."
    )
    sources_lines.append(f"TabPFN v3 per-seed deltas: {[round(float(d), 4) for d in tabpfn_deltas]}")
    sources_lines.append(f"TabPFN v3 mean delta = {tabpfn_mean:.4f}, std (ddof=1) = {tabpfn_std:.4f}")
    sources_lines.append(f"TabICL v2 per-seed deltas: {[round(float(d), 4) for d in tabicl_deltas]}")
    sources_lines.append(f"TabICL v2 mean delta = {tabicl_mean:.4f}, std (ddof=1) = {tabicl_std:.4f}")
    sources_lines.append("")

    SOURCES_PATH.write_text("\n".join(sources_lines), encoding="utf-8")
    print(f"Sources/values written to: {SOURCES_PATH}")


if __name__ == "__main__":
    main()
