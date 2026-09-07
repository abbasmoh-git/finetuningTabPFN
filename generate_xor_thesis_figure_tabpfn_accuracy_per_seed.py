"""
generate_xor_thesis_figure_tabpfn_accuracy_per_seed.py
--------------------------------------------------------
Publication-quality figure for thesis Section 4.4.2: TabPFN v3 XOR
fine-tuning results (test accuracy per seed, 5 strategies), styled to match
the existing TabICL v2 selective fine-tuning figure
(fig4_tabicl_selective_accuracy_per_seed, from
generate_xor_thesis_figures_tabicl_selective.py) as closely as possible:
same figure size, grouped-bar layout, tab10 color cycle, axis labels,
ylim, and below-plot legend style.

This script ONLY reads the already-saved result pickle:
  results/xor_seed_sweep_tabpfn_n800/results.pkl
(produced by run_xor_seed_sweep_tabpfn.py). It does not train, evaluate, or
modify anything -- no experiment code or result file is touched. All plotted
values come directly from that pickle; nothing is hard-coded or invented.

Strategies plotted (keys as stored in results.pkl, confirmed in
run_xor_seed_sweep_tabpfn.py's run_one_dataset()):
  no_finetuning      -> "Baseline"
  full_finetuning    -> "Full FT"
  attention_only     -> "Attention-only"
  mlp_only           -> "MLP-only"
  layerwise_layer0   -> "Layer 0"

Output (NEW file, under results/xor_thesis_figures/ -- same directory as
the other XOR thesis figures, nothing else in it is touched):
  fig6_tabpfn_accuracy_per_seed.svg
  fig6_tabpfn_accuracy_per_seed.png

Usage
-----
    python generate_xor_thesis_figure_tabpfn_accuracy_per_seed.py
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("results/xor_seed_sweep_tabpfn_n800/results.pkl")
OUTPUT_DIR = Path("results/xor_thesis_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCES_PATH = OUTPUT_DIR / "figure_sources.txt"

SEEDS = [1, 2, 3, 4, 5]
DATASET_NAMES = [f"xor_seed{s}_n800" for s in SEEDS]

# (pickle key, display label) -- order also fixes bar/color order, matching
# the "Baseline, Full FT, ..." ordering convention used in the TabICL v2
# selective fine-tuning figure this is styled after.
STRATEGIES = [
    ("no_finetuning", "Baseline"),
    ("full_finetuning", "Full FT"),
    ("attention_only", "Attention-only"),
    ("mlp_only", "MLP-only"),
    ("layerwise_layer0", "Layer 0"),
]


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} does not exist -- run run_xor_seed_sweep_tabpfn.py first."
        )
    with open(RESULTS_PATH, "rb") as f:
        return pickle.load(f)


def main():
    results = load_results()

    acc_by_strategy = {key: [] for key, _ in STRATEGIES}
    for name in DATASET_NAMES:
        res = results[name]
        for key, _ in STRATEGIES:
            acc_by_strategy[key].append(res[key]["test"]["accuracy"])

    n_strategies = len(STRATEGIES)
    n_seeds = len(SEEDS)
    x = np.arange(n_seeds)
    total_width = 0.82
    bar_width = total_width / n_strategies
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (key, label) in enumerate(STRATEGIES):
        offset = (i - (n_strategies - 1) / 2) * bar_width
        ax.bar(x + offset, acc_by_strategy[key], bar_width, label=label, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("TabPFN v3 XOR Fine-Tuning: Accuracy per Seed (n=800)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5)
    plt.tight_layout()

    fig6_svg = OUTPUT_DIR / "fig6_tabpfn_accuracy_per_seed.svg"
    fig6_png = OUTPUT_DIR / "fig6_tabpfn_accuracy_per_seed.png"
    plt.savefig(fig6_svg, bbox_inches="tight")
    plt.savefig(fig6_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig6_svg}, {fig6_png}")

    lines = ["", "TabPFN v3 XOR fine-tuning figure (Section 4.4.2)", "=" * 60,
              f"Source: {RESULTS_PATH}",
              f"Figure: {fig6_svg.name} / {fig6_png.name}", "-" * 60]
    for key, label in STRATEGIES:
        vals = acc_by_strategy[key]
        vals_str = ", ".join(f"seed{s}={v:.4f}" for s, v in zip(SEEDS, vals))
        lines.append(f"  {label} ({key}): {vals_str}")
    lines.append("")

    if SOURCES_PATH.exists():
        with open(SOURCES_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Appended sources/values to: {SOURCES_PATH}")
    else:
        SOURCES_PATH.write_text("\n".join(lines), encoding="utf-8")
        print(f"Sources/values written to: {SOURCES_PATH}")


if __name__ == "__main__":
    main()
