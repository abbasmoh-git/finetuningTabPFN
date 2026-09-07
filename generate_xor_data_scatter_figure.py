"""
generate_xor_data_scatter_figure.py
------------------------------------
Publication-quality illustration of the controlled XOR benchmark itself
(not a result plot): one representative dataset (seed 1, n=800), showing
only the two informative signal features, colored by target class.

Reuses XOR generation UNCHANGED, by importing (not duplicating or editing)
from run_xor_seed_sweep_tabpfn.py:
  * make_xor()   -- copied verbatim from notebooks/decision_boundary_xor_tabpfn_v3.ipynb,
                     unchanged there, only imported here.
  * XOR_CONFIGS  -- the same 5 controlled seed-sweep configs; this script
                     uses only the seed=1 entry ("xor_seed1_n800": n_samples=800,
                     noise=0.01, n_features=10, gap=0.01, random_state=1).

make_xor() returns features in a fixed column order: columns 0 and 1 are
always the two informative (signal) features that define the checkerboard
label; columns 2..9 are Gaussian noise, added only if n_features > 2. This
script plots ONLY columns 0 and 1 -- the 8 noise features are generated
(as part of the unchanged data-generation code) but never plotted.

No training/experiment code is imported, run, or modified -- this script
only calls make_xor() to get X, y and plots them. It does not touch
run_xor_seed_sweep_tabpfn.py, finetuning_engine.py, main.py, or any
existing results file.

Output (NEW files, under results/xor_thesis_figures/ -- same directory as
the other XOR thesis figures, nothing else in it is touched):
  xor_data_scatter_seed1.svg
  xor_data_scatter_seed1.png

Usage
-----
    python generate_xor_data_scatter_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_xor_seed_sweep_tabpfn import make_xor, XOR_CONFIGS

OUTPUT_DIR = Path("results/xor_thesis_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCES_PATH = OUTPUT_DIR / "figure_sources.txt"


def main():
    cfg = next(c for c in XOR_CONFIGS if c["name"] == "xor_seed1_n800")
    X, y = make_xor(
        n_samples=cfg["n_samples"],
        noise=cfg["noise"],
        n_features=cfg["n_features"],
        random_state=cfg["random_state"],
        gap=cfg["gap"],
    )

    # Only the two informative features (columns 0, 1). The remaining
    # n_features - 2 = 8 noise columns in X are not plotted.
    x1, x2 = X[:, 0], X[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {0: "#1f77b4", 1: "#d62728"}  # distinguishable, colorblind-safe-ish blue/red
    for cls in (0, 1):
        mask = y == cls
        ax.scatter(x1[mask], x2[mask], s=14, alpha=0.8, color=colors[cls],
                   label=f"Class {cls}", edgecolors="none")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    svg_path = OUTPUT_DIR / "xor_data_scatter_seed1.svg"
    png_path = OUTPUT_DIR / "xor_data_scatter_seed1.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {svg_path}, {png_path}")

    lines = [
        "", "XOR data illustration (seed 1)", "=" * 60,
        f"Figure: {svg_path.name} / {png_path.name}",
        "-" * 60,
        f"Config: {cfg}",
        f"n_samples plotted: {len(y)} (class 0: {(y == 0).sum()}, class 1: {(y == 1).sum()})",
        "Plotted columns: X[:, 0] (Feature 1), X[:, 1] (Feature 2) -- the "
        f"other {cfg['n_features'] - 2} noise features are not plotted.",
        "",
    ]
    if SOURCES_PATH.exists():
        with open(SOURCES_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Appended sources/values to: {SOURCES_PATH}")
    else:
        SOURCES_PATH.write_text("\n".join(lines), encoding="utf-8")
        print(f"Sources/values written to: {SOURCES_PATH}")


if __name__ == "__main__":
    main()
