"""
generate_xor_thesis_figures_boundaries.py
--------------------------------------------
Thesis figure 3: baseline-vs-full-fine-tuning decision boundary comparison
for TabPFN v3 and TabICL v2, on a single FIXED seed (seed=3, n=800) from
the controlled seed sweep.

Why a refit is unavoidable here
------------------------------------
run_xor_seed_sweep_tabpfn.py / run_xor_seed_sweep_tabicl.py only ever saved
SCALAR metrics (accuracy, log loss, ...) to results.pkl -- they never
pickled the fitted model objects, so there is no saved decision surface to
plot. Producing a decision-boundary figure therefore requires re-fitting
baseline and fully-fine-tuned models on the exact seed=3/n=800 data, purely
to obtain a `.predict_proba()`-capable object for visualization.

This is done with the SAME data-generation code (`make_dataset_splits`,
imported unchanged from run_xor_seed_sweep_tabpfn.py), the SAME engines
(TabPFNFinetuner from finetuning_engine.py, TabICLFinetuner imported
unchanged from run_xor_seed_sweep_tabicl.py), and the SAME hyperparameters
as the actual completed sweep run for xor_seed3_n800 (copied from
run_xor_seed_sweep_tabpfn.py's STRATEGIES["full_finetuning"] /
build_finetuning_config, and run_xor_seed_sweep_tabicl.py's
run_full_finetuning_tabicl hyperparameters). It is NOT a new experiment
design -- no hyperparameter here was chosen freely for this script.

IMPORTANT CAVEAT (recorded in figure_sources.txt, not just here): because
fine-tuning involves epoch-to-epoch stochastic context/query splits and
(on GPU) non-deterministic CUDA kernels, the accuracy number that
`plot_decision_boundary()` computes and prints in each subplot title may
differ slightly from the official value stored in results.pkl for
xor_seed3_n800. The authoritative reported numbers are the ones in
results.pkl / figure_sources.txt from generate_xor_thesis_figures_summary.py;
the in-plot accuracy is a by-product of the plotting function itself and is
for visual reference only.

Plotting logic -- reused verbatim, not invented
---------------------------------------------------
`plot_decision_boundary()` below is copied verbatim (identical function in
both notebooks) from notebooks/decision_boundary_xor_tabpfn_v3.ipynb and
notebooks/decision_boundary_xor_tabicl.ipynb. The two final "side-by-side
decision boundary comparison" cells in those notebooks are also reproduced
as-is (same 1x2 layout, same colorbar setup), just pointed at the
controlled sweep's seed=3/n=800 data instead of each notebook's own default
make_xor() call.

Output isolation
-------------------
Writes ONLY to results/xor_thesis_figures/ (created by
generate_xor_thesis_figures_summary.py if not already present) and appends
to results/xor_thesis_figures/figure_sources.txt. Does not write to
results/xor_seed_sweep_tabpfn_n800/, results/xor_seed_sweep_tabicl_n800/,
or any other existing result directory.

  results/xor_thesis_figures/
    fig3a_decision_boundary_tabpfn_seed3.{png,svg}
    fig3b_decision_boundary_tabicl_seed3.{png,svg}

Usage
-----
    python generate_xor_thesis_figures_boundaries.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from main import preprocess_features, encode_labels_train_only
from finetuning_engine import TabPFNFinetuner
from run_xor_seed_sweep_tabpfn import (
    make_dataset_splits,
    XOR_CONFIGS as PFN_XOR_CONFIGS,
    STRATEGIES as PFN_STRATEGIES,
    build_finetuning_config,
    DEVICE as PFN_DEVICE,
)
from run_xor_seed_sweep_tabicl import TabICLFinetuner, DEVICE as ICL_DEVICE

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("results/xor_thesis_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCES_PATH = OUTPUT_DIR / "figure_sources.txt"

FIXED_SEED_NAME = "xor_seed3_n800"  # seed 3, as instructed (no technical reason found to use another)


# ---------------------------------------------------------------------------
# plot_decision_boundary -- copied verbatim, unchanged, from BOTH
# notebooks/decision_boundary_xor_tabpfn_v3.ipynb and
# notebooks/decision_boundary_xor_tabicl.ipynb (the function is identical
# in both). Do not edit without checking those notebooks first.
# ---------------------------------------------------------------------------
def plot_decision_boundary(clf, X, y, ax, title, resolution=30):
    """Fill a 2D decision boundary for a binary classifier.
    Sweeps over the first two features; all remaining features are fixed at 0."""
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # Pad extra features with zeros so the classifier sees the right shape
    n_features = X.shape[1]
    if n_features > 2:
        padding = np.zeros((len(grid_2d), n_features - 2), dtype=np.float32)
        grid = np.concatenate([grid_2d, padding], axis=1)
    else:
        grid = grid_2d

    proba = clf.predict_proba(grid)[:, 1]  # P(class = 1)
    Z = proba.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=50, cmap=plt.cm.RdBu_r, alpha=0.75, vmin=0, vmax=1)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.2)

    colors = np.where(y == 1, "#d62728", "#1f77b4")
    ax.scatter(X[:, 0], X[:, 1], c=colors, edgecolors="w", linewidths=0.4, s=30, zorder=3)

    acc = (clf.predict(X) == y).mean()
    # NOTE: accuracy is deliberately NOT shown in the subplot title (removed
    # per instruction) -- it is still computed and returned here so callers
    # can record it in figure_sources.txt.
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return float(acc)


def prepare_seed3_data():
    """Identical data preparation to run_one_dataset() in both sweep
    scripts, for xor_seed3_n800 specifically."""
    cfg = next(c for c in PFN_XOR_CONFIGS if c["name"] == FIXED_SEED_NAME)

    X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw = make_dataset_splits(cfg)
    X_train, X_val, X_test = preprocess_features(X_train_raw, X_val_raw, X_test_raw)
    y_train, y_val, y_test, val_mask, test_mask, _ = encode_labels_train_only(
        y_train_raw, y_val_raw, y_test_raw
    )
    X_val = X_val[val_mask.values]
    X_test = X_test[test_mask.values]
    return X_train, X_val, X_test, y_train, y_val, y_test, cfg


def fit_tabpfn_pair(X_train, y_train, X_val, y_val):
    from tabpfn import TabPFNClassifier

    # Vanilla -- exact same instantiation as main.py's run_no_finetuning().
    clf_vanilla = TabPFNClassifier(ignore_pretraining_limits=True, device=PFN_DEVICE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_vanilla.fit(X_train, y_train)

    # Full fine-tuning -- exact same hyperparameters as
    # run_xor_seed_sweep_tabpfn.py's STRATEGIES["full_finetuning"], via the
    # same build_finetuning_config() + TabPFNFinetuner instantiation pattern
    # main.py's run_own_finetuning() uses.
    ft_cfg = build_finetuning_config(PFN_STRATEGIES["full_finetuning"])
    hp = ft_cfg["finetuning_hyperparams"]
    clf_finetuned = TabPFNFinetuner(
        epochs=hp.get("num_epochs", 200),
        learning_rate=hp.get("learning_rate", 1e-5),
        weight_decay=hp.get("weight_decay", 0.01),
        freeze_feature_attn=hp.get("freeze_feature_attn", False),
        freeze_row_attn=hp.get("freeze_row_attn", False),
        freeze_mlp=hp.get("freeze_mlp", False),
        freeze_decoder=hp.get("freeze_decoder", False),
        train_only_layers=hp.get("train_only_layers", None),
        max_context_size=hp.get("max_context_size", None),
        n_estimators=hp.get("n_estimators", 8),
        device=PFN_DEVICE,
        verbose=True,
    )
    clf_finetuned.fit(X_train, y_train, X_val, y_val)

    return clf_vanilla, clf_finetuned


def fit_tabicl_pair(X_train, y_train, X_val, y_val):
    from tabicl import TabICLClassifier

    # Vanilla -- exact same instantiation as
    # run_xor_seed_sweep_tabicl.py's run_no_finetuning_tabicl().
    clf_vanilla = TabICLClassifier(
        n_estimators=1,
        norm_methods=["none"],
        feat_shuffle_method="none",
        class_shuffle_method="none",
        device=ICL_DEVICE,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_vanilla.fit(X_train, y_train)

    # Full fine-tuning -- exact same hyperparameters as
    # run_xor_seed_sweep_tabicl.py's run_full_finetuning_tabicl().
    clf_finetuned = TabICLFinetuner(
        epochs=50,
        learning_rate=1e-4,
        query_ratio=0.3,
        weight_decay=0.01,
        grad_clip=1.0,
        freeze_col_embedder=False,
        freeze_row_interactor=False,
        freeze_icl_predictor=False,
        warmup_proportion=0.1,
        patience=20,
        device=ICL_DEVICE,
        random_state=42,
        verbose=True,
    )
    clf_finetuned.fit(X_train, y_train, X_val, y_val)

    return clf_vanilla, clf_finetuned


def make_comparison_figure(clf_vanilla, clf_finetuned, X_test, y_test, model_name, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    acc_vanilla = plot_decision_boundary(clf_vanilla, X_test, y_test, axes[0], f"Vanilla {model_name}")
    acc_finetuned = plot_decision_boundary(clf_finetuned, X_test, y_test, axes[1], f"Fine-tuned {model_name}")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=mcolors.Normalize(vmin=0, vmax=1))
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.12, 0.02, 0.76])
    fig.colorbar(sm, cax=cbar_ax, label="P(class = 1)")

    png_path = OUTPUT_DIR / f"{out_stem}.png"
    svg_path = OUTPUT_DIR / f"{out_stem}.svg"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}, {svg_path}")
    return acc_vanilla, acc_finetuned, png_path, svg_path


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, cfg = prepare_seed3_data()
    print(f"Using {FIXED_SEED_NAME}: {cfg}")

    print("Fitting TabPFN v3 pair (vanilla + full fine-tuning) ...")
    pfn_vanilla, pfn_finetuned = fit_tabpfn_pair(X_train, y_train, X_val, y_val)
    pfn_acc_v, pfn_acc_f, pfn_png, pfn_svg = make_comparison_figure(
        pfn_vanilla, pfn_finetuned, X_test, y_test,
        "TabPFN v3", "fig3a_decision_boundary_tabpfn_seed3",
    )

    print("Fitting TabICL v2 pair (vanilla + full fine-tuning) ...")
    icl_vanilla, icl_finetuned = fit_tabicl_pair(X_train, y_train, X_val, y_val)
    icl_acc_v, icl_acc_f, icl_png, icl_svg = make_comparison_figure(
        icl_vanilla, icl_finetuned, X_test, y_test,
        "TabICL v2", "fig3b_decision_boundary_tabicl_seed3",
    )

    section = []
    section.append("")
    section.append("Figure 3a: fig3a_decision_boundary_tabpfn_seed3.{png,svg}")
    section.append("-" * 60)
    section.append(
        f"Data: {FIXED_SEED_NAME} (n=800), same make_dataset_splits() as "
        "run_xor_seed_sweep_tabpfn.py -- NOT read from results.pkl, refit "
        "for this figure only (see script docstring for why)."
    )
    section.append(f"Config used: {cfg}")
    section.append(
        f"In-plot test accuracy (from plot_decision_boundary, this refit): "
        f"vanilla = {pfn_acc_v:.4f}, fine-tuned = {pfn_acc_f:.4f}"
    )
    section.append(
        f"Compare to official results.pkl value for {FIXED_SEED_NAME} in "
        "figure_sources.txt's Figure 1 section above -- small differences "
        "are expected (stochastic fine-tuning), see caveat in script docstring."
    )
    section.append("")
    section.append("Figure 3b: fig3b_decision_boundary_tabicl_seed3.{png,svg}")
    section.append("-" * 60)
    section.append(
        f"Data: {FIXED_SEED_NAME} (n=800), identical X_train/X_val/X_test/"
        "y_train/y_val/y_test as Figure 3a (same underlying data for both models)."
    )
    section.append(
        f"In-plot test accuracy (from plot_decision_boundary, this refit): "
        f"vanilla = {icl_acc_v:.4f}, fine-tuned = {icl_acc_f:.4f}"
    )
    section.append(
        f"Compare to official results.pkl value for {FIXED_SEED_NAME} in "
        "figure_sources.txt's Figure 1 section above -- small differences "
        "are expected (stochastic fine-tuning), see caveat in script docstring."
    )
    section.append("")

    if SOURCES_PATH.exists():
        with open(SOURCES_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(section))
        print(f"Appended figure 3 sources/values to: {SOURCES_PATH}")
    else:
        SOURCES_PATH.write_text("\n".join(section), encoding="utf-8")
        print(f"Sources/values written to: {SOURCES_PATH} "
              f"(figure_sources.txt did not exist yet -- run "
              f"generate_xor_thesis_figures_summary.py too for figures 1+2)")


if __name__ == "__main__":
    main()
