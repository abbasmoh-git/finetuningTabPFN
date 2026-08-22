"""
run_xor_sanity_check.py
------------------------
Synthetic XOR sanity check for the TabPFN v3 fine-tuning pipeline.

Why this exists
----------------
The TabArena benchmark results (results/thesis_report/tables.md, Table 5.5 /
5.6) show close-to-zero deltas for every fine-tuning strategy on real-world
tabular datasets. Before concluding "fine-tuning has no measurable effect",
this script checks that the pipeline itself CAN move the needle at all when
a task genuinely benefits from fine-tuning: TabPFN v3's pretrained
in-context learning is not designed for a hard, non-linear XOR/checkerboard
decision boundary, so if fine-tuning is implemented correctly it should
produce a large, clearly non-zero effect here. A near-zero effect on THIS
task (unlike on TabArena) would point to a pipeline bug rather than a
genuine "fine-tuning doesn't help on tabular data" finding.

XOR generation -- reused verbatim, not reinvented
---------------------------------------------------
`make_xor()` below is copied VERBATIM (same parameters, same logic, same
default values) from Amir Rezaei Balef's
`notebooks/decision_boundary_xor_tabpfn_v3.ipynb` (merged into `main` via
the `merge-amir` branch), per explicit instruction to reuse his exact
formulation rather than inventing a different XOR/checkerboard definition.
Do not modify this function without first checking that notebook.

Fine-tuning hyperparameters -- LEARNING RATE DISCREPANCY, FLAGGED
--------------------------------------------------------------------
This script uses the TabPFNFinetuner engine (finetuning_engine.py) with, by
default, TabArena-consistent hyperparameters: learning_rate=1e-5,
num_epochs=200, weight_decay=0.01, max_context_size=3000, n_estimators=8 --
i.e. exactly config_own_finetuning.py / config_attention_only_finetuning.py
/ config_mlp_only_finetuning.py / config_layerwise_finetuning.py, just
pointed at XOR data instead of an OpenML task.

FLAGGED DISCREPANCY (reported here rather than silently resolved, as
instructed): Amir's own notebook demo used learning_rate=1e-4 (not 1e-5),
plus query_ratio=0.3, epochs=50, patience=20, grad_clip=1.0,
warmup_proportion=0.1. TabPFNFinetuner as it exists today does not
implement query_ratio / patience / grad_clip / warmup_proportion at all (it
uses max_context_size's proportional split for validation instead, and has
no early stopping or gradient clipping) -- so an exact reproduction of
Amir's notebook settings is not possible without changing the engine
itself. Given the instruction to keep "the same fine-tuning engine and main
hyperparameters as the TabArena experiments, including learning rate
1e-5", this script defaults to 1e-5. Set XOR_LEARNING_RATE below to 1e-4
and rerun (writes into the same output directory, overwriting only XOR
results) if a comparison against Amir's exact notebook learning rate is
wanted instead -- flagging this choice rather than picking silently.

Guarantee: existing TabArena results untouched
------------------------------------------------
This script never imports/calls anything that writes under
results/finetuning_experiments/ (the TabArena experiment root) or
results/thesis_report/ (the report generator's output). All output goes to
results/xor_sanity_check/, a directory this script owns exclusively:

  results/xor_sanity_check/
    xor_configs.json   -- the 5 dataset configs (seed, size, ...) for reproducibility
    results.pkl         -- full nested results (all metrics, all strategies, all datasets)
    summary.md           -- compact baseline vs. fine-tuned table + deltas

Usage
-----
    python run_xor_sanity_check.py
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from main import (
    carve_out_validation,
    preprocess_features,
    encode_labels_train_only,
    run_no_finetuning,
    run_own_finetuning,
)

OUTPUT_DIR = Path("results/xor_sanity_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- TabArena-consistent hyperparameters (see module docstring for the
#     learning_rate discrepancy vs. Amir's own notebook demo) ---
XOR_LEARNING_RATE = 1e-5
XOR_NUM_EPOCHS = 200
XOR_WEIGHT_DECAY = 0.01
XOR_MAX_CONTEXT_SIZE = 3000  # never actually triggered -- XOR datasets are far below this
XOR_N_ESTIMATORS = 8
DEVICE = "cuda"  # set to "cpu" for a quick local smoke test without a GPU

VALIDATION_FRACTION = 0.2   # carved out of the train+val portion, same as TabArena
TEST_FRACTION = 0.25        # XOR has no official OpenML train/test split, so this
                             # stratified hold-out plays the role that role for real tasks
SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# XOR generation -- copied verbatim from
# notebooks/decision_boundary_xor_tabpfn_v3.ipynb (Amir Rezaei Balef).
# Do not edit without checking the source notebook first.
# ---------------------------------------------------------------------------
N_FEATURES = 10  # total number of features (checker label uses first 2; rest are noise)


def make_xor(n_samples: int = 400, noise: float = 0.01, n_features: int = N_FEATURES,
             random_state: int = 42, gap: float = 0.01):
    """Checkerboard classification with 2 grids per axis (4x4 tiles).
    First 2 features carry the signal; remaining features are Gaussian noise.
    Label = (floor(x1 * 2) mod 2) XOR (floor(x2 * 2) mod 2).
    gap: width of empty margin around each tile boundary (in data units)."""
    rng = np.random.default_rng(random_state)

    boundaries = np.array([-0.5, 0.0, 0.5])
    half_gap = gap / 2
    oversample_factor = int(np.ceil(1 / (1 - gap * len(boundaries)) ** 2)) + 2

    X_signal = rng.uniform(-1, 1, size=(n_samples * oversample_factor, 2))
    dist_x = np.min(np.abs(X_signal[:, 0:1] - boundaries), axis=1)
    dist_y = np.min(np.abs(X_signal[:, 1:2] - boundaries), axis=1)
    keep = (dist_x >= half_gap) & (dist_y >= half_gap)
    X_signal = X_signal[keep][:n_samples]

    grid_x = np.floor(X_signal[:, 0] * 2).astype(int) % 2
    grid_y = np.floor(X_signal[:, 1] * 2).astype(int) % 2
    y = (grid_x ^ grid_y).astype(int)
    X_signal += rng.normal(0, noise, size=X_signal.shape)

    if n_features > 2:
        X_noise = rng.normal(0, 0.5, size=(len(X_signal), n_features - 2))
        X = np.concatenate([X_signal, X_noise], axis=1)
    else:
        X = X_signal

    return X.astype(np.float32), y


# ---------------------------------------------------------------------------
# 5 XOR dataset variants (varying seed and/or size), stored to JSON for
# reproducibility. noise/n_features/gap held fixed at Amir's notebook
# defaults; only random_state and n_samples vary across the 5 variants.
# ---------------------------------------------------------------------------
XOR_CONFIGS = [
    {"name": "xor_seed42_n400",   "n_samples": 400, "noise": 0.01, "n_features": N_FEATURES, "random_state": 42,   "gap": 0.01},
    {"name": "xor_seed123_n400",  "n_samples": 400, "noise": 0.01, "n_features": N_FEATURES, "random_state": 123,  "gap": 0.01},
    {"name": "xor_seed7_n800",    "n_samples": 800, "noise": 0.01, "n_features": N_FEATURES, "random_state": 7,    "gap": 0.01},
    {"name": "xor_seed2026_n200", "n_samples": 200, "noise": 0.01, "n_features": N_FEATURES, "random_state": 2026, "gap": 0.01},
    {"name": "xor_seed99_n600",   "n_samples": 600, "noise": 0.01, "n_features": N_FEATURES, "random_state": 99,   "gap": 0.01},
]

# ---------------------------------------------------------------------------
# Fine-tuning strategies -- the same 4 selective variants used in the
# TabArena experiments (see configs/config_{own,attention_only,mlp_only,
# layerwise}_finetuning.py). Layer-wise uses block 0 only, as instructed.
# ---------------------------------------------------------------------------
STRATEGIES = {
    "full_finetuning": dict(
        freeze_feature_attn=False, freeze_row_attn=False,
        freeze_mlp=False, freeze_decoder=False, train_only_layers=None,
    ),
    "attention_only": dict(
        freeze_feature_attn=False, freeze_row_attn=False,
        freeze_mlp=True, freeze_decoder=True, train_only_layers=None,
    ),
    "mlp_only": dict(
        freeze_feature_attn=True, freeze_row_attn=True,
        freeze_mlp=False, freeze_decoder=True, train_only_layers=None,
    ),
    "layerwise_layer0": dict(
        freeze_feature_attn=True, freeze_row_attn=True,
        freeze_mlp=True, freeze_decoder=True, train_only_layers=[0],
    ),
}


def build_finetuning_config(strategy_flags: dict) -> dict:
    return {
        "device": DEVICE,
        "finetuning_hyperparams": {
            "learning_rate": XOR_LEARNING_RATE,
            "num_epochs": XOR_NUM_EPOCHS,
            "weight_decay": XOR_WEIGHT_DECAY,
            "max_context_size": XOR_MAX_CONTEXT_SIZE,
            "n_estimators": XOR_N_ESTIMATORS,
            **strategy_flags,
        },
    }


def make_dataset_splits(cfg: dict):
    """Generate one XOR dataset and split it train/val/test the same way as
    the TabArena pipeline: a stratified hold-out plays the role of OpenML's
    official test split (XOR has no such official split), and
    carve_out_validation() (imported from main.py, unmodified) then carves
    the validation set out of the remaining training portion only."""
    X, y = make_xor(
        n_samples=cfg["n_samples"], noise=cfg["noise"],
        n_features=cfg["n_features"], random_state=cfg["random_state"], gap=cfg["gap"],
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_s = pd.Series(y, name="target")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_df, y_s, test_size=TEST_FRACTION, stratify=y_s, random_state=SPLIT_SEED,
    )
    X_train, X_val, y_train, y_val = carve_out_validation(
        X_trainval, y_trainval, validation_fraction=VALIDATION_FRACTION, seed=SPLIT_SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_one_dataset(cfg: dict) -> dict:
    print(f"\n=== XOR dataset: {cfg['name']} (n={cfg['n_samples']}, seed={cfg['random_state']}) ===")
    X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw = make_dataset_splits(cfg)

    X_train, X_val, X_test = preprocess_features(X_train_raw, X_val_raw, X_test_raw)
    y_train, y_val, y_test, val_mask, test_mask, _ = encode_labels_train_only(
        y_train_raw, y_val_raw, y_test_raw
    )
    # Masks are boolean pandas Series aligned with y_val_raw/y_test_raw's row
    # order; X_val/X_test are numpy arrays in that same row order (from
    # preprocess_features), so .values indexes them correctly.
    X_val = X_val[val_mask.values]
    X_test = X_test[test_mask.values]

    dataset_result = {"config": cfg}

    print("  running baseline (no fine-tuning) ...")
    t0 = time.perf_counter()
    baseline_res = run_no_finetuning(X_train, y_train, X_val, y_val, X_test, y_test, {"device": DEVICE})
    print(f"  baseline done in {time.perf_counter() - t0:.1f}s "
          f"(test acc={baseline_res['test']['accuracy']:.4f}, "
          f"logloss={baseline_res['test']['logloss']:.4f})")
    dataset_result["no_finetuning"] = baseline_res

    for strategy_name, flags in STRATEGIES.items():
        print(f"  running {strategy_name} ...")
        ft_cfg = build_finetuning_config(flags)
        t0 = time.perf_counter()
        res = run_own_finetuning(X_train, y_train, X_val, y_val, X_test, y_test, ft_cfg)
        print(f"  {strategy_name} done in {time.perf_counter() - t0:.1f}s "
              f"(test acc={res['test']['accuracy']:.4f}, "
              f"logloss={res['test']['logloss']:.4f})")
        dataset_result[strategy_name] = res

    return dataset_result


def summarize_and_save(all_results: dict):
    with open(OUTPUT_DIR / "results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    with open(OUTPUT_DIR / "xor_configs.json", "w", encoding="utf-8") as f:
        json.dump(XOR_CONFIGS, f, indent=2)

    lines = []
    lines.append("# XOR Sanity Check -- Baseline vs. Fine-Tuning\n")
    lines.append(
        f"Learning rate used: {XOR_LEARNING_RATE} (TabArena-consistent; Amir's notebook "
        "demo used 1e-4 -- see module docstring in run_xor_sanity_check.py for the "
        "flagged discrepancy and how to rerun with 1e-4 instead).\n"
    )
    lines.append(
        "| Dataset | Strategy | Accuracy | Bal. Accuracy | ROC-AUC | Neg. Log Loss | "
        "Δ Accuracy | Δ Neg. Log Loss |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    for cfg in XOR_CONFIGS:
        name = cfg["name"]
        res = all_results[name]
        base = res["no_finetuning"]["test"]
        base_neg_ll = -base["logloss"]
        lines.append(
            f"| {name} | baseline | {base['accuracy']:.4f} | {base['balanced_accuracy']:.4f} | "
            f"{base['roc_auc']:.4f} | {base_neg_ll:.4f} | - | - |"
        )
        for strategy_name in STRATEGIES:
            m = res[strategy_name]["test"]
            neg_ll = -m["logloss"]
            d_acc = m["accuracy"] - base["accuracy"]
            d_negll = neg_ll - base_neg_ll
            lines.append(
                f"| {name} | {strategy_name} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | "
                f"{m['roc_auc']:.4f} | {neg_ll:.4f} | {d_acc:+.4f} | {d_negll:+.4f} |"
            )

    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary written to: {summary_path}")
    print(f"Full results: {OUTPUT_DIR / 'results.pkl'}")
    print(f"Dataset configs: {OUTPUT_DIR / 'xor_configs.json'}")


def main():
    all_results = {}
    for cfg in XOR_CONFIGS:
        all_results[cfg["name"]] = run_one_dataset(cfg)
    summarize_and_save(all_results)


if __name__ == "__main__":
    main()
