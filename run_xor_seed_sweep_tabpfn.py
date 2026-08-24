"""
run_xor_seed_sweep_tabpfn.py
------------------------------
Controlled XOR seed sweep for the TabPFN v3 fine-tuning pipeline: SAME
sample size (n=800) across all 5 datasets, only the random seed varies
(1, 2, 3, 4, 5). This isolates seed-to-seed variance from size-to-seed
variance, unlike run_xor_sanity_check.py, which varied both seed and size
together across its 5 configs.

Relationship to run_xor_sanity_check.py
------------------------------------------
This script is a copy of run_xor_sanity_check.py with only the dataset
sweep (XOR_CONFIGS) and output directory changed:
  * `make_xor()` -- copied verbatim, unchanged, identical to the other
    script (same source: notebooks/decision_boundary_xor_tabpfn_v3.ipynb).
  * TabPFNFinetuner engine (finetuning_engine.py) -- unchanged, unmodified.
  * main.py -- unchanged, unmodified. Only imported (carve_out_validation,
    preprocess_features, encode_labels_train_only, run_no_finetuning,
    run_own_finetuning), never edited.
  * Hyperparameters (learning_rate=1e-4, num_epochs=200, weight_decay=0.01,
    max_context_size=3000, n_estimators=8) -- identical to
    run_xor_sanity_check.py, for direct comparability.
  * The 4 fine-tuning strategies (full / attention-only / mlp-only /
    layer-wise block 0) -- identical.

Output isolation
-------------------
Writes ONLY to results/xor_seed_sweep_tabpfn_n800/ -- a new directory. This
script never imports or calls anything that writes under
results/finetuning_experiments/ (TabArena), results/thesis_report/ (report
generator), or results/xor_sanity_check/ (the original XOR sanity check).
Running this script cannot alter or overwrite any of those.

  results/xor_seed_sweep_tabpfn_n800/
    xor_configs.json   -- the 5 dataset configs (seed, size, ...) for reproducibility
    results.pkl         -- full nested results (all metrics, all strategies, all datasets)
    summary.md           -- compact baseline vs. fine-tuned table + deltas

Usage
-----
    python run_xor_seed_sweep_tabpfn.py
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

OUTPUT_DIR = Path("results/xor_seed_sweep_tabpfn_n800")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Identical to run_xor_sanity_check.py, for direct comparability ---
XOR_LEARNING_RATE = 1e-4
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
# XOR generation -- copied verbatim, unchanged, from
# notebooks/decision_boundary_xor_tabpfn_v3.ipynb (same source as
# run_xor_sanity_check.py). Do not edit without checking the source notebook.
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
# 5 XOR dataset variants -- CONTROLLED sweep: n_samples fixed at 800,
# only random_state varies (1, 2, 3, 4, 5). noise/n_features/gap held fixed
# at the reference notebook's defaults, same as run_xor_sanity_check.py.
# ---------------------------------------------------------------------------
XOR_CONFIGS = [
    {"name": "xor_seed1_n800", "n_samples": 800, "noise": 0.01, "n_features": N_FEATURES, "random_state": 1, "gap": 0.01},
    {"name": "xor_seed2_n800", "n_samples": 800, "noise": 0.01, "n_features": N_FEATURES, "random_state": 2, "gap": 0.01},
    {"name": "xor_seed3_n800", "n_samples": 800, "noise": 0.01, "n_features": N_FEATURES, "random_state": 3, "gap": 0.01},
    {"name": "xor_seed4_n800", "n_samples": 800, "noise": 0.01, "n_features": N_FEATURES, "random_state": 4, "gap": 0.01},
    {"name": "xor_seed5_n800", "n_samples": 800, "noise": 0.01, "n_features": N_FEATURES, "random_state": 5, "gap": 0.01},
]

# ---------------------------------------------------------------------------
# Fine-tuning strategies -- identical to run_xor_sanity_check.py / the
# TabArena experiments (see configs/config_{own,attention_only,mlp_only,
# layerwise}_finetuning.py). Layer-wise uses block 0 only.
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
    lines.append("# XOR Controlled Seed Sweep (n=800) -- Baseline vs. Fine-Tuning\n")
    lines.append(
        f"Learning rate used: {XOR_LEARNING_RATE} (1e-4), following the XOR configuration "
        "used in the project's reference notebook (decision_boundary_xor_tabpfn_v3.ipynb), "
        "identical to run_xor_sanity_check.py. The main TabArena experiments used 1e-5.\n"
    )
    lines.append(
        "All 5 datasets use n_samples=800; only random_state (1-5) varies, isolating "
        "seed-to-seed variance from the size-to-seed confound present in "
        "run_xor_sanity_check.py's original 5-dataset sweep.\n"
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
