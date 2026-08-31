"""
run_xor_tabicl_selective_seed_task.py
-----------------------------------------
One SLURM-array task = one XOR seed (1-5). Runs all 7 selective TabICL v2
fine-tuning strategies (attention_only, mlp_only, layerwise_icl0,
layerwise_icl3, layerwise_icl6, layerwise_icl8, layerwise_icl11) for its
assigned seed and writes ONLY its own partial result file. Five of these
tasks (run via the SLURM array in submit_xor_tabicl_selective_array.sh)
together cover all 5 seeds; aggregate_xor_tabicl_extended_results.py then
combines the 5 partial files afterwards.

Reused UNCHANGED from run_xor_seed_sweep_tabicl.py (imported, not
duplicated or modified):
  * make_xor()            -- XOR dataset generation
  * make_dataset_splits() -- same carve_out_validation / preprocess_features /
                              encode_labels_train_only pipeline as the
                              original TabICL sweep
  * XOR_CONFIGS           -- same n=800, seeds 1-5, noise=0.01, n_features=10,
                              gap=0.01
  * DEVICE

New in tabicl_selective_finetuner.py (imported, not duplicated here):
  * TabICLSelectiveFinetuner -- a TabICLFinetuner subclass; only
    `_apply_freezing` differs. fit()/predict()/predict_proba()/the training
    loop/validation/early stopping are all inherited UNCHANGED from
    TabICLFinetuner (itself copied verbatim from the reference notebook).

Fine-tuning hyperparameters -- UNCHANGED from the existing TabICL sweep's
run_full_finetuning_tabicl(): epochs=50, learning_rate=1e-4, query_ratio=0.3,
weight_decay=0.01, grad_clip=1.0, warmup_proportion=0.1, patience=20,
random_state=42.

Output isolation
-------------------
Writes ONLY to results/xor_seed_sweep_tabicl_extended_n800/partial/ -- never
touches results/xor_seed_sweep_tabicl_n800/ (the original, already-completed
TabICL sweep) in any way.

  results/xor_seed_sweep_tabicl_extended_n800/partial/seed{N}_results.pkl

Usage
-----
    python run_xor_tabicl_selective_seed_task.py --seed 3
(the SLURM array maps SLURM_ARRAY_TASK_ID -> --seed automatically; see
submit_xor_tabicl_selective_array.sh)
"""

import argparse
import pickle
import time
import warnings
from pathlib import Path

from main import compute_metrics, preprocess_features, encode_labels_train_only
from run_xor_seed_sweep_tabicl import make_dataset_splits, XOR_CONFIGS, DEVICE
from tabicl_selective_finetuner import STRATEGIES, TabICLSelectiveFinetuner

warnings.filterwarnings("ignore")

PARTIAL_DIR = Path("results/xor_seed_sweep_tabicl_extended_n800/partial")
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

# Unchanged from the existing TabICL sweep's full-fine-tuning configuration.
FIXED_HYPERPARAMS = dict(
    epochs=50,
    learning_rate=1e-4,
    query_ratio=0.3,
    weight_decay=0.01,
    grad_clip=1.0,
    warmup_proportion=0.1,
    patience=20,
    random_state=42,
)


def run_one_strategy(strategy, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"  running {strategy} ...")
    finetuner = TabICLSelectiveFinetuner(
        strategy=strategy,
        device=DEVICE,
        verbose=True,
        **FIXED_HYPERPARAMS,
    )

    t0 = time.perf_counter()
    finetuner.fit(X_train, y_train, X_val, y_val)
    training_time = time.perf_counter() - t0

    def _eval(X, y):
        t0 = time.perf_counter()
        y_pred = finetuner.predict(X)
        y_proba = finetuner.predict_proba(X)
        inference_time = time.perf_counter() - t0
        return compute_metrics(y, y_pred, y_proba), inference_time

    train_metrics, _ = _eval(X_train, y_train)
    val_metrics, _ = _eval(X_val, y_val)
    test_metrics, test_inference_time = _eval(X_test, y_test)

    print(
        f"  {strategy} done in {training_time:.1f}s "
        f"(test acc={test_metrics['accuracy']:.4f}, logloss={test_metrics['logloss']:.4f}, "
        f"trainable={finetuner.verification_info['trainable_params']:,}/"
        f"{finetuner.verification_info['total_params']:,} "
        f"({finetuner.verification_info['pct_trainable']:.2f}%))"
    )

    return {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "training_time": training_time,
        "inference_time": test_inference_time,
        "history": finetuner.history,
        "verification_info": finetuner.verification_info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    dataset_name = f"xor_seed{args.seed}_n800"
    cfg = next(c for c in XOR_CONFIGS if c["name"] == dataset_name)
    print(f"=== SLURM array task: {dataset_name} (config={cfg}) ===")

    X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw = make_dataset_splits(cfg)
    X_train, X_val, X_test = preprocess_features(X_train_raw, X_val_raw, X_test_raw)
    y_train, y_val, y_test, val_mask, test_mask, _ = encode_labels_train_only(
        y_train_raw, y_val_raw, y_test_raw
    )
    X_val = X_val[val_mask.values]
    X_test = X_test[test_mask.values]

    seed_result = {"config": cfg}
    for strategy in STRATEGIES:
        seed_result[strategy] = run_one_strategy(
            strategy, X_train, y_train, X_val, y_val, X_test, y_test
        )

    partial_path = PARTIAL_DIR / f"seed{args.seed}_results.pkl"
    with open(partial_path, "wb") as f:
        pickle.dump({dataset_name: seed_result}, f)
    print(f"\nPartial result written to: {partial_path}")


if __name__ == "__main__":
    main()
