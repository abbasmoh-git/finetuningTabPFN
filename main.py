"""
main.py
-------
Entry point for running TabPFN fine-tuning experiments on OpenML datasets.

Data split: 10 folds (6 train / 2 validation / 2 test), one repetition for now.

Usage
-----
# Baseline (no fine-tuning):
    python main.py

# Full fine-tuning:
    python main.py --finetuning_method full_finetuning

# Different config:
    python main.py --config config_1
"""

import argparse
import importlib
import pickle
import warnings
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TabPFN fine-tuning experiments on OpenML datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config_1",
        help="Config module name inside configs/ (e.g. 'config_1').",
    )
    parser.add_argument(
        "--finetuning_method", type=str, default=None,
        choices=["no_finetuning", "full_finetuning"],
        help="Override the finetuning_method from the config.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the 10-fold split.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_name: str) -> dict:
    """Dynamically import a config module and return config_base dict."""
    module = importlib.import_module(f"configs.{config_name}")
    return module.config_base


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_openml_dataset(task_id: int):
    """Load an OpenML task and return X (numpy), y (numpy), dataset name."""
    print(f"Loading OpenML task {task_id} ...")
    task    = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # Encode target labels to integers
    le = LabelEncoder()
    y  = le.fit_transform(y) # Encode target labels as integers
    


    # Convert features to numpy float32
    if isinstance(X, pd.DataFrame):
        X = X.fillna(X.median(numeric_only=True))
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.to_numpy(dtype=np.float32)
    else:
        X = np.array(X, dtype=np.float32)

    print(f"  Dataset : {dataset.name}")
    print(f"  Shape   : {X.shape}")
    print(f"  Classes : {len(np.unique(y))}")
    return X, y, dataset.name


# ---------------------------------------------------------------------------
# 10-fold split: 6 train / 2 val / 2 test
# ---------------------------------------------------------------------------

def make_train_val_test_split(X, y, seed: int):
    """
    Split data into 10 folds (stratified), then assign:
      folds 0-5 → training   (60 %)
      folds 6-7 → validation (20 %)
      folds 8-9 → test       (20 %)

    Returns index arrays: train_idx, val_idx, test_idx
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    # Collect the test indices for each of the 10 folds
    fold_indices = [test_idx for _, test_idx in skf.split(X, y)]

    train_idx = np.concatenate(fold_indices[0:6])
    val_idx   = np.concatenate(fold_indices[6:8])
    test_idx  = np.concatenate(fold_indices[8:10])

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Return accuracy, balanced_accuracy, roc_auc, logloss."""
    metrics = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "logloss":           log_loss(y_true, y_proba),
    }
    if y_proba.shape[1] == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
    else:
        metrics["roc_auc"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
    return metrics


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_no_finetuning(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """TabPFN without fine-tuning (pure in-context learning)."""
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier()
    clf.fit(X_train, y_train) # erstelle das Model mit Trainingsdaten als Kontext ( in context learning) 

    def _eval(X, y):
        return compute_metrics(y, clf.predict(X), clf.predict_proba(X))

    return {
        "train":      _eval(X_train, y_train),
        "validation": _eval(X_val,   y_val),
        "test":       _eval(X_test,  y_test),
    }


def run_full_finetuning(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """TabPFN with full fine-tuning via TabTune."""
    from tabtune.TabularPipeline.pipeline import TabularPipeline

    # TabTune expects DataFrames / Series , not NumPy-Arrays
    to_df = lambda arr: pd.DataFrame(arr) 
    to_s  = lambda arr: pd.Series(arr)

    pipeline = TabularPipeline(
        model_name="TabPFNv26",
        task_type="classification",
        tuning_strategy="finetune",
        finetune_mode="native",
    )
    pipeline.fit(to_df(X_train), to_s(y_train)) # giving tabtune training data -->start withw training data and labels  ( dataframe/ series format)

    def _eval(X, y): #
        X_df = to_df(X)
        return compute_metrics(y, pipeline.predict(X_df), pipeline.predict_proba(X_df))

    return {
        "train":      _eval(X_train, y_train),
        "validation": _eval(X_val,   y_val),
        "test":       _eval(X_test,  y_test),
    }


def run_experiment(X_train, y_train, X_val, y_val, X_test, y_test,
                   finetuning_method: str) -> dict:
    """Dispatch to the correct runner."""
    if finetuning_method == "no_finetuning":
        return run_no_finetuning(X_train, y_train, X_val, y_val, X_test, y_test)
    elif finetuning_method == "full_finetuning":
        return run_full_finetuning(X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        raise ValueError(f"Unknown finetuning_method: '{finetuning_method}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = _parse_args()
    config = load_config(args.config)

    # CLI can override finetuning_method
    finetuning_method = args.finetuning_method or config.get("finetuning_method", "no_finetuning")
    config["finetuning_method"] = finetuning_method

    print(f"\n{'='*55}")
    print(f"Config:            {args.config}")
    print(f"Task ID:           {config['open_ml_task_id']}")
    print(f"Finetuning method: {finetuning_method}")
    print(f"Seed:              {args.seed}")
    print(f"{'='*55}\n")

    # 1) Load dataset
    X, y, dataset_name = load_openml_dataset(config["open_ml_task_id"])

    # 2) Split: 10 folds → 6 train / 2 val / 2 test  (1 repetition for now)
    train_idx, val_idx, test_idx = make_train_val_test_split(X, y, seed=args.seed)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"\nSplit sizes → train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}\n")

    # 3) Fine-tune (or not) and evaluate
    fold_results = run_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test, finetuning_method
    )

    # Print summary
    for split in ("train", "validation", "test"):
        m = fold_results[split]
        print(f"{split:>12} → accuracy: {m['accuracy']:.4f} | "
              f"roc_auc: {m['roc_auc']:.4f} | logloss: {m['logloss']:.4f}")

    # 4) Collect results and save
    # Structure: results[dataset_name]["fold_1"][train/validation/test]
    results = {
        "config": config,
        dataset_name: {
            "fold_1": fold_results,
        },
    }

    save_dir = Path(config["saving_path"]) / finetuning_method #separierte Ordner
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"results_task_{config['open_ml_task_id']}_{args.config}_{finetuning_method}.pkl"

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
