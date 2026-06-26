"""
main.py
-------
Benchmark-based fine-tuning evaluation pipeline for TabPFN / TabTune,
following the TabArena / OpenML suite (suite_id=457) evaluation protocol.

Key differences vs. the previous single-task version:
  * Datasets come from the TabArena OpenML benchmark suite, not a single
    hand-picked task ID.
  * Train/test splits come directly from OpenML's official task folds/repeats
    (no custom split of any kind). The official test split is never touched.
    A validation split is carved out of the official *training* portion only
    (see `carve_out_validation`), since OpenML tasks only define train/test.
  * LabelEncoder is fit on y_train ONLY; validation and test are transformed
    with that same encoder (see `encode_labels_train_only` for how unseen
    labels are handled).
  * Every fold/repeat run is stored individually, then aggregated
    (mean / std / variance / CI) per dataset, method, and metric.
  * Training and inference time are recorded for every run.
  * Fine-tuning hyperparameters live in the config, not hardcoded.
  * Results are saved both as a pickle (full detail) and a flat CSV
    (one row per dataset/method/fold/repetition/split) for plotting.

Usage
-----
# Baseline (no fine-tuning), full suite:
    python main.py

# Full fine-tuning:
    python main.py --finetuning_method full_finetuning

# Quick debug run on a handful of tasks, 1 fold / 1 repeat each:
    python main.py --max_tasks 3 --lite

# Different config:
    python main.py --config config_1
"""

import argparse
import importlib
import pickle
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import openml
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TabPFN/TabTune fine-tuning experiments on the TabArena/OpenML benchmark suite.",
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
        "--max_tasks", type=int, default=None,
        help="Limit the number of benchmark tasks to run (useful for testing/debugging).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Reserved for reproducibility of any future stochastic step. "
             "Currently unused: all splits come from OpenML's deterministic "
             "official folds/repeats.",
    )
    parser.add_argument(
        "--lite", action="store_true",
        help="Lite evaluation mode: 1 fold / 1 repeat per task, for fast debugging.",
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
# Benchmark task discovery (TabArena / OpenML suite)
# ---------------------------------------------------------------------------

def get_benchmark_task_ids(suite_id: int, max_tasks: Optional[int] = None) -> list:
    """Return classification task IDs from a given OpenML benchmark suite.

    Parameters
    ----------
    suite_id : OpenML study/suite ID (TabArena = 457).
    max_tasks : if given, only the first `max_tasks` *classification* task
        IDs are returned (useful for quick testing/debugging).

    Notes
    -----
    TabArena (suite 457) contains both classification and regression tasks.
    This pipeline only implements classification runners/metrics, so
    non-classification tasks are filtered out here -- using a metadata-only
    `get_task()` call (no dataset download) -- rather than relying on
    `--max_tasks` to "get lucky" and pick a classification task, and rather
    than crashing deep inside TabPFN on a continuous target.
    """
    suite = openml.study.get_suite(suite_id)
    all_task_ids = list(suite.tasks)

    classification_task_ids = []
    for task_id in all_task_ids:
        task = openml.tasks.get_task(task_id)  # metadata only, no dataset download
        if task.task_type == "Supervised Classification":
            classification_task_ids.append(task_id)
        else:
            print(f"  [skip] task {task_id} is '{task.task_type}', not classification")
        if max_tasks is not None and len(classification_task_ids) >= max_tasks:
            break

    return classification_task_ids


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class UnsupportedTaskTypeError(Exception):
    """Raised when a benchmark task is not a (binary/multiclass) classification
    task. Suite 457 (TabArena) contains regression tasks as well; this
    pipeline only implements classification metrics/runners, so such tasks
    are skipped rather than silently mis-treating a continuous target as
    class labels."""


def load_openml_task(task_id: int):
    """Load an OpenML task + dataset, return (task, dataset, X, y, dataset_name).

    X is returned as a pandas DataFrame and y as a pandas Series, both still
    in their *raw* form (no encoding/imputation). This is deliberate: any
    encoding/imputation must be fit on the training split only, which can
    differ per fold/repeat. See `preprocess_features` and
    `encode_labels_train_only`.

    Raises
    ------
    UnsupportedTaskTypeError
        If the task is not a classification task (e.g. regression).
    """
    print(f"Loading OpenML task {task_id} ...")
    task = openml.tasks.get_task(task_id)

    if task.task_type != "Supervised Classification":
        raise UnsupportedTaskTypeError(
            f"task {task_id} is '{task.task_type}', not 'Supervised "
            "Classification' — skipping (this pipeline only supports "
            "classification)."
        )

    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    print(f"  Dataset : {dataset.name}")
    print(f"  Shape   : {X.shape}")
    print(f"  Classes : {y.nunique()}")
    return task, dataset, X, y, dataset.name


# ---------------------------------------------------------------------------
# Fold / repeat bookkeeping (TabArena-style; adapted from
# https://github.com/amirbalef/is_one_layer_enough/blob/78c9061b4f6ba36a847e7700e154a60bd95681cf/Experiments/util.py#L831
# and #L873)
# ---------------------------------------------------------------------------

def determine_repeats_and_folds(task, dataset, lite_evaluation: bool = False) -> tuple:
    """Return (n_repeats, n_folds) to actually run, following a TabArena-style
    protocol:

      * lite_evaluation=True  -> always (1, 1), for fast debugging.
      * otherwise: folds come from the task's own CV definition
        (`task.get_split_dimensions()`), capped at 3; the number of repeats
        is capped by dataset size:
          - small datasets  (< 2 500 samples) -> up to 10 repeats
          - larger datasets (>= 2 500 samples) -> up to 3 repeats
        but never more than what the OpenML task actually provides.
    """
    if lite_evaluation:
        return 1, 1

    n_repeats_available, n_folds_available, _ = task.get_split_dimensions()
    n_samples = dataset.qualities.get("NumberOfInstances", np.inf)

    target_repeats = 10 if n_samples < 2_500 else 3
    n_repeats = max(1, min(target_repeats, n_repeats_available))
    n_folds = max(1, min(3, n_folds_available))

    return n_repeats, n_folds


def get_openml_splits(task, fold: int, repeat: int):
    """Return (train_idx, test_idx) for the requested OpenML fold/repeat,
    using the official task split (no custom split of any kind)."""
    train_idx, test_idx = task.get_train_test_split_indices(fold=fold, repeat=repeat)
    return train_idx, test_idx


def carve_out_validation(
    X_train_raw: pd.DataFrame, y_train_raw: pd.Series,
    validation_fraction: float = 0.2, seed: int = 42,
):
    """Carve a stratified validation split out of the OFFICIAL training
    portion only. OpenML tasks only define train/test indices, so the
    validation set cannot come "from OpenML" directly; instead it is split
    off from the training rows, leaving the official test split untouched.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_raw, y_train_raw,
        test_size=validation_fraction,
        stratify=y_train_raw,
        random_state=seed,
    )
    return X_tr, X_val, y_tr, y_val


# ---------------------------------------------------------------------------
# Preprocessing: fit on TRAIN only, then apply to validation/test
# ---------------------------------------------------------------------------

def preprocess_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """Impute + encode categoricals, fitting all statistics on X_train only.

      * Numeric NaNs are filled with the *training* median.
      * Categorical/object columns are label-encoded using categories seen in
        *training*; categories seen only in validation/test are mapped to a
        dedicated "unknown" code (-1) instead of crashing.
    """
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_val   = X_val.fillna(medians)
    X_test  = X_test.fillna(medians)

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(X_train[col].astype(str))
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}

        def _encode(series, mapping=mapping):
            return series.astype(str).map(lambda v: mapping.get(v, -1))

        X_train[col] = _encode(X_train[col])
        X_val[col]   = _encode(X_val[col])
        X_test[col]  = _encode(X_test[col])

    return (
        X_train.to_numpy(dtype=np.float32),
        X_val.to_numpy(dtype=np.float32),
        X_test.to_numpy(dtype=np.float32),
    )


def encode_labels_train_only(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series):
    """Fit LabelEncoder on y_train ONLY, then transform y_val and y_test.

    Behaviour for labels in validation/test that were never seen during
    training: such rows cannot be meaningfully scored by a model that has no
    notion of that class, so they are DROPPED from that split. A warning
    with the number of dropped rows is printed per split. Returns boolean
    masks (aligned with the original y_val/y_test index) so the caller can
    apply the same filtering to the matching X rows.
    """
    le = LabelEncoder()
    le.fit(y_train)
    known = set(le.classes_)

    y_train_enc = le.transform(y_train)

    def _safe_transform(y_split: pd.Series, split_name: str):
        mask = y_split.isin(known)
        n_dropped = int((~mask).sum())
        if n_dropped > 0:
            print(f"  [warn] dropping {n_dropped} row(s) from {split_name} "
                  f"with label(s) unseen during training")
        return le.transform(y_split[mask]), mask

    y_val_enc, val_mask   = _safe_transform(y_val, "validation")
    y_test_enc, test_mask = _safe_transform(y_test, "test")

    return y_train_enc, y_val_enc, y_test_enc, val_mask, test_mask, le


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Return accuracy, balanced_accuracy, roc_auc, logloss."""
    n_classes = y_proba.shape[1]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "logloss": log_loss(y_true, y_proba, labels=np.arange(n_classes)),
    }
    if n_classes == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
    else:
        metrics["roc_auc"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro",
            labels=np.arange(n_classes),
        )
    return metrics


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_no_finetuning(X_train, y_train, X_val, y_val, X_test, y_test, config: dict) -> dict:
    """TabPFN without fine-tuning (pure in-context learning)."""
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(ignore_pretraining_limits=True)

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    training_time = time.perf_counter() - t0

    def _eval(X, y):
        t0 = time.perf_counter()
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)
        inference_time = time.perf_counter() - t0
        return compute_metrics(y, y_pred, y_proba), inference_time

    train_metrics, _ = _eval(X_train, y_train)
    val_metrics, _ = _eval(X_val, y_val)
    test_metrics, test_inference_time = _eval(X_test, y_test)

    return {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "training_time": training_time,
        "inference_time": test_inference_time,
    }


def run_full_finetuning(X_train, y_train, X_val, y_val, X_test, y_test, config: dict) -> dict:
    """TabPFN with full fine-tuning via TabTune."""
    from tabtune.TabularPipeline.pipeline import TabularPipeline

    to_df = lambda arr: pd.DataFrame(arr)
    to_s = lambda arr: pd.Series(arr)

    ft_cfg = config.get("finetuning_hyperparams", {})

    pipeline = TabularPipeline(
        model_name=config.get("tabtune_model_name", "TabPFNv26"),
        task_type="classification",
        tuning_strategy="finetune",
        finetune_mode="native",
        learning_rate=ft_cfg.get("learning_rate", 1e-5),
        num_epochs=ft_cfg.get("num_epochs", 10),
        batch_size=ft_cfg.get("batch_size", 32),
        **ft_cfg.get("extra_kwargs", {}),
    )

    t0 = time.perf_counter()
    pipeline.fit(to_df(X_train), to_s(y_train))
    training_time = time.perf_counter() - t0

    def _eval(X, y):
        X_df = to_df(X)
        t0 = time.perf_counter()
        y_pred = pipeline.predict(X_df)
        y_proba = pipeline.predict_proba(X_df)
        inference_time = time.perf_counter() - t0
        return compute_metrics(y, y_pred, y_proba), inference_time

    train_metrics, _ = _eval(X_train, y_train)
    val_metrics, _ = _eval(X_val, y_val)
    test_metrics, test_inference_time = _eval(X_test, y_test)

    return {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "training_time": training_time,
        "inference_time": test_inference_time,
    }


def run_experiment(X_train, y_train, X_val, y_val, X_test, y_test,
                    finetuning_method: str, config: dict) -> dict:
    """Dispatch to the correct runner."""
    if finetuning_method == "no_finetuning":
        return run_no_finetuning(X_train, y_train, X_val, y_val, X_test, y_test, config)
    elif finetuning_method == "full_finetuning":
        return run_full_finetuning(X_train, y_train, X_val, y_val, X_test, y_test, config)
    else:
        raise ValueError(f"Unknown finetuning_method: '{finetuning_method}'")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_ci(values: np.ndarray, confidence: float = 0.95) -> dict:
    """Mean, std, variance, and a normal-approximation confidence interval."""
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    variance = std ** 2
    if n > 1:
        z = 1.96  # ~95% normal approximation
        margin = z * std / np.sqrt(n)
        ci = (mean - margin, mean + margin)
    else:
        ci = (mean, mean)
    return {"mean": mean, "std": std, "variance": variance, "confidence_interval": ci}


def summarize_results(runs: dict, split: str = "test") -> dict:
    """Aggregate metrics across all repeat/fold runs for one dataset/method.

    `runs` is the {"repeat_X_fold_Y": {...}} dict for a single dataset/method.
    Returns {metric_name: {mean, std, variance, confidence_interval}}.
    """
    metric_names = ["accuracy", "balanced_accuracy", "roc_auc", "logloss"]
    summary = {}
    for metric in metric_names:
        values = np.array([run[split][metric] for run in runs.values()])
        summary[metric] = _mean_ci(values)
    return summary


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_results(results: dict, save_dir: Path, filename: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / filename
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    return output_file


def export_csv(results: dict, csv_path: Path) -> Path:
    """Flatten results into one row per dataset/method/fold/repetition/split,
    suitable for later plotting (one CSV row per data point)."""
    rows = []
    for dataset_name, dataset_entry in results["datasets"].items():
        task_id = dataset_entry["task_id"]
        method = dataset_entry["method"]
        for run_key, run in dataset_entry["runs"].items():
            # run_key looks like "repeat_0_fold_0"
            parts = run_key.split("_")
            repeat = int(parts[1])
            fold = int(parts[3])
            for split in ("train", "validation", "test"):
                m = run[split]
                rows.append({
                    "dataset_name": dataset_name,
                    "task_id": task_id,
                    "method": method,
                    "repetition": repeat,
                    "fold": fold,
                    "split": split,
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "roc_auc": m["roc_auc"],
                    "logloss": m["logloss"],
                    "training_time": run["training_time"],
                    "inference_time": run["inference_time"],
                })
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Per-task driver
# ---------------------------------------------------------------------------

def run_task(task_id: int, finetuning_method: str, config: dict, lite: bool):
    """Run all required folds/repetitions for a single OpenML task, using
    OpenML's official train/test split for every fold/repeat. A validation
    split is carved out of the official training portion only (see
    `carve_out_validation`) — the official test split is never touched.

    Returns (dataset_result_dict, dataset_name).
    """
    task, dataset, X, y, dataset_name = load_openml_task(task_id)
    n_repeats, n_folds = determine_repeats_and_folds(task, dataset, lite_evaluation=lite)
    print(f"  Repeats x folds to run: {n_repeats} x {n_folds}")

    validation_fraction = config.get("validation_fraction", 0.2)
    validation_seed = config.get("validation_seed", 42)

    runs = {}
    for repeat in range(n_repeats):
        for fold in range(n_folds):
            run_key = f"repeat_{repeat}_fold_{fold}"
            print(f"  -> {run_key}")

            train_idx, test_idx = get_openml_splits(task, fold=fold, repeat=repeat)
            X_train_full, y_train_full = X.iloc[train_idx], y.iloc[train_idx]
            X_test_raw, y_test_raw = X.iloc[test_idx], y.iloc[test_idx]

            # Carve validation out of the official TRAINING rows only;
            # the official OpenML test split stays untouched.
            X_train_raw, X_val_raw, y_train_raw, y_val_raw = carve_out_validation(
                X_train_full, y_train_full,
                validation_fraction=validation_fraction,
                seed=validation_seed,
            )

            y_train, y_val, y_test, val_mask, test_mask, _ = encode_labels_train_only(
                y_train_raw, y_val_raw, y_test_raw
            )
            X_val_raw = X_val_raw[val_mask.to_numpy()]
            X_test_raw = X_test_raw[test_mask.to_numpy()]

            X_train, X_val, X_test = preprocess_features(X_train_raw, X_val_raw, X_test_raw)

            fold_result = run_experiment(
                X_train, y_train, X_val, y_val, X_test, y_test,
                finetuning_method, config,
            )
            runs[run_key] = fold_result

    dataset_result = {
        "task_id": task_id,
        "method": finetuning_method,
        "runs": runs,
        "summary": {
            "test": summarize_results(runs, split="test"),
        },
    }
    return dataset_result, dataset_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    finetuning_method = args.finetuning_method or config.get("finetuning_method", "no_finetuning")
    config["finetuning_method"] = finetuning_method

    suite_id = config.get("suite_id", 457)
    task_ids = get_benchmark_task_ids(suite_id, max_tasks=args.max_tasks)

    print(f"\n{'='*55}")
    print(f"Config:            {args.config}")
    print(f"Benchmark suite:   {suite_id} (TabArena/OpenML)")
    print(f"Num tasks:         {len(task_ids)}")
    print(f"Finetuning method: {finetuning_method}")
    print(f"Seed:              {args.seed}")
    print(f"Lite evaluation:   {args.lite}")
    print(f"{'='*55}\n")

    results = {
        "config": config,
        "benchmark": {
            "suite_id": suite_id,
            "name": "TabArena / OpenML benchmark",
            "task_ids": task_ids,
        },
        "datasets": {},
    }

    for task_id in task_ids:
        try:
            dataset_result, dataset_name = run_task(
                task_id, finetuning_method, config, lite=args.lite
            )
        except UnsupportedTaskTypeError as e:
            print(f"  [skip] {e}")
            continue

        results["datasets"][dataset_name] = dataset_result

        test_summary = dataset_result["summary"]["test"]
        print(f"\n  [{dataset_name}] test summary:")
        for metric, stats in test_summary.items():
            print(f"    {metric:>18}: mean={stats['mean']:.4f}  std={stats['std']:.4f}")

    save_dir = Path(config["saving_path"]) / finetuning_method
    pkl_file = save_results(
        results, save_dir, f"results_suite_{suite_id}_{args.config}_{finetuning_method}.pkl"
    )
    csv_file = export_csv(
        results, save_dir / f"results_suite_{suite_id}_{args.config}_{finetuning_method}.csv"
    )

    print(f"\nResults saved to: {pkl_file}")
    print(f"CSV exported to:  {csv_file}")


if __name__ == "__main__":
    main()
