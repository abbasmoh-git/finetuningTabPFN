"""
run_xor_seed_sweep_tabicl.py
-------------------------------
Same controlled XOR seed sweep as run_xor_seed_sweep_tabpfn.py (n=800,
random_state = 1, 2, 3, 4, 5 -- only the seed varies, size held fixed), but
for TabICL v2 instead of TabPFN v3, using the `TabICLFinetuner` class from
the project's reference notebook `notebooks/decision_boundary_xor_tabicl.ipynb`.

Status: PREPARATION ONLY
--------------------------
This script has been written and is ready to run, but has not been executed
yet. It performs no new scientific interpretation and does not touch any
existing result (TabArena, the two TabPFN XOR sweeps, or anything else).
Run it only when you're ready to generate the TabICL-side comparison.

What was copied verbatim vs. what was changed
--------------------------------------------------
  * `make_xor()` -- copied verbatim, unchanged, from the reference notebook
    (identical to the TabPFN v3 reference notebook's version -- same
    function, same defaults).
  * `CosineWarmupScheduler` and `TabICLFinetuner` -- copied verbatim,
    unchanged, from notebooks/decision_boundary_xor_tabicl.ipynb, including
    its `_load_pretrained` / `_apply_freezing` / `_make_meta_batch` /
    `_train_epoch` / `_validate` internals. No logic in these classes was
    modified.
  * REMOVED: the notebook's hardcoded local development path
        sys.path.insert(0, "/home/amir/GitHub/looped_tfm/temp/tabicl/src")
    (and the accompanying `sys.path.insert(0, "..")`). This script instead
    relies on `tabicl` being installed in the project's own environment
    (`from tabicl import TabICLClassifier`, same import the notebook itself
    already uses in its "vanilla TabICL" cell) -- no local/user-specific
    filesystem paths. Make sure `tabicl` is installed in the active
    environment before running (check with `python -c "import tabicl"`).
  * main.py -- unchanged, unmodified. Only imported (carve_out_validation,
    preprocess_features, encode_labels_train_only, compute_metrics), never
    edited. finetuning_engine.py (the TabPFN engine) is not used/imported
    here at all -- TabICL has its own, separate fine-tuner class.

Fine-tuning configuration
----------------------------
Only ONE fine-tuning variant is run per dataset: full fine-tuning (all
three TabICL components -- col_embedder, row_interactor, icl_predictor --
unfrozen), with epochs=50, learning_rate=1e-4, query_ratio=0.3, patience=20,
grad_clip=1.0, warmup_proportion=0.1 -- exactly the notebook's own
demonstrated configuration, not invented. The notebook does not demonstrate
any selective-freezing (component-only) variants, so none are added here;
a TabICL equivalent of the TabPFN attention-only / mlp-only / layer-wise
strategy grid would need its own deliberate design and is out of scope for
this preparation step.

Output isolation
-------------------
Writes ONLY to results/xor_seed_sweep_tabicl_n800/ -- a new directory.
Never touches results/finetuning_experiments/ (TabArena),
results/thesis_report/, results/xor_sanity_check/, or
results/xor_seed_sweep_tabpfn_n800/.

  results/xor_seed_sweep_tabicl_n800/
    xor_configs.json   -- the 5 dataset configs (seed, size, ...) for reproducibility
    results.pkl         -- full nested results (all metrics, both baseline and fine-tuned, all datasets)
    summary.md           -- compact baseline vs. fine-tuned table + deltas

Usage
-----
    python run_xor_seed_sweep_tabicl.py
"""

import copy
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW

from main import (
    carve_out_validation,
    preprocess_features,
    encode_labels_train_only,
    compute_metrics,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("results/xor_seed_sweep_tabicl_n800")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"  # set to "cpu" for a quick local smoke test without a GPU

VALIDATION_FRACTION = 0.2   # carved out of the train+val portion, same as the TabPFN sweep
TEST_FRACTION = 0.25        # XOR has no official OpenML train/test split, so this
                             # stratified hold-out plays the role that role for real tasks
SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# XOR generation -- copied verbatim, unchanged, from
# notebooks/decision_boundary_xor_tabicl.ipynb (identical to the TabPFN v3
# reference notebook's version). Do not edit without checking that notebook.
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
# 5 XOR dataset variants -- SAME controlled sweep as
# run_xor_seed_sweep_tabpfn.py: n_samples fixed at 800, only random_state
# varies (1, 2, 3, 4, 5). Identical configs -> identical underlying data for
# a direct TabPFN-vs-TabICL comparison.
# ---------------------------------------------------------------------------
XOR_CONFIGS = [
    {"name": "xor_seed1_n800", "n_samples": 800, "noise": 0.01, "n_features": N_FEATURES, "random_state": 1, "gap": 0.01},
    {"name": "xor_seed2_n800", "n_samples": 800, "noise": 0.01, "n_features": N_FEATURES, "random_state": 2, "gap": 0.01},
    {"name": "xor_seed3_n800", "n_samples": 800, "noise": 0.01, "n_features": N_FEATURES, "random_state": 3, "gap": 0.01},
    {"name": "xor_seed4_n800", "n_samples": 800, "noise": 0.01, "n_features": N_FEATURES, "random_state": 4, "gap": 0.01},
    {"name": "xor_seed5_n800", "n_samples": 800, "noise": 0.01, "n_features": N_FEATURES, "random_state": 5, "gap": 0.01},
]


# ---------------------------------------------------------------------------
# CosineWarmupScheduler -- copied verbatim, unchanged, from
# notebooks/decision_boundary_xor_tabicl.ipynb.
# ---------------------------------------------------------------------------
class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        super().__init__(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# TabICLFinetuner -- copied verbatim, unchanged, from
# notebooks/decision_boundary_xor_tabicl.ipynb. The only change anywhere in
# this file relative to the notebook is the removal of the hardcoded
# `/home/amir/...` sys.path insert at the top (see module docstring); the
# class body itself, including `_load_pretrained`'s `from tabicl import
# TabICLClassifier`, is untouched.
# ---------------------------------------------------------------------------
class TabICLFinetuner:
    """Fine-tune TabICL on a single dataset using the same meta-learning setup
    as TabICL's original training:

    Each epoch splits training data into context (support) and query sets.
    The model sees all rows as features with only context labels visible.
    Loss is computed on query predictions and backpropagated.

    TabICL component map:
      col_embedder   — column-wise (feature) embedding
      row_interactor — row-wise interaction between features per row
      icl_predictor  — in-context learning predictor / output head

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    learning_rate : float
        AdamW learning rate. Small values (1e-5 to 1e-4) are recommended.
    query_ratio : float
        Fraction of training samples used as the query set per epoch.
    weight_decay : float
        AdamW weight decay.
    grad_clip : float
        Max gradient norm (0 = disabled).
    freeze_col_embedder : bool
        Freeze the column-wise embedding transformer.
    freeze_row_interactor : bool
        Freeze the row-wise interaction transformer.
    freeze_icl_predictor : bool
        Freeze the in-context learning predictor head.
    warmup_proportion : float
        Fraction of total steps used for LR warmup.
    patience : int
        Early-stopping patience (epochs without val accuracy improvement).
    device : str
        Compute device ("cuda" or "cpu").
    random_state : int
        Seed for context/query splits.
    verbose : bool
        Print per-epoch progress.
    """

    def __init__(
        self,
        *,
        epochs: int = 20,
        learning_rate: float = 1e-5,
        query_ratio: float = 0.2,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        freeze_col_embedder: bool = False,
        freeze_row_interactor: bool = False,
        freeze_icl_predictor: bool = False,
        warmup_proportion: float = 0.1,
        patience: int = 8,
        device: str = "cuda",
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.query_ratio = query_ratio
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.freeze_col_embedder = freeze_col_embedder
        self.freeze_row_interactor = freeze_row_interactor
        self.freeze_icl_predictor = freeze_icl_predictor
        self.warmup_proportion = warmup_proportion
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self._label_encoder: Optional[LabelEncoder] = None
        self._clf = None
        self.history: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "TabICLFinetuner":
        """Fine-tune TabICL on (X_train, y_train), evaluate on (X_val, y_val)."""
        # 1. Encode labels
        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y_train)
        y_val_enc = self._label_encoder.transform(y_val)

        # 2. Load pretrained TabICL and set up preprocessing
        model, clf = self._load_pretrained(X_train, y_train)
        model = model.to(self.device)
        model.train()

        # 3. Preprocess features (identity for numerical numpy arrays)
        X_tr_num = clf.X_encoder_.transform(X_train).astype(np.float32)
        X_val_num = clf.X_encoder_.transform(X_val).astype(np.float32)

        # 4. Freeze selected components
        self._apply_freezing(model)

        # 5. Optimizer + scheduler
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("All parameters frozen — nothing to fine-tune.")
        optimizer = AdamW(trainable, lr=self.learning_rate, weight_decay=self.weight_decay)
        total_steps = self.epochs
        warmup_steps = max(1, int(total_steps * self.warmup_proportion))
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

        # 6. Baseline validation
        model.eval()
        best_val_acc, best_val_loss = self._validate(model, X_tr_num, y_enc, X_val_num, y_val_enc)
        best_state = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(f"  [finetune] baseline val_acc={best_val_acc:.4f} | val_loss={best_val_loss:.4f}")

        train_losses, val_losses, val_accs = [], [], []

        # 7. Training loop
        patience_counter = 0
        for epoch in range(self.epochs):
            model.train()
            t0 = time.perf_counter()
            train_loss = self._train_epoch(model, optimizer, X_tr_num, y_enc, epoch)
            scheduler.step()

            model.eval()
            val_acc, val_loss = self._validate(model, X_tr_num, y_enc, X_val_num, y_val_enc)
            elapsed = time.perf_counter() - t0

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if self.verbose:
                print(
                    f"  [finetune] epoch {epoch+1:3d}/{self.epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | time={elapsed:.1f}s"
                )

            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"  [finetune] early stopping at epoch {epoch+1}")
                    break

        # 8. Restore best weights
        model.load_state_dict(best_state)
        model.eval()
        if self.verbose:
            print(f"  [finetune] best val_acc={best_val_acc:.4f}")

        self.history = {"train_loss": train_losses, "val_loss": val_losses, "val_acc": val_accs}

        # 9. Build inference classifier with fine-tuned weights
        clf.model_.load_state_dict(model.state_dict())
        clf.model_.eval()
        self._clf = clf
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Call fit() before predict().")
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self._clf.predict_proba(X)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pretrained(self, X_train, y_train):
        """Load TabICL via TabICLClassifier.fit() to initialise all preprocessing.

        Returns (raw TabICL nn.Module, fitted TabICLClassifier).
        """
        from tabicl import TabICLClassifier

        # Use a single estimator with no augmentation so the preprocessing
        # is deterministic and easy to replicate in the fine-tuning loop.
        clf = TabICLClassifier(
            n_estimators=1,
            norm_methods=["none"],
            feat_shuffle_method="none",
            class_shuffle_method="none",
            device=self.device,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)

        model = clf.model_
        return model, clf

    def _apply_freezing(self, model: nn.Module) -> None:
        """Disable gradients for selected top-level sub-modules."""
        frozen = 0
        components = [
            (self.freeze_col_embedder,   "col_embedder"),
            (self.freeze_row_interactor, "row_interactor"),
            (self.freeze_icl_predictor,  "icl_predictor"),
        ]
        for should_freeze, name in components:
            if should_freeze:
                module = getattr(model, name)
                for p in module.parameters():
                    p.requires_grad = False
                frozen += 1
        if self.verbose and frozen > 0:
            names = [n for f, n in components if f]
            print(f"  [finetune] frozen: {names}")

    def _make_meta_batch(
        self, X: np.ndarray, y: np.ndarray, epoch: int
    ):
        """Split data into context and query; return tensors in TabICL format.

        TabICL _train_forward expects:
          X       : (B, T, H)      float  — all rows (ctx first, then query)
          y_ctx   : (B, n_ctx)     float  — context labels only
          y_query : (n_query,)     long   — query targets for CE loss
        """
        rng = np.random.default_rng(self.random_state + epoch)
        n = len(X)
        n_query = max(1, int(n * self.query_ratio))

        idx = rng.permutation(n)
        query_idx, ctx_idx = idx[:n_query], idx[n_query:]

        X_all = np.concatenate([X[ctx_idx], X[query_idx]], axis=0)  # ctx first
        x_t = torch.tensor(X_all, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T, H)
        y_ctx_t = torch.tensor(y[ctx_idx], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, n_ctx)
        y_query_t = torch.tensor(y[query_idx], dtype=torch.long, device=self.device)  # (n_query,)

        return x_t, y_ctx_t, y_query_t

    def _train_epoch(
        self, model: nn.Module, optimizer, X: np.ndarray, y: np.ndarray, epoch: int
    ) -> float:
        """One gradient update step using a single context/query split."""
        x_t, y_ctx_t, y_query_t = self._make_meta_batch(X, y, epoch)

        optimizer.zero_grad(set_to_none=True)

        # _train_forward: input (B, T, H), ctx labels (B, n_ctx) → output (B, n_query, n_classes)
        output = model._train_forward(x_t, y_ctx_t)  # (1, n_query, n_classes)
        logits = output[0]  # (n_query, n_classes)

        loss = nn.functional.cross_entropy(logits, y_query_t)

        if not torch.isfinite(loss):
            warnings.warn("Non-finite loss — skipping update.")
            return float("nan")

        loss.backward()

        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

        optimizer.step()
        return float(loss.detach().item())

    def _validate(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train_enc: np.ndarray,
        X_val: np.ndarray,
        y_val_enc: np.ndarray,
    ):
        """Validation: use all training data as context, val as query."""
        with torch.no_grad():
            X_all = np.concatenate([X_train, X_val], axis=0)
            x_t = torch.tensor(X_all, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T, H)
            y_ctx_t = torch.tensor(y_train_enc, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, n_train)
            y_val_t = torch.tensor(y_val_enc, dtype=torch.long, device=self.device)

            # Call _train_forward explicitly to avoid inference-specific paths
            output = model._train_forward(x_t, y_ctx_t)  # (1, n_val, n_classes)
            logits = output[0]  # (n_val, n_classes)

            val_loss = float(nn.functional.cross_entropy(logits, y_val_t).item())
            preds = logits.argmax(dim=-1).cpu().numpy()

        val_acc = float((preds == y_val_enc).mean())
        return val_acc, val_loss


# ---------------------------------------------------------------------------
# Baseline (vanilla, no fine-tuning) -- same TabICLClassifier configuration
# as the notebook's "vanilla TabICL" cell.
# ---------------------------------------------------------------------------
def run_no_finetuning_tabicl(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(
        n_estimators=1,
        norm_methods=["none"],
        feat_shuffle_method="none",
        class_shuffle_method="none",
        device=DEVICE,
        random_state=42,
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def run_full_finetuning_tabicl(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Full fine-tuning (all 3 TabICL components unfrozen) -- exactly the
    notebook's own demonstrated hyperparameters, not invented."""
    finetuner = TabICLFinetuner(
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
        device=DEVICE,
        random_state=42,
        verbose=True,
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

    return {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "training_time": training_time,
        "inference_time": test_inference_time,
        "history": finetuner.history,
    }


def make_dataset_splits(cfg: dict):
    """Identical splitting logic to run_xor_seed_sweep_tabpfn.py, so the same
    seed/size configs produce IDENTICAL train/val/test data for both
    TabPFN and TabICL -- a direct, apples-to-apples comparison."""
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
    X_val = X_val[val_mask.values]
    X_test = X_test[test_mask.values]

    dataset_result = {"config": cfg}

    print("  running baseline (no fine-tuning) ...")
    t0 = time.perf_counter()
    baseline_res = run_no_finetuning_tabicl(X_train, y_train, X_val, y_val, X_test, y_test)
    print(f"  baseline done in {time.perf_counter() - t0:.1f}s "
          f"(test acc={baseline_res['test']['accuracy']:.4f}, "
          f"logloss={baseline_res['test']['logloss']:.4f})")
    dataset_result["no_finetuning"] = baseline_res

    print("  running full_finetuning ...")
    t0 = time.perf_counter()
    ft_res = run_full_finetuning_tabicl(X_train, y_train, X_val, y_val, X_test, y_test)
    print(f"  full_finetuning done in {time.perf_counter() - t0:.1f}s "
          f"(test acc={ft_res['test']['accuracy']:.4f}, "
          f"logloss={ft_res['test']['logloss']:.4f})")
    dataset_result["full_finetuning"] = ft_res

    return dataset_result


def summarize_and_save(all_results: dict):
    with open(OUTPUT_DIR / "results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    with open(OUTPUT_DIR / "xor_configs.json", "w", encoding="utf-8") as f:
        json.dump(XOR_CONFIGS, f, indent=2)

    lines = []
    lines.append("# XOR Controlled Seed Sweep (n=800) -- TabICL v2, Baseline vs. Fine-Tuning\n")
    lines.append(
        "Learning rate used: 0.0001 (1e-4), epochs=50, query_ratio=0.3, patience=20, "
        "grad_clip=1.0, warmup_proportion=0.1 -- exactly the reference notebook's own "
        "demonstrated configuration (notebooks/decision_boundary_xor_tabicl.ipynb), "
        "not invented. Only full fine-tuning (all 3 components unfrozen) was run; no "
        "selective-freezing variants exist for TabICL in this preparation step.\n"
    )
    lines.append(
        "Same 5 dataset configs (n=800, seeds 1-5) as run_xor_seed_sweep_tabpfn.py, "
        "generated identically -- directly comparable to the TabPFN v3 results.\n"
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
        m = res["full_finetuning"]["test"]
        neg_ll = -m["logloss"]
        d_acc = m["accuracy"] - base["accuracy"]
        d_negll = neg_ll - base_neg_ll
        lines.append(
            f"| {name} | full_finetuning | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | "
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
