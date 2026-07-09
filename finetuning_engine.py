"""
finetuning_engine.py
--------------------
Fine-tuning pipeline for TabPFN, inspired by TabICL's fine-tuning approach.

Core idea (same as TabICL):
  1. Split training data into context (support) and query sets each epoch.
  2. Forward pass: model sees context+query features, context labels.
  3. Compute CrossEntropyLoss on query predictions.
  4. Backpropagate and update only the unfrozen parameters.

TabPFN model structure (analogous to TabICL):
  TabICL                TabPFN
  --------------------  ----------------------------
  col_embedder       →  attn_between_features (per layer)
  row_interactor     →  attn_between_items    (per layer)
  icl_predictor      →  decoder head

Freezing flags:
  freeze_feature_attn : freeze all attn_between_features sub-modules
  freeze_row_attn     : freeze all attn_between_items sub-modules
  freeze_decoder      : freeze the output decoder head

Usage
-----
from finetuning_engine import TabPFNFinetuner

finetuner = TabPFNFinetuner(
    epochs=20,
    learning_rate=1e-5,
    freeze_feature_attn=False,
    freeze_row_attn=False,
    device="cuda",
)
finetuner.fit(X_train, y_train, X_val, y_val)
y_pred  = finetuner.predict(X_test)
y_proba = finetuner.predict_proba(X_test)
"""

from __future__ import annotations

import copy
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ---------------------------------------------------------------------------
# Helper: cosine warmup scheduler
# ---------------------------------------------------------------------------

class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup followed by cosine decay (same as TabICL uses)."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        super().__init__(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TabPFNFinetuner:
    """Fine-tune TabPFN on a single dataset.

    Parameters
    ----------
    epochs : int
        Number of training epochs (each epoch = one pass over query/context splits).
    learning_rate : float
        AdamW learning rate. Small values (1e-5 to 1e-4) are recommended to
        avoid destroying the pretrained weights.
    query_ratio : float
        Fraction of training samples used as the query set per step.
        The rest is used as context (support). Same role as
        TabICL's ``finetune_ctx_query_ratio``.
    weight_decay : float
        AdamW weight decay.
    grad_clip : float
        Max gradient norm for clipping (0 = disabled).
    freeze_feature_attn : bool
        If True, freeze all feature-attention sub-modules
        (``attn_between_features`` across all layers).
    freeze_row_attn : bool
        If True, freeze all row-attention sub-modules
        (``attn_between_items`` across all layers).
    freeze_decoder : bool
        If True, freeze the output decoder head.
    warmup_proportion : float
        Fraction of total steps used for LR warmup (same as TabICL).
    patience : int
        Early-stopping patience: stop if val accuracy hasn't improved
        for this many consecutive epochs.
    device : str
        Compute device (``"cuda"`` or ``"cpu"``).
    random_state : int
        Seed for context/query splits.
    verbose : bool
        Print progress per epoch.
    """

    def __init__(
        self,
        *,
        epochs: int = 20,
        learning_rate: float = 1e-5,
        query_ratio: float = 0.2,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        freeze_feature_attn: bool = False,
        freeze_row_attn: bool = False,
        freeze_decoder: bool = False,
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
        self.freeze_feature_attn = freeze_feature_attn
        self.freeze_row_attn = freeze_row_attn
        self.freeze_decoder = freeze_decoder
        self.warmup_proportion = warmup_proportion
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self._label_encoder: Optional[LabelEncoder] = None
        self._model: Optional[nn.Module] = None
        self._clf = None  # fitted TabPFNClassifier for inference

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "TabPFNFinetuner":
        """Fine-tune TabPFN on (X_train, y_train), evaluate on (X_val, y_val).

        Loads the pretrained checkpoint, applies any freezing, runs the
        meta-learning training loop, and restores the best-validation weights
        before returning (same safety net as TabICL).
        """
        # 1. Encode labels (0 .. K-1) -- required for CrossEntropyLoss
        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y_train)
        y_val_enc = self._label_encoder.transform(y_val)
        n_classes = len(self._label_encoder.classes_)

        # 2. Load pretrained TabPFN and extract the raw nn.Module
        model, clf_for_inference = self._load_pretrained()
        model = model.to(self.device)
        model.train()

        # 3. Freeze selected components
        self._apply_freezing(model)

        # 4. Optimizer + scheduler
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "All parameters are frozen — nothing to fine-tune. "
                "Check your freeze_* flags."
            )
        optimizer = AdamW(trainable, lr=self.learning_rate, weight_decay=self.weight_decay)
        total_steps = self.epochs
        warmup_steps = max(1, int(total_steps * self.warmup_proportion))
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

        # 5. Baseline validation (before any weight update)
        model.eval()
        best_val_acc = self._validate(model, X_train, y_enc, X_val, y_val_enc, n_classes)
        best_state = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(f"  [finetune] baseline val_acc = {best_val_acc:.4f}")

        # 6. Training loop
        patience_counter = 0
        for epoch in range(self.epochs):
            model.train()
            t0 = time.perf_counter()

            loss_val = self._train_epoch(
                model, optimizer, X_train, y_enc, n_classes, epoch
            )
            scheduler.step()

            # Validate
            model.eval()
            val_acc = self._validate(model, X_train, y_enc, X_val, y_val_enc, n_classes)
            elapsed = time.perf_counter() - t0

            if self.verbose:
                print(
                    f"  [finetune] epoch {epoch + 1:3d}/{self.epochs} | "
                    f"loss={loss_val:.4f} | val_acc={val_acc:.4f} | "
                    f"time={elapsed:.1f}s"
                )

            # Early stopping + best-model tracking
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"  [finetune] early stopping at epoch {epoch + 1}")
                    break

        # 7. Restore best weights
        model.load_state_dict(best_state)
        model.eval()
        self._model = model
        if self.verbose:
            print(f"  [finetune] best val_acc = {best_val_acc:.4f}")

        # 8. Build final inference estimator with fine-tuned weights
        self._clf = self._build_inference_clf(model, X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels using the fine-tuned model."""
        if self._clf is None:
            raise RuntimeError("Call fit() before predict().")
        return self._clf.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the fine-tuned model."""
        if self._clf is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self._clf.predict_proba(X_test)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pretrained(self):
        """Load TabPFN, do a dummy fit to initialise weights, return (model, clf)."""
        from tabpfn import TabPFNClassifier

        clf = TabPFNClassifier(
            device=self.device,
            ignore_pretraining_limits=True,
            fit_mode="fit_preprocessors",
            inference_precision=torch.float32,
        )
        # Dummy fit to download/load weights (TabPFN is lazy-loaded)
        # Must be non-constant, otherwise TabPFN raises TabPFNValidationError
        rng = np.random.default_rng(0)
        X_dummy = rng.standard_normal((10, 4)).astype(np.float32)
        y_dummy = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_dummy, y_dummy)

        # Extract raw nn.Module (first ensemble member)
        model = clf.models_[0]
        return model, clf

    def _apply_freezing(self, model: nn.Module) -> None:
        """Zero requires_grad on frozen sub-modules.

        Component names in TabPFN (analogous to TabICL):
          - feature attention : sub-modules whose name contains 'feature'
          - row/item attention : sub-modules whose name contains 'item' or 'row'
          - decoder head      : sub-modules whose name contains 'decoder'

        Call this AFTER moving the model to device but BEFORE creating the
        optimizer, so the optimizer only sees trainable parameters.
        """
        frozen_count = 0
        for name, module in model.named_modules():
            name_lower = name.lower()
            should_freeze = (
                (self.freeze_feature_attn and "feature" in name_lower)
                or (self.freeze_row_attn and ("item" in name_lower or "row" in name_lower))
                or (self.freeze_decoder and "decoder" in name_lower)
            )
            if should_freeze:
                for p in module.parameters():
                    p.requires_grad = False
                frozen_count += 1

        if self.verbose and frozen_count > 0:
            print(f"  [finetune] frozen {frozen_count} sub-module(s)")

    def _make_meta_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Split data into context and query; format as tensors.

        Returns (x_tensor, y_ctx_tensor, y_query_tensor, query_indices)
        where x_tensor has shape (n_ctx + n_query, 1, n_features) — the
        format expected by TabPFN's Architecture.forward().
        """
        rng = np.random.default_rng(self.random_state + epoch)
        n = len(X)
        n_query = max(1, int(n * self.query_ratio))

        idx = rng.permutation(n)
        query_idx = idx[:n_query]
        ctx_idx = idx[n_query:]

        X_ctx = X[ctx_idx]
        y_ctx = y[ctx_idx]
        X_query = X[query_idx]
        y_query = y[query_idx]

        # TabPFN expects (seq_len, batch=1, features)
        X_all = np.concatenate([X_ctx, X_query], axis=0)
        x_t = torch.tensor(X_all, dtype=torch.float32, device=self.device).unsqueeze(1)
        y_ctx_t = torch.tensor(y_ctx, dtype=torch.long, device=self.device).unsqueeze(1)
        y_query_t = torch.tensor(y_query, dtype=torch.long, device=self.device)

        return x_t, y_ctx_t, y_query_t, len(ctx_idx)

    def _train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        epoch: int,
    ) -> float:
        """Run one gradient update step (one context/query split per epoch)."""
        x_t, y_ctx_t, y_query_t, n_ctx = self._make_meta_batch(X, y, epoch)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass: TabPFN sees all rows as x, only context labels as y
        # Output shape: (n_query, 1, n_classes)
        output = model(x_t, y_ctx_t, only_return_standard_out=True)
        logits_query = output[n_ctx:, 0, :]  # (n_query, n_classes)

        loss = nn.functional.cross_entropy(logits_query, y_query_t)

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
        n_classes: int,
    ) -> float:
        """Accuracy on val set using current model weights (no grad)."""
        with torch.no_grad():
            x_t, y_ctx_t, _, n_ctx = self._make_meta_batch(
                np.concatenate([X_train, X_val]),
                np.concatenate([y_train_enc, np.zeros(len(y_val_enc), dtype=int)]),
                epoch=999,
            )
            # Use all training data as context, val as query
            X_all = np.concatenate([X_train, X_val], axis=0)
            x_t = torch.tensor(X_all, dtype=torch.float32, device=self.device).unsqueeze(1)
            y_ctx_t = torch.tensor(y_train_enc, dtype=torch.long, device=self.device).unsqueeze(1)

            output = model(x_t, y_ctx_t, only_return_standard_out=True)
            logits_val = output[len(X_train):, 0, :]  # (n_val, n_classes)
            preds = logits_val.argmax(dim=-1).cpu().numpy()

        acc = float((preds == y_val_enc).mean())
        return acc

    def _build_inference_clf(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray):
        """Build a TabPFNClassifier that uses the fine-tuned model weights for prediction."""
        from tabpfn import TabPFNClassifier
        from tabpfn.base import ClassifierModelSpecs

        clf = TabPFNClassifier(
            device=self.device,
            ignore_pretraining_limits=True,
            fit_mode="fit_preprocessors",
            inference_precision=torch.float32,
        )
        # Inject fine-tuned weights before fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)

        # Replace the model weights with our fine-tuned ones
        clf.models_[0].load_state_dict(model.state_dict())
        clf.models_[0].eval()
        return clf
