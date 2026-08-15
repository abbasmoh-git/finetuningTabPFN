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
  freeze_mlp          : freeze all MLP (feed-forward) sub-modules
  freeze_decoder      : freeze the output decoder head
  train_only_layers   : if set, only these transformer_encoder layer indices
                        remain trainable (everything else -- other layers and
                        the decoder -- is frozen), regardless of the flags
                        above. Use this for layer-wise fine-tuning experiments
                        (e.g. train_only_layers=[5] to test layer 5 alone).

Verified against the actual TabPFN architecture (PriorLabs/TabPFN,
architectures/base/layer.py + transformer.py):
  PerFeatureEncoderLayer submodules -> self.self_attn_between_features,
                                        self.self_attn_between_items,
                                        self.mlp (and self.second_mlp)
  PerFeatureTransformer             -> self.transformer_encoder.layers[i]
                                        (nn.ModuleList of PerFeatureEncoderLayer)
                                        self.decoder_dict (output head)
So substring matching on "feature" / "item" / "mlp" / "decoder" in
named_modules() reliably hits the right sub-modules. Note: this was verified
against the TabPFN v2.6 / base architecture; if the cluster's `tabpfn` package
resolves TabPFNv3 to a different architecture module, re-check by running
`for n, _ in model.named_modules(): print(n)` once before trusting the freeze
counts blindly.

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
import matplotlib.pyplot as plt
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
    freeze_mlp : bool
        If True, freeze all MLP / feed-forward sub-modules
        (``mlp`` and ``second_mlp`` across all layers).
    freeze_decoder : bool
        If True, freeze the output decoder head.
    train_only_layers : list[int], optional
        If given, only the transformer encoder layers at these indices remain
        trainable; every other layer and the decoder are frozen, overriding
        the freeze_* flags above. Used for layer-wise fine-tuning experiments.
    max_context_size : int, optional
        Upper bound on how many rows are fed through the model in a single
        forward pass (context + query combined during training; context +
        validation query during validation). ``None`` (default) disables
        this and reproduces the original behaviour of always using every
        row of the dataset in one pass -- fine for small/medium datasets,
        but the memory of a single TabPFN forward+backward pass grows with
        row count, and on datasets with many rows this can exceed GPU
        memory even though plain inference (no gradients) on the same data
        fits fine. When set and a dataset has more rows than this cap, a
        *different* random subset of this size is drawn every epoch (and a
        fixed random subset for validation), so no rows are permanently
        discarded -- across the full training run the model still sees the
        whole dataset, just never all of it in one pass at once. This is
        the same idea as ordinary mini-batch training, not a reduction of
        the dataset used for the experiment.
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
    n_estimators : int
        Number of TabPFN ensemble members used for the FINAL inference
        classifier built after fit() (train/val/test predict/predict_proba).
        Does not affect the training loop itself, which always fine-tunes a
        single raw model. Default is 8 to match TabPFNClassifier's own
        library default, which is what the no-fine-tuning baseline uses
        (implicitly, since it never overrides n_estimators either) -- keep
        this equal to the baseline's value so both are compared under the
        same ensembling conditions.
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
        freeze_mlp: bool = False,
        freeze_decoder: bool = False,
        train_only_layers: Optional[list[int]] = None,
        max_context_size: Optional[int] = None,
        warmup_proportion: float = 0.1,
        patience: int = 8,
        device: str = "cuda",
        random_state: int = 42,
        verbose: bool = True,
        n_estimators: int = 8,
        inference_precision: torch.dtype = torch.float32,
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.query_ratio = query_ratio
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.freeze_feature_attn = freeze_feature_attn
        self.freeze_row_attn = freeze_row_attn
        self.freeze_mlp = freeze_mlp
        self.freeze_decoder = freeze_decoder
        self.train_only_layers = train_only_layers
        self.max_context_size = max_context_size
        self.warmup_proportion = warmup_proportion
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.inference_precision = inference_precision

        self._label_encoder: Optional[LabelEncoder] = None
        self._model: Optional[nn.Module] = None
        self._clf = None  # fitted TabPFNClassifier for inference
        self.history: dict = {}  # tracks train_loss, val_loss, val_acc per epoch

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
        best_val_acc, best_val_loss = self._validate(model, X_train, y_enc, X_val, y_val_enc, n_classes)
        best_state = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(f"  [finetune] baseline val_acc = {best_val_acc:.4f} | val_loss = {best_val_loss:.4f}")

        # history tracking
        train_losses: list[float] = []
        val_losses:   list[float] = []
        val_accs:     list[float] = []

        # 6. Training loop
        patience_counter = 0
        for epoch in range(self.epochs):
            model.train()
            t0 = time.perf_counter()

            train_loss = self._train_epoch(
                model, optimizer, X_train, y_enc, n_classes, epoch
            )
            scheduler.step()

            # Validate
            model.eval()
            val_acc, val_loss = self._validate(model, X_train, y_enc, X_val, y_val_enc, n_classes)
            elapsed = time.perf_counter() - t0

            # Save history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if self.verbose:
                print(
                    f"  [finetune] epoch {epoch + 1:3d}/{self.epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | time={elapsed:.1f}s"
                )

            # Early stopping + best-model tracking (based on val_acc)
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

        # Store history for later plotting
        self.history = {
            "train_loss": train_losses,
            "val_loss":   val_losses,
            "val_acc":    val_accs,
        }

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

    def plot_history(self, save_path: Optional[str] = None) -> None:
        """Plot training loss and validation loss per epoch.

        Shows whether the model is learning (loss going down) or overfitting
        (train_loss goes down but val_loss goes up).

        Parameters
        ----------
        save_path : str, optional
            If given, save the plot as an image file (e.g. "loss_curve.png").
            Otherwise the plot is shown interactively.
        """
        if not self.history:
            raise RuntimeError("No history found. Call fit() first.")

        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: Loss curves
        axes[0].plot(epochs, self.history["train_loss"], label="Train Loss", marker="o", markersize=3)
        axes[0].plot(epochs, self.history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].set_title("Training Loss vs. Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Right: Validation accuracy
        axes[1].plot(epochs, self.history["val_acc"], label="Val Accuracy", color="green", marker="o", markersize=3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy per Epoch")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [finetune] plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

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
            n_estimators=self.n_estimators,
            inference_precision=self.inference_precision,
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

        Component names in TabPFN (analogous to TabICL), verified against the
        actual PriorLabs/TabPFN source (architectures/base/layer.py):
          - feature attention : sub-modules whose name contains 'feature'
            (matches self.self_attn_between_features)
          - row/item attention : sub-modules whose name contains 'item' or 'row'
            (matches self.self_attn_between_items)
          - MLP                : sub-modules whose name contains 'mlp'
            (matches self.mlp and self.second_mlp)
          - decoder head       : sub-modules whose name contains 'decoder'
            (matches self.decoder_dict)

        If `train_only_layers` is set, it takes precedence: every parameter is
        frozen except those belonging to transformer_encoder.layers[i] for i
        in train_only_layers. This implements layer-wise fine-tuning (testing
        which single encoder layer, or subset of layers, matters most) and
        ignores the freeze_* flags above for the frozen/trainable decision.

        Call this AFTER moving the model to device but BEFORE creating the
        optimizer, so the optimizer only sees trainable parameters.
        """
        if self.train_only_layers is not None:
            self._apply_layerwise_freezing(model, self.train_only_layers)
            return

        frozen_count = 0
        for name, module in model.named_modules():
            name_lower = name.lower()
            should_freeze = (
                (self.freeze_feature_attn and "feature" in name_lower)
                or (self.freeze_row_attn and ("item" in name_lower or "row" in name_lower))
                or (self.freeze_mlp and "mlp" in name_lower)
                or (self.freeze_decoder and "decoder" in name_lower)
            )
            if should_freeze:
                for p in module.parameters():
                    p.requires_grad = False
                frozen_count += 1

        if self.verbose:
            if frozen_count > 0:
                print(f"  [finetune] frozen {frozen_count} sub-module(s)")
            elif self.freeze_feature_attn or self.freeze_row_attn or self.freeze_mlp or self.freeze_decoder:
                # A freeze flag was requested but nothing matched -- almost
                # certainly a naming mismatch with the installed TabPFN
                # version, not "nothing to freeze". Fail loud instead of
                # silently running full fine-tuning under a selective label.
                warnings.warn(
                    "A freeze_* flag was set but no sub-module name matched "
                    "('feature' / 'item' / 'row' / 'mlp' / 'decoder'). This "
                    "run is NOT selectively fine-tuning anything -- check "
                    "`for n, _ in model.named_modules(): print(n)` against "
                    "the installed tabpfn version."
                )

    def _apply_layerwise_freezing(self, model: nn.Module, train_only_layers: list) -> None:
        """Freeze everything except the given transformer encoder layer indices.

        Matches module names of the form 'transformer_encoder.layers.<i>' (the
        PerFeatureEncoderLayer instances inside the encoder's nn.ModuleList)
        and keeps only the requested indices trainable. All other parameters,
        including the decoder head and any other layers, are frozen.
        """
        import re

        target_indices = {int(i) for i in train_only_layers}
        matched_indices = set()

        # First, freeze everything.
        for p in model.parameters():
            p.requires_grad = False

        # Then, re-enable gradients only for the requested layer indices.
        for name, module in model.named_modules():
            m = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            if m and int(m.group(1)) in target_indices:
                # Only unfreeze at the PerFeatureEncoderLayer level itself
                # (name ends exactly at the index), not sub-matches, to avoid
                # redundant work -- but unfreezing repeatedly is harmless, so
                # this is a minor efficiency choice, not a correctness one.
                for p in module.parameters():
                    p.requires_grad = True
                matched_indices.add(int(m.group(1)))

        if self.verbose:
            print(
                f"  [finetune] layer-wise: training only layer(s) "
                f"{sorted(matched_indices)} (requested: {sorted(target_indices)})"
            )
        missing = target_indices - matched_indices
        if missing:
            warnings.warn(
                f"train_only_layers requested indices {sorted(missing)} but no "
                "matching 'transformer_encoder.layers.<i>' sub-module was "
                "found for them -- check the layer count of the installed "
                "TabPFN model (e.g. `len(model.transformer_encoder.layers)`)."
            )

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

        If ``max_context_size`` is set and the dataset has more rows than
        that, a random subset of that size is drawn first (a fresh subset
        every epoch, via the epoch-dependent seed below) and the context/
        query split is taken from within that subset only. This bounds the
        size of the single forward+backward pass without ever permanently
        dropping rows from the experiment.
        """
        rng = np.random.default_rng(self.random_state + epoch)
        n = len(X)

        if self.max_context_size is not None and n > self.max_context_size:
            sample_idx = rng.choice(n, size=self.max_context_size, replace=False)
            X = X[sample_idx]
            y = y[sample_idx]
            n = self.max_context_size

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
        # Output shape: (n_query, 1, n_classes) — TabPFN only returns query predictions
        output = model(x_t, y_ctx_t, only_return_standard_out=True)
        logits_query = output[:, 0, :]  # (n_query, n_classes)

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
    ) -> tuple[float, float]:
        """Accuracy and loss on val set using current model weights (no grad).

        Returns (val_acc, val_loss).
        Train is used as context, val as query — same setup as in _train_epoch.

        If ``max_context_size`` is set, context and query are capped using
        the same split ratio as training (``query_ratio``, e.g. ~80% context
        / ~20% query of the budget) rather than always keeping the full
        validation set. Keeping the query 100% intact and only shrinking
        context sounds appealing, but on datasets where the validation set
        alone is close to or bigger than ``max_context_size`` it can shrink
        the context all the way to zero rows -- which crashes the model
        (there is nothing left to condition the in-context predictions on).
        A fixed proportional split avoids that regardless of dataset size.
        The seed is fixed (random_state, not epoch-dependent) so every
        validation call during one run uses the same subsets.
        """
        with torch.no_grad():
            X_ctx, y_ctx_enc = X_train, y_train_enc
            X_q, y_q_enc = X_val, y_val_enc

            if self.max_context_size is not None and len(X_ctx) + len(X_q) > self.max_context_size:
                rng = np.random.default_rng(self.random_state)

                q_budget = max(1, min(len(X_q), int(self.max_context_size * self.query_ratio)))
                ctx_budget = max(1, min(len(X_ctx), self.max_context_size - q_budget))

                if len(X_q) > q_budget:
                    warnings.warn(
                        f"Validation set ({len(X_q)} rows) + training context "
                        f"({len(X_ctx)} rows) exceed max_context_size="
                        f"{self.max_context_size}; scoring on a fixed random "
                        f"subset of {q_budget} validation rows instead of the "
                        "full validation set."
                    )
                    sub = rng.choice(len(X_q), size=q_budget, replace=False)
                    X_q, y_q_enc = X_q[sub], y_q_enc[sub]

                if len(X_ctx) > ctx_budget:
                    sub = rng.choice(len(X_ctx), size=ctx_budget, replace=False)
                    X_ctx, y_ctx_enc = X_ctx[sub], y_ctx_enc[sub]

            # Use (possibly capped) training data as context, val as query
            X_all = np.concatenate([X_ctx, X_q], axis=0)
            x_t = torch.tensor(X_all, dtype=torch.float32, device=self.device).unsqueeze(1)
            y_ctx_t = torch.tensor(y_ctx_enc, dtype=torch.long, device=self.device).unsqueeze(1)
            y_val_t = torch.tensor(y_q_enc, dtype=torch.long, device=self.device)

            # TabPFN output already contains only test-row predictions: (n_val, 1, n_classes)
            output = model(x_t, y_ctx_t, only_return_standard_out=True)
            logits_val = output[:, 0, :]  # (n_val, n_classes)

            val_loss = float(nn.functional.cross_entropy(logits_val, y_val_t).item())
            preds = logits_val.argmax(dim=-1).cpu().numpy()

        acc = float((preds == y_q_enc).mean())
        return acc, val_loss

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        device: str = "cpu",
        n_estimators: int = 1,
        inference_precision: torch.dtype = torch.float32,
        base_clf=None,
    ) -> "TabPFNFinetuner":
        """Wrap an existing model (e.g. quantized) for sklearn-compatible inference.

        Skips training entirely. Unlike ``fit()``, this does a direct model
        assignment rather than ``load_state_dict``, so it works for quantized
        models whose state dict has a different format (int8 tensors, etc.).

        Parameters
        ----------
        model : nn.Module
            A pretrained or quantized TabPFN model (already in eval mode).
        X_train : np.ndarray
            Training features — used as context during inference.
        y_train : np.ndarray
            Training labels — used as context during inference.
        device : str
            Compute device (must be ``"cpu"`` for quantized models).
        n_estimators : int
            Number of ensemble members passed to TabPFNClassifier.
        inference_precision : torch.dtype
            Precision used by TabPFNClassifier during inference.
        base_clf : TabPFNClassifier, optional
            A pre-fitted TabPFNClassifier to reuse (deepcopied internally).
            Avoids reloading pretrained weights from disk on every call —
            useful when wrapping multiple models in a loop.
        """
        from tabpfn import TabPFNClassifier

        instance = cls.__new__(cls)
        instance.device = device
        instance.n_estimators = n_estimators
        instance.inference_precision = inference_precision
        instance._model = model.eval()
        instance._label_encoder = None
        instance.history = {}

        if base_clf is not None:
            clf = copy.deepcopy(base_clf)
        else:
            clf = TabPFNClassifier(
                device=device,
                ignore_pretraining_limits=True,
                fit_mode="fit_preprocessors",
                n_estimators=n_estimators,
                inference_precision=inference_precision,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_train, y_train)

        # Inject weights in-place so TabPFN's inference executor sees the change.
        # load_state_dict works for float32 models (including simulated N-bit quantization).
        # For true int8 quantized models (quantize_dynamic) the state dict is incompatible,
        # so we fall back to element replacement in the existing list.
        try:
            clf.models_[0].load_state_dict(model.state_dict())
            clf.models_[0].eval()
        except (RuntimeError, TypeError):
            clf.models_[0] = model.eval()

        instance._clf = clf
        return instance

    def _build_inference_clf(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray):
        """Build a TabPFNClassifier that uses the fine-tuned model weights for prediction.

        With ``n_estimators > 1``, TabPFNClassifier.fit() populates
        ``clf.models_`` with one entry per ensemble member -- all starting
        from the *same* pretrained checkpoint, differing only in the data
        permutation/config each member applies at inference time, not in
        their weights. So every member must receive the fine-tuned weights,
        not just the first one: loading only ``models_[0]`` would silently
        average 1 fine-tuned member with (n_estimators - 1) un-fine-tuned
        ones, diluting the fine-tuning effect in every reported metric.
        """
        from tabpfn import TabPFNClassifier

        clf = TabPFNClassifier(
            device=self.device,
            ignore_pretraining_limits=True,
            fit_mode="fit_preprocessors",
            n_estimators=self.n_estimators,
            inference_precision=self.inference_precision,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)

        # Replace ALL ensemble members' weights with our fine-tuned ones.
        fine_tuned_state = model.state_dict()
        for m in clf.models_:
            m.load_state_dict(fine_tuned_state)
            m.eval()
        return clf