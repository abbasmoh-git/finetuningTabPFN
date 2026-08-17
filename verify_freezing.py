"""
verify_freezing.py
-------------------
Quick, GPU-free sanity check: for each fine-tuning config, load the
pretrained TabPFN v3 model, apply the same freezing logic main.py would use,
and report how many parameters end up trainable vs frozen.

This exists because two silent bugs (MLP-only leaving attention trainable,
layer-wise touching the wrong 3-layer sub-module instead of the real
24-block icl_blocks stack) were only caught by manually introspecting the
model -- this script would have caught both immediately. Run this any time
the TabPFN version changes or a new selective config is added, BEFORE
spending GPU time on it.

Usage:
    python verify_freezing.py
"""

import importlib
import warnings

from finetuning_engine import TabPFNFinetuner


def total_params(model):
    return sum(p.numel() for p in model.parameters())


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_config(config_module_name: str, label: str):
    cfg = importlib.import_module(config_module_name).config_base
    ft = cfg.get("finetuning_hyperparams", {})

    finetuner = TabPFNFinetuner(
        freeze_feature_attn=ft.get("freeze_feature_attn", False),
        freeze_row_attn=ft.get("freeze_row_attn", False),
        freeze_mlp=ft.get("freeze_mlp", False),
        freeze_decoder=ft.get("freeze_decoder", False),
        train_only_layers=ft.get("train_only_layers", None),
        device="cpu",
        verbose=False,
    )
    model, _ = finetuner._load_pretrained()
    finetuner._apply_freezing(model)

    tot = total_params(model)
    train = trainable_params(model)
    print(f"{label:25s} trainable={train:>12,} / {tot:>12,}  ({100*train/tot:5.1f}%)")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"{'Config':25s} {'Trainable / Total params':40s}")
        print("-" * 70)
        check_config("configs.config_own_finetuning", "Full fine-tuning")
        check_config("configs.config_attention_only_finetuning", "Attention-only")
        check_config("configs.config_mlp_only_finetuning", "MLP-only")
        check_config("configs.config_layerwise_finetuning", "Layer-wise (layer 0)")
