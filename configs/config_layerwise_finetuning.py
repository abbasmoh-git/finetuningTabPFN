config_base = {}

config_base["model"] = "tabpfn_v3"
config_base["tabtune_model_name"] = "TabPFNv3"
config_base["suite_id"] = 457

config_base["validation_fraction"] = 0.2
config_base["validation_seed"] = 42

config_base["finetuning_method"] = "own_finetuning"

# --- Selective fine-tuning: LAYER-WISE ---
# Only the icl_blocks[i] transformer block indices listed in
# "train_only_layers" stay trainable; every other block, the decoder, and
# the auxiliary embedding modules are frozen. This trains only block 0.
# See config_layerwise_layer6.py / _layer11 / _layer17 / _layer23 for the
# other tested blocks (verified 2026-08-16: model.icl_blocks has 24 blocks,
# indices 0-23 -- check `len(model.icl_blocks)` again if the tabpfn
# package version changes).
config_base["finetuning_hyperparams"] = {
    "learning_rate": 1e-5,
    "num_epochs": 200,
    "weight_decay": 0.01,
    "train_only_layers": [0],
    # See config_own_finetuning.py for why this exists -- same value used
    # across all four fine-tuning variants to keep the method consistent.
    "max_context_size": 3000,
    "n_estimators": 8,
}

config_base["device"] = "cuda"

config_base["saving_path"] = "results/finetuning_experiments/layerwise_layer0"
