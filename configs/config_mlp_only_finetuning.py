config_base = {}

config_base["model"] = "tabpfn_v3"

# Model name expected by TabTune's TabularPipeline (used only for full_finetuning)
config_base["tabtune_model_name"] = "TabPFNv3"

# --- Benchmark suite (TabArena / OpenML) ---
config_base["suite_id"] = 457  # TabArena benchmark suite

# --- Validation split ---
config_base["validation_fraction"] = 0.2
config_base["validation_seed"] = 42

# --- Fine-tuning method ---
config_base["finetuning_method"] = "own_finetuning"

# --- Selective fine-tuning: MLP ONLY ---
# "X only" means freezing everything except X: both attention types
# (feature + row/item) and the decoder are frozen, only the MLP /
# feed-forward sub-modules remain trainable.
config_base["finetuning_hyperparams"] = {
    "learning_rate": 1e-5,
    "num_epochs": 200,
    "weight_decay": 0.01,
    "freeze_feature_attn": True,
    "freeze_row_attn": True,
    "freeze_mlp": False,
    "freeze_decoder": True,
}

config_base["device"] = "cuda"

config_base["saving_path"] = "results/finetuning_experiments/mlp_only"
