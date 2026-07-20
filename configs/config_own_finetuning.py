config_base = {}

config_base["model"] = "tabpfn_v3"

# Model name expected by TabTune's TabularPipeline (used only for full_finetuning)
config_base["tabtune_model_name"] = "TabPFNv26"

# --- Benchmark suite (TabArena / OpenML) ---
config_base["suite_id"] = 457  # TabArena benchmark suite

# --- Validation split ---
config_base["validation_fraction"] = 0.2
config_base["validation_seed"] = 42

# --- Fine-tuning method ---
config_base["finetuning_method"] = "own_finetuning"

# --- Fine-tuning hyperparameters for own_finetuning ---
config_base["finetuning_hyperparams"] = {
    "learning_rate": 1e-5,
    "num_epochs": 200,          # enough epochs to see full loss curve
    "weight_decay": 0.01,
    "freeze_feature_attn": False,
    "freeze_row_attn": False,
    "freeze_decoder": False,
}

config_base["device"] = "cuda"

config_base["saving_path"] = "results/finetuning_experiments/own_finetuning"
