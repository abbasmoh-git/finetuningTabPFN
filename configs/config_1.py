config_base = {}

config_base["model"] = "tabpfn_v3"

# Model name expected by TabTune's TabularPipeline (used only for full_finetuning)
config_base["tabtune_model_name"] = "TabPFNv26"

# --- Benchmark suite (TabArena / OpenML) ---
config_base["suite_id"] = 457  # TabArena benchmark suite

# --- Validation split ---
# OpenML tasks only define official train/test indices. The validation set
# is carved out of the official TRAINING portion only (stratified); the
# official test split is never touched.
config_base["validation_fraction"] = 0.2
config_base["validation_seed"] = 42

# --- Fine-tuning method ---
config_base["finetuning_method"] = "no_finetuning"  # or "full_finetuning"

# --- Fine-tuning hyperparameters (used only when finetuning_method == "full_finetuning") ---
config_base["finetuning_hyperparams"] = {
    "learning_rate": 1e-5,
    "num_epochs": 10,
    "batch_size": 32,
    "extra_kwargs": {},  # any additional TabTune-specific kwargs, e.g. {"weight_decay": 0.01}
}

config_base["saving_path"] = "../results/finetuning_experiments/" + config_base["model"]
