config_base = {}

config_base["model"] = "tabpfn_v3"
config_base["tabtune_model_name"] = "TabPFNv26"
config_base["suite_id"] = 457

config_base["validation_fraction"] = 0.2
config_base["validation_seed"] = 42

config_base["finetuning_method"] = "own_finetuning"

config_base["finetuning_hyperparams"] = {
    "learning_rate": 1e-4,
    "num_epochs": 200,
    "weight_decay": 0.01,
    "freeze_feature_attn": False,
    "freeze_row_attn": False,
    "freeze_decoder": False,
}

config_base["device"] = "cuda"
config_base["saving_path"] = "results/finetuning_experiments/lr_1e4"
