config_base = {}

config_base["model"] = "tabpfn_v3"
config_base["tabtune_model_name"] = "TabPFNv3"
config_base["suite_id"] = 457

config_base["validation_fraction"] = 0.2
config_base["validation_seed"] = 42

config_base["finetuning_method"] = "own_finetuning"

# --- Selective fine-tuning: LAYER-WISE ---
# Only the transformer encoder layer indices listed in "train_only_layers"
# stay trainable; every other layer and the decoder are frozen. This example
# trains only layer 0 -- copy this file per layer index (or write a small
# loop over --config) to sweep across all layers.
#
# Before running a full sweep, check how many layers the installed TabPFN
# model actually has, e.g.:
#   len(finetuner._model.transformer_encoder.layers)
# so you don't request an out-of-range index.
config_base["finetuning_hyperparams"] = {
    "learning_rate": 1e-5,
    "num_epochs": 200,
    "weight_decay": 0.01,
    "train_only_layers": [0],
}

config_base["device"] = "cuda"

config_base["saving_path"] = "results/finetuning_experiments/layerwise_layer0"
