from pathlib import Path

config_base = {}

config_base["model"] = "tabpfn_v3"
config_base["open_ml_task_id"] = 13 
config_base["finetuning_method"] = "no_finetuning" #"full_finetuning" 


#config_base["benchmarks"] = "tabarena" # 

config_base["saving_path"] = "../results/finetuning_experiments/" + config_base["model"] + "/" + config_base["finetuning_method"] + "/"