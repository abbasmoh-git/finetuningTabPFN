import pickle
from pathlib import Path

results_root = Path("../results/finetuning_experiments")

print("dataset,method,test_accuracy,test_balanced_accuracy,test_roc_auc,test_logloss")

for pkl_file in results_root.rglob("*.pkl"):
    with open(pkl_file, "rb") as f:
        results = pickle.load(f)

    config = results["config"]
    method = config["finetuning_method"]

    dataset_name = [k for k in results.keys() if k != "config"][0]
    test_metrics = results[dataset_name]["fold_1"]["test"]

    print(
        f"{dataset_name},"
        f"{method},"
        f"{test_metrics['accuracy']:.4f},"
        f"{test_metrics['balanced_accuracy']:.4f},"
        f"{test_metrics['roc_auc']:.4f},"
        f"{test_metrics['logloss']:.4f}"
    )