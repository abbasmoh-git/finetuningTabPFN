
import argparse

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tabular ML probing experiments on OpenML benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",            type=str,  default="tabpfn_v3.config_c0",
                        help="Config module path (e.g. 'tabpfn_v3.config_c0').")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    
    #configs, classifier_classes = get_classifier_config_list(f"configs.{args.config}")

    # 1) load config
    # 2) load model and dataset 
    #    a) use openml_task_id to load dataset and splits using openml API
    #    b) train, validation, and test splits (10 folds, 6, 2, 2) for each dataset for now one repetition, later more repetitions

    # For now tabtune can handle step 3:
    #   3) fine tuning amd evaluate the model with differnt metrics: accuracy, balanced accuracy, roc_auc, logloss, etc. on the train, validation and test set.

    # 4) put all results along with configs into a dict and save results to config_base["saving_path"]
    # dictionary should be like this:
    # results = {
    #     "config": {configurations},
    #     "dataset_name": {
    #         "fold_1": {
    #             "train": {
    #                 "accuracy": 0.9,
    #                 "balanced_accuracy": 0.8,
    #                 "roc_auc": 0.85,
    #                 "logloss": 0.2,   
    #             },    
    #             "validation": {
    #                 "accuracy": 0.85,
    #                 "balanced_accuracy": 0.75,
    #                 "roc_auc": 0.8,
    #                 "logloss": 0.3,
    #             },

    #             "test": {
    #                 "accuracy": 0.8,
    #                 "balanced_accuracy": 0.7,
    #                 "roc_auc": 0.75,
    #                 "logloss": 0.4,
    #             },
 

if __name__ == "__main__":
    main()