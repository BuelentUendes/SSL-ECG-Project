import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from metaflow import FlowSpec, step, Parameter, project, current
from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH
import mlflow

from utils.torch_utilities import (load_processed_data, split_indices_by_participant, set_seed, create_directory)

@project(name="ecg_majority_baseline")
class MajorityBaselineFlow(FlowSpec):

    mlflow_tracking_uri = Parameter("mlflow_tracking_uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    window_data_path = Parameter("window_data_path",
        default=f"{os.path.join(DATA_PATH, 'interim', 'windowed_data.h5')}")
    seed = Parameter("seed", default=42)

    @step
    def start(self):
        """Initialize MLflow and set seed."""
        set_seed(self.seed)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            raise RuntimeError(f"MLflow connection failed: {str(e)}")

        self.next(self.evaluate_baselines)

    @step
    def evaluate_baselines(self):
        X, y, groups = load_processed_data(self.window_data_path,
            label_map={"baseline": 0, "mental_stress": 1})
        _, _, test_idx = split_indices_by_participant(groups, seed=42)
        y_test = y[test_idx].astype(np.int32)

        # Majority class baseline
        majority_class = int(np.mean(y_test) >= 0.5)
        y_pred_majority = np.full_like(y_test, majority_class)

        acc_maj = accuracy_score(y_test, y_pred_majority)
        try:
            auc_maj = roc_auc_score(y_test, y_pred_majority)
        except:
            auc_maj = 0.5
        f1_macro_maj = f1_score(y_test, y_pred_majority, average='macro')
        f1_class0_maj = f1_score(y_test, y_pred_majority, pos_label=0)
        f1_class1_maj = f1_score(y_test, y_pred_majority, pos_label=1)

        print("\n[Majority Class Baseline Results]")
        print(f" Majority class : {majority_class}")
        print(f" Accuracy: {acc_maj:.4f}")
        print(f" AUC-ROC: {auc_maj:.4f}")
        print(f" F1 (macro): {f1_macro_maj:.4f}")
        print(f" F1 (class 0): {f1_class0_maj:.4f}")
        print(f" F1 (class 1): {f1_class1_maj:.4f}")

        # Random baseline
        rng = np.random.default_rng(self.seed)
        y_pred_rand = rng.integers(0, 2, size=len(y_test))

        acc_rand = accuracy_score(y_test, y_pred_rand)
        try:
            auc_rand = roc_auc_score(y_test, y_pred_rand)
        except:
            auc_rand = 0.5
        f1_macro_rand = f1_score(y_test, y_pred_rand, average='macro')
        f1_class0_rand = f1_score(y_test, y_pred_rand, pos_label=0)
        f1_class1_rand = f1_score(y_test, y_pred_rand, pos_label=1)

        print("\n[Random Class Baseline Results]")
        print(f" Accuracy: {acc_rand:.4f}")
        print(f" AUC-ROC: {auc_rand:.4f}")
        print(f" F1 (macro): {f1_macro_rand:.4f}")
        print(f" F1 (class 0): {f1_class0_rand:.4f}")
        print(f" F1 (class 1): {f1_class1_rand:.4f}")

        # Log to MLflow
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics({
                "baseline_majority_accuracy": acc_maj,
                "baseline_majority_auc": auc_maj,
                "baseline_majority_f1_macro": f1_macro_maj,
                "baseline_majority_f1_class0": f1_class0_maj,
                "baseline_majority_f1_class1": f1_class1_maj,
                "baseline_random_accuracy": acc_rand,
                "baseline_random_auc": auc_rand,
                "baseline_random_f1_macro": f1_macro_rand,
                "baseline_random_f1_class0": f1_class0_rand,
                "baseline_random_f1_class1": f1_class1_rand,
            })

        self.next(self.end)

    @step
    def end(self):
        print("Baseline evaluation complete.")
        mlflow.end_run()

if __name__ == "__main__":
    MajorityBaselineFlow()
