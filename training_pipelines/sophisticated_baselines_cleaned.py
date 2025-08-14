#!/usr/bin/env python
import os
import argparse
import logging
import gc

import numpy as np
import mlflow

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
from sklearn.dummy import DummyClassifier

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant_groups,
    set_seed,
    create_directory,
)

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH


def main(
        mlflow_tracking_uri: str,
        seed: int,
        pretrain_all_conditions: bool,
        label_fraction: float,
        dataset: str,
):
    # ── Step 0: Setup ────────────────────────────────────────────────────────────
    set_seed(seed)

    logging.basicConfig(level=logging.INFO)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Baseline_Evaluation")

    # Start top‑level run
    run = mlflow.start_run(run_name=f"baseline_evaluation_{seed}")
    run_id = run.info.run_id
    logging.info(f"MLflow run_id: {run_id}")

    # Check if directory for saving results exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"
    if dataset == "ours":
        folder_path = "Ours_Baseline"
        window_data_path = os.path.join(DATA_PATH, "interim", "ECG", '1000', 'windowed_data.h5')
    elif dataset == "stressid":
        folder_path = "StressID"
        window_data_path = os.path.join(DATA_PATH, "interim", "STRESSID", "ECG", '500', '10', '5', 'windowed_data.h5')
    elif dataset == "wesad":
        folder_path = "WESAD"
        window_data_path = os.path.join(DATA_PATH, "interim", "WESAD", "ECG", '700', "10", "5", 'windowed_data.h5')

    model_save_path = os.path.join(SAVED_MODELS_PATH, "ECG", "Baseline", pretrain_data, f"{seed}")
    results_save_path = os.path.join(SAVED_MODELS_PATH, folder_path, pretrain_data, f"{seed}")
    create_directory(model_save_path)
    create_directory(results_save_path)

    # ── Step 1: Preprocess ───────────────────────────────────────────────────────
    if pretrain_all_conditions:
        label_map = {
            "baseline": 0, "mental_stress": 1,
            "low_physical_activity": 2,
            "moderate_physical_activity": 3,
            "high_physical_activity": 4
        }
    else:
        label_map = {"baseline": 0, "mental_stress": 1}

    # Data path
    X, y, groups = load_processed_data(window_data_path, label_map=label_map)
    y = y.astype(np.int32)

    # Split by participant to get train/test split
    train_idx, train_p, test_idx, test_p = split_indices_by_participant_groups(
        groups,
        train_ratio=0.8,
        label_fraction=label_fraction,
        seed=seed
    )

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_test = X[test_idx]

    # Get splits - for baselines we only need test set
    print(f"windows: train {len(train_idx)}, test {len(test_idx)}")

    # Keep binary‐task mask for evaluation (filter to baseline vs mental_stress only)
    downstream_mask = {
        "train": np.isin(y[train_idx], [0, 1]),
        "test": np.isin(y[test_idx], [0, 1]),
    }

    # Filter test set to binary classification task
    y_test = y[test_idx][downstream_mask["test"]]

    print(f"Test set size for baseline evaluation: {len(y_test)}")
    print(f"Class distribution: {np.bincount(y_test)}")

    # ── Step 2: Evaluate Baselines ──────────────────────────────────────────────

    # Get the dummy classifier
    dummy_classifier = DummyClassifier(strategy="most_frequent", random_state=seed)

    dummy_classifier.fit(X_train, y_train)
    y_pred = dummy_classifier.predict(X_test)
    y_pred_proba = dummy_classifier.predict_proba(X_test)[:, 1]

    # Majority class baseline
    majority_class = int(np.mean(y_test) >= 0.5)

    accuracy_score_dummy = accuracy_score(y_test, y_pred)
    roc_auc_score_dummy = roc_auc_score(y_test, y_pred_proba)
    pr_auc_score_dummy = average_precision_score(y_test, y_pred_proba)

    f1_macro_maj = f1_score(y_test, y_pred_proba, average='macro')
    f1_class0_maj = f1_score(y_test, y_pred_proba, pos_label=0)
    f1_class1_maj = f1_score(y_test, y_pred_proba, pos_label=1)

    print("\n[Majority Class Baseline Results]")
    print(f" Majority class: {majority_class}")
    print(f" Accuracy: {accuracy_score_dummy:.4f}")
    print(f" AUC-ROC: {roc_auc_score_dummy:.4f}")
    print(f" PR-AUC: {pr_auc_score_dummy:.4f}")
    print(f" F1 (macro): {f1_macro_maj:.4f}")
    print(f" F1 (class 0): {f1_class0_maj:.4f}")
    print(f" F1 (class 1): {f1_class1_maj:.4f}")


    # ── Step 3: Log Results ─────────────────────────────────────────────────────
    baseline_params = {
        "seed": seed,
        "label_fraction": label_fraction,
        "pretrain_all_conditions": pretrain_all_conditions,
        "test_set_size": len(y_test),
        "class_0_count": int(np.sum(y_test == 0)),
        "class_1_count": int(np.sum(y_test == 1)),
    }
    mlflow.log_params(baseline_params)

    # Log all baseline metrics
    mlflow.log_metrics({
        # Majority baseline
        "baseline_majority_accuracy": accuracy_score_dummy,
        "baseline_majority_auroc": roc_auc_score_dummy,
        "baseline_majority_pr_auc": pr_auc_score_dummy,
        "baseline_majority_f1_macro": f1_macro_maj,
        "baseline_majority_f1_class0": f1_class0_maj,
        "baseline_majority_f1_class1": f1_class1_maj,
    })

    # Save results locally as well
    results_file = os.path.join(results_save_path, "baseline_results.txt")
    with open(results_file, "w") as f:
        f.write("=== Baseline Evaluation Results ===\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Test set size: {len(y_test)}\n")
        f.write(f"Class distribution: {np.bincount(y_test)}\n\n")

        f.write("[Majority Class Baseline]\n")
        f.write(f"Accuracy: {accuracy_score_dummy:.4f}\n")
        f.write(f"AUC-ROC: {roc_auc_score_dummy:.4f}\n")
        f.write(f"PR-AUC: {pr_auc_score_dummy:.4f}\n")
        f.write(f"F1 (macro): {f1_macro_maj:.4f}\n\n")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    for _ in range(3):
        gc.collect()

    print(f"\n=== Baseline Evaluation Complete ===")
    print(f"Results saved to: {results_file}")
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Evaluation Pipeline")
    parser.add_argument("--mlflow_tracking_uri",
                        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain_all_conditions", action="store_true")
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument("--dataset", choices=("stressid", "wesad", "ours"), default="wesad")

    args = parser.parse_args()
    main(**vars(args))