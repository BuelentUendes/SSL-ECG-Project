#!/usr/bin/env python
import os
import sys
import argparse
import logging
import gc

import numpy as np
import mlflow

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant,
    set_seed,
    create_directory,
)

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH


def main(
        window_data_path: str,
        mlflow_tracking_uri: str,
        seed: int,
        pretrain_all_conditions: bool,
        label_fraction: float,
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

    # We save results here via seeds
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"
    results_save_path = os.path.join(SAVED_MODELS_PATH, "Baseline", pretrain_data, f"{seed}")
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

    X, y, groups = load_processed_data(window_data_path, label_map=label_map)
    y = y.astype(np.int32)

    # Get splits - for baselines we only need test set
    train_idx, val_idx, test_idx = split_indices_by_participant(
        groups, label_fraction=label_fraction, self_supervised_method=False, seed=seed
    )
    print(f"windows: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    # Keep binary‐task mask for evaluation (filter to baseline vs mental_stress only)
    downstream_mask = {
        "train": np.isin(y[train_idx], [0, 1]),
        "val": np.isin(y[val_idx], [0, 1]),
        "test": np.isin(y[test_idx], [0, 1]),
    }

    # Filter test set to binary classification task
    y_test = y[test_idx][downstream_mask["test"]]

    print(f"Test set size for baseline evaluation: {len(y_test)}")
    print(f"Class distribution: {np.bincount(y_test)}")

    # ── Step 2: Evaluate Baselines ──────────────────────────────────────────────

    # Majority class baseline
    majority_class = int(np.mean(y_test) >= 0.5)
    y_pred_majority = np.full_like(y_test, majority_class)

    # Create probability scores for majority baseline (1.0 for predicted class, 0.0 for other)
    y_prob_majority = np.where(y_pred_majority == 1, 1.0, 0.0)

    acc_maj = accuracy_score(y_test, y_pred_majority)
    try:
        auc_maj = roc_auc_score(y_test, y_prob_majority) if len(np.unique(y_test)) > 1 else 0.5
        pr_auc_maj = average_precision_score(y_test, y_prob_majority) if len(np.unique(y_test)) > 1 else np.mean(y_test)
    except:
        auc_maj = 0.5
        pr_auc_maj = np.mean(y_test)

    f1_macro_maj = f1_score(y_test, y_pred_majority, average='macro')
    f1_class0_maj = f1_score(y_test, y_pred_majority, pos_label=0)
    f1_class1_maj = f1_score(y_test, y_pred_majority, pos_label=1)

    print("\n[Majority Class Baseline Results]")
    print(f" Majority class: {majority_class}")
    print(f" Accuracy: {acc_maj:.4f}")
    print(f" AUC-ROC: {auc_maj:.4f}")
    print(f" PR-AUC: {pr_auc_maj:.4f}")
    print(f" F1 (macro): {f1_macro_maj:.4f}")
    print(f" F1 (class 0): {f1_class0_maj:.4f}")
    print(f" F1 (class 1): {f1_class1_maj:.4f}")

    # Random baseline
    rng = np.random.default_rng(seed)
    y_pred_rand = rng.integers(0, 2, size=len(y_test))

    # Create random probability scores
    y_prob_rand = rng.uniform(0, 1, size=len(y_test))

    acc_rand = accuracy_score(y_test, y_pred_rand)
    try:
        auc_rand = roc_auc_score(y_test, y_prob_rand) if len(np.unique(y_test)) > 1 else 0.5
        pr_auc_rand = average_precision_score(y_test, y_prob_rand) if len(np.unique(y_test)) > 1 else np.mean(y_test)
    except:
        auc_rand = 0.5
        pr_auc_rand = np.mean(y_test)

    f1_macro_rand = f1_score(y_test, y_pred_rand, average='macro')
    f1_class0_rand = f1_score(y_test, y_pred_rand, pos_label=0)
    f1_class1_rand = f1_score(y_test, y_pred_rand, pos_label=1)

    print("\n[Random Class Baseline Results]")
    print(f" Accuracy: {acc_rand:.4f}")
    print(f" AUC-ROC: {auc_rand:.4f}")
    print(f" PR-AUC: {pr_auc_rand:.4f}")
    print(f" F1 (macro): {f1_macro_rand:.4f}")
    print(f" F1 (class 0): {f1_class0_rand:.4f}")
    print(f" F1 (class 1): {f1_class1_rand:.4f}")

    # Stratified baseline (predicts based on class proportions)
    class_proportions = np.bincount(y_test) / len(y_test)
    y_pred_stratified = rng.choice(2, size=len(y_test), p=class_proportions)
    y_prob_stratified = rng.uniform(0, 1, size=len(y_test))

    acc_strat = accuracy_score(y_test, y_pred_stratified)
    try:
        auc_strat = roc_auc_score(y_test, y_prob_stratified) if len(np.unique(y_test)) > 1 else 0.5
        pr_auc_strat = average_precision_score(y_test, y_prob_stratified) if len(np.unique(y_test)) > 1 else np.mean(
            y_test)
    except:
        auc_strat = 0.5
        pr_auc_strat = np.mean(y_test)

    f1_macro_strat = f1_score(y_test, y_pred_stratified, average='macro')
    f1_class0_strat = f1_score(y_test, y_pred_stratified, pos_label=0)
    f1_class1_strat = f1_score(y_test, y_pred_stratified, pos_label=1)

    print("\n[Stratified Random Baseline Results]")
    print(f" Class proportions: {class_proportions}")
    print(f" Accuracy: {acc_strat:.4f}")
    print(f" AUC-ROC: {auc_strat:.4f}")
    print(f" PR-AUC: {pr_auc_strat:.4f}")
    print(f" F1 (macro): {f1_macro_strat:.4f}")
    print(f" F1 (class 0): {f1_class0_strat:.4f}")
    print(f" F1 (class 1): {f1_class1_strat:.4f}")

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
        "baseline_majority_accuracy": acc_maj,
        "baseline_majority_auroc": auc_maj,
        "baseline_majority_pr_auc": pr_auc_maj,
        "baseline_majority_f1_macro": f1_macro_maj,
        "baseline_majority_f1_class0": f1_class0_maj,
        "baseline_majority_f1_class1": f1_class1_maj,

        # Random baseline
        "baseline_random_accuracy": acc_rand,
        "baseline_random_auroc": auc_rand,
        "baseline_random_pr_auc": pr_auc_rand,
        "baseline_random_f1_macro": f1_macro_rand,
        "baseline_random_f1_class0": f1_class0_rand,
        "baseline_random_f1_class1": f1_class1_rand,

        # Stratified baseline
        "baseline_stratified_accuracy": acc_strat,
        "baseline_stratified_auroc": auc_strat,
        "baseline_stratified_pr_auc": pr_auc_strat,
        "baseline_stratified_f1_macro": f1_macro_strat,
        "baseline_stratified_f1_class0": f1_class0_strat,
        "baseline_stratified_f1_class1": f1_class1_strat,
    })

    # Save results locally as well
    results_file = os.path.join(results_save_path, "baseline_results.txt")
    with open(results_file, "w") as f:
        f.write("=== Baseline Evaluation Results ===\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Test set size: {len(y_test)}\n")
        f.write(f"Class distribution: {np.bincount(y_test)}\n\n")

        f.write("[Majority Class Baseline]\n")
        f.write(f"Accuracy: {acc_maj:.4f}\n")
        f.write(f"AUC-ROC: {auc_maj:.4f}\n")
        f.write(f"PR-AUC: {pr_auc_maj:.4f}\n")
        f.write(f"F1 (macro): {f1_macro_maj:.4f}\n\n")

        f.write("[Random Baseline]\n")
        f.write(f"Accuracy: {acc_rand:.4f}\n")
        f.write(f"AUC-ROC: {auc_rand:.4f}\n")
        f.write(f"PR-AUC: {pr_auc_rand:.4f}\n")
        f.write(f"F1 (macro): {f1_macro_rand:.4f}\n\n")

        f.write("[Stratified Random Baseline]\n")
        f.write(f"Accuracy: {acc_strat:.4f}\n")
        f.write(f"AUC-ROC: {auc_strat:.4f}\n")
        f.write(f"PR-AUC: {pr_auc_strat:.4f}\n")
        f.write(f"F1 (macro): {f1_macro_strat:.4f}\n")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    for _ in range(3):
        gc.collect()

    print(f"\n=== Baseline Evaluation Complete ===")
    print(f"Results saved to: {results_file}")
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Evaluation Pipeline")
    parser.add_argument("--window_data_path",
                        default=f"{os.path.join(DATA_PATH, 'interim', 'windowed_data.h5')}")
    parser.add_argument("--mlflow_tracking_uri",
                        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain_all_conditions", action="store_true")
    parser.add_argument("--label_fraction", type=float, default=1.0)

    args = parser.parse_args()
    main(**vars(args))