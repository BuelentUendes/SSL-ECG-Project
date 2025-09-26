#!/usr/bin/env python
import os
import argparse
import logging
import gc
import json
from typing import List, Dict, Any

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

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH, RESULTS_PATH


def run_single_seed(
        mlflow_tracking_uri: str,
        seed: int,
        pretrain_all_conditions: bool,
        label_fraction: float,
        dataset: str,
        save_individual: bool = True,
) -> Dict[str, Any]:
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
        folder_path = "ECG"
        window_data_path = os.path.join(DATA_PATH, "interim", "ECG", '1000', 'windowed_data.h5')
    elif dataset == "stressid":
        folder_path = "StressID"
        window_data_path = os.path.join(DATA_PATH, "interim", "STRESSID", "ECG", '500', '10', '5', 'windowed_data.h5')
    elif dataset == "wesad":
        folder_path = "WESAD"
        window_data_path = os.path.join(DATA_PATH, "interim", "WESAD", "ECG", '700', "10", "5", 'windowed_data.h5')

    model_save_path = os.path.join(SAVED_MODELS_PATH, "ECG", "Baseline", pretrain_data, f"{seed}")
    results_save_path = os.path.join(RESULTS_PATH, folder_path, pretrain_data, f"{seed}")
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

    # This is the dummy classifier that does predict simply either 0 or 1 random
    dummy_classifier_random = DummyClassifier(strategy="uniform", random_state=seed)

    dummy_classifier.fit(X_train, y_train)
    dummy_classifier_random.fit(X_train, y_train)

    y_pred = dummy_classifier.predict(X_test)
    y_pred_proba = dummy_classifier.predict_proba(X_test)[:, 1]

    # Get what the majority class is
    majority_class = int(np.mean(y_test) >= 0.5)

    accuracy_score_dummy = accuracy_score(y_test, y_pred)
    roc_auc_score_dummy = roc_auc_score(y_test, y_pred_proba)
    pr_auc_score_dummy = average_precision_score(y_test, y_pred_proba)

    f1_macro_maj = f1_score(y_test, y_pred_proba, average='macro')
    f1_class0_maj = f1_score(y_test, y_pred_proba, pos_label=0)
    f1_class1_maj = f1_score(y_test, y_pred_proba, pos_label=1)

    # For this we want to check the performance on the whole target dataet
    # This is for table Zero-shot performance
    # Given that we test on the whole dataset, there is no randomness here

    y_pred_random = dummy_classifier_random.predict(X)
    y_pred_proba_random = dummy_classifier_random.predict_proba(X)[:, 1]

    accuracy_score_dummy_random = accuracy_score(y, y_pred_random)
    roc_auc_score_dummy_random = roc_auc_score(y, y_pred_proba_random)
    pr_auc_score_dummy_random = average_precision_score(y, y_pred_proba_random)

    print("\n[Majority Class Baseline Results]")
    print(f" Majority class: {majority_class}")
    print(f" Accuracy: {accuracy_score_dummy:.4f}")
    print(f" AUC-ROC: {roc_auc_score_dummy:.4f}")
    print(f" PR-AUC: {pr_auc_score_dummy:.4f}")
    print(f" F1 (macro): {f1_macro_maj:.4f}")
    print(f" F1 (class 0): {f1_class0_maj:.4f}")
    print(f" F1 (class 1): {f1_class1_maj:.4f}")

    print("\n[Majority Class Baseline Results]")
    print(f" Random Accuracy: {accuracy_score_dummy_random:.4f}")
    print(f" Random AUC-ROC: {roc_auc_score_dummy_random:.4f}")
    print(f" Random PR-AUC: {pr_auc_score_dummy_random:.4f}")

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

    # Prepare results data
    results_data = {
        "experiment_info": {
            "seed": seed,
            "test_set_size": len(y_test),
            "class_distribution": {
                "class_0": int(np.sum(y_test == 0)),
                "class_1": int(np.sum(y_test == 1))
            }
        },
        "majority_class_baseline": {
            "majority_class": majority_class,
            "accuracy": float(accuracy_score_dummy),
            "auroc": float(roc_auc_score_dummy),
            "pr_auc": float(pr_auc_score_dummy),
            "f1_macro": float(f1_macro_maj),
            "f1_class0": float(f1_class0_maj),
            "f1_class1": float(f1_class1_maj)
        },
        "random_class_baseline": {
            "accuracy": float(accuracy_score_dummy_random),
            "auroc": float(roc_auc_score_dummy_random),
            "pr_auc": float(pr_auc_score_dummy_random)
        }
    }
    
    # Save individual seed results if requested
    if save_individual:
        results_file = os.path.join(results_save_path, "baseline_results.json")
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to: {results_file}")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    for _ in range(3):
        gc.collect()

    mlflow.end_run()
    
    return results_data


def compute_stats_across_seeds(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and standard error across seeds for all metrics."""
    
    # Extract metrics from all seeds
    majority_metrics = {}
    random_metrics = {}
    
    for metric_name in ['accuracy', 'auroc', 'pr_auc', 'f1_macro', 'f1_class0', 'f1_class1']:
        values = [result['majority_class_baseline'][metric_name] for result in all_results]
        majority_metrics[f"{metric_name}_mean"] = float(np.mean(values))
        majority_metrics[f"{metric_name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        majority_metrics[f"{metric_name}_stderr"] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    
    for metric_name in ['accuracy', 'auroc', 'pr_auc']:
        values = [result['random_class_baseline'][metric_name] for result in all_results]
        random_metrics[f"{metric_name}_mean"] = float(np.mean(values))
        random_metrics[f"{metric_name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        random_metrics[f"{metric_name}_stderr"] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    
    # Get experiment info from first result (should be consistent across seeds)
    first_result = all_results[0]
    
    aggregated_results = {
        "experiment_info": {
            "seeds": [result['experiment_info']['seed'] for result in all_results],
            "num_seeds": len(all_results),
            "test_set_size": first_result['experiment_info']['test_set_size'],
            "class_distribution": first_result['experiment_info']['class_distribution']
        },
        "majority_class_baseline_stats": majority_metrics,
        "random_class_baseline_stats": random_metrics,
        "individual_seed_results": all_results
    }
    
    return aggregated_results


def main(
    mlflow_tracking_uri: str,
    seeds: List[int],
    pretrain_all_conditions: bool,
    label_fraction: float,
    dataset: str,
    save_individual: bool = True,
):
    """Run baseline evaluation across multiple seeds and compute statistics."""
    
    logging.basicConfig(level=logging.INFO)
    print(f"Running baseline evaluation across seeds: {seeds}")
    
    all_results = []
    
    # Run evaluation for each seed
    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        result = run_single_seed(
            mlflow_tracking_uri=mlflow_tracking_uri,
            seed=seed,
            pretrain_all_conditions=pretrain_all_conditions,
            label_fraction=label_fraction,
            dataset=dataset,
            save_individual=save_individual
        )
        all_results.append(result)
        print(f"Completed seed {seed}")
    
    # Compute aggregated statistics
    aggregated_results = compute_stats_across_seeds(all_results)
    
    # Determine save path for aggregated results (parent folder, not seed subfolder)
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"
    if dataset == "ours":
        folder_path = "ECG"
    elif dataset == "stressid":
        folder_path = "StressID"
    elif dataset == "wesad":
        folder_path = "WESAD"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    aggregated_save_path = os.path.join(RESULTS_PATH, folder_path, pretrain_data)
    create_directory(aggregated_save_path)
    
    # Save aggregated results
    aggregated_file = os.path.join(aggregated_save_path, "baseline_results_aggregated.json")
    with open(aggregated_file, "w") as f:
        json.dump(aggregated_results, f, indent=2)
    
    # Print summary statistics
    print(f"\n=== Aggregated Results Across {len(seeds)} Seeds ===")
    maj_stats = aggregated_results['majority_class_baseline_stats']
    rand_stats = aggregated_results['random_class_baseline_stats']
    
    print("\n[Majority Class Baseline - Mean ± SE]")
    print(f"  Accuracy: {maj_stats['accuracy_mean']:.4f} ± {maj_stats['accuracy_stderr']:.4f}")
    print(f"  AUROC: {maj_stats['auroc_mean']:.4f} ± {maj_stats['auroc_stderr']:.4f}")
    print(f"  PR-AUC: {maj_stats['pr_auc_mean']:.4f} ± {maj_stats['pr_auc_stderr']:.4f}")
    print(f"  F1 (macro): {maj_stats['f1_macro_mean']:.4f} ± {maj_stats['f1_macro_stderr']:.4f}")
    
    print("\n[Random Class Baseline - Mean ± SE]")
    print(f"  Accuracy: {rand_stats['accuracy_mean']:.4f} ± {rand_stats['accuracy_stderr']:.4f}")
    print(f"  AUROC: {rand_stats['auroc_mean']:.4f} ± {rand_stats['auroc_stderr']:.4f}")
    print(f"  PR-AUC: {rand_stats['pr_auc_mean']:.4f} ± {rand_stats['pr_auc_stderr']:.4f}")
    
    print(f"\nAggregated results saved to: {aggregated_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Evaluation Pipeline")
    parser.add_argument("--mlflow_tracking_uri",
                        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    
    # Seed options - either single seed or multiple seeds
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, 
                           help="Single seed to run (mutually exclusive with --seeds)")
    seed_group.add_argument("--seeds", type=int, nargs='+', default=[3, 5, 7, 9, 42],
                           help="Multiple seeds to run with aggregated statistics (default: [3, 5, 7, 9, 42])")
    
    parser.add_argument("--pretrain_all_conditions", action="store_true")
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument("--dataset", choices=("stressid", "wesad", "ours"), default="wesad")
    parser.add_argument("--save_individual", action="store_true", default=True,
                       help="Save individual seed results (default: True)")

    args = parser.parse_args()
    
    # Determine which seeds to use
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = args.seeds
    
    # Remove seed/seeds from args and add the processed seeds list
    delattr(args, 'seed')
    delattr(args, 'seeds')
    args.seeds = seeds
    
    main(**vars(args))