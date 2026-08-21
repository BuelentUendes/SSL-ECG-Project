#!/usr/bin/env python
import os
import argparse
import logging
import gc
import json
from typing import List, Dict, Any

import numpy as np

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score, balanced_accuracy_score,
)
from sklearn.dummy import DummyClassifier

from utils.torch_utilities import (
    load_processed_data_with_conditions,
    split_indices_by_participant_groups,
    set_seed,
    create_directory,
)

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH, RESULTS_PATH


STRESSOR_GROUPS = [
    ("TA", ["TA", "TA_repeat"]),
    ("Pasat", ["Pasat", "Pasat_repeat"]),
    ("Raven", ["Raven"]),
    ("SSST", ["SSST_Sing_countdown"]),
]


def _per_condition_metrics_dummy(model, X_test, y_test, conditions_test):
    """Per-condition binary (vs baseline) metrics for a DummyClassifier."""
    baseline_mask = y_test == 0
    out = {}
    for cond in np.unique(conditions_test[y_test == 1]):
        cond_mask = (conditions_test == cond) & (y_test == 1)
        mask = baseline_mask | cond_mask
        if mask.sum() < 2 or cond_mask.sum() == 0:
            continue
        y_s = y_test[mask].astype(int)
        X_s = X_test[mask]
        proba = model.predict_proba(X_s)[:, 1]
        pred = model.predict(X_s)
        try:
            auroc = float(roc_auc_score(y_s, proba))
        except ValueError:
            auroc = float("nan")
        out[cond] = {
            "n_stress_samples": int(cond_mask.sum()),
            "auroc": auroc,
            "pr_auc": float(average_precision_score(y_s, proba)),
            "accuracy": float(accuracy_score(y_s, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_s, pred)),
            "f1": float(f1_score(y_s, pred, zero_division=0)),
        }
    return out


# Helper function to resample the positive and negative samples to keep the overall prevalence the same as in the overall test set performance
# On this way, I can see how the method performs across different stressors and if some stressors are more difficult classify or not

def _pr_auc_ratio_corrected(model, X, y, overall_ratio, seed=42):
    stress_idx = np.where(np.asarray(y) == 1)[0]
    baseline_idx = np.where(np.asarray(y) == 0)[0]
    n_stress = len(stress_idx)
    if n_stress == 0 or len(baseline_idx) == 0:
        return float("nan"), 0
    # This is the number of baseline values to sample from to keep the overall ratio between positive and negative as the desired rate
    n_baseline = int((1 - overall_ratio) * n_stress / overall_ratio)
    n_baseline = min(max(n_baseline, 1), len(baseline_idx))
    rng = np.random.RandomState(seed)
    sampled = rng.choice(baseline_idx, size=n_baseline, replace=False)
    combined = np.concatenate([stress_idx, sampled])
    y_s = np.asarray(y)[combined]
    if len(np.unique(y_s)) < 2:
        return float("nan"), n_baseline
    X_s = X[combined] if not hasattr(X, "iloc") else X.iloc[combined]
    proba = model.predict_proba(X_s)[:, 1]
    return float(average_precision_score(y_s, proba)), n_baseline


def run_single_seed(
        seed: int,
        pretrain_all_conditions: bool,
        label_fraction: float,
        dataset: str,
        save_individual: bool = True,
        leave_one_stressor_out: bool = False,
) -> Dict[str, Any]:
    # ── Step 0: Setup ────────────────────────────────────────────────────────────
    set_seed(seed)

    logging.basicConfig(level=logging.INFO)

    # Check if directory for saving results exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"
    if dataset == "ours":
        folder_path = "ECG"
        window_data_path = os.path.join(DATA_PATH, "interim", "ECG", '1000', '10', '5', 'windowed_data.h5')
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

    X, y, groups, conditions = load_processed_data_with_conditions(window_data_path, label_map=label_map)
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

    print(f"windows: train {len(train_idx)}, test {len(test_idx)}")

    # Filter test to binary classification task (baseline vs mental_stress)
    test_binary_mask = np.isin(y[test_idx], [0, 1])
    X_test = X[test_idx][test_binary_mask]
    y_test = y[test_idx][test_binary_mask]
    conditions_test = conditions[test_idx][test_binary_mask]

    # Binary training slice (needed for LOSO)
    train_binary_mask = np.isin(y[train_idx], [0, 1])
    X_train_binary = X_train[train_binary_mask]
    y_train_binary = y_train[train_binary_mask]
    conditions_train_binary = conditions[train_idx][train_binary_mask]

    print(f"Test set size for baseline evaluation: {len(y_test)}")
    print(f"Class distribution: {np.bincount(y_test)}")

    # ── Step 2: Evaluate Baselines ──────────────────────────────────────────────

    dummy_classifier = DummyClassifier(strategy="most_frequent", random_state=seed)
    dummy_classifier_random = DummyClassifier(strategy="uniform", random_state=seed)

    dummy_classifier.fit(X_train, y_train)
    dummy_classifier_random.fit(X_train, y_train)

    y_pred = dummy_classifier.predict(X_test)
    y_pred_proba = dummy_classifier.predict_proba(X_test)[:, 1]

    majority_class = int(np.mean(y_test) >= 0.5)

    accuracy_score_dummy = accuracy_score(y_test, y_pred)
    roc_auc_score_dummy = roc_auc_score(y_test, y_pred_proba)
    pr_auc_score_dummy = average_precision_score(y_test, y_pred_proba)

    f1_macro_maj = f1_score(y_test, y_pred, average='macro')
    f1_class0_maj = f1_score(y_test, y_pred, pos_label=0)
    f1_class1_maj = f1_score(y_test, y_pred, pos_label=1)

    # Random baseline evaluated on the binary test slice (consistent with majority baseline)
    y_pred_random = dummy_classifier_random.predict(X_test)
    y_pred_proba_random = dummy_classifier_random.predict_proba(X_test)[:, 1]

    accuracy_score_dummy_random = accuracy_score(y_test, y_pred_random)
    roc_auc_score_dummy_random = roc_auc_score(y_test, y_pred_proba_random)
    pr_auc_score_dummy_random = average_precision_score(y_test, y_pred_proba_random)

    print("\n[Majority Class Baseline Results]")
    print(f" Majority class: {majority_class}")
    print(f" Accuracy: {accuracy_score_dummy:.4f}")
    print(f" AUC-ROC: {roc_auc_score_dummy:.4f}")
    print(f" PR-AUC: {pr_auc_score_dummy:.4f}")
    print(f" F1 (macro): {f1_macro_maj:.4f}")
    print(f" F1 (class 0): {f1_class0_maj:.4f}")
    print(f" F1 (class 1): {f1_class1_maj:.4f}")

    print("\n[Random Class Baseline Results]")
    print(f" Random Accuracy: {accuracy_score_dummy_random:.4f}")
    print(f" Random AUC-ROC: {roc_auc_score_dummy_random:.4f}")
    print(f" Random PR-AUC: {pr_auc_score_dummy_random:.4f}")

    # ── Per-Condition Metrics ────────────────────────────────────────────────────
    per_condition_results = _per_condition_metrics_dummy(
        dummy_classifier, X_test, y_test, conditions_test
    )
    print("\n[Per-Condition Metrics (majority baseline)]")
    for cond, m in per_condition_results.items():
        print(f"  {cond}: AUROC={m['auroc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
              f"n_stress={m['n_stress_samples']}")

    # ── Leave-One-Stressor-Out ───────────────────────────────────────────────────
    loso_results = {}
    if leave_one_stressor_out and dataset == "ours":
        baseline_test_mask = y_test == 0
        for group_name, stressor_conditions in STRESSOR_GROUPS:

            held_out_train = (
                np.isin(conditions_train_binary, stressor_conditions) & (y_train_binary == 1)
            )
            X_tr_loso = X_train_binary[~held_out_train]
            y_tr_loso = y_train_binary[~held_out_train]

            held_out_test = np.isin(conditions_test, stressor_conditions) & (y_test == 1)
            loso_test_mask = baseline_test_mask | held_out_test
            X_te_loso = X_test[loso_test_mask]
            y_te_loso = y_test[loso_test_mask]

            if held_out_test.sum() == 0 or X_tr_loso.shape[0] == 0:
                print(f"LOSO [{group_name}]: skipping — no samples")
                continue

            clf_loso = DummyClassifier(strategy="most_frequent", random_state=seed)
            clf_loso.fit(X_tr_loso, y_tr_loso)

            y_pred_loso = clf_loso.predict(X_te_loso)
            y_proba_loso = clf_loso.predict_proba(X_te_loso)[:, 1]

            try:
                auroc_loso = float(roc_auc_score(y_te_loso, y_proba_loso))
            except ValueError:
                auroc_loso = float("nan")

            loso_metrics = {
                "auroc": auroc_loso,
                "pr_auc": float(average_precision_score(y_te_loso, y_proba_loso)),
                "accuracy": float(accuracy_score(y_te_loso, y_pred_loso)),
                "balanced_accuracy": float(balanced_accuracy_score(y_te_loso, y_pred_loso)),
                "f1": float(f1_score(y_te_loso, y_pred_loso, zero_division=0)),
            }
            overall_stress_ratio = float((y_test == 1).sum()) / len(y_test)
            pr_auc_corrected, n_baseline_used = _pr_auc_ratio_corrected(
                clf_loso, X_te_loso, y_te_loso,
                overall_ratio=overall_stress_ratio, seed=seed,
            )
            loso_results[group_name] = {
                "held_out_stressor": stressor_conditions,
                "n_train_stress": int((y_tr_loso == 1).sum()),
                "n_test_stress": int(held_out_test.sum()),
                "test_metrics": loso_metrics,
                "test_metrics_ratio_corrected": {
                    "pr_auc": pr_auc_corrected,
                    "n_baseline_samples_used": n_baseline_used,
                    "overall_stress_ratio_used": round(overall_stress_ratio, 4),
                },
            }
            print(f"LOSO [{group_name}]: AUROC={auroc_loso:.4f}, "
                  f"PR-AUC (raw)={loso_metrics['pr_auc']:.4f}, "
                  f"PR-AUC (ratio-corrected)={pr_auc_corrected:.4f}")

        if loso_results:
            loso_file = os.path.join(results_save_path, "loso_stressor_results.json")
            with open(loso_file, "w") as f:
                json.dump(loso_results, f, indent=2)
            print(f"LOSO results saved to: {loso_file}")

    # ── Step 3: Log Results ─────────────────────────────────────────────────────
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
        },
        "per_condition_metrics": per_condition_results,
        "loso_results": loso_results,
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

    return results_data


def _aggregate_metrics(all_results, key, metric_names):
    """Return mean/std/stderr for each metric across seeds for a given result key."""
    out = {}
    for metric_name in metric_names:
        values = [r[key][metric_name] for r in all_results if metric_name in r.get(key, {})]
        if not values:
            continue
        n = len(values)
        out[f"{metric_name}_mean"] = float(np.mean(values))
        out[f"{metric_name}_std"] = float(np.std(values, ddof=1)) if n > 1 else 0.0
        out[f"{metric_name}_stderr"] = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return out


def _aggregate_per_condition(all_results):
    """Aggregate per-condition metrics across seeds."""
    all_conditions = set()
    for r in all_results:
        all_conditions.update(r.get("per_condition_metrics", {}).keys())

    out = {}
    for cond in sorted(all_conditions):
        cond_metrics: Dict[str, List[float]] = {}
        for r in all_results:
            cond_data = r.get("per_condition_metrics", {}).get(cond)
            if cond_data is None:
                continue
            for k, v in cond_data.items():
                if k == "n_stress_samples":
                    continue
                cond_metrics.setdefault(k, []).append(v)
        aggregated_cond: Dict[str, Any] = {}
        for metric, values in cond_metrics.items():
            n = len(values)
            aggregated_cond[f"{metric}_mean"] = float(np.mean(values))
            aggregated_cond[f"{metric}_std"] = float(np.std(values, ddof=1)) if n > 1 else 0.0
            aggregated_cond[f"{metric}_stderr"] = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out[cond] = aggregated_cond
    return out


def _aggregate_loso(all_results):
    """Aggregate leave-one-stressor-out metrics across seeds."""
    all_groups = set()
    for r in all_results:
        all_groups.update(r.get("loso_results", {}).keys())

    out = {}
    for group in sorted(all_groups):
        group_metrics: Dict[str, List[float]] = {}
        held_out_stressor = None
        for r in all_results:
            group_data = r.get("loso_results", {}).get(group)
            if group_data is None:
                continue
            held_out_stressor = group_data["held_out_stressor"]
            for k, v in group_data["test_metrics"].items():
                group_metrics.setdefault(k, []).append(v)
        corrected_pr_aucs = []
        for r in all_results:
            group_data = r.get("loso_results", {}).get(group)
            if group_data is None:
                continue
            v = group_data.get("test_metrics_ratio_corrected", {}).get("pr_auc")
            if v is not None:
                corrected_pr_aucs.append(v)

        aggregated_group: Dict[str, Any] = {"held_out_stressor": held_out_stressor}
        for metric, values in group_metrics.items():
            n = len(values)
            aggregated_group[f"{metric}_mean"] = float(np.mean(values))
            aggregated_group[f"{metric}_std"] = float(np.std(values, ddof=1)) if n > 1 else 0.0
            aggregated_group[f"{metric}_stderr"] = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        if corrected_pr_aucs:
            n = len(corrected_pr_aucs)
            aggregated_group["pr_auc_ratio_corrected_mean"] = float(np.mean(corrected_pr_aucs))
            aggregated_group["pr_auc_ratio_corrected_std"] = float(np.std(corrected_pr_aucs, ddof=1)) if n > 1 else 0.0
            aggregated_group["pr_auc_ratio_corrected_stderr"] = float(np.std(corrected_pr_aucs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out[group] = aggregated_group
    return out


def compute_stats_across_seeds(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and standard error across seeds for all metrics."""
    majority_metrics = _aggregate_metrics(
        all_results, "majority_class_baseline",
        ["accuracy", "auroc", "pr_auc", "f1_macro", "f1_class0", "f1_class1"],
    )
    random_metrics = _aggregate_metrics(
        all_results, "random_class_baseline",
        ["accuracy", "auroc", "pr_auc"],
    )

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
        "per_condition_metrics_stats": _aggregate_per_condition(all_results),
        "loso_stats": _aggregate_loso(all_results),
        "individual_seed_results": all_results,
    }

    return aggregated_results


def main(
    seeds: List[int],
    pretrain_all_conditions: bool,
    label_fraction: float,
    dataset: str,
    save_individual: bool = True,
    leave_one_stressor_out: bool = False,
):
    """Run baseline evaluation across multiple seeds and compute statistics."""

    logging.basicConfig(level=logging.INFO)
    print(f"Running baseline evaluation across seeds: {seeds}")

    all_results = []

    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        result = run_single_seed(
            seed=seed,
            pretrain_all_conditions=pretrain_all_conditions,
            label_fraction=label_fraction,
            dataset=dataset,
            save_individual=save_individual,
            leave_one_stressor_out=leave_one_stressor_out,
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

    cond_stats = aggregated_results.get("per_condition_metrics_stats", {})
    if cond_stats:
        print("\n[Per-Condition Metrics - Mean ± SE]")
        for cond, m in cond_stats.items():
            print(f"  {cond}: AUROC={m.get('auroc_mean', float('nan')):.4f} "
                  f"± {m.get('auroc_stderr', float('nan')):.4f}")

    loso_stats = aggregated_results.get("loso_stats", {})
    if loso_stats:
        print("\n[Leave-One-Stressor-Out - Mean ± SE]")
        for group, m in loso_stats.items():
            print(f"  {group}: AUROC={m.get('auroc_mean', float('nan')):.4f} "
                  f"± {m.get('auroc_stderr', float('nan')):.4f}, "
                  f"PR-AUC (raw)={m.get('pr_auc_mean', float('nan')):.4f} "
                  f"± {m.get('pr_auc_stderr', float('nan')):.4f}, "
                  f"PR-AUC (corrected)={m.get('pr_auc_ratio_corrected_mean', float('nan')):.4f} "
                  f"± {m.get('pr_auc_ratio_corrected_stderr', float('nan')):.4f}")

    print(f"\nAggregated results saved to: {aggregated_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Evaluation Pipeline")
    # Seed options - either single seed or multiple seeds
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, 
                           help="Single seed to run (mutually exclusive with --seeds)")
    seed_group.add_argument("--seeds", type=int, nargs='+', default=[3, 5, 7, 9, 11, 13, 15, 17, 19, 42],
                           help="Multiple seeds to run with aggregated statistics (default: [3, 5, 7, 9, 42])")
    
    parser.add_argument("--pretrain_all_conditions", action="store_true")
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument("--dataset", choices=("stressid", "wesad", "ours"), default="ours")
    parser.add_argument("--save_individual", action="store_true", default=True,
                       help="Save individual seed results (default: True)")
    parser.add_argument("--leave_one_stressor_out", action="store_true",
                       help="Run leave-one-stressor-out analysis (only for --dataset ours)")

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