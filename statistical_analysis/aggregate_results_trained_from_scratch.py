"""
Aggregate "trained from scratch" results across seeds for all methods and datasets.

Methods covered:
  - CNN (supervised)                  : results/{dataset}/Supervised/cnn/
  - Feature-engineered (LR)          : results/{dataset}_features/10/5/logistic_regression/
  - TSTCC (SSL, trained from scratch) : results/Transfer_learning/{dataset}/trained_from_scratch/TSTCC/logistic_regression/
  - TSTCC_128                         : results/Transfer_learning/{dataset}/trained_from_scratch/TSTCC_128/logistic_regression/
  - TSTCC_S3                          : results/Transfer_learning/{dataset}/trained_from_scratch/TSTCC_S3/logistic_regression/

Datasets: WESAD, StressID
Seeds used: [3, 5, 7, 9, 42]  (common to both datasets)
Label fraction: 1.0 | Window: 10 | Step: 5
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from utils.helper_paths import RESULTS_PATH

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------

def _supervised_path(results_path, dataset, seed):
    """CNN supervised: results/{dataset}/Supervised/cnn/{seed}/1.0/10/5/test_results.json"""
    return os.path.join(
        results_path, dataset, "Supervised", "cnn",
        str(seed), LABEL_FRACTION,
        str(WINDOW_SIZE), str(STEP_SIZE),
        "test_results.json",
    )


def _features_path(results_path, dataset, seed):
    """Feature-engineered: results/{dataset}_features/10/5/logistic_regression/{seed}/1.0/test_results.json"""
    return os.path.join(
        results_path, f"{dataset}_features",
        str(WINDOW_SIZE), str(STEP_SIZE),
        "logistic_regression",
        str(seed), LABEL_FRACTION,
        "test_results.json",
    )


def _tstcc_scratch_path(results_path, dataset, model, seed):
    """TSTCC trained from scratch: results/Transfer_learning/{dataset}/trained_from_scratch/{model}/logistic_regression/{seed}/1.0/10/5/test_results.json"""
    return os.path.join(
        results_path, "Transfer_learning", dataset,
        "trained_from_scratch", model,
        "logistic_regression",
        str(seed), LABEL_FRACTION,
        str(WINDOW_SIZE), str(STEP_SIZE),
        "test_results.json",
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_metrics(file_path):
    """Load auroc and pr_auc from a test_results.json file."""
    with open(file_path) as f:
        data = json.load(f)
    m = data["test_metrics"]
    return {"auroc": m["auroc"], "pr_auc": m["pr_auc"]}


def load_results_for_model(path_fn, seeds):
    """Iterate seeds, load metrics; return list of dicts with seed + metrics."""
    results = []
    for seed in seeds:
        file_path = path_fn(seed)
        if os.path.exists(file_path):
            try:
                results.append({"seed": seed, **_load_metrics(file_path)})
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [WARN] Error reading {file_path}: {e}")
        else:
            print(f"  [MISSING] {file_path}")
    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_stats(values):
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    stderr = std / np.sqrt(n)
    return {"mean": float(mean), "std": float(std), "stderr": float(stderr), "n": n}


def _print_model(model_name, results):
    if not results:
        print(f"  {model_name}: no results found")
        return
    auroc = [r["auroc"] for r in results]
    pr_auc = [r["pr_auc"] for r in results]
    n = len(auroc)
    print(f"\n  {model_name}  (n={n}, seeds={[r['seed'] for r in results]})")
    print(f"    AUROC : {np.mean(auroc):.3f} ± {np.std(auroc, ddof=1):.3f}  "
          f"(stderr: {np.std(auroc, ddof=1)/np.sqrt(n):.4f})")
    print(f"    PR-AUC: {np.mean(pr_auc):.3f} ± {np.std(pr_auc, ddof=1):.3f}  "
          f"(stderr: {np.std(pr_auc, ddof=1)/np.sqrt(n):.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def aggregate_dataset(results_path, dataset):
    """Aggregate all methods for one dataset; return (all_results_df, summary_df)."""

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}")
    print(f"  Seeds: {SEEDS}  |  Label fraction: {LABEL_FRACTION}  |  Window: {WINDOW_SIZE}/{STEP_SIZE}")
    print(f"{'='*60}")

    # Define all models and their path functions
    models = {
        "CNN": lambda seed: _supervised_path(results_path, dataset, seed),
        "ECG_Features_10s_5s": lambda seed: _features_path(results_path, dataset, seed),
        "TSTCC": lambda seed: _tstcc_scratch_path(results_path, dataset, "TSTCC", seed),
    }

    all_rows = []
    summary_rows = []

    for model_name, path_fn in models.items():
        results = load_results_for_model(path_fn, SEEDS)
        _print_model(model_name, results)

        if not results:
            continue

        auroc = [r["auroc"] for r in results]
        pr_auc = [r["pr_auc"] for r in results]

        for r in results:
            all_rows.append({"dataset": dataset, "model": model_name, **r})

        for metric, values in [("AUROC", auroc), ("PR_AUC", pr_auc)]:
            stats = _compute_stats(values)
            summary_rows.append({
                "dataset": dataset,
                "model": model_name,
                "metric": metric,
                **stats,
            })

    return pd.DataFrame(all_rows), pd.DataFrame(summary_rows)


def main(dataset_list):
    all_individual = []
    all_summary = []

    for dataset in dataset_list:
        ind_df, sum_df = aggregate_dataset(RESULTS_PATH, dataset)
        all_individual.append(ind_df)
        all_summary.append(sum_df)

    individual_df = pd.concat(all_individual, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)

    out_dir = os.path.dirname(os.path.abspath(__file__))

    individual_csv = os.path.join(out_dir, "trained_from_scratch_individual_results.csv")
    summary_csv = os.path.join(out_dir, "trained_from_scratch_summary_statistics.csv")

    individual_df.to_csv(individual_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nIndividual results  -> {individual_csv}")
    print(f"Summary statistics  -> {summary_csv}")

    # Also print a clean summary table
    print(f"\n{'='*60}")
    print("Summary (mean ± stderr)")
    print(f"{'='*60}")
    for _, row in summary_df.iterrows():
        print(f"  {row['dataset']:10s} | {row['model']:25s} | {row['metric']:6s} : "
              f"{row['mean']:.3f} ± {row['stderr']:.4f}  (std={row['std']:.3f}, n={row['n']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate trained-from-scratch results across seeds for all methods."
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("wesad", "stressid", "all"),
        help="Dataset to aggregate (default: all)",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["WESAD", "StressID"]
    elif args.dataset == "wesad":
        datasets = ["WESAD"]
    else:
        datasets = ["StressID"]

    SEEDS = [3, 5, 7, 9, 42]
    LABEL_FRACTION = "1.0"
    WINDOW_SIZE = 10
    STEP_SIZE = 5

    main(datasets)