#!/usr/bin/env python
import os
import json
import argparse
import logging
import gc

import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader, TensorDataset

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant_groups,
    set_seed,
    create_directory,
    get_participant_cv_splitter,
    run_logistic_regression_with_gridsearch,
    run_logistic_regression_with_gridsearch_verbose,
    run_mlp_with_cv_and_test,
)

from utils.helper_paths import DATA_PATH, RESULTS_PATH


def create_data_loaders(X, y, batch_size, device, shuffle=True):
    """Create PyTorch data loaders from numpy arrays"""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def handle_missing_data(data, drop_values=True, verbose=True):
    """Handle missing values and infinity values in the data."""
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
        was_numpy = True
    else:
        df = data.copy()
        was_numpy = False

    original_data_len = len(df)

    # Identify rows and columns with infinity values
    inf_mask = df.isin([np.inf, -np.inf])
    rows_with_inf = inf_mask.any(axis=1)
    cols_with_inf = inf_mask.any(axis=0)

    if verbose:
        print(f"Rows with infinity values: {rows_with_inf.sum()}")
        print(f"Columns with infinity values:")
        for col in df.columns[cols_with_inf]:
            inf_count = inf_mask[col].sum()
            print(f"  - {col}: {inf_count} infinity values ({(inf_count / len(df)) * 100:.2f}%)")

    # Identify rows and columns with NaN values
    nan_mask = df.isna()
    rows_with_nan = nan_mask.any(axis=1)
    cols_with_nan = nan_mask.any(axis=0)

    if verbose:
        print(f"Rows with NaN values: {rows_with_nan.sum()}")
        print(f"Columns with NaN values:")
        for col in df.columns[cols_with_nan]:
            nan_count = nan_mask[col].sum()
            print(f"  - {col}: {nan_count} NaN values ({(nan_count / len(df)) * 100:.2f}%)")

    if drop_values:
        clean_data = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
        clean_data = clean_data.dropna()

        dropped_percent = ((original_data_len - len(clean_data)) / original_data_len) * 100
        if verbose:
            print(f"Dropping these rows removed {np.round(dropped_percent, 4)}% of the original data")

        if was_numpy:
            return clean_data.values
        else:
            return clean_data
    else:
        if was_numpy:
            return df.values
        else:
            return df

def main(
        mlflow_tracking_uri: str,
        fs: str,
        gpu: int,
        seed: int,
        classifier_model: str,
        window_size: int,
        classifier_epochs: int,
        label_fraction: float,
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
        verbose: bool = False,
):
    # ── Step 0: Setup ────────────────────────────────────────────────────────────
    set_seed(seed)

    # device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logging.basicConfig(level=logging.INFO)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Baseline Classifiers CV")

    # Start top‑level run
    run = mlflow.start_run(run_name=f"cv_{classifier_model}_{seed}_lf_{label_fraction}")
    run_id = run.info.run_id
    logging.info(f"MLflow run_id: {run_id}")
    print(f"Using device: {device}")

    # Check if directory for saving results exist
    create_directory(RESULTS_PATH)
    results_save_path = os.path.join(RESULTS_PATH, "ECG_features", classifier_model, f"{seed}",
                                     f"{label_fraction}")
    create_directory(results_save_path)

    # ── Step 1: Load and Preprocess Data ────────────────────────────────────────
    label_map = {"baseline": 0, "mental_stress": 1}
    window_data_path = os.path.join(DATA_PATH, "interim", "ECG_features", str(window_size), 'windowed_data.h5')

    X, y, groups, feature_names = load_processed_data(window_data_path, label_map=label_map, domain_features=True)
    y = y.astype(np.float32)

    # Handle missing values
    print("=== Handling missing values (Drop missing values) ===")
    X_clean = handle_missing_data(X, drop_values=True, verbose=True)

    if len(X_clean) != len(X):
        print(f"Updating labels and groups after dropping {len(X) - len(X_clean)} samples")
        X_df = pd.DataFrame(X)
        valid_rows = ~(X_df.isin([np.inf, -np.inf]).any(axis=1) | X_df.isna().any(axis=1))
        y = y[valid_rows]
        groups = groups[valid_rows]
        X = X_clean

    # Split by participant to get train/test split
    train_idx, train_p, test_idx, test_p = split_indices_by_participant_groups(
        groups,
        train_ratio=0.8,
        label_fraction=label_fraction,
        seed=seed
    )

    X_train_all = X[train_idx]
    y_train_all = y[train_idx]
    groups_train_all = groups[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    # Filter to binary classification for both train and test
    train_binary_mask = np.isin(y_train_all, [0, 1])
    test_binary_mask = np.isin(y_test, [0, 1])

    X_train_all = X_train_all[train_binary_mask]
    y_train_all = y_train_all[train_binary_mask]
    groups_train_all = groups_train_all[train_binary_mask]

    X_test = X_test[test_binary_mask]
    y_test = y_test[test_binary_mask]

    print(f"Training data: {X_train_all.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Training participants: {len(np.unique(groups_train_all))}")
    print(f"Test participants: {len(np.unique(groups[test_idx][test_binary_mask]))}")

    # ── Step 2: Set up Cross-Validation Splitter ───────────────────────────────
    cv_splitter, n_splits = get_participant_cv_splitter(
        groups_train_all,
        min_participants_for_kfold=min_participants_for_kfold,
        k=k_folds
    )

    # ── Step 3: Run Model Selection + Final Training + Test Evaluation ─────────
    if classifier_model == "logistic_regression":

        #Verbose option:
        if verbose:
            results = run_logistic_regression_with_gridsearch_verbose(
                X_train_all, y_train_all, groups_train_all, X_test, y_test,
                feature_names, cv_splitter, True, seed
            )

        else:
            results = run_logistic_regression_with_gridsearch(
                X_train_all, y_train_all, groups_train_all,
                X_test, y_test, feature_names, cv_splitter, True, seed,
            )

        # Log metrics
        mlflow.log_metrics({
            "best_cv_auroc": results['best_cv_score'] if cv_splitter is not None else 0,
            "test_accuracy": results['test_metrics']['accuracy'],
            "test_auroc": results['test_metrics']['auroc'],
            "test_f1": results['test_metrics']['f1'],
            "test_pr_auc": results['test_metrics']['pr_auc'],
        })

        mlflow.log_params(results['best_params'])

    elif classifier_model == "mlp":
        results = run_mlp_with_cv_and_test(
            X_train_all, y_train_all, groups_train_all,
            X_test, y_test, feature_names, cv_splitter,
            device, classifier_epochs, seed
        )

        # Log metrics
        mlflow.log_metrics({
            "best_cv_auroc": results['best_cv_score'],
            "test_accuracy": results['test_metrics']['accuracy'],
            "test_auroc": results['test_metrics']['auroc'],
            "test_f1": results['test_metrics']['f1'],
            "test_pr_auc": results['test_metrics']['pr_auc'],
        })

        mlflow.log_params(results['best_params'])

    # ── Step 4: Save Results ────────────────────────────────────────────────────
    with open(os.path.join(results_save_path, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Log parameters
    mlflow.log_params({
        "classifier_model": classifier_model,
        "label_fraction": label_fraction,
        "seed": seed,
        "k_folds": k_folds,
        "n_cv_splits": n_splits,
        "window_size": window_size,
    })

    # ── Cleanup ────────────────────────────────────────────────────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mlflow.end_run()
    print(f"=== Cross-Validation Complete! Results saved to {results_save_path} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Classifier CV Pipeline")
    parser.add_argument("--mlflow_tracking_uri",
                        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--fs", default=1000, type=str, help="Sample frequency")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier_model", type=str, default="logistic_regression",
                        choices=("logistic_regression", "mlp"))
    parser.add_argument("--window_size", help="Window size in seconds", default=30, type=int)
    parser.add_argument("--classifier_epochs", type=int, default=25)
    parser.add_argument("--label_fraction", type=float, default=0.1)
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for CV")
    parser.add_argument("--min_participants_for_kfold", type=int, default=5,
                        help="Minimum participants needed for k-fold (otherwise use LOGO)")
    parser.add_argument("--verbose", action="store_true",
                        help="If set, we show a verbose output of CV. Only applicable for LR. "
                             "Important: This slows down the fitting!")

    args = parser.parse_args()
    main(**vars(args))