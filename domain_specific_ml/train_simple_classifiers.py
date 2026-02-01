#!/usr/bin/env python
import os
import json
import argparse
import gc

import numpy as np
import pandas as pd
import torch

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
    evaluate_zero_shot_model_performance,
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
        # Reshape 3D arrays to 2D by flattening the last dimension
        if data.ndim == 3:
            original_shape = data.shape
            data_reshaped = data.reshape(data.shape[0], -1)
            df = pd.DataFrame(data_reshaped)
            was_3d = True
        else:
            df = pd.DataFrame(data)
            was_3d = False
        was_numpy = True
    else:
        df = data.copy()
        was_numpy = False
        was_3d = False

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
        clean_data = clean_data.dropna(axis=0)

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
        fs: str,
        gpu: int,
        seed: int,
        classifier_model: str,
        window_size: int,
        step_size: int,
        classifier_epochs: int,
        label_fraction: float,
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
        verbose: bool = False,
        zero_shot_evaluation: bool=False,
        zero_shot_dataset: str="wesad",
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

    print(f"Using device: {device}")

    # Check if directory for saving results exist
    create_directory(RESULTS_PATH)
    results_save_path = os.path.join(RESULTS_PATH, "ECG_features", classifier_model, f"{seed}",
                                     f"{label_fraction}", str(window_size), str(step_size))
    create_directory(results_save_path)

    if zero_shot_evaluation:
        target_domain = "StressID" if zero_shot_dataset == "stressid" else "WESAD"
        zero_shot_results_path = os.path.join(
            RESULTS_PATH, "Transfer_learning", target_domain, "zero_shot_performance", "feature_engineered", classifier_model,
            f"{seed}", f"{label_fraction}", str(window_size), str(step_size)
    )

        create_directory(zero_shot_results_path)

    # ── Step 1: Load and Preprocess Data ────────────────────────────────────────
    label_map = {"baseline": 0, "mental_stress": 1}
    window_data_path = os.path.join(
        DATA_PATH, "interim", "ECG_features", str(fs), str(window_size), str(step_size), 'windowed_data.h5'
    )

    X, y, groups, feature_names = load_processed_data(window_data_path, label_map=label_map, domain_features=True)
    y = y.astype(np.float32)

    # Load the zero-shot window dataset
    if zero_shot_evaluation:
        if zero_shot_dataset == "wesad":
            if int(fs) == 700:
                zero_shot_window_data_path = os.path.join(
                    DATA_PATH, "interim", "WESAD_features", "ECG", str(fs), str(window_size), str(step_size), 'windowed_data.h5')
            else:
                raise ValueError("For zero-shot evaluation for wesad the frequency needs to be 700")

        elif zero_shot_dataset == "stressid":
            if int(fs) == 500:
                zero_shot_window_data_path = os.path.join(
                DATA_PATH, "interim", "STRESSID_features", "ECG", str(fs), str(window_size), str(step_size), 'windowed_data.h5')
            else:
                raise ValueError("For zero-shot evaluation for stressid the frequency needs to be 500")
        else:
            raise ValueError('Please use a proper dataset "wesad" or "stressid"')

        X_zero_shot, y_zero_shot, groups_shot = load_processed_data(
            zero_shot_window_data_path, label_map={"baseline": 0, "mental_stress": 1}
        )
        y_zero_shot = y_zero_shot.astype(np.float32)

        # Handle missing data
        X_zero_shot_clean = handle_missing_data(X_zero_shot, drop_values=True, verbose=True)

        if len(X_zero_shot_clean) != len(X_zero_shot):
            print(f"Updating labels and groups after dropping {len(X_zero_shot) - len(X_zero_shot_clean)} samples")
            # Handle 3D array case by reshaping for DataFrame operations
            if X_zero_shot.ndim == 3:
                X_zero_shot_reshaped = X_zero_shot.reshape(X_zero_shot.shape[0], -1)
                X_zero_shot_df = pd.DataFrame(X_zero_shot_reshaped)
            else:
                X_zero_shot_df = pd.DataFrame(X_zero_shot)
            valid_rows = ~(X_zero_shot_df.isin([np.inf, -np.inf]).any(axis=1) | X_zero_shot_df.isna().any(axis=1))
            y_zero_shot = y_zero_shot[valid_rows]
            groups_shot = groups_shot[valid_rows]
            X_zero_shot = X_zero_shot_clean

    X_df = pd.DataFrame(X, columns=feature_names)
    missing_percentages = (X_df.isnull().sum() / len(X_df)) * 100
    print("=== Missing values percentage per feature ===")
    for feature, percentage in missing_percentages.items():
        if percentage > 0:
            print(f"{feature}: {percentage:.2f}%")
    print(f"Features with missing values: {(missing_percentages > 0).sum()}/{len(feature_names)}")

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
    if classifier_model in ["logistic_regression", "random_forest", "xgboost"]:

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
                classifier_model=classifier_model
            )

    elif classifier_model == "mlp":
        results = run_mlp_with_cv_and_test(
            X_train_all, y_train_all, groups_train_all,
            X_test, y_test, feature_names, cv_splitter,
            device, classifier_epochs, seed
        )

    if zero_shot_evaluation:
        # Load the best-trained model and scaler
        classifier_model = results["model"]
        scaler = results.get("scaler", None)
        
        # Apply the same standardization to zero-shot data if scaler exists
        if scaler is not None:
            standard_scaler, minmax_scaler = scaler
            X_zero_shot_scaled = X_zero_shot.copy()
            
            # Apply the same feature standardization as training
            min_max_scaler_names = ["nn20", "nn50", "wmax"]
            nn_indices = []
            standard_indices = []
            
            for i, name in enumerate(feature_names):
                if name.lower() in min_max_scaler_names:
                    nn_indices.append(i)
                else:
                    standard_indices.append(i)
            
            # Transform zero-shot data using fitted scalers
            if standard_indices:
                X_zero_shot_scaled[:, standard_indices] = standard_scaler.transform(X_zero_shot[:, standard_indices])
            if nn_indices:
                X_zero_shot_scaled[:, nn_indices] = minmax_scaler.transform(X_zero_shot[:, nn_indices])
            
            X_zero_shot = X_zero_shot_scaled
        
        # Then test the performance
        zero_shot_results = evaluate_zero_shot_model_performance(classifier_model, X_zero_shot, y_zero_shot)

        # Save results
        with open(os.path.join(zero_shot_results_path, "zero_shot_results.json"), 'w') as f:
            json.dump(zero_shot_results, f, indent=2, default=str)

    # ── Step 4: Save Results ────────────────────────────────────────────────────
    with open(os.path.join(results_save_path, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # mlflow.end_run()
    print(f"=== Cross-Validation Complete! Results saved to {results_save_path} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Classifier CV Pipeline")
    parser.add_argument("--fs", default=1000, type=str, help="Sample frequency")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier_model", type=str, default="logistic_regression",
                        choices=("logistic_regression", "mlp", "random_forest", "xgboost"))
    parser.add_argument("--window_size", help="Window size in seconds", default=30, type=int)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--classifier_epochs", type=int, default=25)
    parser.add_argument("--label_fraction", type=float, default=0.01)
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for CV")
    parser.add_argument("--min_participants_for_kfold", type=int, default=5,
                        help="Minimum participants needed for k-fold (otherwise use Leave one participant out)")
    parser.add_argument("--verbose", action="store_true",
                        help="If set, we show a verbose output of CV. Only applicable for LR. "
                             "Important: This slows down the fitting!")
    parser.add_argument("--zero_shot_evaluation", action="store_true",
                        help="If set, we do downstream zero-shot evaluation")
    parser.add_argument("--zero_shot_dataset", type=str,
                                 choices=("stressid", "wesad"), default="wesad")

    args = parser.parse_args()

    main(**vars(args))