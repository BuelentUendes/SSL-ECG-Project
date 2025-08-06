#!/usr/bin/env python
import os
import json
import sys
import argparse
import logging
import tempfile
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, ParameterGrid, GridSearchCV

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant,
    split_indices_by_participant_groups,
    set_seed,
    create_directory,
    train_classifier_for_optuna,
)

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH, RESULTS_PATH

from models.supervised import LinearClassifier, MLPClassifier
from models.tstcc import (
    build_linear_loaders,
    evaluate_classifier,
    train_linear_classifier,
)


def create_data_loaders(X, y, batch_size, device, shuffle=True):
    """Create PyTorch data loaders from numpy arrays"""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def standardize_features(X_train, X_val, X_test, feature_names):
    # Initialize scalers
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    # Create copies to avoid modifying original data
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy() if X_val is not None else None
    X_test_scaled = X_test.copy() if X_test is not None else None

    # Identify nn20 and nn50 indices
    nn_indices = []
    standard_indices = []

    min_max_scaler_names = ["nn20", "nk_pnn20", "nn50", "nk_pnn50"]
    for i, name in enumerate(feature_names):
        if name.lower() in min_max_scaler_names:
            nn_indices.append(i)
        else:
            standard_indices.append(i)

    # Apply StandardScaler to most features (fit only on train)
    if standard_indices:
        standard_scaler.fit(X_train[:, standard_indices])
        X_train_scaled[:, standard_indices] = standard_scaler.transform(X_train[:, standard_indices])
        if X_val is not None:
            X_val_scaled[:, standard_indices] = standard_scaler.transform(X_val[:, standard_indices])
        if X_test is not None:
            X_test_scaled[:, standard_indices] = standard_scaler.transform(X_test[:, standard_indices])

    # Apply MinMaxScaler to nn20/nn50 features (fit only on train)
    if nn_indices:
        minmax_scaler.fit(X_train[:, nn_indices])
        X_train_scaled[:, nn_indices] = minmax_scaler.transform(X_train[:, nn_indices])
        if X_val is not None:
            X_val_scaled[:, nn_indices] = minmax_scaler.transform(X_val[:, nn_indices])
        if X_test is not None:
            X_test_scaled[:, nn_indices] = minmax_scaler.transform(X_test[:, nn_indices])

    return X_train_scaled, X_val_scaled, X_test_scaled


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


def get_participant_cv_splitter(groups, min_participants_for_kfold=5, k=5):
    """Get appropriate cross-validation splitter based on number of participants."""
    unique_participants = np.unique(groups)
    n_participants = len(unique_participants)

    print(f"Total participants: {n_participants}")

    if n_participants < min_participants_for_kfold:
        if n_participants == 1:
            print(f"We only have 1 participant, return None and fit a default LR!")
            return None, None

        cv_splitter = LeaveOneGroupOut()
        n_splits = n_participants
        print(f"Using Leave-One-Group-Out CV ({n_splits} splits)")
    else:
        actual_k = min(k, n_participants)
        cv_splitter = GroupKFold(n_splits=actual_k)
        n_splits = actual_k
        print(f"Using {actual_k}-Fold Group CV ({n_splits} splits)")

        if n_participants % actual_k != 0:
            print(f"Note: {n_participants} participants don't divide evenly by {actual_k}")
            print("Some folds will have different numbers of participants")

    return cv_splitter, n_splits


def run_logistic_regression_with_gridsearch(X_train, y_train, groups_train, X_test, y_test,
                                            feature_names, cv_splitter, seed=42):
    """Run GridSearchCV for Logistic Regression, then evaluate on test set."""

    # Standardize features (fit on train, transform both)
    X_train_scaled, _, X_test_scaled = standardize_features(X_train, None, X_test, feature_names)

    # Parameter grid for grid search
    param_grid = {
        'C': [0.0001, 0.001, 0.01],
        'penalty': ['l2', 'l1'],
        'max_iter': [5000],
    }

    default_best_params = {
        'C': 0.001,
        'penalty': 'l2',
        'max_iter': 5000,
        'random_state': seed,
        'n_jobs': -1,
        'solver': 'saga'
    }

    if cv_splitter is not None:
        # Create base model for grid search
        base_model = LogisticRegression(random_state=seed, n_jobs=-1, solver="saga")

        # GridSearchCV with GroupKFold
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_splitter,
            scoring='roc_auc',  # Use AUROC as the scoring metric
            n_jobs=-1,
            verbose=3,
        )

        print("Running GridSearchCV...")
        # Fit grid search (this does 5-fold CV internally)
        grid_search.fit(X_train_scaled, y_train, groups=groups_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score (AUROC): {grid_search.best_score_:.4f}")

        # Get the best model (already trained on full training set)
        cv_results = grid_search.cv_results_
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

    else:
        # Use default best parameters when cv_splitter is None
        print("cv_splitter is None (single participant). Using default best parameters...")
        print(f"Default parameters: {default_best_params}")

        best_model = LogisticRegression(**default_best_params)
        best_model.fit(X_train_scaled, y_train)

        best_params = default_best_params
        best_cv_score = None
        cv_results = None

    # Evaluate on test set
    y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = best_model.predict(X_test_scaled)

    # Calculate test metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auroc = roc_auc_score(y_test, y_test_proba)
    test_f1 = f1_score(y_test, y_test_pred)
    test_pr_auc = average_precision_score(y_test, y_test_proba)

    print(f"\n=== Test Set Results ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")

    return {
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'cv_results': cv_results,
        'test_metrics': {
            'accuracy': test_acc,
            'auroc': test_auroc,
            'f1': test_f1,
            'pr_auc': test_pr_auc
        },
        'model': best_model
    }


def run_logistic_regression_with_gridsearch_verbose(X_train, y_train, groups_train, X_test, y_test,
                                                    feature_names, cv_splitter, seed=42):
    """Run GridSearchCV for Logistic Regression with detailed logging of splits and fitting."""

    # Standardize features (fit on train, transform both)
    X_train_scaled, _, X_test_scaled = standardize_features(X_train, None, X_test, feature_names)

    # Parameter grid for grid search
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0],
        'max_iter': [1000],
    }

    print("=== DETAILED CV PROCESS ===")
    print(f"Total training samples: {len(X_train_scaled)}")
    print(f"Unique training participants: {np.unique(groups_train)}")
    print(f"Parameter grid: {param_grid}")
    print(f"Total parameter combinations: {len(param_grid['C']) * len(param_grid['max_iter'])}")

    # First, let's manually inspect the CV splits
    print(f"\n=== CV SPLIT INSPECTION ===")
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_scaled, y_train, groups_train)):
        train_participants = np.unique(groups_train[train_idx])
        val_participants = np.unique(groups_train[val_idx])

        print(f"Fold {fold_idx + 1}:")
        print(f"  Train participants: {train_participants} ({len(train_participants)} participants)")
        print(f"  Val participants: {val_participants} ({len(val_participants)} participants)")
        print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        print(f"  Train label distribution: {np.bincount(y_train[train_idx].astype(int))}")
        print(f"  Val label distribution: {np.bincount(y_train[val_idx].astype(int))}")
        print()

    # Custom scorer that provides more details
    def detailed_auroc_scorer(estimator, X, y):
        y_pred_proba = estimator.predict_proba(X)[:, 1]
        auroc = roc_auc_score(y, y_pred_proba)
        return auroc

    # Create base model
    base_model = LogisticRegression(random_state=seed, n_jobs=1)  # n_jobs=1 for cleaner output

    # Manual grid search with detailed logging
    print("=== MANUAL GRID SEARCH WITH DETAILED LOGGING ===")

    best_score = -np.inf
    best_params = None
    all_results = []

    total_fits = len(param_grid['C']) * len(param_grid['max_iter']) * cv_splitter.get_n_splits()
    fit_count = 0

    for C in param_grid['C']:
        for max_iter in param_grid['max_iter']:
            print(f"\n--- Testing C={C}, max_iter={max_iter} ---")

            current_params = {'C': C, 'max_iter': max_iter}
            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_scaled, y_train, groups_train)):
                fit_count += 1
                print(f"  Fold {fold_idx + 1}/{cv_splitter.get_n_splits()} (Fit {fit_count}/{total_fits})")

                # Get fold data
                X_fold_train = X_train_scaled[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train_scaled[val_idx]
                y_fold_val = y_train[val_idx]

                # Create and train model for this fold
                model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)

                # Fit model
                print(f"    Fitting model on {len(X_fold_train)} samples...")
                model.fit(X_fold_train, y_fold_train)

                # Evaluate
                y_val_proba = model.predict_proba(X_fold_val)[:, 1]
                auroc = roc_auc_score(y_fold_val, y_val_proba)
                fold_scores.append(auroc)

                # Additional metrics for this fold
                y_val_pred = model.predict(X_fold_val)
                acc = accuracy_score(y_fold_val, y_val_pred)
                f1 = f1_score(y_fold_val, y_val_pred)

                print(f"    Fold {fold_idx + 1} results: AUROC={auroc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
                print(f"    Participants - Train: {np.unique(groups_train[train_idx])}")
                print(f"    Participants - Val: {np.unique(groups_train[val_idx])}")

            # Calculate mean CV score for this parameter combination
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            print(f"  Mean CV AUROC: {mean_score:.4f} (±{std_score:.4f})")
            print(f"  Individual fold scores: {[f'{score:.4f}' for score in fold_scores]}")

            # Track results
            result = {
                'params': current_params,
                'mean_test_score': mean_score,
                'std_test_score': std_score,
                'fold_scores': fold_scores
            }
            all_results.append(result)

            # Update best parameters
            if mean_score > best_score:
                best_score = mean_score
                best_params = current_params
                print(f"  *** NEW BEST PARAMETERS! ***")

    print(f"\n=== GRID SEARCH COMPLETE ===")
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {best_score:.4f}")

    # Train final model with best parameters
    print(f"\n=== TRAINING FINAL MODEL ===")
    print(f"Training on all {len(X_train_scaled)} training samples with best params: {best_params}")

    final_model = LogisticRegression(**best_params, random_state=seed)
    final_model.fit(X_train_scaled, y_train)

    print(f"Final model coefficients shape: {final_model.coef_.shape}")
    print(f"Final model intercept: {final_model.intercept_}")

    # Show feature importance (top 10 most important features)
    if len(feature_names) == len(final_model.coef_[0]):
        feature_importance = np.abs(final_model.coef_[0])
        top_indices = np.argsort(feature_importance)[-10:][::-1]

        print(f"\nTop 10 most important features:")
        for i, idx in enumerate(top_indices):
            print(f"  {i + 1}. {feature_names[idx]}: {final_model.coef_[0][idx]:.4f}")

    # Test evaluation
    print(f"\n=== FINAL TEST EVALUATION ===")
    print(f"Testing on {len(X_test_scaled)} test samples")

    y_test_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = final_model.predict(X_test_scaled)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_auroc = roc_auc_score(y_test, y_test_proba)
    test_f1 = f1_score(y_test, y_test_pred)
    test_pr_auc = average_precision_score(y_test, y_test_proba)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")

    # Summary table of all parameter combinations
    print(f"\n=== COMPLETE RESULTS SUMMARY ===")
    print("Params | Mean AUROC | Std AUROC | Fold Scores")
    print("-" * 60)
    for result in sorted(all_results, key=lambda x: x['mean_test_score'], reverse=True):
        params_str = f"C={result['params']['C']}"
        scores_str = ", ".join([f"{score:.3f}" for score in result['fold_scores']])
        print(
            f"{params_str:8} | {result['mean_test_score']:.4f}     | {result['std_test_score']:.4f}     | [{scores_str}]")

    return {
        'best_params': best_params,
        'best_cv_score': best_score,
        'all_results': all_results,
        'test_metrics': {
            'accuracy': test_acc,
            'auroc': test_auroc,
            'f1': test_f1,
            'pr_auc': test_pr_auc
        },
        'final_model': final_model
    }

def run_mlp_with_cv_and_test(X_train, y_train, groups_train, X_test, y_test,
                             feature_names, cv_splitter, device, classifier_epochs=25, seed=42):
    """Run CV for MLP hyperparameter selection, then train final model and test."""

    # Standardize features
    X_train_scaled, _, X_test_scaled = standardize_features(X_train, None, X_test, feature_names)

    # Simple hyperparameter options for MLP
    hidden_dims = [16, 32, 64]
    dropout_rates = [0.1, 0.2, 0.3, 0.5]

    best_params = None
    best_cv_score = 0

    print("Running manual CV for MLP hyperparameters...")

    # Manual grid search (since sklearn GridSearchCV doesn't work with PyTorch)
    for hidden_dim in hidden_dims:
        for dropout_rate in dropout_rates:
            print(f"Testing hidden_dim={hidden_dim}, dropout={dropout_rate}")

            fold_scores = []

            # Run CV for this parameter combination
            for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_scaled, y_train, groups_train)):
                X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                # Create and train model
                input_dim = X_fold_train.shape[-1]
                model = MLPClassifier(input_dim, hidden_dim=hidden_dim, dropout=dropout_rate).to(device)

                tr_loader = build_linear_loaders(X_fold_train, y_fold_train, 32, device)
                val_loader = build_linear_loaders(X_fold_val, y_fold_val, 32, device, shuffle=False)

                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                loss_fn = torch.nn.BCEWithLogitsLoss()

                # Training loop
                for epoch in range(classifier_epochs):
                    model.train()
                    for X_batch, y_batch in tr_loader:
                        optimizer.zero_grad()
                        logits = model(X_batch).squeeze(-1)
                        loss = loss_fn(logits, y_batch)
                        loss.backward()
                        optimizer.step()

                # Validation evaluation
                model.eval()
                val_probs = []
                val_labels = []

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        logits = model(X_batch).squeeze(-1)
                        probs = torch.sigmoid(logits)
                        val_probs.extend(probs.cpu().numpy())
                        val_labels.extend(y_batch.cpu().numpy())

                fold_auroc = roc_auc_score(val_labels, val_probs)
                fold_scores.append(fold_auroc)

            # Average CV score for this parameter combination
            mean_cv_score = np.mean(fold_scores)
            print(f"  Mean CV AUROC: {mean_cv_score:.4f}")

            if mean_cv_score > best_cv_score:
                best_cv_score = mean_cv_score
                best_params = {'hidden_dim': hidden_dim, 'dropout': dropout_rate}

    print(f"\nBest parameters: {best_params}")
    print(f"Best CV score: {best_cv_score:.4f}")

    # Train final model with best parameters on full training set
    print("Training final model on full training set...")
    input_dim = X_train_scaled.shape[-1]
    final_model = MLPClassifier(
        input_dim,
        hidden_dim=best_params['hidden_dim'],
        dropout=best_params['dropout']
    ).to(device)

    tr_loader = build_linear_loaders(X_train_scaled, y_train, 32, device)
    te_loader = build_linear_loaders(X_test_scaled, y_test, 32, device, shuffle=False)

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train final model
    for epoch in range(classifier_epochs):
        final_model.train()
        for X_batch, y_batch in tr_loader:
            optimizer.zero_grad()
            logits = final_model(X_batch).squeeze()
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

    # Test evaluation
    final_model.eval()
    test_probs = []
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for X_batch, y_batch in te_loader:
            logits = final_model(X_batch).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y_batch.cpu().numpy())

    test_probs = np.array(test_probs)
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    test_acc = accuracy_score(test_labels, test_preds)
    test_auroc = roc_auc_score(test_labels, test_probs)
    test_f1 = f1_score(test_labels, test_preds)
    test_pr_auc = average_precision_score(test_labels, test_probs)

    print(f"\n=== Test Set Results ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")

    return {
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'test_metrics': {
            'accuracy': test_acc,
            'auroc': test_auroc,
            'f1': test_f1,
            'pr_auc': test_pr_auc
        },
        'model': final_model
    }


def main(
        mlflow_tracking_uri: str,
        fs: str,
        gpu: int,
        seed: int,
        classifier_model: str,
        window_size: int,
        classifier_epochs: int,
        label_fraction: float,
        missing_value_strategy: str = "drop",
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
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
    if missing_value_strategy == "drop":
        print("=== Handling missing values ===")
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
    print(f"Training participants: {np.unique(groups_train_all)}")
    print(f"Test participants: {np.unique(groups[test_idx][test_binary_mask])}")

    # ── Step 2: Set up Cross-Validation Splitter ───────────────────────────────
    cv_splitter, n_splits = get_participant_cv_splitter(
        groups_train_all,
        min_participants_for_kfold=min_participants_for_kfold,
        k=k_folds
    )

    # ── Step 3: Run Model Selection + Final Training + Test Evaluation ─────────
    if classifier_model == "logistic_regression":

        # More verbose solution:
        # results = run_logistic_regression_with_gridsearch_verbose(
        #     X_train_all, y_train_all, groups_train_all, X_test, y_test,
        #     feature_names, cv_splitter, seed=42
        # )

        # This worked
        results = run_logistic_regression_with_gridsearch(
            X_train_all, y_train_all, groups_train_all,
            X_test, y_test, feature_names, cv_splitter, seed
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
        "missing_value_strategy": missing_value_strategy,
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
    parser.add_argument("--missing_value_strategy", type=str, default="drop",
                        choices=("drop", "knn", "iterative"))
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for CV")
    parser.add_argument("--min_participants_for_kfold", type=int, default=5,
                        help="Minimum participants needed for k-fold (otherwise use LOGO)")

    args = parser.parse_args()
    main(**vars(args))