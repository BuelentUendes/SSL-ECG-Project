#!/usr/bin/env python
import os
import sys
import argparse
import logging
import tempfile
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import optuna
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant,
    set_seed,
    create_directory,
    train_classifier_for_optuna,
)

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH

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
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    # Identify nn20 and nn50 indices
    nn_indices = []
    standard_indices = []

    for i, name in enumerate(feature_names):
        if name.lower() == 'nn20' or  name.lower() == 'nn50':
            nn_indices.append(i)
        else:
            standard_indices.append(i)

    # Apply StandardScaler to most features (fit only on train)
    if standard_indices:
        standard_scaler.fit(X_train[:, standard_indices])
        X_train_scaled[:, standard_indices] = standard_scaler.transform(X_train[:, standard_indices])
        X_val_scaled[:, standard_indices] = standard_scaler.transform(X_val[:, standard_indices])
        X_test_scaled[:, standard_indices] = standard_scaler.transform(X_test[:, standard_indices])

    # Apply MinMaxScaler to nn20/nn50 features (fit only on train)
    if nn_indices:
        minmax_scaler.fit(X_train[:, nn_indices])
        X_train_scaled[:, nn_indices] = minmax_scaler.transform(X_train[:, nn_indices])
        X_val_scaled[:, nn_indices] = minmax_scaler.transform(X_val[:, nn_indices])
        X_test_scaled[:, nn_indices] = minmax_scaler.transform(X_test[:, nn_indices])

    return X_train_scaled, X_val_scaled, X_test_scaled



def optuna_objective(trial, X_train, y_train, X_val, y_val,
                     classifier_batch_size, device, classifier_model, seed):
    """Optuna objective function for hyperparameter tuning"""
    set_seed(seed)

    # Data is already in the right format (n_samples, num_features)
    input_dim = X_train.shape[-1]

    # Suggest hyperparameters
    if classifier_model == "mlp":
        hidden_dim = trial.suggest_int('hidden_dim', 16, 128, step=16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

        classifier = MLPClassifier(
            input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout_rate,
        ).to(device)

    else:  # linear
        classifier = LinearClassifier(input_dim).to(device)

    # Build data loaders
    tr_loader = build_linear_loaders(X_train, y_train,
                                     classifier_batch_size, device)
    va_loader = build_linear_loaders(X_val, y_val,
                                     classifier_batch_size, device,
                                     shuffle=False)

    # Optimizer
    optimizer = optim.AdamW(classifier.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train and get validation score
    val_score = train_classifier_for_optuna(
        classifier, tr_loader, va_loader, optimizer, loss_fn,
        num_epochs=25, device=device, early_stopping_patience=7
    )

    return val_score


def main(
        mlflow_tracking_uri: str,
        fs: str,
        gpu: int,
        seed: int,
        classifier_model: str,
        classifier_epochs: int,
        classifier_lr: float,
        classifier_batch_size: int,
        label_fraction: float,
        do_hyperparameter_tuning: bool = False,
        n_trials: int = 50,
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
    mlflow.set_experiment("Baseline Classifiers")

    # Start top‑level run
    run = mlflow.start_run(run_name=f"baseline_{classifier_model}_{seed}")
    run_id = run.info.run_id
    logging.info(f"MLflow run_id: {run_id}")
    print(f"Using device: {device}")

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)

    model_save_path = os.path.join(SAVED_MODELS_PATH, "ECG_features", str(fs), "Feature_based_classifiers", classifier_model, f"{seed}")
    create_directory(model_save_path)

    # ── Step 1: Preprocess ───────────────────────────────────────────────────────
    # Binary classification: baseline vs mental stress
    label_map = {"baseline": 0, "mental_stress": 1}

    # Data Path
    window_data_path = os.path.join(DATA_PATH, "interim", "ECG_features",'windowed_data.h5')

    X, y, groups, feature_names = load_processed_data(window_data_path, label_map=label_map, domain_features=True)
    y = y.astype(np.float32)

    # Split data by participant
    train_idx, val_idx, test_idx = split_indices_by_participant(
        groups, label_fraction=label_fraction, self_supervised_method=False, seed=seed
    )
    print(f"windows: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    # Filter to binary classification samples only
    downstream_mask = {
        "train": np.isin(y[train_idx], [0, 1]),
        "val": np.isin(y[val_idx], [0, 1]),
        "test": np.isin(y[test_idx], [0, 1]),
    }

    # Extract data splits
    X_train = X[train_idx][downstream_mask["train"]]
    y_train = y[train_idx][downstream_mask["train"]]
    X_val = X[val_idx][downstream_mask["val"]]
    y_val = y[val_idx][downstream_mask["val"]]
    X_test = X[test_idx][downstream_mask["test"]]
    y_test = y[test_idx][downstream_mask["test"]]

    X_train, X_val, X_test = standardize_features(X_train, X_val, X_test, feature_names)

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape}")

    # ── Step 2: Prepare Features ────────────────────────────────────────────────
    # Data is already in the right format (n_samples, num_features)
    input_dim = X_train.shape[-1]
    print(f"Input dimension: {input_dim}")

    # ── Step 3: Classifier Training with Hyperparameter Tuning ─────────────────
    set_seed(seed)

    # Initialize variables for best parameters
    best_params = None
    best_val_score = 0

    if do_hyperparameter_tuning:
        print(f"Starting hyperparameter tuning with Optuna ({n_trials} trials)...")

        # Create Optuna study
        study_name = f"baseline_tuning_{classifier_model}_{seed}"
        study = optuna.create_study(
            direction='maximize',  # Maximize AUROC score
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Run optimization
        study.optimize(
            lambda trial: optuna_objective(
                trial, X_train, y_train, X_val, y_val,
                classifier_batch_size, device, classifier_model, seed
            ),
            n_trials=n_trials,
            show_progress_bar=True
        )

        # Get best parameters
        best_params = study.best_params
        best_val_score = study.best_value

        print(f"Best validation AUROC score: {best_val_score:.4f}")
        print("Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        # Log best parameters to MLflow
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_auroc_optuna", best_val_score)

        # Create final classifier with best parameters
        if classifier_model == "mlp":
            classifier = MLPClassifier(
                input_dim,
                hidden_dim=best_params['hidden_dim'],
                dropout=best_params['dropout_rate'],
            ).to(device)
        else:
            classifier = LinearClassifier(input_dim).to(device)

    else:
        # Use default parameters without hyperparameter tuning
        if classifier_model == "linear":
            classifier = LinearClassifier(input_dim).to(device)
        else:
            classifier = MLPClassifier(input_dim).to(device)

    # Optimizer
    opt_clf = optim.AdamW(classifier.parameters(), lr=classifier_lr)

    # Build data loaders for final training
    tr_loader = build_linear_loaders(X_train, y_train,
                                     classifier_batch_size, device)
    va_loader = build_linear_loaders(X_val, y_val,
                                     classifier_batch_size, device,
                                     shuffle=False)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Log parameters
    clf_params = {
        "model_type": "baseline",
        "classifier_model": "LinearClassifier" if classifier_model == "linear" else "MLP",
        "classifier_epochs": classifier_epochs,
        "classifier_lr": classifier_lr,
        "classifier_batch_size": classifier_batch_size,
        "label_fraction": label_fraction,
        "seed": seed,
        "input_dim": input_dim,
        "do_hyperparameter_tuning": do_hyperparameter_tuning,
        "n_trials": n_trials if do_hyperparameter_tuning else 0,
    }
    mlflow.log_params(clf_params)

    # Train classifier and get best threshold
    print("Training baseline classifier...")
    classifier, best_thr = train_linear_classifier(
        classifier, tr_loader, va_loader,
        opt_clf, loss_fn,
        classifier_epochs, device
    )

    # Save the trained model
    model_path = os.path.join(model_save_path, f"baseline_{classifier_model}.pt")
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'best_threshold': best_thr,
        'input_dim': input_dim,
        'model_type': classifier_model,
        'best_params': best_params,
    }, model_path)

    mlflow.log_artifact(model_path, artifact_path="baseline_model")

    # ── Step 4: Evaluation ──────────────────────────────────────────────────────
    te_loader = build_linear_loaders(X_test, y_test,
                                     classifier_batch_size, device,
                                     shuffle=False)

    acc, auroc, pr_auc, f1 = evaluate_classifier(
        model=classifier,
        test_loader=te_loader,
        device=device,
        threshold=best_thr,
        loss_fn=loss_fn,
    )

    mlflow.log_metrics({
        "test_accuracy": acc,
        "test_auroc": auroc,
        "test_pr_auc": pr_auc,
        "test_f1": f1,
    })

    # ── Cleanup ────────────────────────────────────────────────────────────────
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"=== Done! Test Acc: {acc:.4f}, AUROC: {auroc:.4f}, PR-AUC: {pr_auc:.4f}, F1: {f1:.4f} ===")

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Classifier Training Pipeline")
    parser.add_argument("--mlflow_tracking_uri",
                        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--fs", default=1000, type=str, help="What sample frequency used for training")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier_model", type=str, default="linear", choices=("linear", "mlp"))
    parser.add_argument("--classifier_epochs", type=int, default=25)
    parser.add_argument("--classifier_lr", type=float, default=1e-4)
    parser.add_argument("--classifier_batch_size", type=int, default=32)
    parser.add_argument("--label_fraction", type=float, default=0.1)

    parser.add_argument("--do_hyperparameter_tuning", action="store_true",
                        help="Enable hyperparameter tuning with Optuna")
    parser.add_argument("--n_trials", type=int, default=25,
                        help="Number of Optuna trials for hyperparameter tuning")

    args = parser.parse_args()
    main(**vars(args))