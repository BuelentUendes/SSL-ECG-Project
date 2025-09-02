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
import optuna
import mlflow
import mlflow.pytorch

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

from models.simclr import (
    get_simclr_model,
    NTXentLoss,
    simclr_data_loaders,
    pretrain_one_epoch,
    encode_representations,
    train_linear_classifier,
    evaluate_classifier,
    show_shape,
    build_simclr_fingerprint,
    search_encoder_fp,
)

from models.supervised import (
    LinearClassifier,
    MLPClassifier,
)


def optuna_objective(trial, train_repr, y_train, val_repr, y_val,
                     classifier_batch_size, device, classifier_model, seed):
    """Optuna objective function for hyperparameter tuning"""
    set_seed(seed)

    # Suggest hyperparameters
    if classifier_model == "mlp":
        hidden_dim = trial.suggest_int('hidden_dim', 16, 64, step=16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)

        classifier = MLPClassifier(
            train_repr.shape[-1],
            hidden_dim=hidden_dim,
            dropout=dropout_rate,
        ).to(device)

    else:  # linear
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)

        classifier = LinearClassifier(
            train_repr.shape[-1],
        ).to(device)

    # Create data loaders
    tr_dl = DataLoader(
        TensorDataset(torch.from_numpy(train_repr).float().to(device),
                      torch.from_numpy(y_train).float().to(device)),
        batch_size=classifier_batch_size, shuffle=True)
    va_dl = DataLoader(
        TensorDataset(torch.from_numpy(val_repr).float().to(device),
                      torch.from_numpy(y_val).float().to(device)),
        batch_size=classifier_batch_size, shuffle=False)

    # Optimizer
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train and get validation score
    val_score = train_classifier_for_optuna(
        classifier, tr_dl, va_dl, optimizer, loss_fn,
        num_epochs=25, device=device, early_stopping_patience=7
    )

    return val_score


def main(
        mlflow_tracking_uri: str,
        fs: str,
        gpu: int,
        seed: int,
        force_retraining: bool,
        pretrain_all_conditions: bool,
        epochs: int,
        lr: float,
        batch_size: int,
        temperature: float,
        classifier_epochs: int,
        classifier_model: str,
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
        torch.cuda.set_device(f"cuda:{gpu}")
        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logging.basicConfig(level=logging.INFO)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("SimCLR")

    # Start top‑level run
    run = mlflow.start_run(run_name=f"simclr_training_{seed}")
    run_id = run.info.run_id
    logging.info(f"MLflow run_id: {run_id}")
    print(f"Using device: {device}")

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"

    model_save_path = os.path.join(SAVED_MODELS_PATH, "ECG", str(fs), "SimCLR", pretrain_data, f"{seed}")
    create_directory(model_save_path)

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
    window_data_path = os.path.join(DATA_PATH, "interim", "ECG", str(fs), 'windowed_data.h5')

    X, y, groups = load_processed_data(window_data_path, label_map=label_map)
    y = y.astype(np.float32)
    win_len = X.shape[1]

    # We first get all train idx for the SSL method (label fraction 1.0) as we do not use the labels
    # train_idx_all (represents all training samples as we do not use their labels)
    train_idx, train_idx_all, val_idx, test_idx = split_indices_by_participant(
        groups, label_fraction=label_fraction, self_supervised_method=True, seed=seed
    )
    print(f"windows: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    # Keep binary‐task mask for later
    downstream_mask = {
        "train": np.isin(y[train_idx], [0, 1]),
        "val":   np.isin(y[val_idx],   [0, 1]),
        "test":  np.isin(y[test_idx],  [0, 1]),
    }

    # ── Step 2: SimCLR Pretraining ──────────────────────────────────────────────
    torch.cuda.empty_cache()
    set_seed(seed)

    # Build fingerprint and MLflow lookup
    fp = {
        "model_name": "SimCLR",
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "temperature": temperature,
        "window_len": win_len,
    }
    fp = build_simclr_fingerprint(fp)

    cached = search_encoder_fp(fp,
                               experiment_name="SimCLR",
                               tracking_uri=mlflow_tracking_uri)

    model = get_simclr_model(window=win_len, device=device)

    # IF we have forced retraining we will always retrain
    if (cached or os.path.exists(os.path.join(model_save_path, "simclr_encoder.pt"))) and not (force_retraining):
        if cached:
            print(f"Found cached encoder run {cached}; downloading…")
            uri = f"runs:/{cached}/ssl_model"
            ckpt_dir = mlflow.artifacts.download_artifacts(uri)
            ckpt_path = os.path.join(ckpt_dir, "simclr_encoder.pt")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print("We found a pretrained model. Load the pretrained weights")
            model_path = os.path.join(model_save_path, "simclr_encoder.pt")
            model.load_state_dict(torch.load(model_path, map_location=device))

    else:
        print("No cached encoder; training SimCLR from scratch")

        # Load only training data for pretraining
        # For the training from scratch we use the train_idx_all
        X_train = X[train_idx_all].astype(np.float32)
        X_val = X[val_idx].astype(np.float32)

        loss_fn = NTXentLoss(batch_size, temperature)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        tr_dl, _ = simclr_data_loaders(X_train, X_val, batch_size)

        print(f"Created SimCLR model on device: {next(model.parameters()).device}")

        mlflow.log_params(fp)

        # Train SimCLR
        for ep in range(1, epochs + 1):
            tr_loss = pretrain_one_epoch(model, tr_dl, loss_fn, opt, device)
            mlflow.log_metric("ssl_train_loss", tr_loss, step=ep)
            if ep == 1 or ep % 25 == 0:
                print(f"Epoch {ep}/{epochs}: loss={tr_loss:.4f}")

        # Save model locally
        saved_results = os.path.join(model_save_path, "simclr_encoder.pt")
        torch.save(model.state_dict(), saved_results)

        # Save encoder weights to MLflow
        ckpt = os.path.join(tempfile.mkdtemp(), "simclr_encoder.pt")
        torch.save(model.state_dict(), ckpt)
        mlflow.log_artifact(ckpt, artifact_path="ssl_model")

    # ── Step 3: Extract Representations ─────────────────────────────────────────
    print("\nExtracting representations...")

    # Get SimCLR embeddings
    with torch.no_grad():
        train_repr = encode_representations(
            model, X[train_idx].astype(np.float32), batch_size, device)
        val_repr = encode_representations(
            model, X[val_idx].astype(np.float32), batch_size, device)
        test_repr = encode_representations(
            model, X[test_idx].astype(np.float32), batch_size, device)

    # filter to binary downstream samples
    train_repr = train_repr[downstream_mask["train"]]
    y_train     = y[train_idx][downstream_mask["train"]]
    val_repr   = val_repr[downstream_mask["val"]]
    y_val       = y[val_idx][downstream_mask["val"]]
    test_repr  = test_repr[downstream_mask["test"]]
    y_test      = y[test_idx][downstream_mask["test"]]

    show_shape("repr", (train_repr, val_repr, test_repr))
    print(f"Extracted SimCLR representations: train_repr shape={train_repr.shape}")

    # ── Step 4: Classifier Fine‑Tuning ──────────────────────────────────────────
    set_seed(seed)

    # Initialize variables for best parameters
    best_params = None
    best_val_score = 0

    if do_hyperparameter_tuning:
        print(f"Starting hyperparameter tuning with Optuna ({n_trials} trials)...")

        # Create Optuna study
        study_name = f"classifier_tuning_{classifier_model}_{seed}"
        study = optuna.create_study(
            direction='maximize',  # Maximize AUROC score
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Run optimization
        study.optimize(
            lambda trial: optuna_objective(
                trial, train_repr, y_train, val_repr, y_val,
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
                train_repr.shape[-1],
                hidden_dim=best_params['hidden_dim'],
                dropout=best_params['dropout_rate'],
            ).to(device)
        else:
            classifier = LinearClassifier(
                train_repr.shape[-1],
            ).to(device)

        opt_clf = optim.AdamW(
            classifier.parameters(),
            lr=best_params['lr'],
        )

    else:
        # Use original approach without hyperparameter tuning
        if classifier_model == "linear":
            classifier = LinearClassifier(train_repr.shape[-1]).to(device)
        else:
            classifier = MLPClassifier(train_repr.shape[-1]).to(device)

        opt_clf = optim.AdamW(classifier.parameters(), lr=classifier_lr)

    # Build data loaders for final training
    # Create data loaders
    tr_dl = DataLoader(
        TensorDataset(torch.from_numpy(train_repr).float(),
                      torch.from_numpy(y_train).float()),
        batch_size=classifier_batch_size, shuffle=True)
    va_dl = DataLoader(
        TensorDataset(torch.from_numpy(val_repr).float(),
                      torch.from_numpy(y_val).float()),
        batch_size=classifier_batch_size, shuffle=False)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Log classifier parameters
    clf_params = {
        "classifier_model": "LinearClassifier" if classifier_model == "linear" else "MLP",
        "classifier_epochs": classifier_epochs,
        "classifier_lr": classifier_lr if not do_hyperparameter_tuning else best_params['lr'],
        "classifier_batch_size": classifier_batch_size,
        "label_fraction": label_fraction,
        "seed": seed,
        "do_hyperparameter_tuning": do_hyperparameter_tuning,
        "n_trials": n_trials if do_hyperparameter_tuning else 0,
    }
    mlflow.log_params(clf_params)

    # Train classifier and get best threshold
    classifier, best_thr = train_linear_classifier(
        classifier, tr_dl, va_dl,
        opt_clf, loss_fn,
        classifier_epochs, device
    )

    # ── Step 5: Evaluation ──────────────────────────────────────────────────────
    te_dl = DataLoader(
        TensorDataset(torch.from_numpy(test_repr).float(),
                      torch.from_numpy(y_test).float()),
        batch_size=classifier_batch_size, shuffle=False)

    acc, auroc, pr_auc, f1 = evaluate_classifier(
        model=classifier,
        test_loader=te_dl,
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
    parser = argparse.ArgumentParser(description="SimCLR Training Pipeline")
    parser.add_argument("--mlflow_tracking_uri",
                        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--fs", default=1000, type=str, help="What sample frequency used for training")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_retraining", action="store_true")
    parser.add_argument("--pretrain_all_conditions", action="store_true")

    # SimCLR pretraining parameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Epochs for SimCLR pretraining")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for SimCLR")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for SimCLR")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature parameter for contrastive loss")

    # classifier fine-tuning
    parser.add_argument("--classifier_epochs", type=int, default=25)
    parser.add_argument("--classifier_model", type=str, default="linear", choices=("linear", "mlp"))
    parser.add_argument("--classifier_lr", type=float, default=1e-4)
    parser.add_argument("--classifier_batch_size", type=int, default=32)
    parser.add_argument("--label_fraction", type=float, default=1.0)

    # Optuna hyperparameter tuning
    parser.add_argument("--do_hyperparameter_tuning", action="store_true",
                        help="Enable hyperparameter tuning with Optuna")
    parser.add_argument("--n_trials", type=int, default=25,
                        help="Number of Optuna trials for hyperparameter tuning")

    args = parser.parse_args()
    main(**vars(args))