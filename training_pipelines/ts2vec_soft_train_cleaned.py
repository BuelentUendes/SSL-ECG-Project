#!/usr/bin/env python
import os
import sys
import argparse
import logging
import tempfile
import gc

import numpy as np
import torch
import torch.optim as optim
import optuna
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant,
    set_seed,
    create_directory,
    train_classifier_for_optuna,
)

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH

from models.ts2vec_soft import (
    TS2Vec_soft,
    save_sim_mat,
    densify,
    train_linear_classifier,
    evaluate_classifier,
    build_fingerprint,
    search_encoder_fp,
    compute_soft_labels,
    build_linear_loaders,
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

    # Build data loaders
    tr_loader = build_linear_loaders(train_repr, y_train,
                                     classifier_batch_size, device)
    va_loader = build_linear_loaders(val_repr, y_val,
                                     classifier_batch_size, device,
                                     shuffle=False)

    # Optimizer
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train and get validation score
    val_score = train_classifier_for_optuna(
        classifier, tr_loader, va_loader, optimizer, loss_fn,
        num_epochs=25, device=device, early_stopping_patience=7
    )

    return val_score

def main(
        window_data_path: str,
        mlflow_tracking_uri: str,
        gpu: int,
        seed: int,
        force_retraining: bool,
        pretrain_all_conditions: bool,
        ts2vec_epochs: int,
        ts2vec_lr: float,
        ts2vec_batch_size: int,
        ts2vec_output_dims: int,
        ts2vec_hidden_dims: int,
        ts2vec_depth: int,
        ts2vec_max_train_length: int,
        ts2vec_temporal_unit: int,
        ts2vec_dist_type: str,
        ts2vec_tau_inst: float,
        ts2vec_tau_temp: float,
        ts2vec_alpha: float,
        ts2vec_lambda: float,
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
    mlflow.set_experiment("SoftTS2Vec")

    # Start top‑level run
    run = mlflow.start_run(run_name=f"soft_ts2vec_training_{seed}")
    run_id = run.info.run_id
    logging.info(f"MLflow run_id: {run_id}")
    print(f"Using device: {device}")

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)

    # We save the model here via seeds
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"
    model_save_path = os.path.join(SAVED_MODELS_PATH, "SoftTS2Vec", pretrain_data, f"{seed}")
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

    X, y, groups = load_processed_data(window_data_path, label_map=label_map)
    y = y.astype(np.float32)
    n_features = X.shape[2]

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

    # ── Step 2: Soft TS2Vec Pretraining ─────────────────────────────────────────
    torch.cuda.empty_cache()
    set_seed(seed)

    # Fingerprint & search
    fp = {
        "model_name": "TS2Vec_soft",
        "seed": seed,
        "ts2vec_epochs": ts2vec_epochs,
        "ts2vec_output_dims": ts2vec_output_dims,
        "ts2vec_hidden_dims": ts2vec_hidden_dims,
        "ts2vec_depth": ts2vec_depth,
        "ts2vec_dist_type": ts2vec_dist_type,
        "ts2vec_tau_inst": ts2vec_tau_inst,
        "ts2vec_tau_temp": ts2vec_tau_temp,
        "ts2vec_alpha": ts2vec_alpha,
        "ts2vec_lambda": ts2vec_lambda,
        "ts2vec_max_train_length": ts2vec_max_train_length,
        "ts2vec_temporal_unit": ts2vec_temporal_unit,
    }
    fp = build_fingerprint(fp)

    cached = search_encoder_fp(fp,
                               experiment_name="SoftTS2Vec",
                               tracking_uri=mlflow_tracking_uri)

    # IF we have forced retraining we will always retrain
    if (cached or os.path.exists(os.path.join(model_save_path, "ts2vec_soft_model.pth"))) and not (force_retraining):
        if cached:
            print(f"Found cached encoder run {cached}; downloading…")
            uri = f"runs:/{cached}/ts2vec_soft_model"
            net = mlflow.pytorch.load_model(uri, map_location=device)

            ts2vec_soft = TS2Vec_soft(
                input_dims=n_features,
                output_dims=ts2vec_output_dims,
                hidden_dims=ts2vec_hidden_dims,
                depth=ts2vec_depth,
                device=device,
                lr=ts2vec_lr,
                batch_size=ts2vec_batch_size,
                lambda_=ts2vec_lambda,
                tau_temp=ts2vec_tau_temp,
                max_train_length=ts2vec_max_train_length,
                temporal_unit=ts2vec_temporal_unit,
                soft_instance=True,
                soft_temporal=True,
            )
            ts2vec_soft.net = ts2vec_soft._net = net
        else:
            print("We found a pretrained model. Load the pretrained weights")
            model_path = os.path.join(model_save_path, "ts2vec_soft_model.pth")

            ts2vec_soft = TS2Vec_soft(
                input_dims=n_features,
                output_dims=ts2vec_output_dims,
                hidden_dims=ts2vec_hidden_dims,
                depth=ts2vec_depth,
                device=device,
                lr=ts2vec_lr,
                batch_size=ts2vec_batch_size,
                lambda_=ts2vec_lambda,
                tau_temp=ts2vec_tau_temp,
                max_train_length=ts2vec_max_train_length,
                temporal_unit=ts2vec_temporal_unit,
                soft_instance=True,
                soft_temporal=True,
            )
            ts2vec_soft.net = ts2vec_soft._net = torch.load(model_path, map_location=device)

    else:
        print("No cached encoder; training Soft TS2Vec from scratch")

        # Load only training data for pretraining
        # For the training from scratch we use the train_idx_all
        X_train = X[train_idx_all].astype(np.float32)

        # Compute soft labels
        print("Computing soft labels...")
        soft_labels = compute_soft_labels(
            X_train, ts2vec_tau_inst, ts2vec_alpha,
            ts2vec_dist_type, ts2vec_max_train_length
        )

        ts2vec_soft = TS2Vec_soft(
            input_dims=n_features,
            output_dims=ts2vec_output_dims,
            hidden_dims=ts2vec_hidden_dims,
            depth=ts2vec_depth,
            device=device,
            lr=ts2vec_lr,
            batch_size=ts2vec_batch_size,
            lambda_=ts2vec_lambda,
            tau_temp=ts2vec_tau_temp,
            max_train_length=ts2vec_max_train_length,
            temporal_unit=ts2vec_temporal_unit,
            soft_instance=True,
            soft_temporal=True,
        )

        print(f"Created Soft TS2Vec model on device: {next(ts2vec_soft.net.parameters()).device}")

        mlflow.log_params(fp)

        # Train Soft TS2Vec
        run_dir = tempfile.mkdtemp(prefix="ts2vec_soft_")
        ts2vec_soft.fit(
            X_train, soft_labels,
            run_dir=run_dir,
            n_epochs=ts2vec_epochs,
            verbose=True
        )

        # Save model
        mlflow.pytorch.log_model(
            pytorch_model=ts2vec_soft.net,
            artifact_path="ts2vec_soft_model"
        )

        saved_results = os.path.join(model_save_path, "ts2vec_soft_model.pth")
        torch.save(ts2vec_soft.net, saved_results)

    # ── Step 3: Extract Representations ─────────────────────────────────────────
    print("\nExtracting representations...")

    # Get Soft TS2Vec embeddings
    train_repr = ts2vec_soft.encode(X[train_idx].astype(np.float32), encoding_window="full_series")
    val_repr = ts2vec_soft.encode(X[val_idx].astype(np.float32), encoding_window="full_series")
    test_repr = ts2vec_soft.encode(X[test_idx].astype(np.float32), encoding_window="full_series")

    # filter to binary downstream samples
    train_repr = train_repr[downstream_mask["train"]]
    y_train     = y[train_idx][downstream_mask["train"]]
    val_repr   = val_repr[downstream_mask["val"]]
    y_val       = y[val_idx][downstream_mask["val"]]
    test_repr  = test_repr[downstream_mask["test"]]
    y_test      = y[test_idx][downstream_mask["test"]]

    print(f"Extracted Soft TS2Vec representations: train_repr shape={train_repr.shape}")

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
    tr_loader = build_linear_loaders(train_repr, y_train,
                                     classifier_batch_size, device)
    va_loader = build_linear_loaders(val_repr, y_val,
                                     classifier_batch_size, device,
                                     shuffle=False)

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
        classifier, tr_loader, va_loader,
        opt_clf, loss_fn,
        classifier_epochs, device
    )

    # ── Step 5: Evaluation ──────────────────────────────────────────────────────
    te_loader = build_linear_loaders(test_repr, y_test,
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
    parser = argparse.ArgumentParser(description="Soft TS2Vec Training Pipeline")
    parser.add_argument("--window_data_path",
                        default=f"{os.path.join(DATA_PATH, 'interim', 'windowed_data.h5')}")
    parser.add_argument("--mlflow_tracking_uri",
                        default=os.getenv("MLFLOW_tracking_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_retraining", action="store_true")
    parser.add_argument("--pretrain_all_conditions", action="store_true")

    # Soft TS2Vec pretraining parameters
    parser.add_argument("--ts2vec_epochs", type=int, default=50,
                        help="Epochs for Soft TS2Vec pretraining")
    parser.add_argument("--ts2vec_lr", type=float, default=0.001,
                        help="Learning rate for Soft TS2Vec")
    parser.add_argument("--ts2vec_batch_size", type=int, default=8,
                        help="Batch size for Soft TS2Vec")
    parser.add_argument("--ts2vec_output_dims", type=int, default=320,
                        help="Representation dimension (Co)")
    parser.add_argument("--ts2vec_hidden_dims", type=int, default=64,
                        help="Hidden dimension (Ch)")
    parser.add_argument("--ts2vec_depth", type=int, default=10,
                        help="Depth (# dilated conv blocks)")
    parser.add_argument("--ts2vec_max_train_length", type=int, default=5000,
                        help="Max training length")
    parser.add_argument("--ts2vec_temporal_unit", type=int, default=0,
                        help="Temporal unit for hierarchical pooling")

    # Soft contrastive learning hyperparameters
    parser.add_argument("--ts2vec_dist_type", type=str, default="EUC",
                        choices=["DTW", "EUC", "COS", "TAM", "GAK"],
                        help="Distance metric for soft labels")
    parser.add_argument("--ts2vec_tau_inst", type=float, default=50.0,
                        help="Temperature parameter tau_inst for soft instance CL")
    parser.add_argument("--ts2vec_tau_temp", type=float, default=2.5,
                        help="Temperature parameter tau_temp for soft temporal CL")
    parser.add_argument("--ts2vec_alpha", type=float, default=0.5,
                        help="Alpha for densification of soft labels")
    parser.add_argument("--ts2vec_lambda", type=float, default=0.5,
                        help="Weight lambda for instance vs temporal CL")

    # classifier fine-tuning
    parser.add_argument("--classifier_epochs", type=int, default=25)
    parser.add_argument("--classifier_model", type=str, default="linear", choices=("linear", "mlp"))
    parser.add_argument("--classifier_lr", type=float, default=0.0001)
    parser.add_argument("--classifier_batch_size", type=int, default=32)
    parser.add_argument("--label_fraction", type=float, default=1.0)

    # Optuna hyperparameter tuning
    parser.add_argument("--do_hyperparameter_tuning", action="store_true",
                        help="Enable hyperparameter tuning with Optuna")
    parser.add_argument("--n_trials", type=int, default=25,
                        help="Number of Optuna trials for hyperparameter tuning")

    args = parser.parse_args()
    main(**vars(args))