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

from models.tstcc import (
    data_generator_from_arrays,
    train_linear_classifier,
    evaluate_classifier,
    Trainer,
    base_Model,
    TC,
    Config as ECGConfig,
    encode_representations,
    show_shape,
    build_tstcc_fingerprint,
    search_encoder_fp,
    build_linear_loaders,
)
from models.supervised import LinearClassifier, MLPClassifier


def optuna_objective(trial, train_repr, y_train, val_repr, y_val,
                     classifier_batch_size, device, classifier_model, seed):
    """Optuna objective function for hyperparameter tuning"""
    set_seed(seed)

    # Suggest hyperparameters
    if classifier_model == "mlp":
        hidden_dim = trial.suggest_int('hidden_dim', 16, 64, step=16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

        classifier = MLPClassifier(
            train_repr.shape[-1],
            hidden_dim=hidden_dim,
            dropout=dropout_rate,
        ).to(device)

    else:  # linear
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
        force_retraining: bool,
        tcc_epochs: int,
        tcc_lr: float,
        tcc_batch_size: int,
        pretrain_all_conditions: bool,
        tc_timesteps: int,
        tc_hidden_dim: int,
        cc_temperature: float,
        cc_use_cosine: bool,
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
        # Important note:
        # For TSTCC the MPS is not supported due to some binary operation that does not work on MPS.
    else:
        device = torch.device("cpu")

    logging.basicConfig(level=logging.INFO)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("TSTCC cleaned")

    # Start top‑level run
    run = mlflow.start_run(run_name=f"tstcc_training_{seed}")
    run_id = run.info.run_id
    logging.info(f"MLflow run_id: {run_id}")
    print(f"Using device: {device}")

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"

    model_save_path = os.path.join(SAVED_MODELS_PATH, "ECG", str(fs), "TSTCC", pretrain_data, f"{seed}")
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

    # Data Path
    window_data_path = os.path.join(DATA_PATH, "interim", "ECG", str(fs), 'windowed_data.h5')

    X, y, groups = load_processed_data(window_data_path, label_map=label_map)
    y = y.astype(np.float32)

    # We first get all train idx for the SSL method (label fraction 1.0) as we do not use the labels
    # train_idx_all (represents all training samples as we do not use their labels)
    train_idx, train_idx_all, val_idx, test_idx = split_indices_by_participant(
        groups, label_fraction=label_fraction, self_supervised_method=True, seed=seed
    )
    print(f"windows: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    # Keep binary‐task mask for later
    downstream_mask = {
        "train": np.isin(y[train_idx], [0, 1]),
        "val": np.isin(y[val_idx], [0, 1]),
        "test": np.isin(y[test_idx], [0, 1]),
    }

    # ── Step 2: TS‑TCC Pretraining ───────────────────────────────────────────────
    torch.cuda.empty_cache()
    set_seed(seed)

    # Fingerprint & search
    fp = build_tstcc_fingerprint({
        "model_name": "TSTCC",
        "seed": seed,
        "pretrain_all_conditions": pretrain_all_conditions,
        "tcc_epochs": tcc_epochs,
        "tcc_lr": tcc_lr,
        "tcc_batch_size": tcc_batch_size,
        "tc_timesteps": tc_timesteps,
        "tc_hidden_dim": tc_hidden_dim,
        "cc_temperature": cc_temperature,
        "cc_use_cosine": cc_use_cosine,
    })

    cached = search_encoder_fp(
        fp, experiment_name="TSTCC", tracking_uri=mlflow_tracking_uri
    )

    # IF we have forced retraining we will always retraining
    if (cached or os.path.exists(os.path.join(model_save_path, "tstcc.pt"))) and not (force_retraining):
        if cached:
            print(f"Found cached encoder run {cached}; downloading…")
            uri = f"runs:/{cached}/tstcc_model"
            ckpt_dir = mlflow.artifacts.download_artifacts(uri)
            ckpt_path = os.path.join(ckpt_dir, "tstcc.pt")
        else:
            print("We found a pretrained model. Load the pretrained weights")
            ckpt_path = os.path.join(model_save_path, "tstcc.pt")

        # rebuild model
        cfg = ECGConfig()
        cfg.num_epoch = tcc_epochs
        cfg.batch_size = tcc_batch_size
        cfg.TC.timesteps = tc_timesteps
        cfg.TC.hidden_dim = tc_hidden_dim
        cfg.Context_Cont.temperature = cc_temperature
        cfg.Context_Cont.use_cosine_similarity = cc_use_cosine

        model = base_Model(cfg).to(device)
        tc_head = TC(cfg, device).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["encoder"])
        tc_head.load_state_dict(state["tc_head"])

    else:
        print("No cached encoder; training TS-TCC from scratch")
        cfg = ECGConfig()
        cfg.num_epoch = tcc_epochs
        cfg.batch_size = tcc_batch_size
        cfg.TC.timesteps = tc_timesteps
        cfg.TC.hidden_dim = tc_hidden_dim
        cfg.Context_Cont.temperature = cc_temperature
        cfg.Context_Cont.use_cosine_similarity = cc_use_cosine

        # data loaders
        Xtr = X[train_idx_all].astype(np.float32)
        Xva = X[val_idx].astype(np.float32)
        Xte = X[test_idx].astype(np.float32)
        tr_dl, va_dl, te_dl = data_generator_from_arrays(
            Xtr, y[train_idx_all], Xva, y[val_idx], Xte, y[test_idx],
            cfg, training_mode="self_supervised"
        )

        # models & optimizers
        model = base_Model(cfg).to(device)
        tc_head = TC(cfg, device).to(device)
        opt_m = optim.AdamW(model.parameters(), lr=tcc_lr, weight_decay=3e-4)
        opt_tc = optim.AdamW(tc_head.parameters(), lr=tcc_lr, weight_decay=3e-4)

        # Deleted second start of the run
        mlflow.log_params(fp)
        workdir = tempfile.mkdtemp(prefix="tstcc_")
        Trainer(
            model=model,
            temporal_contr_model=tc_head,
            model_optimizer=opt_m,
            temp_cont_optimizer=opt_tc,
            train_dl=tr_dl, valid_dl=va_dl, test_dl=te_dl,
            device=device, config=cfg,
            experiment_log_dir=workdir,
            training_mode="self_supervised",
        )
        ckpt = os.path.join(workdir, "tstcc.pt")
        torch.save(
            {"encoder": model.state_dict(),
             "tc_head": tc_head.state_dict()},
            ckpt
        )

        mlflow.log_artifact(ckpt, artifact_path="tstcc_model")

        saved_results = os.path.join(model_save_path, "tstcc.pt")
        torch.save(
            {"encoder": model.state_dict(),
             "tc_head": tc_head.state_dict()},
            saved_results
        )

    # ── Step 3: Extract Representations ─────────────────────────────────────────
    model.eval();
    tc_head.eval()

    with torch.no_grad():
        train_repr, _ = encode_representations(X[train_idx], y[train_idx],
                                               model, tc_head, tcc_batch_size, device)
        val_repr, _ = encode_representations(X[val_idx], y[val_idx],
                                             model, tc_head, tcc_batch_size, device)
        test_repr, _ = encode_representations(X[test_idx], y[test_idx],
                                              model, tc_head, tcc_batch_size, device)

    # filter to binary downstream samples
    train_repr = train_repr[downstream_mask["train"]]
    y_train = y[train_idx][downstream_mask["train"]]
    val_repr = val_repr[downstream_mask["val"]]
    y_val = y[val_idx][downstream_mask["val"]]
    test_repr = test_repr[downstream_mask["test"]]
    y_test = y[test_idx][downstream_mask["test"]]

    print(f"train_repr shape = {train_repr.shape}")
    show_shape("val/test repr", (val_repr, test_repr))

    # ── Step 4: Classifier Fine‑Tuning with Hyperparameter Tuning ───────────────
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
            lr=classifier_lr,
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
        "classifier_lr": classifier_lr,
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
    parser = argparse.ArgumentParser(description="TS-TCC Training Pipeline (cleaned)")
    parser.add_argument("--mlflow_tracking_uri",
                        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--fs", default=1000, type=str, help="What sample frequency used for training")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_retraining", action="store_true")
    parser.add_argument("--tcc_epochs", type=int, default=40)
    parser.add_argument("--tcc_lr", type=float, default=3e-4)
    parser.add_argument("--tcc_batch_size", type=int, default=128)
    parser.add_argument("--pretrain_all_conditions", action="store_true")
    parser.add_argument("--tc_timesteps", type=int, default=70)
    parser.add_argument("--tc_hidden_dim", type=int, default=128)
    parser.add_argument("--cc_temperature", type=float, default=0.07)
    parser.add_argument("--cc_use_cosine", action="store_true")
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