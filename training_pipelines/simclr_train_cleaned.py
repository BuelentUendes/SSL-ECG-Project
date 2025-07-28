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

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant,
    set_seed,
    create_directory,
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


def main(
        window_data_path: str,
        mlflow_tracking_uri: str,
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

    # We save the model here via seeds
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"
    model_save_path = os.path.join(SAVED_MODELS_PATH, "SimCLR", pretrain_data, f"{seed}")
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

    if classifier_model == "linear":
        classifier = LinearClassifier(train_repr.shape[-1]).to(device)
    else:
        classifier = MLPClassifier(train_repr.shape[-1]).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    opt_clf = optim.AdamW(classifier.parameters(), lr=classifier_lr)

    # Create data loaders
    tr_dl = DataLoader(
        TensorDataset(torch.from_numpy(train_repr).float(),
                      torch.from_numpy(y_train).float()),
        batch_size=classifier_batch_size, shuffle=True)
    va_dl = DataLoader(
        TensorDataset(torch.from_numpy(val_repr).float(),
                      torch.from_numpy(y_val).float()),
        batch_size=classifier_batch_size, shuffle=False)

    clf_params = {
        "classifier_model": "LinearClassifier" if classifier_model == "linear" else "MLP",
        "classifier_epochs": classifier_epochs,
        "classifier_lr": classifier_lr,
        "classifier_batch_size": classifier_batch_size,
        "label_fraction": label_fraction,
        "seed": seed,
    }
    mlflow.log_params(clf_params)

    # train & get best threshold
    classifier, best_thr = train_linear_classifier(
        classifier, tr_dl, va_dl,
        opt_clf, loss_fn,
        classifier_epochs, device
    )

    print("Classifier training complete.")

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
        torch.cuda.ipc_collect()

    print(f"=== Done! Test Acc: {acc:.4f}, AUROC: {auroc:.4f}, PR-AUC: {pr_auc:.4f}, F1: {f1:.4f} ===")
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Training Pipeline")
    parser.add_argument("--window_data_path",
                        default=f"{os.path.join(DATA_PATH, 'interim', 'windowed_data.h5')}")
    parser.add_argument("--mlflow_tracking_uri",
                        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
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

    args = parser.parse_args()
    main(**vars(args))