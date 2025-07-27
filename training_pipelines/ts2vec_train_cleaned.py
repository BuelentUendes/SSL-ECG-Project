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
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant,
    set_seed,
    create_directory,
)

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH

from models.ts2vec import (
    TS2Vec,
    build_fingerprint,
    search_encoder_fp,
    build_linear_loaders,
    train_linear_classifier,
    evaluate_classifier,
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
        ts2vec_epochs: int,
        ts2vec_lr: float,
        ts2vec_batch_size: int,
        ts2vec_output_dims: int,
        ts2vec_hidden_dims: int,
        ts2vec_depth: int,
        ts2vec_max_train_length: int,
        ts2vec_temporal_unit: int,
        classifier_epochs: int,
        classifier_model: bool,
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
    mlflow.set_experiment("TS2Vec")

    # Start top‑level run
    run = mlflow.start_run(run_name=f"ts2vec_training_{seed}")
    run_id = run.info.run_id
    logging.info(f"MLflow run_id: {run_id}")
    print(f"Using device: {device}")

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)

    # We save the model here via seeds
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"
    model_save_path = os.path.join(SAVED_MODELS_PATH, "TSTCC", pretrain_data, f"{seed}")
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
    train_idx, val_idx, test_idx = split_indices_by_participant(groups, seed=seed)
    print(f"windows: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    # Keep binary‐task mask for later
    downstream_mask = {
        "train": np.isin(y[train_idx], [0, 1]),
        "val":   np.isin(y[val_idx],   [0, 1]),
        "test":  np.isin(y[test_idx],  [0, 1]),
    }

    # ── Step 2: TS2Vec Pretraining ──────────────────────────────────────────────
    torch.cuda.empty_cache()
    set_seed(seed)

    # Fingerprint & search
    fp = {
        "model_name": "TS2Vec",
        "seed": seed,
        "ts2vec_epochs": ts2vec_epochs,
        "ts2vec_output_dims": ts2vec_output_dims,
        "ts2vec_hidden_dims": ts2vec_hidden_dims,
        "ts2vec_depth": ts2vec_depth,
        "ts2vec_max_train_length": ts2vec_max_train_length,
        "ts2vec_temporal_unit": ts2vec_temporal_unit,
    }
    fp = build_fingerprint(fp)

    cached = search_encoder_fp(fp,
                               experiment_name="TS2Vec",
                               tracking_uri=mlflow_tracking_uri)

    # IF we have forced retraining we will always retrain
    if (cached or os.path.exists(os.path.join(model_save_path, "ts2vec_model.pth"))) and not (force_retraining):
        if cached:
            print(f"Found cached encoder run {cached}; downloading…")
            uri = f"runs:/{cached}/ts2vec_model"
            net = mlflow.pytorch.load_model(uri, map_location=device)

            ts2vec = TS2Vec(
                input_dims=n_features,
                output_dims=ts2vec_output_dims,
                hidden_dims=ts2vec_hidden_dims,
                depth=ts2vec_depth,
                device=device,
                lr=ts2vec_lr,
                batch_size=ts2vec_batch_size,
                max_train_length=ts2vec_max_train_length,
                temporal_unit=ts2vec_temporal_unit,
            )
            ts2vec.net = ts2vec._net = net
        else:
            print("We found a pretrained model. Load the pretrained weights")
            model_path = os.path.join(model_save_path, "ts2vec_model.pth")

            ts2vec = TS2Vec(
                input_dims=n_features,
                output_dims=ts2vec_output_dims,
                hidden_dims=ts2vec_hidden_dims,
                depth=ts2vec_depth,
                device=device,
                lr=ts2vec_lr,
                batch_size=ts2vec_batch_size,
                max_train_length=ts2vec_max_train_length,
                temporal_unit=ts2vec_temporal_unit,
            )
            ts2vec.net = ts2vec._net = torch.load(model_path, map_location=device)

    else:
        print("No cached encoder; training TS2Vec from scratch")

        # Load only training data for pretraining
        X_train = X[train_idx].astype(np.float32)

        ts2vec = TS2Vec(
            input_dims=n_features,
            output_dims=ts2vec_output_dims,
            hidden_dims=ts2vec_hidden_dims,
            depth=ts2vec_depth,
            device=device,
            lr=ts2vec_lr,
            batch_size=ts2vec_batch_size,
            max_train_length=ts2vec_max_train_length,
            temporal_unit=ts2vec_temporal_unit,
        )

        print(f"Created TS2Vec model on device: {next(ts2vec.net.parameters()).device}")

        mlflow.log_params(fp)

        # Train TS2Vec
        loss_log = ts2vec.fit(
            X_train,
            n_epochs=ts2vec_epochs,
            verbose=True
        )

        # Save model
        mlflow.pytorch.log_model(
            pytorch_model=ts2vec.net,
            artifact_path="ts2vec_model"
        )

        saved_results = os.path.join(model_save_path, "ts2vec_model.pth")
        torch.save(ts2vec.net, saved_results)

    # ── Step 3: Extract Representations ─────────────────────────────────────────
    print("\nExtracting representations...")

    # Get TS2Vec embeddings
    train_repr = ts2vec.encode(X[train_idx].astype(np.float32), encoding_window="full_series")
    val_repr = ts2vec.encode(X[val_idx].astype(np.float32), encoding_window="full_series")
    test_repr = ts2vec.encode(X[test_idx].astype(np.float32), encoding_window="full_series")

    # filter to binary downstream samples
    train_repr = train_repr[downstream_mask["train"]]
    y_train     = y[train_idx][downstream_mask["train"]]
    val_repr   = val_repr[downstream_mask["val"]]
    y_val       = y[val_idx][downstream_mask["val"]]
    test_repr  = test_repr[downstream_mask["test"]]
    y_test      = y[test_idx][downstream_mask["test"]]

    print(f"Extracted TS2Vec representations: train_repr shape={train_repr.shape}")

    # ── Step 4: Classifier Fine‑Tuning ──────────────────────────────────────────
    set_seed(seed)
    if label_fraction < 1.0:
        sub_idx, _ = train_test_split(
            np.arange(len(y_train)),
            train_size=label_fraction,
            stratify=y_train,
            random_state=seed
        )
    else:
        sub_idx = np.arange(len(y_train))

    tr_loader = build_linear_loaders(train_repr[sub_idx], y_train[sub_idx],
                                     classifier_batch_size, device)
    va_loader = build_linear_loaders(val_repr, y_val,
                                     classifier_batch_size, device,
                                     shuffle=False)

    if classifier_model == "linear":
        classifier = LinearClassifier(train_repr.shape[-1]).to(device)
    else:
        classifier = MLPClassifier(train_repr.shape[-1]).to(device)

    opt_clf = optim.Adam(classifier.parameters(), lr=classifier_lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    clf_params = {
        "classifier_model": "LinearClassifier" if classifier_model == "linear" else "MLP",
        "classifier_epochs": classifier_epochs,
        "classifier_lr": classifier_lr,
        "classifier_batch_size": classifier_batch_size,
        "label_fraction": label_fraction,
        "seed": seed,
    }
    mlflow.log_params(clf_params)

    # train & get best threshold (Note: this function can also train MLP model, a bit of a misnomer the function)
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
        torch.cuda.ipc_collect()

    print(f"=== Done! Test Acc: {acc:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f} ===")
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TS2Vec Training Pipeline")
    parser.add_argument("--window_data_path",
                        default=f"{os.path.join(DATA_PATH, 'interim', 'windowed_data.h5')}")
    parser.add_argument("--mlflow_tracking_uri",
                        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_retraining", action="store_true")
    parser.add_argument("--pretrain_all_conditions", action="store_true")

    # TS2Vec pretraining parameters
    parser.add_argument("--ts2vec_epochs", type=int, default=50,
                        help="Epochs for TS2Vec pretraining")
    parser.add_argument("--ts2vec_lr", type=float, default=0.001,
                        help="Learning rate for TS2Vec")
    parser.add_argument("--ts2vec_batch_size", type=int, default=8,
                        help="Batch size for TS2Vec")
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

    # classifier fine-tuning
    parser.add_argument("--classifier_epochs", type=int, default=25)
    parser.add_argument("--classifier_model", type=str, default="linear", choices=("linear", "mlp"))
    parser.add_argument("--classifier_lr", type=float, default=0.0001)
    parser.add_argument("--classifier_batch_size", type=int, default=32)
    parser.add_argument("--label_fraction", type=float, default=1.0)

    args = parser.parse_args()
    main(**vars(args))