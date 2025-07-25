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
)
from models.tstcc import (
    data_generator_from_arrays,
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
from models.supervised import LinearClassifier


def main(
    window_data_path: str,
    mlflow_tracking_uri: str,
    seed: int,
    tcc_epochs: int,
    tcc_lr: float,
    tcc_batch_size: int,
    pretrain_all_conditions: bool,
    tc_timesteps: int,
    tc_hidden_dim: int,
    cc_temperature: float,
    cc_use_cosine: bool,
    classifier_epochs: int,
    classifier_lr: float,
    classifier_batch_size: int,
    label_fraction: float,
):
    # ── Step 0: Setup ────────────────────────────────────────────────────────────
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    logging.basicConfig(level=logging.INFO)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("TSTCC")

    # Start top‑level run
    run = mlflow.start_run(run_name=f"tstcc_training_{seed}")
    run_id = run.info.run_id
    logging.info(f"MLflow run_id: {run_id}")
    print(f"Using device: {device}")

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
    train_idx, val_idx, test_idx = split_indices_by_participant(groups, seed=seed)
    print(f"windows: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    # Keep binary‐task mask for later
    downstream_mask = {
        "train": np.isin(y[train_idx], [0, 1]),
        "val":   np.isin(y[val_idx],   [0, 1]),
        "test":  np.isin(y[test_idx],  [0, 1]),
    }

    # ── Step 2: TS‑TCC Pretraining ───────────────────────────────────────────────
    torch.cuda.empty_cache()
    set_seed(seed)

    # Fingerprint & search
    fp = build_tstcc_fingerprint({
        "model_name":            "TSTCC",
        "seed":                  seed,
        "pretrain_all_conditions": pretrain_all_conditions,
        "tcc_epochs":            tcc_epochs,
        "tcc_lr":                tcc_lr,
        "tcc_batch_size":        tcc_batch_size,
        "tc_timesteps":          tc_timesteps,
        "tc_hidden_dim":         tc_hidden_dim,
        "cc_temperature":        cc_temperature,
        "cc_use_cosine":         cc_use_cosine,
    })

    cached = search_encoder_fp(fp,
                               experiment_name="TSTCC",
                               tracking_uri=mlflow_tracking_uri)

    if cached:
        print(f"Found cached encoder run {cached}; downloading…")
        uri = f"runs:/{cached}/tstcc_model"
        ckpt_dir = mlflow.artifacts.download_artifacts(uri)
        ckpt_path = os.path.join(ckpt_dir, "tstcc.pt")

        # rebuild model
        cfg = ECGConfig()
        cfg.num_epoch = tcc_epochs
        cfg.batch_size = tcc_batch_size
        cfg.TC.timesteps = tc_timesteps
        cfg.TC.hidden_dim = tc_hidden_dim
        cfg.Context_Cont.temperature = cc_temperature
        cfg.Context_Cont.use_cosine_similarity = cc_use_cosine

        model  = base_Model(cfg).to(device)
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
        Xtr = X[train_idx].astype(np.float32)
        Xva = X[val_idx].astype(np.float32)
        Xte = X[test_idx].astype(np.float32)
        tr_dl, va_dl, te_dl = data_generator_from_arrays(
            Xtr, y[train_idx], Xva, y[val_idx], Xte, y[test_idx],
            cfg, training_mode="self_supervised"
        )

        # models & optimizers
        model   = base_Model(cfg).to(device)
        tc_head = TC(cfg, device).to(device)
        opt_m   = optim.AdamW(model.parameters(), lr=tcc_lr, weight_decay=3e-4)
        opt_tc  = optim.AdamW(tc_head.parameters(), lr=tcc_lr, weight_decay=3e-4)

        with mlflow.start_run(run_id=run_id):
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

    # ── Step 3: Extract Representations ─────────────────────────────────────────
    model.eval(); tc_head.eval()
    with torch.no_grad():
        train_repr, _ = encode_representations(X[train_idx], y[train_idx],
                                               model, tc_head, tcc_batch_size, device)
        val_repr,   _ = encode_representations(X[val_idx],   y[val_idx],
                                               model, tc_head, tcc_batch_size, device)
        test_repr,  _ = encode_representations(X[test_idx],  y[test_idx],
                                               model, tc_head, tcc_batch_size, device)

    # filter to binary downstream samples
    train_repr = train_repr[downstream_mask["train"]]
    y_train     = y[train_idx][downstream_mask["train"]]
    val_repr   = val_repr[downstream_mask["val"]]
    y_val       = y[val_idx][downstream_mask["val"]]
    test_repr  = test_repr[downstream_mask["test"]]
    y_test      = y[test_idx][downstream_mask["test"]]

    print(f"train_repr shape = {train_repr.shape}")
    show_shape("val/test repr", (val_repr, test_repr))

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
    classifier = LinearClassifier(train_repr.shape[-1]).to(device)
    opt_clf = optim.AdamW(classifier.parameters(), lr=classifier_lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with mlflow.start_run(run_id=run_id):
        clf_params = {
            "classifier_model":     "LinearClassifier",
            "classifier_epochs":    classifier_epochs,
            "classifier_lr":        classifier_lr,
            "classifier_batch_size":classifier_batch_size,
            "label_fraction":       label_fraction,
            "seed":                 seed,
        }
        mlflow.log_params(clf_params)

        # train & get best threshold
        classifier, best_thr = Trainer.train_linear_classifier(
            classifier, tr_loader, va_loader,
            opt_clf, loss_fn,
            classifier_epochs, device
        )

    # ── Step 5: Evaluation ──────────────────────────────────────────────────────
    te_loader = build_linear_loaders(test_repr, y_test,
                                     classifier_batch_size, device,
                                     shuffle=False)
    with mlflow.start_run(run_id=run_id):
        acc, auroc, pr_auc, f1 = Trainer.evaluate_classifier(
            model=classifier,
            test_loader=te_loader,
            device=device,
            threshold=best_thr,
            loss_fn=loss_fn,
        )
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_auroc":    auroc,
            "test_pr_auc":   pr_auc,
            "test_f1":       f1,
        })

    # ── Cleanup ────────────────────────────────────────────────────────────────
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"=== Done! Test Acc: {acc:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f} ===")
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TS-TCC Training Pipeline (cleaned)")
    parser.add_argument("--window_data_path",     default="../data/interim/windowed_data.h5")
    parser.add_argument("--mlflow_tracking_uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--tcc_epochs",          type=int,   default=40)
    parser.add_argument("--tcc_lr",              type=float, default=3e-4)
    parser.add_argument("--tcc_batch_size",      type=int,   default=128)
    parser.add_argument("--pretrain_all_conditions", action="store_true")
    parser.add_argument("--tc_timesteps",        type=int,   default=70)
    parser.add_argument("--tc_hidden_dim",       type=int,   default=128)
    parser.add_argument("--cc_temperature",      type=float, default=0.07)
    parser.add_argument("--cc_use_cosine",       action="store_true")
    parser.add_argument("--classifier_epochs",   type=int,   default=25)
    parser.add_argument("--classifier_lr",       type=float, default=1e-4)
    parser.add_argument("--classifier_batch_size", type=int, default=32)
    parser.add_argument("--label_fraction",      type=float, default=1.0)

    args = parser.parse_args()
    main(**vars(args))
