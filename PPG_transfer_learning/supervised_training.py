# ______________________________________________________________________________
# Simple script for the training of supervised methods on the PPG data
# For comparison and benchmarking against SSL and Pre-Trained TSTCC (on ECG)
# ______________________________________________________________________________

import os
import gc
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant,
    build_supervised_fingerprint,
    PhysiologicalDataset,
    set_seed,
    create_directory,
    train_one_epoch,
    test,
    EarlyStopping,
)

from models.supervised import (
    Improved1DCNN_v2,
    TCNClassifier,
    TransformerECGClassifier,
)

def main(
        mlflow_tracking_uri: str,
        fs: str,
        model_type: str = "cnn",
        gpu: int = 0,
        seed: int = 42,
        force_retraining: bool = True,
        lr: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 25,
        patience: int = 5,
        scheduler_mode: str = "min",
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 2,
        scheduler_min_lr: float = 1e-11,
        label_fraction: float = 1.0,
):
    set_seed(seed)

    # device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_cuda = (device.type == "cuda")
    pin_memory = use_cuda

    # mlflow setup
    exp_map = {
        "cnn": "Supervised_CNN (PPG)",
        "tcn": "Supervised_TCN (PPG)",
        "transformer": "Supervised_Transformer (PPG)",
    }
    experiment_name = exp_map.get(model_type.lower())
    if experiment_name is None:
        raise ValueError(f"Unknown model_type '{model_type}'")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=None)

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    model_save_path = os.path.join(SAVED_MODELS_PATH, "PPG", str(fs), f"{model_type}", f"{seed}", f"{label_fraction}")
    create_directory(model_save_path)

    # Data path
    window_data_path = os.path.join(DATA_PATH, "interim", "ECG", str(fs), 'windowed_data.h5')

    X, y, groups = load_processed_data(
        window_data_path,
        label_map={"baseline": 0, "mental_stress": 1},
        ppg_data=True
    )
    y = y.astype(np.float32)
    n_features = X.shape[2]

    # train/val/test split
    tr_idx, val_idx, te_idx = split_indices_by_participant(groups, label_fraction=label_fraction, seed=seed)
    print(f"windows: train {len(tr_idx)}, val {len(val_idx)}, test {len(te_idx)}")

    #
    tr_ds = PhysiologicalDataset(X[tr_idx], y[tr_idx])
    va_ds = PhysiologicalDataset(X[val_idx], y[val_idx])
    te_ds = PhysiologicalDataset(X[te_idx], y[te_idx])

    num_workers = min(8, os.cpu_count() or 2)

    # loaders
    tr_loader = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    va_loader = DataLoader(
        va_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        te_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # model
    if model_type.lower() == "cnn":
        model = Improved1DCNN_v2()
    elif model_type.lower() == "tcn":
        model = TCNClassifier()
    else:
        model = TransformerECGClassifier()
    model = model.to(device)

    # count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable: {trainable_params}")

    # criterion, optimizer, scheduler
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_mode, factor=scheduler_factor,
        patience=scheduler_patience, min_lr=scheduler_min_lr,
    )
    es = EarlyStopping(patience)

    # fingerprint
    fp = build_supervised_fingerprint({
        "model_name": model_type.lower(),
        "seed": seed,
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "patience": patience,
        "scheduler_mode": scheduler_mode,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "scheduler_min_lr": scheduler_min_lr,
        "label_fraction": label_fraction,
    })
    mlflow.log_params(fp)

    # train (if force retraining)

    if force_retraining:
        best_t, best_f1 = 0.5, -1.0
        for ep in range(1, num_epochs + 1):
            print(f"\nEpoch {ep}/{num_epochs}")
            val_loss, best_t, best_f1 = train_one_epoch(
                model, tr_loader, va_loader,
                optimizer, loss_fn, device, ep,
                best_threshold_so_far=best_t,
                best_f1_so_far=best_f1,
                log_interval=100,
            )
            scheduler.step(val_loss)
            es(val_loss)
            if es.early_stop:
                print("Early stopping")
                break

        mlflow.log_param("chosen_threshold", best_t)

        saved_results = os.path.join(model_save_path, f"{model_type}.pt")
        torch.save(
            {"model_parameters": model.state_dict(),
             "best_threshold": best_t,
             "best_f1": best_f1},
            saved_results
        )

        # save model
        if np.isclose(label_fraction, 1.0):
            mlflow.pytorch.log_model(model, artifact_path="supervised_model")

    else:
        # Load parameters from saved results
        saved_results = os.path.join(model_save_path, f"{model_type}.pt")

        if os.path.exists(saved_results):
            print(f"Loading saved model parameters from: {saved_results}")
            checkpoint = torch.load(saved_results, map_location=device, weights_only=False)

            # Load model state dict
            model.load_state_dict(checkpoint["model_parameters"])

            # Load best threshold and F1 score if available
            best_t = checkpoint.get("best_threshold", 0.5)
            best_f1 = checkpoint.get("best_f1", -1.0)

            print(f"Loaded model with best_threshold: {best_t:.4f}, best_f1: {best_f1:.4f}")
        else:
            print(f"No saved model found at {saved_results}. Please run with --force_retraining")
            raise FileNotFoundError(f"Model file not found: {saved_results}")

    # test
    loss, acc, auroc, prauc, f1 = test(
        model, test_loader, device,
        threshold=best_t, loss_fn=loss_fn,
    )
    print(f"Test acc: {acc:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}")
    mlflow.log_metrics({"test_loss": loss, "test_acc": acc, "test_auroc": auroc, "test_f1": f1})

    # cleanup
    for loader in (tr_loader, va_loader, test_loader):
        if hasattr(loader, '_iterator') and loader._iterator is not None:
            loader._iterator._shutdown_workers()
    del tr_loader, va_loader, test_loader
    for _ in range(3): gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train ECG classifier")
    parser.add_argument("--mlflow_tracking_uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--fs", default=1000, type=str, help="What sample frequency used for training")
    parser.add_argument("--model_type", choices=["cnn", "tcn", "transformer"], default="cnn")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_retraining", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5) #lr 1e-4 was good
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--scheduler_mode", default="min")
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=2)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-9)
    parser.add_argument("--label_fraction", type=float, default=1.0,
                        help="Percent of labeled participants in the training stage.")
    args = parser.parse_args()
    main(**vars(args))
