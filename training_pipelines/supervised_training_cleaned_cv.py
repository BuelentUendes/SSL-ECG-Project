####
# This is the script of supervised_training with cross validation
###

import os
import json
import gc
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH, RESULTS_PATH

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant_groups,
    get_participant_cv_splitter,
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


def run_supervised_model_with_cv_and_test(
        model_type, X_train, y_train, groups_train, X_test, y_test,
        cv_splitter, device, classifier_epochs=25, classifier_batch_size=32,
        classifier_lr=1e-4, pin_memory=False
):
    """Run CV for Supervised model  then train final model and test."""

    lr_rates = [1e-3, 1e-4]
    dropout_rates = [0.1]

    num_workers = min(8, os.cpu_count() or 2)

    best_params = None
    best_cv_score = 0

    default_best_params = {
        "dropout": [0.3],
        "lr": [1e-5],
    }

    print(f"Running manual CV for supervised model {model_type} hyperparameters...")

    if cv_splitter is None:
        print("cv_splitter is None (single participant). Using default best parameters...")
        print(f"Default parameters: {default_best_params}")

        best_params = default_best_params
        #ToDo!
        # model_factory = create_default_model_factory(model_type)
        best_cv_score = 0.0

    else:
        for lr in lr_rates:
            for dropout_rate in dropout_rates:
                print(f"Testing lr={lr}, dropout={dropout_rate}")

                fold_scores = []

                # Run CV for this parameter combination
                for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train, groups_train), 1):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                    # model
                    if model_type.lower() == "cnn":
                        model = Improved1DCNN_v2()
                    elif model_type.lower() == "tcn":
                        model = TCNClassifier()
                    else:
                        model = TransformerECGClassifier()

                    model = model.to(device)

                    optimizer = optim.AdamW(model.parameters(), lr=lr)
                    loss_fn = torch.nn.BCEWithLogitsLoss()

                    # Create proper datasets for supervised models (not SSL representations)
                    tr_ds = PhysiologicalDataset(X_fold_train, y_fold_train)
                    val_ds = PhysiologicalDataset(X_fold_val, y_fold_val)
                    tr_loader = DataLoader(
                        tr_ds, batch_size=classifier_batch_size, shuffle=True, 
                        drop_last=True, pin_memory=pin_memory, num_workers=num_workers
                    )
                    val_loader = DataLoader(
                        val_ds, batch_size=classifier_batch_size, shuffle=False, 
                        drop_last=False, pin_memory=pin_memory, num_workers=num_workers
                    )

                    non_blocking_bool = torch.cuda.is_available()

                    # Training loop
                    for idx, epoch in enumerate(range(classifier_epochs), 1):
                        print(f"Fold: {fold}: Processing Epoch {idx} / {classifier_epochs}")
                        model.train()
                        for X_batch, y_batch in tr_loader:
                            X_batch = X_batch.to(device, non_blocking=non_blocking_bool).permute(0, 2, 1)  # (B,C,L)
                            y_batch = y_batch.to(device, non_blocking=non_blocking_bool).float()
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
                            X_batch = X_batch.to(device, non_blocking=non_blocking_bool).permute(0, 2, 1)  # (B,C,L)
                            y_batch = y_batch.to(device, non_blocking=non_blocking_bool).float()
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
                    best_params = {'lr': lr, 'dropout': dropout_rate}

    print(f"\nBest parameters: {best_params}")
    print(f"Best CV score: {best_cv_score:.4f}")

    # Train final model with best parameters on full training set
    print("Training final model on full training set...")

    #ToDo: Add here the model factory!
    if model_type.lower() == "cnn":
        final_model = Improved1DCNN_v2().to(device)
    elif model_type.lower() == "tcn":
        final_model = TCNClassifier().to(device)
    else:
        final_model = TransformerECGClassifier().to(device)

    total_params = sum(p.numel() for p in final_model.parameters())
    trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable: {trainable_params}")

    tr_ds = PhysiologicalDataset(X_train, y_train)
    te_ds = PhysiologicalDataset(X_test, y_test)
    tr_loader = DataLoader(
        tr_ds, batch_size=32, shuffle=True, drop_last=True, 
        pin_memory=pin_memory, num_workers=num_workers
    )
    te_loader = DataLoader(
        te_ds, batch_size=32, shuffle=False, drop_last=False, 
        pin_memory=pin_memory, num_workers=num_workers
    )

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params["lr"])
    
    # Move non_blocking_bool definition here for final model training
    non_blocking_bool = torch.cuda.is_available()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train final model
    for idx, epoch in enumerate(range(classifier_epochs), start=1):
        print(f"Epoch {idx} / {classifier_epochs}")
        final_model.train()
        for X_batch, y_batch in tr_loader:
            X_batch = X_batch.to(device, non_blocking=non_blocking_bool).permute(0, 2, 1)  # (B,C,L)
            y_batch = y_batch.to(device, non_blocking=non_blocking_bool).float()
            optimizer.zero_grad()
            logits = final_model(X_batch).squeeze(-1)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            if epoch % 5 == 0:
                print(f"The loss is {loss}")
            optimizer.step()

    # Test evaluation
    final_model.eval()
    test_probs = []
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for X_batch, y_batch in te_loader:
            X_batch = X_batch.to(device, non_blocking=non_blocking_bool).permute(0, 2, 1)  # (B,C,L)
            y_batch = y_batch.to(device, non_blocking=non_blocking_bool).float()
            logits = final_model(X_batch).squeeze(-1)
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
        'model': final_model,
        'total_params': total_params,
    }

def main(
        mlflow_tracking_uri: str,
        fs: str,
        model_type: str = "cnn",
        label_fraction: float = 0.1,
        window_size: int = 10,
        step_size: int =5,
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
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
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
        "cnn": "Supervised_CNN",
        "tcn": "Supervised_TCN",
        "transformer": "Supervised_Transformer",
    }
    experiment_name = exp_map.get(model_type.lower())
    if experiment_name is None:
        raise ValueError(f"Unknown model_type '{model_type}'")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=None)

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)
    create_directory(RESULTS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    model_save_path = os.path.join(
        SAVED_MODELS_PATH, "ECG", str(fs), f"{model_type}", f"{seed}", f"{label_fraction}", f"{window_size}", f"{step_size}"
    )
    results_save_path = os.path.join(
        RESULTS_PATH, "ECG", "Supervised", model_type, f"{seed}", f"{label_fraction}", f"{window_size}", f"{step_size}"
    )

    create_directory(model_save_path)
    create_directory(results_save_path)

    # Data path
    window_data_path = os.path.join(
        DATA_PATH, "interim", "ECG", str(fs), f"{window_size}", f"{step_size}", 'windowed_data.h5'
    )

    # load data
    X, y, groups = load_processed_data(
        window_data_path,
        label_map={"baseline": 0, "mental_stress": 1},
    )
    y = y.astype(np.float32)

    # train/val/test split
    # Split by participant to get train/test split
    train_idx, train_p, test_idx, test_p = split_indices_by_participant_groups(
        groups,
        train_ratio=0.8,
        label_fraction=label_fraction,
        seed=seed
    )

    print(f" Labeled windows: train {len(train_idx)}, test {len(test_idx)}")

    X_train = X[train_idx]
    y_train = y[train_idx]
    groups_train = groups[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    # Filter to binary classification for both train and test
    train_binary_mask = np.isin(y_train, [0, 1])
    test_binary_mask = np.isin(y_test, [0, 1])

    X_train = X_train[train_binary_mask]
    y_train = y_train[train_binary_mask]
    groups_train = groups_train[train_binary_mask]

    X_test = X_test[test_binary_mask]
    y_test = y_test[test_binary_mask]

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Training participants: {len(np.unique(groups_train))}")
    print(f"Test participants: {len(np.unique(groups[test_idx][test_binary_mask]))}")

    # ── Step 2: Set up Cross-Validation Splitter ───────────────────────────────
    cv_splitter, n_splits = get_participant_cv_splitter(
        groups_train,
        min_participants_for_kfold=min_participants_for_kfold,
        k=k_folds
    )

    # --Step 3: Training if set (force retraining) --------
    if force_retraining:
        results = run_supervised_model_with_cv_and_test(
            model_type, X_train, y_train, groups_train, X_test, y_test,
            cv_splitter, device, classifier_epochs=num_epochs, classifier_batch_size=batch_size,
            classifier_lr=lr, pin_memory=pin_memory)

        # log the results:
        mlflow.log_metrics({
            "test_accuracy": results["test_metrics"]["accuracy"],
            "test_aurco": results["test_metrics"]["auroc"],
            "test_f1": results["test_metrics"]["f1"],
            "test_pr_auc": results["test_metrics"]["pr_auc"],
        })
        # Save the results:
        with open(os.path.join(results_save_path, "test_results.json"), "w") as f:
            json.dump(results, f)

    else:
        # Load parameters from saved results
        saved_results = os.path.join(model_save_path, f"{model_type}.pt")

        # model
        if model_type.lower() == "cnn":
            model = Improved1DCNN_v2()
        elif model_type.lower() == "tcn":
            model = TCNClassifier()
        else:
            model = TransformerECGClassifier()
        model = model.to(device)

        if os.path.exists(saved_results):
            print(f"Loading saved model parameters from: {saved_results}")
            checkpoint = torch.load(saved_results, map_location=device, weights_only=True)

            # Load model state dict
            model.load_state_dict(checkpoint["model_parameters"])

        else:
            print(f"No saved model found at {saved_results}. Please run with --force_retraining")
            raise FileNotFoundError(f"Model file not found: {saved_results}")

        test_ds = PhysiologicalDataset(X_test, y_test)
        test_loader = DataLoader(
            test_ds, batch_size=32, shuffle=False, drop_last=False,
            pin_memory=pin_memory, num_workers=num_workers
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()

        #ToDo: Change the best t for f1 score
        loss, acc, auroc, prauc, f1 = test(
            model, test_loader, device,
            threshold=0.5, loss_fn=loss_fn,
        )

        print(f"Test acc: {acc:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}, PR-AUC: {prauc:.4f}")
        mlflow.log_metrics({"test_loss": loss, "test_acc": acc, "test_auroc": auroc, "test_f1": f1})

    # cleanup
    for _ in range(3): gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train ECG classifier")
    parser.add_argument("--mlflow_tracking_uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--fs", default=1000, type=str, help="What sample frequency used for training")
    parser.add_argument("--model_type", choices=["cnn", "tcn", "transformer"], default="cnn")
    parser.add_argument("--label_fraction", type=float, default=0.1,
                        help="Percent of labeled participants in the training stage.")
    parser.add_argument("--window_size", type=int, default=10,
                           help="Window size in seconds")
    parser.add_argument("--step_size", type=int, default=5,
                           help="Step size in seconds for sliding window")
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
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for CV")
    parser.add_argument("--min_participants_for_kfold", type=int, default=5,
                        help="Minimum participants needed for k-fold (otherwise use Leave one participant out)")

    args = parser.parse_args()

    args.force_retraining = True
    main(**vars(args))