####
# This is the script of supervised_training with cross validation
###

import os
import copy
import json
import gc
import argparse
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH, RESULTS_PATH

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant_groups,
    get_participant_cv_splitter,
    PhysiologicalDataset,
    set_seed,
    create_directory,
    test,
    run_logistic_regression_with_gridsearch,
    run_mlp_with_cv_and_test,
    run_linear_classifier_with_cv_and_test,
)

from models.supervised import (
    Improved1DCNN_v2,
    TCNClassifier,
    TransformerECGClassifier,
    DeepECGNet,
    PatchTSTECGClassifier,
    MomentFMClassifier,
    FineTunedCNNNet,
    freeze_and_unfreeze_encoder,
)


def run_supervised_model_with_cv_and_test(
        model_type, X_train, y_train, groups_train, X_test, y_test,
        cv_splitter,
        device,
        classifier_epochs=25,
        classifier_batch_size=32,
        disable_hyperparameter_tuning=False,
        dropout_rate=0.3,
        lr=0.0001,
        pin_memory=False,
        scoring_metric="roc_auc",
        use_s3_layers: bool = False,
        initial_num_segments: int = 2,
        num_s3_layers: int = 2,
        segment_multiplier: int = 2,
        results_save_path: str = RESULTS_PATH,
):
    """Run CV for Supervised model  then train final model and test."""

    torch.autograd.set_detect_anomaly(True)

    lr_rates = [1e-4, 1e-5]
    dropout_rates = [0.2, 0.3, 0.5]

    num_workers = min(8, os.cpu_count() or 2)

    best_params = None
    best_cv_score = 0

    default_best_params = {
        "dropout": 0.5,
        "lr": 1e-5,
    }

    print(f"Running manual CV for supervised model {model_type} hyperparameters...")

    if not disable_hyperparameter_tuning:
        if cv_splitter is None:
            print("cv_splitter is None (single participant). Using default best parameters...")
            print(f"Default parameters: {default_best_params}")

            best_params = default_best_params
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
                            model = Improved1DCNN_v2(
                                dropout=dropout_rate,
                                use_s3_layers=use_s3_layers,
                                initial_num_segments=initial_num_segments,
                                num_s3_layers=num_s3_layers,
                                segment_multiplier=segment_multiplier,
                            )
                        elif model_type.lower() == "tcn":
                            model = TCNClassifier(
                                dropout=dropout_rate,
                                use_s3_layers=use_s3_layers,
                                initial_num_segments=initial_num_segments,
                                num_s3_layers=num_s3_layers,
                                segment_multiplier=segment_multiplier,
                            )
                        elif model_type.lower() == "deep_ecg_net":
                            model = DeepECGNet(
                                dropout_rate=dropout_rate,
                                use_s3_layers = use_s3_layers,
                                initial_num_segments = initial_num_segments,
                                num_s3_layers = num_s3_layers,
                                segment_multiplier = segment_multiplier,
                            )
                        elif model_type.lower() == "patchtst":
                            model = PatchTSTECGClassifier(dropout=dropout_rate)
                        elif model_type.lower() == "moment":
                            model = MomentFMClassifier(dropout=dropout_rate)
                        else:
                            model = TransformerECGClassifier(dropout=dropout_rate)

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
                            print(f"Fold: {fold}: Processing Epoch {idx} / {classifier_epochs}", flush=True, end="\r")
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

                                # Check for NaN in logits
                                if torch.isnan(logits).any():
                                    print(f"NaN detected in logits! Fold {fold}, batch size: {X_batch.size()}")
                                    print(f"Logits: {logits}")
                                    raise ValueError("Model produced NaN logits")

                                probs = torch.sigmoid(logits)

                                # Check for NaN in probabilities
                                if torch.isnan(probs).any():
                                    print(f"NaN detected in probs! Fold {fold}")
                                    print(f"Probs: {probs}")
                                    raise ValueError("Sigmoid produced NaN probabilities")

                                val_probs.extend(probs.cpu().numpy())
                                val_labels.extend(y_batch.cpu().numpy())

                        # Calculate fold score based on selected metric
                        if scoring_metric == "roc_auc":
                            fold_score = roc_auc_score(val_labels, val_probs)
                        elif scoring_metric == "average_precision":
                            fold_score = average_precision_score(val_labels, val_probs)
                        elif scoring_metric == "f1":
                            val_preds = (np.array(val_probs) > 0.5).astype(int)
                            fold_score = f1_score(val_labels, val_preds)
                        elif scoring_metric == "balanced_accuracy":
                            val_preds = (np.array(val_probs) > 0.5).astype(int)
                            fold_score = balanced_accuracy_score(val_labels, val_preds)
                        else:
                            raise ValueError(f"Unknown scoring metric: {scoring_metric}")

                        fold_scores.append(fold_score)

                    # Average CV score for this parameter combination
                    mean_cv_score = np.mean(fold_scores)
                    print()
                    print(f"  Mean CV {scoring_metric.upper()}: {mean_cv_score:.4f}")

                    if mean_cv_score > best_cv_score:
                        best_cv_score = mean_cv_score
                        best_params = {'lr': lr, 'dropout': dropout_rate}

        print(f"\nBest parameters: {best_params}")
        print(f"Best CV score: {best_cv_score:.4f}")

    else:
        print(f"Hyperparameter tuning is disabled. Run with selected hyperparameters")
        best_params = {
            "dropout": dropout_rate,
            "lr": lr,
        }

        print(best_params)

    # Train final model with best parameters on full training set
    print("Training final model on full training set...")

    if model_type.lower() == "cnn":
        final_model = Improved1DCNN_v2(dropout=best_params["dropout"]).to(device)
    elif model_type.lower() == "tcn":
        final_model = TCNClassifier(dropout=best_params["dropout"]).to(device)
    elif model_type.lower() == "deep_ecg_net":
        final_model = DeepECGNet(dropout_rate=best_params["dropout"]).to(device)
    elif model_type.lower() == "patchtst":
        final_model = PatchTSTECGClassifier(dropout=best_params["dropout"]).to(device)
    elif model_type.lower() == "moment":
        final_model = MomentFMClassifier(dropout=best_params["dropout"]).to(device)
    else:
        final_model = TransformerECGClassifier(dropout=best_params["dropout"]).to(device)

    total_params = sum(p.numel() for p in final_model.parameters())
    trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable: {trainable_params}")

    tr_ds = PhysiologicalDataset(X_train, y_train)
    te_ds = PhysiologicalDataset(X_test, y_test)
    tr_loader = DataLoader(
        tr_ds, batch_size=classifier_batch_size, shuffle=True, drop_last=True,
        pin_memory=pin_memory, num_workers=num_workers
    )
    te_loader = DataLoader(
        te_ds, batch_size=classifier_batch_size, shuffle=False, drop_last=False,
        pin_memory=pin_memory, num_workers=num_workers
    )

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params["lr"])
    
    # Move non_blocking_bool definition here for final model training
    non_blocking_bool = torch.cuda.is_available()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Save the memory consumption and runtimes per model here as well
    epoch_peak_memory = {}
    epoch_runtimes = {}
    average_epoch_loss = []

    # Train final model
    for idx, epoch in enumerate(range(classifier_epochs), start=1):
        print()
        print(f"Final training: Epoch {idx} / {classifier_epochs}", end="\r")
        final_model.train()
        epoch_start_time = time.time()
        epoch_loss = []

        for X_batch, y_batch in tr_loader:
            X_batch = X_batch.to(device, non_blocking=non_blocking_bool).permute(0, 2, 1)  # (B,C,L)
            y_batch = y_batch.to(device, non_blocking=non_blocking_bool).float()
            optimizer.zero_grad()
            logits = final_model(X_batch).squeeze(-1)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()

        average_epoch_loss.append(np.mean(epoch_loss))
        print(f"Epoch: {idx}: average loss: {np.mean(epoch_loss)}")

        # Calculate epoch runtime
        epoch_runtime = time.time() - epoch_start_time
        epoch_runtimes[f"{epoch}"] = epoch_runtime

        # Log memory usage for CUDA
        if torch.cuda.is_available():
            epoch_peak_memory[f"{epoch}"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Peak memory allocated during epoch {epoch}: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")

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

    # Save the results for peak memory time
    run_time_save_name = "runtime_per_epoch.json"
    peak_memory_save_name = "peak_memory_consumption_epochs.json"

    with open(os.path.join(results_save_path, run_time_save_name), "w") as f:
        json.dump(epoch_runtimes, f, indent=2)

    if torch.cuda.is_available():
        with open(os.path.join(results_save_path, peak_memory_save_name), "w") as f:
            json.dump(epoch_peak_memory, f, indent=2)

    return {
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'test_metrics': {
            'accuracy': test_acc,
            'auroc': test_auroc,
            'f1': test_f1,
            'pr_auc': test_pr_auc
        },
        'total_params': total_params,
        'average_epoch_loss': average_epoch_loss,
    }, final_model


def get_paths(
        dataset:str,
        fs: str,
        model_type: str,
        seed: int,
        label_fraction: float,
        window_size: int,
        step_size: int
) -> [str, str, str]:
    """
    Returns the model path, result path and data path based on the dataset
    """

    if dataset == "ours":
        # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
        model_save_path = os.path.join(
            SAVED_MODELS_PATH, "ECG", str(fs), f"{model_type}", f"{seed}", f"{label_fraction}", f"{window_size}", f"{step_size}"
        )
        results_save_path = os.path.join(
            RESULTS_PATH, "ECG", "Supervised", model_type, f"{seed}", f"{label_fraction}", f"{window_size}", f"{step_size}"
        )
        window_data_path = os.path.join(
            DATA_PATH, "interim", "ECG", str(fs), f"{window_size}", f"{step_size}", 'windowed_data.h5'
        )

    elif dataset == "stressid":
        # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
        model_save_path = os.path.join(
            SAVED_MODELS_PATH, "StressID", str(fs), f"{model_type}", f"{seed}", f"{label_fraction}", f"{window_size}",
            f"{step_size}"
        )
        results_save_path = os.path.join(
            RESULTS_PATH, "StressID", "Supervised", model_type, f"{seed}", f"{label_fraction}", f"{window_size}",
            f"{step_size}"
        )
        window_data_path = os.path.join(
            DATA_PATH, "interim", "STRESSID", "ECG", str(fs), f"{window_size}", f"{step_size}", 'windowed_data.h5'
        )

    elif dataset == "wesad":
        # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
        model_save_path = os.path.join(
            SAVED_MODELS_PATH, "WESAD", str(fs), f"{model_type}", f"{seed}", f"{label_fraction}", f"{window_size}",
            f"{step_size}"
        )
        results_save_path = os.path.join(
            RESULTS_PATH, "WESAD", "Supervised", model_type, f"{seed}", f"{label_fraction}", f"{window_size}",
            f"{step_size}"
        )
        window_data_path = os.path.join(
            DATA_PATH, "interim", "WESAD", "ECG", str(fs), f"{window_size}", f"{step_size}", 'windowed_data.h5'
        )

    else:
        raise AttributeError(f"{dataset} is not available.")

    return model_save_path, results_save_path, window_data_path


def load_and_return_saved_model(model_type, model_weights_path, device):
    # model
    if model_type.lower() == "cnn":
        model = Improved1DCNN_v2()
    elif model_type.lower() == "tcn":
        model = TCNClassifier()
    elif model_type.lower() == "deep_ecg_net":
        model = DeepECGNet()
    elif model_type.lower() == "patchtst":
        model = PatchTSTECGClassifier()
    elif model_type.lower() == "moment":
        model = MomentFMClassifier()
    else:
        model = TransformerECGClassifier()

    model = model.to(device)

    if os.path.exists(model_weights_path):
        print(f"Loading saved model parameters from: {model_weights_path}")
        checkpoint = torch.load(model_weights_path, map_location=device, weights_only=True)

        # Load model state dict
        model.load_state_dict(checkpoint["model_parameters"])

        return model

    else:
        print(f"No saved model found at {model_weights_path}. Please run with --force_retraining")
        raise FileNotFoundError(f"Model file not found: {model_weights_path}")


def get_representations_in_batches(model, data, batch_size=32):
    model.eval()
    representations = []
    total_len = (len(data) // batch_size) + 1

    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches", total=total_len, unit="batch"):
        batch = data[i:i + batch_size]
        with torch.no_grad():
            batch_repr = model.get_encoder_representations(batch)
            representations.append(batch_repr.cpu().numpy())

            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return np.concatenate(representations, axis=0)



def main(
        fs: int,
        dataset: str,
        model_type: str = "cnn",
        label_fraction: float = 0.1,
        window_size: int = 10,
        step_size: int =5,
        gpu: int = 0,
        seed: int = 42,
        force_retraining: bool = True,
        disable_hyperparameter_tuning: bool = False,
        dropout_rate: float=0.3,
        lr: float=0.0001,
        batch_size: int = 32,
        num_epochs: int = 25,
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
        scoring_metric: str = "roc_auc",
        use_s3_layers: bool = False,
        initial_num_segments: int=2,
        num_s3_layers: int = 2,
        segment_multiplier: int =2,
        zero_shot_evaluation: bool=False,
        zero_shot_dataset: str="wesad",
        use_pretrained_encoder: bool = False,
        fine_tune_encoder: bool = False,
        classifier_head: str = "logistic_regression",
        classifier_epochs: int = 25,
        classifier_lr: float = 1e-4,
        classifier_batch_size: int =32,
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

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)
    create_directory(RESULTS_PATH)

    model_save_path, results_save_path, window_data_path = get_paths(
        dataset, fs, model_type, seed, label_fraction, window_size,step_size
    )
    create_directory(model_save_path)
    create_directory(results_save_path)

    if use_pretrained_encoder:
        if fine_tune_encoder:
            subfolder_name = "fine_tuned_encoder_new_head"
        else:
            subfolder_name = "pretrained_encoder_new_head"
    else:
        if fine_tune_encoder:
            subfolder_name = "trained_from_scratch_fine_tuned_encoder"
        else:
            subfolder_name = "trained_from_scratch"

    # Get the transfer learning results
    if use_pretrained_encoder:
        pretrained_model_save_path = os.path.join(
            SAVED_MODELS_PATH, "ECG", str(fs), f"{model_type}", f"{seed}", f"{label_fraction}",
            f"{window_size}", f"{step_size}"
        )
        dataset_name = "WESAD" if dataset == "wesad" else "StressID"
        transfer_learning_results_save_path = os.path.join(
            RESULTS_PATH, "Transfer_learning", dataset_name, subfolder_name, model_type,
            f"{seed}", f"{label_fraction}", f"{window_size}", f"{step_size}", classifier_head
    )

        create_directory(transfer_learning_results_save_path)

    # Create zero-shot results path
    if zero_shot_evaluation:
        target_domain = "StressID" if zero_shot_dataset == "stressid" else "WESAD"
        zero_shot_results_path = os.path.join(
            RESULTS_PATH, "Transfer_learning", target_domain, "zero_shot_performance", model_type,
            f"{seed}", f"{label_fraction}",
    )

        create_directory(zero_shot_results_path)

    # If zero shot evaluation is set true, we load the StressID and WESAD dataset
    if zero_shot_evaluation:
        if zero_shot_dataset == "wesad":
            if int(fs) == 700:
                zero_shot_window_data_path = os.path.join(
                    DATA_PATH, "interim", "WESAD", "ECG", str(fs), str(window_size), str(step_size), 'windowed_data.h5')
            else:
                raise ValueError("For zero-shot evaluation for wesad the frequency needs to be 700")

        elif zero_shot_dataset == "stressid":
            if int(fs) == 500:
                zero_shot_window_data_path = os.path.join(
                DATA_PATH, "interim", "STRESSID", "ECG", str(fs), str(window_size), str(step_size), 'windowed_data.h5')
            else:
                raise ValueError("For zero-shot evaluation for stressid the frequency needs to be 500")
        else:
            raise ValueError('Please use a proper dataset "wesad" or "stressid"')

        X_zero_shot, y_zero_shot, groups_shot = load_processed_data(
            zero_shot_window_data_path, label_map={"baseline": 0, "mental_stress": 1}
        )
        y_zero_shot = y_zero_shot.astype(np.float32)

    # load data
    X, y, groups = load_processed_data(
        window_data_path,
        label_map={"baseline": 0, "mental_stress": 1},
    )
    y = y.astype(np.float32)

    # Here we can now check class distribution
    class_1_overall = np.mean(y)
    class_0_overall = 1-class_1_overall

    print(f"The total number of segments is: {len(y)}, class 1: {class_1_overall}, class 0: {class_0_overall}")

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
        results, model = run_supervised_model_with_cv_and_test(
            model_type, X_train, y_train, groups_train, X_test, y_test,
            cv_splitter, device, classifier_epochs=num_epochs, classifier_batch_size=batch_size,
            disable_hyperparameter_tuning=disable_hyperparameter_tuning,
            dropout_rate=dropout_rate,
            lr=lr,
            pin_memory=pin_memory, scoring_metric=scoring_metric,
            use_s3_layers=use_s3_layers, initial_num_segments=initial_num_segments,
            num_s3_layers=num_s3_layers, segment_multiplier=segment_multiplier,
            results_save_path=results_save_path
        )

        # Save the results:
        with open(os.path.join(results_save_path, "test_results.json"), "w") as f:
            json.dump(results, f)

        saved_results = os.path.join(model_save_path, f"{model_type}.pt")
        torch.save(
            {"model_parameters": model.state_dict()},
            saved_results
        )

    else:
        if use_pretrained_encoder:
            # We need to load the corresponding architecture
            pretrained_model_weights_path = os.path.join(
                pretrained_model_save_path, f"{model_type}.pt"
            )
            pretrained_model = load_and_return_saved_model(model_type, pretrained_model_weights_path, device)

            if fine_tune_encoder:
                # If we fine-tune the encoder we learn a representation right away and save the results
                cv_splitter, n_splits = get_participant_cv_splitter(
                    groups_train,
                    min_participants_for_kfold=min_participants_for_kfold,
                    k=k_folds
                )

                # LP+FT pipeline
                # Step 1: First train the head
                # Here we first freeze the backbone and train first classifier head only
                # We tune a total of 25 epochs, as this is the same as we used for our supervised baselines

                # Here we use the option for the classifier head option to vary the head
                fine_tune_model = FineTunedCNNNet(backbone=pretrained_model, classifier_head=classifier_head).to(device)

                fine_tune_model = freeze_and_unfreeze_encoder(fine_tune_model, freeze=True)

                # This fine-tunes the classifier head first (with the frozen backbone)
                fine_tuned_results = run_linear_classifier_with_cv_and_test(
                    X_train, y_train, groups_train, X_test, y_test, fine_tune_model,
                    feature_names=None, cv_splitter=cv_splitter, device=device, classifier_epochs=10,
                    classifier_batch_size=classifier_batch_size,
                    standardize=False, seed=42
                )

                # Get the updated model (this has the head already fine-tuned)
                fine_tune_model = copy.deepcopy(fine_tuned_results["model"])

                # Now unfreeze the weights so the encoder and the head will be trained
                fine_tune_model = freeze_and_unfreeze_encoder(fine_tune_model, freeze=False)

                fine_tuned_results = run_linear_classifier_with_cv_and_test(
                    X_train, y_train, groups_train, X_test, y_test, fine_tune_model,
                    feature_names=None, cv_splitter=cv_splitter, device=device, classifier_epochs=15,
                    classifier_batch_size=classifier_batch_size,
                    standardize=False, seed=42
                )

                print(f"We finished the fine-tuning stage (LP + FT)")
                print(fine_tuned_results["test_metrics"])

                with open(os.path.join(transfer_learning_results_save_path, "test_results_lp_ft.json"), "w") as f:
                    json.dump(fine_tuned_results, f, indent=2, default=str)

            else:
                # Here we use the encoder and learn a logistic regression based on the encoder
                # and see what generalizes better
                pretrained_model.eval().to(device)
                X_train = torch.from_numpy(X_train).to(device).permute(0, 2, 1)
                X_test = torch.from_numpy(X_test).to(device).permute(0, 2, 1)

                train_repr = get_representations_in_batches(pretrained_model, X_train, batch_size)
                test_repr = get_representations_in_batches(pretrained_model, X_test, batch_size)

                set_seed(seed)

                # Create feature names for representations (just numbered features)
                feature_names = [f"repr_{i}" for i in range(train_repr.shape[1])]

                if classifier_head in ["logistic_regression", "random_forest", "xgboost"]:
                    # IMPORTANT: The encoder already normalizes the features, so no need to standardize again
                    # Verbose option:
                    results = run_logistic_regression_with_gridsearch(
                        train_repr, y_train, groups_train,
                        test_repr, y_test, feature_names, cv_splitter, standardize=True, seed=seed,
                        scoring_metric=scoring_metric, classifier_model=classifier_head
                    )

                    # Log metrics locally
                    print(f"Best CV AUROC: {results['best_cv_score'] if cv_splitter is not None else 0:.4f}")
                    print(f"Test metrics - Accuracy: {results['test_metrics']['accuracy']:.4f}, "
                          f"AUROC: {results['test_metrics']['auroc']:.4f}, F1: {results['test_metrics']['f1']:.4f}, "
                          f"PR-AUC: {results['test_metrics']['pr_auc']:.4f}")
                    print(f"Best hyperparameters: {results['best_params']}")

                else:
                    results = run_mlp_with_cv_and_test(
                        train_repr, y_train, groups_train,
                        test_repr, y_test, feature_names, cv_splitter,
                        device, classifier_epochs, classifier_batch_size, classifier_lr, standardize=True,
                        seed=seed
                    )

                    # Log metrics locally
                    print(f"Best CV AUROC: {results['best_cv_score']:.4f}")
                    print(f"Test metrics - Accuracy: {results['test_metrics']['accuracy']:.4f}, "
                          f"AUROC: {results['test_metrics']['auroc']:.4f}, F1: {results['test_metrics']['f1']:.4f}, "
                          f"PR-AUC: {results['test_metrics']['pr_auc']:.4f}")
                    print(f"Best hyperparameters: {results['best_params']}")

                with open(os.path.join(transfer_learning_results_save_path, "test_results.json"), "w") as f:
                    json.dump(results, f, indent=2, default=str)

        else:
            # Load parameters from saved results
            saved_results = os.path.join(model_save_path, f"{model_type}.pt")
            model = load_and_return_saved_model(model_type, saved_results, device)

            test_ds = PhysiologicalDataset(X_test, y_test)
            test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                pin_memory=pin_memory
            )
            loss_fn = torch.nn.BCEWithLogitsLoss()

            loss, acc, auroc, prauc, f1 = test(
                model, test_loader, device,
                threshold=0.5, loss_fn=loss_fn,
            )

            print(f"Test acc: {acc:.4f}, AUROC: {auroc:.4f}, F1: {f1:.4f}, PR-AUC: {prauc:.4f}")

    # Then test the performance
    if zero_shot_evaluation:
        test_ds = PhysiologicalDataset(X_zero_shot, y_zero_shot)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
            pin_memory=pin_memory
        )
        loss, acc, auroc, prauc, f1 = test(
            model, test_loader, device,
            threshold=0.5, loss_fn=loss_fn,
        )

        zero_shot_results = {"zero_shot_accuracy": acc,
                             "zero_shot_roc_auc": auroc,
                             "zero_shot_pr_auc": prauc,
                             }
        print(f"\n=== Zero-shot Test Set Results ===")
        print(zero_shot_results)

        # Save the results then:
        # Save results
        with open(os.path.join(zero_shot_results_path, "zero_shot_results.json"), 'w') as f:
            json.dump(zero_shot_results, f, indent=2, default=str)

    # cleanup
    for _ in range(3): gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train ECG classifier")
    parser.add_argument("--fs", default=1000, type=int, help="What sample frequency used for training")
    parser.add_argument("--dataset", choices=("stressid", "wesad", "ours"), default="ours", type=str)
    parser.add_argument("--model_type",
                        choices=["cnn", "tcn", "transformer", "deep_ecg_net", "patchtst", "moment"], default="cnn")
    parser.add_argument("--label_fraction", type=float, default=1.0,
                        help="Percent of labeled participants in the training stage.")
    parser.add_argument("--window_size", type=int, default=10,
                           help="Window size in seconds")
    parser.add_argument("--step_size", type=int, default=5,
                           help="Step size in seconds for sliding window")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_retraining", action="store_true")
    parser.add_argument("--disable_hyperparameter_tuning", action="store_true",
                        help="If set, we do not use hyperparameter_tuning and we use default lr and dropout rates")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="If disable_hyperparameter_tuning is set, what dropout rate to use. Otherwise it gets tuned")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="If disable_hyperparameter_tuning is set, what lr to use. Otherwise it gets tuned")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for CV")
    parser.add_argument("--min_participants_for_kfold", type=int, default=5,
                        help="Minimum participants needed for k-fold (otherwise use Leave one participant out)")
    parser.add_argument("--scoring_metric", type=str, default="roc_auc",
                         choices=["roc_auc", "average_precision", "f1", "balanced_accuracy"],
                         help="Scoring metric for cross-validation hyperparameter selection")

    # S3 Module architecture
    parser.add_argument("--use_s3_layers", action="store_true",
                                  help="If set, we use the S3 layer")
    parser.add_argument("--initial_num_segments", type=int, default=2)
    parser.add_argument("--num_s3_layers", type=int, default=2)
    parser.add_argument("--segment_multiplier", type=int, default=2)

    # Zero-shot evaluation
    parser.add_argument("--zero_shot_evaluation", action="store_true",
                                 help="If set, we do downstream zero-shot evaluation.")
    parser.add_argument("--zero_shot_dataset", type=str,
                                 choices=("stressid", "wesad"), default="wesad")

    # Pretrained encoder + Fine Tune
    parser.add_argument("--use_pretrained_encoder",action="store_true",
                                  help="If set, we use the pre-trained encoder from our dataset")
    parser.add_argument("--fine_tune_encoder", action="store_true",
                                  help="If set, we fine-tune also the encoder and not only the logistic regression.")
    parser.add_argument("--classifier_head", default="logistic_regression",
                        choices=("mlp", "logistic_regression"),
                        help="If fine_tune_encoder is set, what classifier head model to use. "
                             "Important: By default the CNN uses not a logistic regression but MLP layer. "
                             "One needs to use the use simple layer head argument")
    parser.add_argument("--classifier_epochs", type=int, default=25,
                                 help="Number of epochs for MLP classifier training or fine-tuning of the encoder and TC head")
    parser.add_argument("--classifier_lr", type=float, default=1e-4,
                                 help="Learning rate for MLP classifier")
    parser.add_argument("--classifier_batch_size", type=int, default=32,
                                 help="Batch size for fine-tuning")

    args = parser.parse_args()

    main(**vars(args))