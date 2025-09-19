#!/usr/bin/env python
import os
import time
import json
import argparse
import logging
import gc

import numpy as np
import torch
import torch.optim as optim

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant_groups,
    set_seed,
    create_directory,
    get_participant_cv_splitter,
    run_logistic_regression_with_gridsearch,
    run_logistic_regression_with_gridsearch_verbose,
    run_mlp_with_cv_and_test
)

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH, RESULTS_PATH

from models.simclr import (
    get_simclr_model,
    NTXentLoss,
    simclr_data_loaders,
    pretrain_one_epoch,
    encode_representations,
    build_simclr_fingerprint,
)


def main(
        fs: str,
        window_size: int,
        step_size: int,
        gpu: int,
        seed: int,
        force_retraining: bool,
        pretrain_all_conditions: bool,
        train_ratio_encoder: float,
        epochs: int,
        lr: float,
        batch_size: int,
        temperature: float,
        use_tstcc_encoder: bool,
        use_s3_layers: bool,
        initial_num_segments: int,
        num_s3_layers: int,
        segment_multiplier: int,
        classifier_model: str,
        classifier_epochs: int,
        classifier_lr: float,
        classifier_batch_size: int,
        label_fraction: float,
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
        verbose: bool = False,
        scoring_metric: str = "roc_auc",
):
    # ── Step 0: Setup ────────────────────────────────────────────────────────────
    set_seed(seed)

    # device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(f"cuda:{gpu}")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting SimCLR training with CV {classifier_model}, seed {seed}, label_fraction {label_fraction}")
    print(f"Using device: {device}")

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)
    create_directory(RESULTS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"

    if use_s3_layers:
        if use_tstcc_encoder:
            model_name = "SimCLR_S3_TSTCC_Encoder"
        else:
            model_name = "SimCLR_S3"
    else:
        if use_tstcc_encoder:
            model_name = "SimCLR_TSTCC_Encoder"
        else:
            model_name = "SimCLR"

    model_save_path = os.path.join(
        SAVED_MODELS_PATH, "ECG", str(fs), model_name, pretrain_data, f"{seed}", f"{window_size}", f"{step_size}",
        str(train_ratio_encoder)
    )
    results_save_path = os.path.join(
        RESULTS_PATH, "ECG", model_name, classifier_model, f"{seed}", f"{label_fraction}", f"{window_size}",
        f"{step_size}", str(train_ratio_encoder)
    )

    create_directory(model_save_path)
    create_directory(results_save_path)

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
    window_data_path = os.path.join(
        DATA_PATH, "interim", "ECG", str(fs), str(window_size), str(step_size), 'windowed_data.h5'
    )

    X, y, groups = load_processed_data(window_data_path, label_map=label_map)
    y = y.astype(np.float32)
    win_len = X.shape[1]

    # Split by participant to get train/test split
    train_idx, train_p, all_train_p, all_train_idx, test_idx, test_p = split_indices_by_participant_groups(
        groups,
        train_ratio=0.8,
        label_fraction=label_fraction,
        seed=seed,
        return_all_train_p=True
    )
    
    # Now we can split the training data for encoder training
    groups_train_all_encoder = groups[all_train_idx]

    # Split for encoder training (we don't need labels for SSL, so label fraction is 1.0)
    train_idx_encoder, train_p_rep, val_idx_encoder, val_p = split_indices_by_participant_groups(
        groups_train_all_encoder,
        train_ratio=train_ratio_encoder,  # This will give a split of 60/20/20 when 0.75
        label_fraction=1.0,  # We will discard anyways all labels
        seed=seed,
        return_all_train_p=False,
    )

    # Map back to original indices
    groups_train_idx_encoder = groups_train_all_encoder[train_idx_encoder]  # 60% of original data
    groups_val_idx_encoder = groups_train_all_encoder[val_idx_encoder]  # 20% of original data

    # Test that we have all 127 participants moved in one of the categories
    assert (len(np.unique(groups_train_idx_encoder)) + len(np.unique(groups_val_idx_encoder)) +
            len(np.unique(groups[test_idx])) == 127), \
        "Something went wrong with the participant split!"

    print(f"Labelled windows for training classifier: train {len(train_idx)}, test {len(test_idx)}")

    # Keep binary‐task mask for later
    downstream_mask = {
        "train": np.isin(y[train_idx], [0, 1]),
        "test": np.isin(y[test_idx], [0, 1]),
    }

    # ── Step 2: SimCLR Pretraining ──────────────────────────────────────────────
    torch.cuda.empty_cache()
    set_seed(seed)

    # Build fingerprint and MLflow lookup
    fp = {
        "model_name": model_name,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "temperature": temperature,
        "window_len": win_len,
        "train_ratio_encoder": train_ratio_encoder,
    }
    fp = build_simclr_fingerprint(fp)

    model = get_simclr_model(
        window=win_len,
        device=device,
        use_s3_layers=use_s3_layers,
        num_s3_layers=num_s3_layers,
        initial_num_segments=initial_num_segments,
        segment_multiplier=segment_multiplier,
        use_tstcc_encoder=use_tstcc_encoder,
    )

    # Check for local pretrained model
    if os.path.exists(os.path.join(model_save_path, "simclr_encoder.pt")) and not force_retraining:
        print("Found pretrained model. Loading weights...")
        model_path = os.path.join(model_save_path, "simclr_encoder.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))

    else:
        print(f"No cached encoder; training {model_name} from scratch")

        # Load data for encoder pretraining
        X_train_encoder = X[train_idx_encoder].astype(np.float32)
        X_val_encoder = X[val_idx_encoder].astype(np.float32)

        loss_fn = NTXentLoss(batch_size, temperature)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        tr_dl, _ = simclr_data_loaders(X_train_encoder, X_val_encoder, batch_size)

        print(f"Created {model_name} model on device: {next(model.parameters()).device}")

        logging.info(f"Training parameters: {fp}")

        epoch_peak_memory = {}
        epoch_runtimes = {}

        # Start recording memory snapshot history
        if torch.cuda.is_available():
            # Add this to your training loop for systematic reporting
            torch.cuda.reset_peak_memory_stats()

        # Train SimCLR
        for ep in range(1, epochs + 1):
            epoch_start_time = time.time()
            print(f"Please wait: Run epoch: {ep}")
            tr_loss = pretrain_one_epoch(model, tr_dl, loss_fn, opt, device)
            # Calculate epoch runtime
            epoch_runtime = time.time() - epoch_start_time
            epoch_runtimes[f"{ep}"] = epoch_runtime

            # Log memory usage for CUDA
            if torch.cuda.is_available():
                epoch_peak_memory[f"{ep}"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"Peak memory allocated during epoch {ep}: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")

            logging.info(f"SSL train loss: {tr_loss}")
            print(f"Epoch {ep}/{epochs}: loss={tr_loss:.4f}")

        with open(os.path.join(results_save_path, "runtime_per_epoch.json"), "w") as f:
            json.dump(epoch_runtimes, f, indent=2)

        if torch.cuda.is_available():
            with open(os.path.join(results_save_path, "peak_memory_consumption_epochs.json"), "w") as f:
                json.dump(epoch_peak_memory, f, indent=2)

        # Save model locally
        saved_results = os.path.join(model_save_path, "simclr_encoder.pt")
        torch.save(model.state_dict(), saved_results)

        logging.info("Encoder training complete")

    # ── Step 3: Extract Representations ─────────────────────────────────────────
    print("\nExtracting representations...")

    # Get SimCLR embeddings
    with torch.no_grad():
        train_repr = encode_representations(
            model, X[train_idx].astype(np.float32), batch_size, device)
        test_repr = encode_representations(
            model, X[test_idx].astype(np.float32), batch_size, device)

    # filter to binary downstream samples
    train_repr = train_repr[downstream_mask["train"]]
    y_train = y[train_idx][downstream_mask["train"]]
    groups_train = groups[train_idx][downstream_mask["train"]]

    test_repr = test_repr[downstream_mask["test"]]
    y_test = y[test_idx][downstream_mask["test"]]

    print(f"Extracted SimCLR representations: train_repr shape={train_repr.shape}")

    # ── Step 4: Set up Cross-Validation Splitter ───────────────────────────────
    cv_splitter, n_splits = get_participant_cv_splitter(
        groups_train,
        min_participants_for_kfold=min_participants_for_kfold,
        k=k_folds
    )

    # ── Step 5: Run CV with Logistic Regression or MLP ─────────────────────────────────
    set_seed(seed)

    # Create feature names for representations (just numbered features)
    feature_names = [f"repr_{i}" for i in range(train_repr.shape[1])]

    if classifier_model in ["logistic_regression", "random_forest", "xgboost"]:
        # IMPORTANT: The encoder already normalizes the features, so no need to standardize again
        # Verbose option:
        if verbose:
            results = run_logistic_regression_with_gridsearch_verbose(
                train_repr, y_train, groups_train, test_repr, y_test,
                feature_names, cv_splitter, False, seed
            )
        else:
            results = run_logistic_regression_with_gridsearch(
                train_repr, y_train, groups_train,
                test_repr, y_test, feature_names, cv_splitter, False, seed,
                scoring_metric=scoring_metric, classifier_model=classifier_model
            )

        # Log metrics locally
        logging.info(f"Best CV AUROC: {results['best_cv_score'] if cv_splitter is not None else 0}")
        logging.info(f"Test metrics - Accuracy: {results['test_metrics']['accuracy']}, "
                     f"AUROC: {results['test_metrics']['auroc']}, "
                     f"F1: {results['test_metrics']['f1']}, "
                     f"PR-AUC: {results['test_metrics']['pr_auc']}")
        logging.info(f"Best parameters: {results['best_params']}")

    else:
        results = run_mlp_with_cv_and_test(
            train_repr, y_train, groups_train,
            test_repr, y_test, feature_names, cv_splitter,
            device, classifier_epochs, classifier_batch_size, classifier_lr, False, seed
        )

        # Log metrics locally
        logging.info(f"Best CV AUROC: {results['best_cv_score']}")
        logging.info(f"Test metrics - Accuracy: {results['test_metrics']['accuracy']}, "
                     f"AUROC: {results['test_metrics']['auroc']}, "
                     f"F1: {results['test_metrics']['f1']}, "
                     f"PR-AUC: {results['test_metrics']['pr_auc']}")
        logging.info(f"Best parameters: {results['best_params']}")

    # ── Step 6: Save Results ────────────────────────────────────────────────────
    with open(os.path.join(results_save_path, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Log additional parameters locally
    logging.info(f"Final parameters - Classifier: {classifier_model}, "
                 f"Label fraction: {label_fraction}, "
                 f"Seed: {seed}, K-folds: {k_folds}, "
                 f"CV splits: {n_splits}, "
                 f"Pretrain all: {pretrain_all_conditions}, "
                 f"Train ratio encoder: {train_ratio_encoder}")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"=== Done! Test Acc: {results['test_metrics']['accuracy']:.4f}, "
          f"AUROC: {results['test_metrics']['auroc']:.4f}, "
          f"PR-AUC: {results['test_metrics']['pr_auc']:.4f}, "
          f"F1: {results['test_metrics']['f1']:.4f} ===")
    logging.info("Training pipeline complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SimCLR Training Pipeline with CV and Logistic Regression",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ══════════════════════════════════════════════════════════════════════════════
    # General Setup
    # ══════════════════════════════════════════════════════════════════════════════
    general_group = parser.add_argument_group('General Setup')
    general_group.add_argument("--gpu", type=int, default=0,
                              help="GPU device ID to use")
    general_group.add_argument("--seed", type=int, default=42,
                              help="Random seed for reproducibility")
    general_group.add_argument("--verbose", action="store_true",
                              help="Show verbose output of CV for logistic regression")
    general_group.add_argument("--force_retraining", action="store_true",
                              help="Force retraining even if cached model exists")

    # ══════════════════════════════════════════════════════════════════════════════
    # Data Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument("--fs", default=1000, type=str,
                           help="Sampling frequency used for training")
    data_group.add_argument("--window_size", type=int, default=10,
                           help="Window size in seconds")
    data_group.add_argument("--step_size", type=int, default=5,
                           help="Step size in seconds for sliding window")
    data_group.add_argument("--label_fraction", type=float, default=0.1,
                           help="Fraction of labeled participants to use (0.0-1.0)")
    data_group.add_argument("--pretrain_all_conditions", action="store_true",
                           help="Pretrain on all conditions (not just baseline/mental_stress)")
    data_group.add_argument("--train_ratio_encoder", default=1.0, type=float,
                            help="If set to 0.75, it will result in 60/20/20 split and have a validation set for SimCLR,"
                                 "Alternatively, set to 1.0 to train on all unlabelled training instances.")

    # ══════════════════════════════════════════════════════════════════════════════
    # SimCLR Encoder Training
    # ══════════════════════════════════════════════════════════════════════════════
    simclr_group = parser.add_argument_group('SimCLR Encoder Training')
    simclr_group.add_argument("--epochs", type=int, default=40,
                             help="Number of epochs for SimCLR pretraining")
    simclr_group.add_argument("--lr", type=float, default=1e-3,
                             help="Learning rate for SimCLR training")
    simclr_group.add_argument("--batch_size", type=int, default=256,
                             help="Batch size for SimCLR training")
    simclr_group.add_argument("--temperature", type=float, default=0.2,
                             help="Temperature parameter for contrastive loss")
    simclr_group.add_argument("--use_tstcc_encoder", action="store_true",
                              help="If set, we use the tstcc encoder (differs from the original architecture.")

    # S3 configurations
    simclr_group.add_argument("--use_s3_layers", action="store_true",
                                  help="If set, we use the S3 layer")
    simclr_group.add_argument("--initial_num_segments", type=int, default=2)
    simclr_group.add_argument("--num_s3_layers", type=int, default=2)
    simclr_group.add_argument("--segment_multiplier", type=int, default=1)

    # ══════════════════════════════════════════════════════════════════════════════
    # Downstream Classifier Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    classifier_group = parser.add_argument_group('Downstream Classifier')
    classifier_group.add_argument("--classifier_model", type=str,default="logistic_regression",
                                 choices=("logistic_regression", "mlp", "random_forest", "xgboost"),
                                 help="Type of downstream classifier to use")
    classifier_group.add_argument("--classifier_epochs", type=int, default=25,
                                 help="Number of epochs for MLP classifier training")
    classifier_group.add_argument("--classifier_lr", type=float, default=1e-4,
                                 help="Learning rate for MLP classifier")
    classifier_group.add_argument("--classifier_batch_size", type=int, default=32,
                                 help="Batch size for MLP classifier training")

    # ══════════════════════════════════════════════════════════════════════════════
    # Cross-Validation Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    cv_group = parser.add_argument_group('Cross-Validation')
    cv_group.add_argument("--k_folds", type=int, default=5,
                         help="Number of folds for cross-validation")
    cv_group.add_argument("--min_participants_for_kfold", type=int, default=5,
                         help="Minimum participants needed for k-fold (otherwise use Leave-one-participant-out-CV)")
    cv_group.add_argument("--scoring_metric", type=str, default="roc_auc",
                         choices=["roc_auc", "average_precision", "f1", "balanced_accuracy"],
                         help="Scoring metric for cross-validation hyperparameter selection")

    # Parse arguments and run main function
    args = parser.parse_args()

    # Important:
    args.pretrain_all_conditions = True

    main(**vars(args))