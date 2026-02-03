#!/usr/bin/env python
import os
from datetime import datetime
import json
import argparse
import logging
import gc
import uuid

import numpy as np
import torch

from utils.torch_utilities import (
    load_processed_data,
    return_track_data_file,
    split_indices_by_participant_groups,
    set_seed,
    create_directory,
    get_participant_cv_splitter,
    run_logistic_regression_with_gridsearch,
    run_logistic_regression_with_gridsearch_verbose,
    run_mlp_with_cv_and_test
)

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH, RESULTS_PATH

from models.infots import (
    InfoTS,
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
        infots_epochs: int,
        infots_lr: float,
        infots_meta_lr: float,
        infots_batch_size: int,
        infots_output_dims: int,
        infots_hidden_dims: int,
        infots_depth: int,
        infots_max_train_length: int,
        infots_dropout: float,
        infots_aug_p1: float,
        infots_aug_p2: float,
        classifier_model: str,
        classifier_epochs: int,
        classifier_lr: float,
        classifier_batch_size: int,
        label_fraction: float,
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
        verbose: bool = False,
        scoring_metric: str = "roc_auc",
        optimize_hyperparameters: bool = False,
        use_s3_layers: bool = False,
        num_s3_layers: int = 2,
        initial_num_segments: int = 2,
        shuffle_vector_dim: int = 1,
        segment_multiplier: int = 2,
):
    # ── Step 0: Setup ────────────────────────────────────────────────────────────
    set_seed(seed)

    # device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(f"cuda:{gpu}")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.backends.cuda.matmul.allow_tf32 = True

    elif torch.backends.mps.is_available():
        if use_s3_layers:
            # mps has some issues with long Conv1 layer somehow
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logging.basicConfig(level=logging.INFO)
    model_name = "InfoTS_S3" if use_s3_layers else "InfoTS"

    print(f"Starting {model_name} training with CV {classifier_model}, seed={seed}, {label_fraction}")

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)
    create_directory(RESULTS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"

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
    # Makes it more optimized
    X = np.ascontiguousarray(X)  # Ensure contiguous memory layout

    y = y.astype(np.float32)
    n_features = X.shape[2]

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

    # ── Step 2: InfoTS Pretraining ──────────────────────────────────────────────
    torch.cuda.empty_cache()
    set_seed(seed)

    if optimize_hyperparameters:
        model_save_name = "infots_model_hyperparameter.pt"
    else:
        model_save_name = "infots_model.pt"

    # Check if we have a locally saved model and no forced retraining
    if os.path.exists(os.path.join(model_save_path, model_save_name)) and not force_retraining and not optimize_hyperparameters:
        print("We found a pretrained model. Load the pretrained weights")
        model_path = os.path.join(model_save_path, model_save_name)

        infots = InfoTS(
            input_dims=n_features,
            output_dims=infots_output_dims,
            hidden_dims=infots_hidden_dims,
            depth=infots_depth,
            device=device,
            lr=infots_lr,
            meta_lr=infots_meta_lr,
            batch_size=infots_batch_size,
            max_train_length=infots_max_train_length,
            dropout=infots_dropout,
            aug_p1=infots_aug_p1,
            aug_p2=infots_aug_p2,
            use_s3_layers=use_s3_layers,
            num_s3_layers=num_s3_layers,
            initial_num_segments=initial_num_segments,
            shuffle_vector_dim=shuffle_vector_dim,
            segment_multiplier=segment_multiplier,
        )
        infots.net = infots._net = torch.load(model_path, map_location=device, weights_only=False)

    else:
        print("No cached encoder; training InfoTS from scratch")

        if optimize_hyperparameters:
            # Generate random id for experiment tracking
            run_id = str(uuid.uuid4())
            hyperparameter_file_name = os.path.join(
                results_save_path, "hyperparameter_tuning_results.json"
            )

        # Load data for encoder pretraining
        X_train_encoder = X[train_idx_encoder].astype(np.float32)

        infots = InfoTS(
            input_dims=n_features,
            output_dims=infots_output_dims,
            hidden_dims=infots_hidden_dims,
            depth=infots_depth,
            device=device,
            lr=infots_lr,
            meta_lr=infots_meta_lr,
            batch_size=infots_batch_size,
            max_train_length=infots_max_train_length,
            dropout=infots_dropout,
            aug_p1=infots_aug_p1,
            aug_p2=infots_aug_p2,
            use_s3_layers=use_s3_layers,
            num_s3_layers=num_s3_layers,
            initial_num_segments=initial_num_segments,
            shuffle_vector_dim=shuffle_vector_dim,
            segment_multiplier=segment_multiplier,
            verbose=True,
        )

        print(f"Created {model_name} model on device: {next(infots.net.parameters()).device}")

        # Train InfoTS - Note: InfoTS uses unsupervised meta-learning, so we don't provide labels
        loss_log = infots.fit(
            X_train_encoder,
            n_epochs=infots_epochs,
            verbose=True,
            supervised_meta=False,
            batch_size=infots_batch_size,
            results_save_path=results_save_path,
            train_labels=None# InfoTS uses unsupervised meta-learning by default
        )

        # Save model
        saved_results = os.path.join(model_save_path, model_save_name)
        torch.save(infots.net, saved_results)

    # ── Step 3: Extract Representations ─────────────────────────────────────────
    print("\nExtracting representations...")

    # Get InfoTS embeddings
    train_repr = infots.encode(X[train_idx].astype(np.float32))
    test_repr = infots.encode(X[test_idx].astype(np.float32))

    # filter to binary downstream samples
    train_repr = train_repr[downstream_mask["train"]]
    y_train = y[train_idx][downstream_mask["train"]]
    groups_train = groups[train_idx][downstream_mask["train"]]

    test_repr = test_repr[downstream_mask["test"]]
    y_test = y[test_idx][downstream_mask["test"]]

    print(f"Extracted InfoTS representations: train_repr shape={train_repr.shape}")

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
        print(f"Best CV AUROC: {results['best_cv_score'] if cv_splitter is not None else 0:.4f}")
        print(f"Test metrics - Accuracy: {results['test_metrics']['accuracy']:.4f}, "
              f"AUROC: {results['test_metrics']['auroc']:.4f}, F1: {results['test_metrics']['f1']:.4f}, "
              f"PR-AUC: {results['test_metrics']['pr_auc']:.4f}")
        print(f"Best hyperparameters: {results['best_params']}")

    else:
        results = run_mlp_with_cv_and_test(
            train_repr, y_train, groups_train,
            test_repr, y_test, feature_names, cv_splitter,
            device, classifier_epochs, classifier_batch_size, classifier_lr, False, seed
        )

        # Log metrics locally
        print(f"Best CV AUROC: {results['best_cv_score']:.4f}")
        print(f"Test metrics - Accuracy: {results['test_metrics']['accuracy']:.4f}, "
              f"AUROC: {results['test_metrics']['auroc']:.4f}, F1: {results['test_metrics']['f1']:.4f}, "
              f"PR-AUC: {results['test_metrics']['pr_auc']:.4f}")
        print(f"Best hyperparameters: {results['best_params']}")

    # ── Step 6: Save Results ────────────────────────────────────────────────────

    # Track hyperparameter results:
    if optimize_hyperparameters:
        hyperparameter_save_file = return_track_data_file(hyperparameter_file_name)

        hyperparameter_save_file["runs"][run_id] = {
            "hyperparameters": infots.to_config_dict(),
            "training_loss (last 5 scores)": loss_log[-5:],
            "CV score": results['best_cv_score'],
            "Test_metrics": results["test_metrics"],
            "timestamp": datetime.now().isoformat(),
        }

        with open(hyperparameter_file_name, "w") as f:
            json.dump(hyperparameter_save_file, f, indent=4)

    with open(os.path.join(results_save_path, "test_results.json" ), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Log additional parameters locally
    print(f"Additional parameters - Classifier: {classifier_model}, Label fraction: {label_fraction}, "
          f"Seed: {seed}, K-folds: {k_folds}, CV splits: {n_splits}, "
          f"Pretrain all conditions: {pretrain_all_conditions}, Train ratio encoder: {train_ratio_encoder}, "
          f"Use S3 layers: {use_s3_layers}, Num S3 layers: {num_s3_layers}, "
          f"Initial num segments: {initial_num_segments}, Segment multiplier: {segment_multiplier}")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"=== Done! Test Acc: {results['test_metrics']['accuracy']:.4f}, "
          f"AUROC: {results['test_metrics']['auroc']:.4f}, "
          f"PR-AUC: {results['test_metrics']['pr_auc']:.4f}, "
          f"F1: {results['test_metrics']['f1']:.4f} ===")
    print("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="InfoTS Training Pipeline with CV and Logistic Regression",
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
                            help="If set to 0.75, it will result in 60/20/20 split and "
                                 "have a validation set for InfoTS,"
                                 "Alternatively, set to 1.0 to train on all unlabelled training instances.")

    # ══════════════════════════════════════════════════════════════════════════════
    # InfoTS Encoder Training
    # ══════════════════════════════════════════════════════════════════════════════
    infots_group = parser.add_argument_group('InfoTS Encoder Training')
    infots_group.add_argument("--infots_epochs", type=int, default=40,
                             help="Number of epochs for InfoTS pretraining")
    infots_group.add_argument("--infots_lr", type=float, default=0.001,
                             help="Learning rate for InfoTS encoder training")
    infots_group.add_argument("--infots_meta_lr", type=float, default=0.01,
                             help="Learning rate for InfoTS meta-learner")
    infots_group.add_argument("--infots_batch_size", type=int, default=32,
                             help="Batch size for InfoTS training")
    infots_group.add_argument("--infots_output_dims", type=int, default=320,
                             help="InfoTS representation dimension")
    infots_group.add_argument("--infots_hidden_dims", type=int, default=64,
                             help="InfoTS hidden dimension")
    infots_group.add_argument("--infots_depth", type=int, default=10,
                             help="InfoTS depth (# dilated conv blocks)")
    infots_group.add_argument("--infots_max_train_length", type=int, default=10_000,
                             help="InfoTS max training length")
    infots_group.add_argument("--infots_dropout", type=float, default=0.1,
                             help="InfoTS dropout rate")
    infots_group.add_argument("--infots_aug_p1", type=float, default=0.2,
                             help="InfoTS augmentation probability 1")
    infots_group.add_argument("--infots_aug_p2", type=float, default=0.0,
                             help="InfoTS augmentation probability 2")

    # ══════════════════════════════════════════════════════════════════════════════
    # S3 Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    s3_group = parser.add_argument_group('S3 Configuration')
    # We use the default hyperparameters that were tested and performed well in the S3 paper
    s3_group.add_argument("--use_s3_layers", action="store_true",
                         help="Use S3 layers in the encoder")
    s3_group.add_argument("--num_s3_layers", type=int, default=2,
                         help="Number of S3 layers")
    s3_group.add_argument("--initial_num_segments", type=int, default=2,
                         help="Initial number of segments for S3")
    s3_group.add_argument("--shuffle_vector_dim", type=int, default=1,
                         help="Shuffle vector dimension for S3")
    s3_group.add_argument("--segment_multiplier", type=int, default=1,
                         help="Segment multiplier for S3")

    # ══════════════════════════════════════════════════════════════════════════════
    # Downstream Classifier Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    classifier_group = parser.add_argument_group('Downstream Classifier')
    classifier_group.add_argument("--classifier_model", type=str, default="logistic_regression",
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

    # ══════════════════════════════════════════════════════════════════════════════
    # Hyperparameter Optimization Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    hp_group = parser.add_argument_group('Hyperparameter Optimization')
    hp_group.add_argument("--optimize_hyperparameters", action="store_true",
                         help="Enable hyperparameter optimization for InfoTS augmentation parameters."
                              "Due to high computational costs, we resort to grid-search based tuning, selected "
                              "via the command-line arguments, for 20 epochs and 10% label fraction.")

    # Parse arguments and run main function
    args = parser.parse_args()

    # Important:
    args.pretrain_all_conditions = True

    main(**vars(args))