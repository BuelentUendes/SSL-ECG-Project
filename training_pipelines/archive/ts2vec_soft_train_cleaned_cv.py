#!/usr/bin/env python
import os
import json
import argparse
import logging
import tempfile
import gc

import numpy as np
import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch

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

from models.ts2vec_soft import (
    TS2Vec_soft,
    build_fingerprint,
    search_encoder_fp,
    compute_soft_labels,
)


def main(
        mlflow_tracking_uri: str,
        fs: str,
        window_size: int,
        step_size: int,
        gpu: int,
        seed: int,
        force_retraining: bool,
        pretrain_all_conditions: bool,
        train_ratio_encoder: float,
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
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(f"SoftTS2Vec with CV {classifier_model}")

    # Start top‑level run
    run = mlflow.start_run(run_name=f"soft_ts2vec_cv_{classifier_model}_{seed}_lf_{label_fraction}")
    run_id = run.info.run_id
    logging.info(f"MLflow run_id: {run_id}")
    print(f"Using device: {device}")

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)
    create_directory(RESULTS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"

    model_save_path = os.path.join(
        SAVED_MODELS_PATH, "ECG", str(fs), "TS2Vec_soft", pretrain_data, f"{seed}", f"{window_size}", f"{step_size}",
        str(train_ratio_encoder)
    )
    results_save_path = os.path.join(
        RESULTS_PATH, "ECG", "TS2Vec_soft", classifier_model, f"{seed}", f"{label_fraction}", f"{window_size}",
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
    assert len(np.unique(groups_train_idx_encoder)) + len(np.unique(groups_val_idx_encoder)) + len(np.unique(groups[test_idx])) == 127, \
        "Something went wrong with the participant split!"

    print(f"Labelled windows for training classifier: train {len(train_idx)}, test {len(test_idx)}")

    # Keep binary‐task mask for later
    downstream_mask = {
        "train": np.isin(y[train_idx], [0, 1]),
        "test": np.isin(y[test_idx], [0, 1]),
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
        "train_ratio_encoder": train_ratio_encoder,
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

        # Load data for encoder pretraining
        X_train_encoder = X[train_idx_encoder].astype(np.float32)

        # Compute soft labels
        print("Computing soft labels...")
        soft_labels = compute_soft_labels(
            X_train_encoder, ts2vec_tau_inst, ts2vec_alpha,
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
            X_train_encoder, soft_labels,
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
    test_repr = ts2vec_soft.encode(X[test_idx].astype(np.float32), encoding_window="full_series")

    # filter to binary downstream samples
    train_repr = train_repr[downstream_mask["train"]]
    y_train = y[train_idx][downstream_mask["train"]]
    groups_train = groups[train_idx][downstream_mask["train"]]

    test_repr = test_repr[downstream_mask["test"]]
    y_test = y[test_idx][downstream_mask["test"]]

    print(f"Extracted Soft TS2Vec representations: train_repr shape={train_repr.shape}")

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

        # Log metrics
        mlflow.log_metrics({
            "best_cv_auroc": results['best_cv_score'] if cv_splitter is not None else 0,
            "test_accuracy": results['test_metrics']['accuracy'],
            "test_auroc": results['test_metrics']['auroc'],
            "test_f1": results['test_metrics']['f1'],
            "test_pr_auc": results['test_metrics']['pr_auc'],
        })

        mlflow.log_params(results['best_params'])

    else:
        results = run_mlp_with_cv_and_test(
            train_repr, y_train, groups_train,
            test_repr, y_test, feature_names, cv_splitter,
            device, classifier_epochs, classifier_batch_size, classifier_lr, False, seed
        )

        # Log metrics
        mlflow.log_metrics({
            "best_cv_auroc": results['best_cv_score'],
            "test_accuracy": results['test_metrics']['accuracy'],
            "test_auroc": results['test_metrics']['auroc'],
            "test_f1": results['test_metrics']['f1'],
            "test_pr_auc": results['test_metrics']['pr_auc'],
        })

        mlflow.log_params(results['best_params'])

    # ── Step 6: Save Results ────────────────────────────────────────────────────
    with open(os.path.join(results_save_path, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Log additional parameters
    mlflow.log_params({
        "classifier_model": classifier_model,
        "label_fraction": label_fraction,
        "seed": seed,
        "k_folds": k_folds,
        "n_cv_splits": n_splits,
        "pretrain_all_conditions": pretrain_all_conditions,
        "train_ratio_encoder": train_ratio_encoder,
    })

    # ── Cleanup ────────────────────────────────────────────────────────────────
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"=== Done! Test Acc: {results['test_metrics']['accuracy']:.4f}, "
          f"AUROC: {results['test_metrics']['auroc']:.4f}, "
          f"PR-AUC: {results['test_metrics']['pr_auc']:.4f}, "
          f"F1: {results['test_metrics']['f1']:.4f} ===")
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Soft TS2Vec Training Pipeline with CV and Logistic Regression",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ══════════════════════════════════════════════════════════════════════════════
    # General Setup
    # ══════════════════════════════════════════════════════════════════════════════
    general_group = parser.add_argument_group('General Setup')
    general_group.add_argument("--mlflow_tracking_uri",
                              default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
                              help="MLflow tracking URI for experiment logging")
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
    data_group.add_argument("--train_ratio_encoder", default=0.75, type=float,
                            help="If set to 0.75, it will result in 60/20/20 split and have a validation set for TS2Vec,"
                                 "Alternatively, set to 1.0 to train on all unlabelled training instances.")

    # ══════════════════════════════════════════════════════════════════════════════
    # Soft TS2Vec Encoder Training
    # ══════════════════════════════════════════════════════════════════════════════
    ts2vec_group = parser.add_argument_group('Soft TS2Vec Encoder Training')
    ts2vec_group.add_argument("--ts2vec_epochs", type=int, default=50,
                             help="Number of epochs for Soft TS2Vec pretraining")
    ts2vec_group.add_argument("--ts2vec_lr", type=float, default=0.001,
                             help="Learning rate for Soft TS2Vec training")
    ts2vec_group.add_argument("--ts2vec_batch_size", type=int, default=8,
                             help="Batch size for Soft TS2Vec training")
    ts2vec_group.add_argument("--ts2vec_output_dims", type=int, default=320,
                             help="Soft TS2Vec representation dimension (Co)")
    ts2vec_group.add_argument("--ts2vec_hidden_dims", type=int, default=64,
                             help="Soft TS2Vec hidden dimension (Ch)")
    ts2vec_group.add_argument("--ts2vec_depth", type=int, default=10,
                             help="Soft TS2Vec depth (# dilated conv blocks)")
    ts2vec_group.add_argument("--ts2vec_max_train_length", type=int, default=5000,
                             help="Soft TS2Vec max training length")
    ts2vec_group.add_argument("--ts2vec_temporal_unit", type=int, default=3,
                             help="Soft TS2Vec temporal unit for hierarchical pooling")

    # Soft contrastive learning hyperparameters
    ts2vec_soft_group = parser.add_argument_group('Soft Contrastive Learning Parameters')
    ts2vec_soft_group.add_argument("--ts2vec_dist_type", type=str, default="COS",
                                  choices=["DTW", "EUC", "COS", "TAM", "GAK"],
                                  help="Distance metric for soft labels")
    ts2vec_soft_group.add_argument("--ts2vec_tau_inst", type=float, default=50.0,
                                  help="Temperature parameter tau_inst for soft instance CL")
    ts2vec_soft_group.add_argument("--ts2vec_tau_temp", type=float, default=2.5,
                                  help="Temperature parameter tau_temp for soft temporal CL")
    ts2vec_soft_group.add_argument("--ts2vec_alpha", type=float, default=0.5,
                                  help="Alpha for densification of soft labels")
    ts2vec_soft_group.add_argument("--ts2vec_lambda", type=float, default=0.5,
                                  help="Weight lambda for instance vs temporal CL")

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

    # Parse arguments and run main function
    args = parser.parse_args()

    # Important:
    args.pretrain_all_conditions = True

    main(**vars(args))