#!/usr/bin/env python
import os
import json
import sys
import argparse
import logging
import tempfile
import gc

import numpy as np
import pandas as pd
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
)


def handle_missing_data(data, drop_values=True, verbose=True):
    """Handle missing values and infinity values in the data."""
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
        was_numpy = True
    else:
        df = data.copy()
        was_numpy = False

    original_data_len = len(df)

    # Identify rows and columns with infinity values
    inf_mask = df.isin([np.inf, -np.inf])
    rows_with_inf = inf_mask.any(axis=1)
    cols_with_inf = inf_mask.any(axis=0)

    if verbose:
        print(f"Rows with infinity values: {rows_with_inf.sum()}")
        print(f"Columns with infinity values:")
        for col in df.columns[cols_with_inf]:
            inf_count = inf_mask[col].sum()
            print(f"  - {col}: {inf_count} infinity values ({(inf_count / len(df)) * 100:.2f}%)")

    # Identify rows and columns with NaN values
    nan_mask = df.isna()
    rows_with_nan = nan_mask.any(axis=1)
    cols_with_nan = nan_mask.any(axis=0)

    if verbose:
        print(f"Rows with NaN values: {rows_with_nan.sum()}")
        print(f"Columns with NaN values:")
        for col in df.columns[cols_with_nan]:
            nan_count = nan_mask[col].sum()
            print(f"  - {col}: {nan_count} NaN values ({(nan_count / len(df)) * 100:.2f}%)")

    if drop_values:
        clean_data = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
        clean_data = clean_data.dropna()

        dropped_percent = ((original_data_len - len(clean_data)) / original_data_len) * 100
        if verbose:
            print(f"Dropping these rows removed {np.round(dropped_percent, 4)}% of the original data")

        if was_numpy:
            return clean_data.values
        else:
            return clean_data
    else:
        if was_numpy:
            return df.values
        else:
            return df


def main(
        mlflow_tracking_uri: str,
        fs: str,
        window_size:int,
        step_size: int,
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
        use_spectral_augmentation: bool,
        freq_mask_ratio_weak: float,
        freq_mask_ratio_strong: float,
        freq_max_seq: int,
        classifier_model: str,
        classifier_epochs: int,
        classifier_lr: float,
        classifier_batch_size: int,
        label_fraction: float,
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
        verbose: bool = False,
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
    mlflow.set_experiment(f"TSTCC with CV {classifier_model}")

    # Start top‑level run
    run = mlflow.start_run(run_name=f"tstcc_cv_{classifier_model}_{seed}_lf_{label_fraction}")
    run_id = run.info.run_id
    logging.info(f"MLflow run_id: {run_id}")
    print(f"Using device: {device}")

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)
    create_directory(RESULTS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"

    model_save_path = os.path.join(
        SAVED_MODELS_PATH, "ECG", str(fs), "TSTCC", pretrain_data, f"{seed}", f"{window_size}", f"{step_size}"
    )
    results_save_path = os.path.join(
        RESULTS_PATH, "ECG", "TSTCC", classifier_model, f"{seed}", f"{label_fraction}", f"{window_size}", f"{step_size}"
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

    # We first get all train idx for the SSL method (label fraction 1.0) as we do not use the labels
    # train_idx_all (represents all training samples as we do not use their labels)
    # Split by participant to get train/test split
    # train_idx to the labeled ones!
    # train_p refers to the labeled training participant!
    # all_train_idx refer to all the training samples (irrespective of labeled or not)
    train_idx, train_p, all_train_p, all_train_idx, test_idx, test_p = split_indices_by_participant_groups(
        groups,
        train_ratio=0.8,
        label_fraction=label_fraction,
        seed=seed,
        return_all_train_p=True
    )
    # Now we can split the
    # This is the dataset we use for training of the encoder!
    groups_train_all_encoder = groups[all_train_idx]

    # Rep is the one that we train the encoder on, for these we do not need the labels, so label fraction is set to 1.0
    train_idx_encoder, train_p_rep, val_idx_encoder, val_p  = split_indices_by_participant_groups(
        groups_train_all_encoder,
        train_ratio=0.75, #This will give a split of 60/20/20
        label_fraction=1.0, # We will discard anyways all labels
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
        cfg = ECGConfig(fs, window_size)
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
        cfg = ECGConfig(fs, window_size)
        cfg.num_epoch = tcc_epochs
        cfg.batch_size = tcc_batch_size

        #Augmentation used
        cfg.augmentation.use_spectral_aug = use_spectral_augmentation

        # For spectral augmentations
        cfg.augmentation.freq_mask_ratio_weak = freq_mask_ratio_weak
        cfg.augmentation.freq_mask_ratio_strong = freq_mask_ratio_strong
        cfg.augmentation.freq_max_seg = freq_max_seq

        cfg.TC.timesteps = tc_timesteps
        cfg.TC.hidden_dim = tc_hidden_dim
        cfg.Context_Cont.temperature = cc_temperature
        cfg.Context_Cont.use_cosine_similarity = cc_use_cosine

        # data loaders
        Xtr = X[train_idx_encoder].astype(np.float32)
        Xva = X[val_idx_encoder].astype(np.float32)
        Xte = X[test_idx].astype(np.float32)
        tr_dl, va_dl, te_dl = data_generator_from_arrays(
            Xtr, y[train_idx_encoder], Xva, y[val_idx_encoder], Xte, y[test_idx],
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
        model_file_name = "tstcc_spectral.pt" if use_spectral_augmentation else "tstcc.pt"
        ckpt = os.path.join(workdir, model_file_name)
        torch.save(
            {"encoder": model.state_dict(),
             "tc_head": tc_head.state_dict()},
            ckpt
        )

        mlflow.log_artifact(ckpt, artifact_path="tstcc_model")

        saved_results = os.path.join(model_save_path, model_file_name)
        torch.save(
            {"encoder": model.state_dict(),
             "tc_head": tc_head.state_dict()},
            model_file_name
        )

    # ── Step 3: Extract Representations ─────────────────────────────────────────
    model.eval()
    tc_head.eval()

    with torch.no_grad():
        train_repr, _ = encode_representations(X[train_idx], y[train_idx],
                                               model, tc_head, tcc_batch_size, device)
        test_repr, _ = encode_representations(X[test_idx], y[test_idx],
                                              model, tc_head, tcc_batch_size, device)

    # filter to binary downstream samples
    train_repr = train_repr[downstream_mask["train"]]
    y_train = y[train_idx][downstream_mask["train"]]
    groups_train = groups[train_idx][downstream_mask["train"]]

    test_repr = test_repr[downstream_mask["test"]]
    y_test = y[test_idx][downstream_mask["test"]]

    print(f"train_repr shape = {train_repr.shape}")

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

    if classifier_model == "logistic_regression":
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
            device, classifier_epochs, classifier_batch_size,classifier_lr, False, seed
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

    # ── Step 7: Save Results ────────────────────────────────────────────────────
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
        description="TS-TCC Training Pipeline with CV and Logistic Regression",
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
    general_group.add_argument("--seed", type=int, default=1,
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

    # ══════════════════════════════════════════════════════════════════════════════
    # TS-TCC Encoder Training
    # ══════════════════════════════════════════════════════════════════════════════
    tstcc_group = parser.add_argument_group('TS-TCC Encoder Training')
    tstcc_group.add_argument("--tcc_epochs", type=int, default=40,
                            help="Number of epochs for TS-TCC pretraining")
    tstcc_group.add_argument("--tcc_lr", type=float, default=3e-4,
                            help="Learning rate for TS-TCC training")
    tstcc_group.add_argument("--tcc_batch_size", type=int, default=128,
                            help="Batch size for TS-TCC training")

    # TS-TCC Architecture Parameters
    tstcc_arch_group = parser.add_argument_group('TS-TCC Architecture')
    tstcc_arch_group.add_argument("--tc_timesteps", type=int, default=70,
                                 help="Number of timesteps for temporal contrasting")
    tstcc_arch_group.add_argument("--tc_hidden_dim", type=int, default=128,
                                 help="Hidden dimension for temporal contrasting")
    tstcc_arch_group.add_argument("--cc_temperature", type=float, default=0.07,
                                 help="Temperature parameter for contrastive learning")
    tstcc_arch_group.add_argument("--cc_use_cosine", action="store_true",
                                 help="Use cosine similarity for contrastive learning")

    # Augmentation used
    tstcc_arch_group.add_argument("--use_spectral_augmentation", action="store_true",
                                  help="If set, we use the spectral augmentation (frequency masking)")
    tstcc_arch_group.add_argument("--freq_mask_ratio_weak", default=0.1, type=float)
    tstcc_arch_group.add_argument("--freq_mask_ratio_strong", default=0.2, type=float)
    tstcc_arch_group.add_argument("--freq_max_seq", default=8, type=int)

    # ══════════════════════════════════════════════════════════════════════════════
    # Downstream Classifier Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    classifier_group = parser.add_argument_group('Downstream Classifier')
    classifier_group.add_argument("--classifier_model", type=str, default="logistic_regression",
                                 choices=("logistic_regression", "mlp"),
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

    # Parse arguments and run main function
    args = parser.parse_args()

    #Important:
    args.pretrain_all_conditions = True

    main(**vars(args))