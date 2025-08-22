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

from scipy.stats import uniform
from sklearn.model_selection import ParameterSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

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


def optimize_tstcc_hyperparameters(
    X_train, y_train, groups_train, X_val, y_val, groups_val,
    device, fs, window_size, base_config, n_trials=20, n_epochs_hp=10, seed=42
):
    """
    Perform hyperparameter optimization for TSTCC using random search.
    
    Returns the best hyperparameters and their corresponding validation score.
    """
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    
    # Define hyperparameter search space
    param_distributions = {
        'jitter_ratio': uniform(0.0001, 0.01),  # 0.0001 to 0.0101
        'jitter_scale_ratio': uniform(0.0001, 0.01),  # 0.0001 to 0.0101  
        'max_segment': [4, 6, 8, 10, 12, 16]  # discrete values
    }
    
    # Generate parameter combinations
    param_sampler = ParameterSampler(
        param_distributions, n_iter=n_trials, random_state=seed
    )
    
    best_score = -np.inf
    best_params = None
    best_model = None
    best_tc_head = None
    
    trial_results = []
    
    for trial_idx, params in enumerate(param_sampler):
        print(f"\nTrial {trial_idx + 1}/{n_trials}:")
        print(f"  jitter_ratio: {params['jitter_ratio']:.6f}")
        print(f"  jitter_scale_ratio: {params['jitter_scale_ratio']:.6f}")
        print(f"  max_segment: {params['max_segment']}")
        
        try:
            # Create config for this trial
            cfg = ECGConfig(fs, window_size)
            cfg.num_epoch = n_epochs_hp
            cfg.batch_size = base_config['tcc_batch_size']
            cfg.TC.timesteps = base_config['tc_timesteps']
            cfg.TC.hidden_dim = base_config['tc_hidden_dim']
            cfg.Context_Cont.temperature = base_config['cc_temperature']
            cfg.Context_Cont.use_cosine_similarity = base_config['cc_use_cosine']
            
            # Set hyperparameters being tuned
            cfg.augmentation.jitter_ratio = params['jitter_ratio']
            cfg.augmentation.jitter_scale_ratio = params['jitter_scale_ratio']
            cfg.augmentation.max_seg = params['max_segment']
            
            # Create data loaders
            tr_dl, va_dl, te_dl = data_generator_from_arrays(
                X_train, y_train, X_val, y_val, X_val, y_val,  # Use val as test for HP search
                cfg, training_mode="self_supervised"
            )
            
            # Initialize model
            set_seed(seed)
            model = base_Model(cfg).to(device)
            tc_head = TC(cfg, device).to(device)
            opt_m = optim.AdamW(model.parameters(), lr=base_config['tcc_lr'], weight_decay=3e-4)
            opt_tc = optim.AdamW(tc_head.parameters(), lr=base_config['tcc_lr'], weight_decay=3e-4)
            
            # Train TSTCC with current hyperparameters
            workdir = tempfile.mkdtemp(prefix=f"tstcc_hp_trial_{trial_idx}_")
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
            
            # Extract representations from validation set
            model.eval()
            tc_head.eval()
            with torch.no_grad():
                val_repr, _ = encode_representations(
                    X_val, y_val, model, tc_head, base_config['tcc_batch_size'], device
                )
            
            # Filter to binary task (baseline vs mental_stress)
            val_mask = np.isin(y_val, [0, 1])
            val_repr_filtered = val_repr[val_mask]
            y_val_filtered = y_val[val_mask]
            groups_val_filtered = groups_val[val_mask]
            
            # Quick logistic regression evaluation
            if len(np.unique(y_val_filtered)) >= 2 and len(val_repr_filtered) >= 10:
                # Use simple cross-validation on validation set for scoring
                cv_splitter, _ = get_participant_cv_splitter(
                    groups_val_filtered, min_participants_for_kfold=3, k=3
                )
                
                if cv_splitter is not None:
                    lr = LogisticRegression(random_state=seed, max_iter=1000)
                    cv_scores = cross_val_score(
                        lr, val_repr_filtered, y_val_filtered, 
                        cv=cv_splitter, scoring='roc_auc', groups=groups_val_filtered
                    )
                    trial_score = np.mean(cv_scores)
                else:
                    # Fallback: simple train on validation set
                    lr = LogisticRegression(random_state=seed, max_iter=1000)
                    lr.fit(val_repr_filtered, y_val_filtered)
                    y_pred_proba = lr.predict_proba(val_repr_filtered)[:, 1]
                    trial_score = roc_auc_score(y_val_filtered, y_pred_proba)
            else:
                trial_score = 0.0  # Invalid trial
            
            print(f"  Trial score (AUROC): {trial_score:.4f}")
            
            trial_results.append({
                'trial_idx': trial_idx,
                'params': params.copy(),
                'score': trial_score
            })
            
            # Update best if this trial is better
            if trial_score > best_score:
                best_score = trial_score
                best_params = params.copy()
                # Keep the best model
                best_model = model.state_dict().copy()
                best_tc_head = tc_head.state_dict().copy()
                print(f"  *** New best score: {best_score:.4f} ***")
            
            # Cleanup
            del model, tc_head, opt_m, opt_tc, tr_dl, va_dl, te_dl
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  Trial failed with error: {e}")
            trial_results.append({
                'trial_idx': trial_idx,
                'params': params.copy(),
                'score': -1.0,  # Mark as failed
                'error': str(e)
            })
    
    print(f"\nHyperparameter optimization completed!")
    print(f"Best score: {best_score:.4f}")
    print(f"Best params: {best_params}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_model_state': best_model,
        'best_tc_head_state': best_tc_head,
        'all_trials': trial_results
    }


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
        jitter_scale_ratio: float,
        jitter_ratio: float,
        max_segment: int,
        classifier_model: str,
        classifier_epochs: int,
        classifier_lr: float,
        classifier_batch_size: int,
        use_pretrained_encoder: bool,
        label_fraction: float,
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
        verbose: bool = False,
        scoring_metric: str = "roc_auc",
        optimize_hyperparameters: bool = False,
        hp_n_trials: int = 20,
        hp_n_epochs: int = 10,
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

    if use_pretrained_encoder:
        model_save_path = os.path.join(
            SAVED_MODELS_PATH, "ECG", str(fs), "TSTCC", pretrain_data, f"{seed}", f"{window_size}", f"{step_size}"
        )

    else:
        model_save_path = os.path.join(
            SAVED_MODELS_PATH, "StressID", "TSTCC", f"{seed}", f"{window_size}", f"{step_size}",
        )

    #Save the results based on either pretrained from our dataset or trained from scratch
    subfolder_name = "pretrained_encoder" if use_pretrained_encoder else "trained_from_scratch"

    results_save_path = os.path.join(
        RESULTS_PATH, "Transfer_learning", "StressID", subfolder_name, "TSTCC", classifier_model,
        f"{seed}", f"{label_fraction}", f"{window_size}", f"{step_size}",
    )

    create_directory(model_save_path)
    create_directory(results_save_path)

    # ── Step 1: Preprocess ───────────────────────────────────────────────────────


    if pretrain_all_conditions:
        label_map = {"baseline": 0, "mental_stress": 1, "relax": 2, "other": 3}
    else:
        label_map = {"baseline": 0, "mental_stress": 1}

    # Data path
    window_data_path = os.path.join(
        DATA_PATH, "interim", "STRESSID", "ECG", str(fs), str(window_size), str(step_size), 'windowed_data.h5'
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

    print(f"Class distribution in training data:")
    train_labels = y[train_idx]
    print(f"  Class 0 (baseline): {np.sum(train_labels == 0)} samples")
    print(f"  Class 1 (stress): {np.sum(train_labels == 1)} samples")
    print(
        f"  Class balance ratio: {np.sum(train_labels == 0) / len(train_labels):.3f} / {np.sum(train_labels == 1) / len(train_labels):.3f}")

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
    # We could use these for the hyperparameter tuning
    groups_val_idx_encoder = groups_train_all_encoder[val_idx_encoder]  # 20% of original data

    assert len(np.unique(groups_train_idx_encoder)) + len(np.unique(groups_val_idx_encoder)) + len(np.unique(groups[test_idx])) == 65, \
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
        "jitter_ratio": jitter_ratio,
        "jitter_scale_ratio": jitter_scale_ratio,
        "max_seg": max_segment,
    })

    cached = search_encoder_fp(
        fp, experiment_name="TSTCC", tracking_uri=mlflow_tracking_uri
    )

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
        
        if optimize_hyperparameters:
            print("=== Hyperparameter Optimization Mode ===")
            # Prepare data for hyperparameter optimization
            Xtr = X[train_idx_encoder].astype(np.float32)
            Xva = X[val_idx_encoder].astype(np.float32)
            ytr = y[train_idx_encoder]
            yva = y[val_idx_encoder]
            groups_tr = groups_train_idx_encoder
            groups_va = groups_val_idx_encoder
            
            # Base configuration for hyperparameter optimization
            base_config = {
                'tcc_batch_size': tcc_batch_size,
                'tcc_lr': tcc_lr,
                'tc_timesteps': tc_timesteps,
                'tc_hidden_dim': tc_hidden_dim,
                'cc_temperature': cc_temperature,
                'cc_use_cosine': cc_use_cosine,
            }
            
            # Run hyperparameter optimization
            hp_results = optimize_tstcc_hyperparameters(
                Xtr, ytr, groups_tr, Xva, yva, groups_va,
                device, fs, window_size, base_config,
                n_trials=hp_n_trials, n_epochs_hp=hp_n_epochs, seed=seed
            )
            
            # Log hyperparameter optimization results
            mlflow.log_params({
                'hp_optimization': True,
                'hp_n_trials': hp_n_trials,
                'hp_n_epochs': hp_n_epochs,
                'hp_best_score': hp_results['best_score'],
                **{f"hp_best_{k}": v for k, v in hp_results['best_params'].items()}
            })
            
            # Use the best hyperparameters to train final model with full epochs
            print(f"Training final model with best hyperparameters: {hp_results['best_params']}")
            jitter_ratio = hp_results['best_params']['jitter_ratio']
            jitter_scale_ratio = hp_results['best_params']['jitter_scale_ratio']
            max_segment = hp_results['best_params']['max_segment']
            
            # Save hyperparameter optimization results
            hp_results_path = os.path.join(results_save_path, "hyperparameter_optimization.json")
            with open(hp_results_path, "w") as f:
                json.dump(hp_results, f, indent=2, default=str)
            
        cfg = ECGConfig(fs, window_size)
        cfg.num_epoch = tcc_epochs
        cfg.batch_size = tcc_batch_size
        cfg.TC.timesteps = tc_timesteps
        cfg.TC.hidden_dim = tc_hidden_dim
        cfg.Context_Cont.temperature = cc_temperature
        cfg.Context_Cont.use_cosine_similarity = cc_use_cosine

        # Here we can set the augmentations (potentially optimized)
        cfg.augmentation.jitter_ratio = jitter_ratio
        cfg.augmentation.jitter_scale_ratio = jitter_scale_ratio
        cfg.augmentation.max_seg = max_segment

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
        ckpt = os.path.join(workdir, "tstcc.pt")
        torch.save(
            {"encoder": model.state_dict(),
             "tc_head": tc_head.state_dict()},
            ckpt
        )

        mlflow.log_artifact(ckpt, artifact_path="tstcc_model")

        saved_results = os.path.join(model_save_path, "tstcc.pt")
        torch.save(
            {"encoder": model.state_dict(),
             "tc_head": tc_head.state_dict()},
            saved_results
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
    
    # Check class balance in each CV fold
    if cv_splitter is not None:
        print("Checking class balance in CV folds...")
        problematic_folds = []
        for fold_idx, (train_cv_idx, val_cv_idx) in enumerate(cv_splitter.split(train_repr, y_train, groups_train)):
            y_train_fold = y_train[train_cv_idx]
            y_val_fold = y_train[val_cv_idx]
            
            # Calculate percentages
            train_class0 = np.sum(y_train_fold == 0)
            train_class1 = np.sum(y_train_fold == 1)
            train_total = len(y_train_fold)
            train_class0_pct = train_class0 / train_total * 100
            train_class1_pct = train_class1 / train_total * 100
            
            val_class0 = np.sum(y_val_fold == 0)
            val_class1 = np.sum(y_val_fold == 1)
            val_total = len(y_val_fold)
            val_class0_pct = val_class0 / val_total * 100 if val_total > 0 else 0
            val_class1_pct = val_class1 / val_total * 100 if val_total > 0 else 0
            
            print(f"  Fold {fold_idx+1}: Train class 0: {train_class0} ({train_class0_pct:.1f}%), class 1: {train_class1} ({train_class1_pct:.1f}%)")
            print(f"  Fold {fold_idx+1}: Val   class 0: {val_class0} ({val_class0_pct:.1f}%), class 1: {val_class1} ({val_class1_pct:.1f}%)")
            
            # Check for severely imbalanced folds (>85% one class in validation)
            if val_class0_pct > 85 or val_class1_pct > 85:
                problematic_folds.append(fold_idx + 1)
                print(f"    WARNING: Fold {fold_idx+1} validation set is severely imbalanced!")
            
            # Check for empty classes
            if len(np.unique(y_train_fold)) < 2 or len(np.unique(y_val_fold)) < 2:
                print(f"    CRITICAL: Fold {fold_idx+1} has missing classes!")
                problematic_folds.append(fold_idx + 1)
        
        if problematic_folds:
            print(f"\n🔍 DIAGNOSIS: Fold(s) {problematic_folds} have severe class imbalance.")
        print()

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
                scoring_metric=scoring_metric,
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
        description="TS-TCC Training Pipeline with CV and Logistic Regression for Stress ID Dataset",
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
    data_group.add_argument("--fs", default=500, type=str,
                           help="Sampling frequency used for training")
    data_group.add_argument("--window_size", type=int, default=10,
                           help="Window size in seconds")
    data_group.add_argument("--step_size", type=int, default=5,
                           help="Step size in seconds for sliding window")
    data_group.add_argument("--label_fraction", type=float, default=0.25,
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

    # For tuning the augmentations (we tune the jitter ratio and the segments)
    # Random search as it is more efficient and faster, only for maybe 10 epochs
    # My hypothesis is that this would outperform the trained from scratch architecture
    # Add on, add the S3 layer on top
    tstcc_arch_group.add_argument("--jitter_scale_ratio", default=0.001, type=float)
    tstcc_arch_group.add_argument("--jitter_ratio", default=0.001, type=float)
    tstcc_arch_group.add_argument("--max_segment", default = 8, type=int)

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
    classifier_group.add_argument("--use_pretrained_encoder",action="store_true",
                                  help="If set, we use the pre-trained encoder from our dataset")

    # ══════════════════════════════════════════════════════════════════════════════
    # Cross-Validation Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    cv_group = parser.add_argument_group('Cross-Validation')
    cv_group.add_argument("--k_folds", type=int, default=5,
                         help="Number of folds for cross-validation")
    cv_group.add_argument("--min_participants_for_kfold", type=int, default=5,
                         help="Minimum participants needed for k-fold (otherwise use Leave-one-participant-out-CV)")
    cv_group.add_argument("--scoring_metric", type=str, default="f1",
                         choices=["roc_auc", "average_precision", "f1", "balanced_accuracy"],
                         help="Scoring metric for cross-validation hyperparameter selection")

    # ══════════════════════════════════════════════════════════════════════════════
    # Hyperparameter Optimization Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    hp_group = parser.add_argument_group('Hyperparameter Optimization')
    hp_group.add_argument("--optimize_hyperparameters", action="store_true",
                         help="Enable hyperparameter optimization for TSTCC augmentation parameters")
    hp_group.add_argument("--hp_n_trials", type=int, default=5,
                         help="Number of trials for hyperparameter optimization")
    hp_group.add_argument("--hp_n_epochs", type=int, default=5,
                         help="Number of epochs for each hyperparameter optimization trial")

    # Parse arguments and run main function
    args = parser.parse_args()

    #Important:
    args.pretrain_all_conditions = True

    args.optimize_hyperparameters = True

    main(**vars(args))