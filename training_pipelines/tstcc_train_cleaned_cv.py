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
    optimize_tstcc_hyperparameters,
)


def main(
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
        train_ratio_encoder: float,
        tc_timesteps: int,
        tc_hidden_dim: int,
        cc_temperature: float,
        cc_use_cosine: bool,
        use_s3_layers: bool,
        initial_num_segments: int,
        num_s3_layers: int,
        segment_multiplier: int,
        jitter_scale_ratio: float,
        jitter_ratio: float,
        max_segment: int,
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
        scoring_metric: str = "roc_auc",
        optimize_hyperparameters: bool = False,
        hp_n_trials: int = 20,
        hp_n_epochs: int = 10,
        hp_search_type: str = "random",
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
    print(f"Starting TSTCC training with CV {classifier_model}, seed={seed}, label_fraction={label_fraction}")
    print(f"Using device: {device}")

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)
    create_directory(RESULTS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"

    model_save_path = os.path.join(
        SAVED_MODELS_PATH, "ECG", str(fs), "TSTCC", pretrain_data, f"{seed}", f"{window_size}", f"{step_size}",
        str(train_ratio_encoder)
    )
    results_save_path = os.path.join(
        RESULTS_PATH, "ECG", "TSTCC", classifier_model, f"{seed}", f"{label_fraction}", f"{window_size}",
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
        train_ratio=train_ratio_encoder, #This will give a split of 60/20/20 0.75 achieves this!
        label_fraction=1.0, # We will discard anyways all labels
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

    # ── Step 2: TS‑TCC Pretraining ───────────────────────────────────────────────
    torch.cuda.empty_cache()
    set_seed(seed)

    # Model fingerprinting removed (was used for MLflow caching)

    model_file_name = "tstcc_spectral.pt" if use_spectral_augmentation else "tstcc.pt"

    # Check if we have a locally saved model and no forced retraining
    if os.path.exists(os.path.join(model_save_path, model_file_name)) and not force_retraining:
        print("We found a pretrained model. Load the pretrained weights")
        ckpt_path = os.path.join(model_save_path, model_file_name)

        # rebuild model
        cfg = ECGConfig(fs, window_size)
        cfg.num_epoch = tcc_epochs
        cfg.batch_size = tcc_batch_size
        cfg.TC.timesteps = tc_timesteps
        cfg.TC.hidden_dim = tc_hidden_dim
        cfg.Context_Cont.temperature = cc_temperature
        cfg.Context_Cont.use_cosine_similarity = cc_use_cosine
        cfg.use_s3_layers = use_s3_layers
        cfg.initial_num_segments = initial_num_segments
        cfg.num_s3_layers = num_s3_layers
        cfg.segment_multiplier = segment_multiplier

        # Here we can set the augmentations (potentially optimized)
        cfg.augmentation.jitter_ratio = jitter_ratio
        cfg.augmentation.jitter_scale_ratio = jitter_scale_ratio
        cfg.augmentation.max_seg = max_segment

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
            # groups_tr = groups_train_idx_encoder  # Not used
            groups_va = groups_val_idx_encoder

            # Base configuration for hyperparameter optimization
            base_config = {
                'tcc_batch_size': tcc_batch_size,
                'tcc_lr': tcc_lr,
                'tc_timesteps': tc_timesteps,
                'tc_hidden_dim': tc_hidden_dim,
                'cc_temperature': cc_temperature,
                'cc_use_cosine': cc_use_cosine,
                "use_s3_layers": use_s3_layers,
                "initial_num_segments": initial_num_segments,
                "num_s3_layers": num_s3_layers,
                "segment_multiplier": segment_multiplier,
            }

            # Run hyperparameter optimization
            hp_results = optimize_tstcc_hyperparameters(
                Xtr, ytr, Xva, yva, groups_va,
                device, fs, window_size, base_config,
                n_trials=hp_n_trials, n_epochs_hp=hp_n_epochs, seed=seed,
                search_type=hp_search_type
            )

            # Log hyperparameter optimization results locally
            print(f"Hyperparameter optimization completed with best score: {hp_results['best_score']:.4f}")
            print(f"Best hyperparameters: {hp_results['best_params']}")

            # Use the best hyperparameters to train final model with full epochs
            print(f"Training final model with best hyperparameters: {hp_results['best_params']}")
            jitter_ratio = hp_results['best_params']['jitter_ratio']
            jitter_scale_ratio = hp_results['best_params']['jitter_scale_ratio']
            max_segment = hp_results['best_params']['max_segment']

            initial_num_segments = hp_results['best_params']["initial_num_segments"]
            num_s3_layers = hp_results["best_params"]["num_s3_layers"]
            segment_multiplier = hp_results["best_params"]["segment_multiplier"]

            # Save hyperparameter optimization results
            hp_results_path = os.path.join(results_save_path, "hyperparameter_optimization.json")
            with open(hp_results_path, "w") as f:
                json.dump(hp_results, f, indent=2, default=str)

        cfg = ECGConfig(fs, window_size)
        cfg.num_epoch = tcc_epochs
        cfg.batch_size = tcc_batch_size
        cfg.use_s3_layers = use_s3_layers
        cfg.initial_num_segments = initial_num_segments
        cfg.num_s3_layers = num_s3_layers
        cfg.segment_multiplier = segment_multiplier

        # Here we can set the augmentations (potentially optimized)
        cfg.augmentation.jitter_ratio = jitter_ratio
        cfg.augmentation.jitter_scale_ratio = jitter_scale_ratio
        cfg.augmentation.max_seg = max_segment

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

        # Create temporary working directory
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

        # Model artifact logging removed (replaced with direct file saving)

        saved_results = os.path.join(model_save_path, model_file_name)
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
            device, classifier_epochs, classifier_batch_size,classifier_lr, False, seed
        )

        # Log metrics locally
        print(f"Best CV AUROC: {results['best_cv_score']:.4f}")
        print(f"Test metrics - Accuracy: {results['test_metrics']['accuracy']:.4f}, "
              f"AUROC: {results['test_metrics']['auroc']:.4f}, F1: {results['test_metrics']['f1']:.4f}, "
              f"PR-AUC: {results['test_metrics']['pr_auc']:.4f}")
        print(f"Best hyperparameters: {results['best_params']}")

    # ── Step 7: Save Results ────────────────────────────────────────────────────
    test_results_name = "test_results_spectral.json" if use_spectral_augmentation else "test_results.json"

    with open(os.path.join(results_save_path, test_results_name), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Log additional parameters locally
    print(f"Additional parameters - Classifier: {classifier_model}, Label fraction: {label_fraction}, "
          f"Seed: {seed}, K-folds: {k_folds}, CV splits: {n_splits}, "
          f"Pretrain all conditions: {pretrain_all_conditions}")

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
        description="TS-TCC Training Pipeline with CV and Logistic Regression",
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
    data_group.add_argument("--train_ratio_encoder", default=0.75, type=float,
                            help="If set to 0.75, it will result in 60/20/20 split and have a validation set for TSTCC,"
                                 "Alternatively, set to 1.0 to train on all unlabelled training instances.")
    # ══════════════════════════════════════════════════════════════════════════════
    # TS-TCC Encoder Training
    # ══════════════════════════════════════════════════════════════════════════════
    tstcc_group = parser.add_argument_group('TS-TCC Encoder Training')
    tstcc_group.add_argument("--tcc_epochs", type=int, default=40,
                            help="Number of epochs for TS-TCC pretraining")
    tstcc_group.add_argument("--tcc_lr", type=float, default=3e-4,
                            help="Learning rate for TS-TCC training")
    tstcc_group.add_argument("--tcc_batch_size", type=int, default=128,
                            help="Batch size for TS-TCC training") # reduce to 16 (was 128)

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

    tstcc_arch_group.add_argument("--use_s3_layers", action="store_true",
                                  help="If set, we use the S3 layer", default=True)
    tstcc_arch_group.add_argument("--initial_num_segments", type=int, default=2)
    tstcc_arch_group.add_argument("--num_s3_layers", type=int, default=1)
    tstcc_arch_group.add_argument("--segment_multiplier", type=int, default=1)

    tstcc_arch_group.add_argument("--jitter_scale_ratio", default=0.01570, type=float) #Approximately best tuned parameters!
    tstcc_arch_group.add_argument("--jitter_ratio", default=0.01570, type=float) # Approximately best tuned parameters
    tstcc_arch_group.add_argument("--max_segment", default = 8, type=int)

    # Augmentation used
    tstcc_arch_group.add_argument("--use_spectral_augmentation", action="store_true",
                                  help="If set, we use the spectral augmentation (frequency masking)")
    tstcc_arch_group.add_argument("--freq_mask_ratio_weak", default=0.1, type=float)
    tstcc_arch_group.add_argument("--freq_mask_ratio_strong", default=0.3, type=float)
    tstcc_arch_group.add_argument("--freq_max_seq", default=8, type=int)

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
                         help="Enable hyperparameter optimization for TSTCC augmentation parameters")
    hp_group.add_argument("--hp_n_trials", type=int, default=30,
                         help="Number of trials for hyperparameter optimization")
    hp_group.add_argument("--hp_n_epochs", type=int, default=10,
                         help="Number of epochs for each hyperparameter optimization trial")
    hp_group.add_argument("--hp_search_type", type=str, default="grid",
                         choices=["random", "grid"],
                         help="Search strategy: 'random' for random search, 'grid' for grid search")


    # Parse arguments and run main function
    args = parser.parse_args()

    #Important:
    args.pretrain_all_conditions = True
    args.use_s3_layers = True

    main(**vars(args))