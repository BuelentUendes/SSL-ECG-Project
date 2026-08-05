# Simple script to run and train it fine-tuned
#!/usr/bin/env python
import copy
import os
from datetime import datetime
import json
import argparse
import logging
import tempfile
import gc
import uuid

import numpy as np
import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch

from utils.torch_utilities import (
    load_processed_data,
    return_track_data_file,
    split_indices_by_participant_groups,
    set_seed,
    create_directory,
    get_participant_cv_splitter,
    run_logistic_regression_with_gridsearch,
    run_logistic_regression_with_gridsearch_verbose,
    run_mlp_with_cv_and_test,
    run_linear_classifier_with_cv_and_test,
)

from utils.helper_paths import SAVED_MODELS_PATH, DATA_PATH, RESULTS_PATH

from models.tstcc import (
    data_generator_from_arrays,
    Trainer,
    base_Model,
    TC,
    Config as ECGConfig,
    encode_representations,
    FineTunedNet,
    freeze_and_unfreeze_encoder,
)


# Numeric index → WESAD participant ID (sorted numerically, S12 is missing)
WESAD_PARTICIPANT_MAP = {
    0: "S2", 1: "S3", 2: "S4", 3: "S5", 4: "S6",
    5: "S7", 6: "S8", 7: "S9", 8: "S10", 9: "S11",
    10: "S13", 11: "S14", 12: "S15", 13: "S16", 14: "S17",
}


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
        save_embeddings: bool,
        tc_timesteps: int,
        tc_hidden_dim: int,
        cc_temperature: float,
        cc_disable_cosine: bool,
        use_s3_layers: bool,
        initial_num_segments: int,
        num_s3_layers: int,
        segment_multiplier: int,
        jitter_scale_ratio: float,
        jitter_ratio: float,
        max_segment: int,
        classifier_model: str,
        classifier_epochs: int,
        classifier_lr: float,
        classifier_batch_size: int,
        use_pretrained_encoder: bool,
        fine_tune_encoder: bool,
        label_fraction: float,
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
        verbose: bool = False,
        scoring_metric: str = "roc_auc",
        optimize_hyperparameters: bool = False,
        loso: bool = False,
        held_out_participant: int = 0,
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
    print(f"Using device: {device}")

    # Check if directory for saving model parameters exist, otherwise create it
    create_directory(SAVED_MODELS_PATH)
    create_directory(RESULTS_PATH)

    # We save the model here via seeds, we create a separate folder for pretraining on all labels and on only task-related data
    pretrain_data = "all_labels" if pretrain_all_conditions else "mental_stress_baseline"

    #IMPORTANT: If we use a pretrained encoder then we use a different save path so it loads a different model!
    if use_pretrained_encoder:
        if use_s3_layers:
            model_save_path = os.path.join(
                SAVED_MODELS_PATH, "ECG", str(fs), "TSTCC_S3", pretrain_data, f"{seed}",
                f"{window_size}", f"{step_size}", f"{train_ratio_encoder}",
            )
        else:
            model_save_path = os.path.join(
                SAVED_MODELS_PATH, "ECG", str(fs), "TSTCC", pretrain_data, f"{seed}",
                f"{window_size}", f"{step_size}", f"{train_ratio_encoder}",
            )

    else:
        if use_s3_layers:
            model_save_path = os.path.join(
                SAVED_MODELS_PATH, "WESAD", "TSTCC_S3", f"{seed}", f"{window_size}", f"{step_size}"
            )
        else:
            model_save_path = os.path.join(
                SAVED_MODELS_PATH, "WESAD", "TSTCC", f"{seed}", f"{window_size}", f"{step_size}"
            )

    # Save the results based on either pretrained from our dataset or trained from scratch
    # use pretrained encoder -> train a new head
    # fine_tune_encoder -> fine tune encoder and a new head
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

    model_name = "TSTCC_S3" if use_s3_layers else "TSTCC"

    if loso:
        results_save_path = os.path.join(
            RESULTS_PATH, "Transfer_learning", "WESAD_LOSO", subfolder_name, model_name,
            classifier_model, f"{seed}", f"participant_index_{held_out_participant}",
            f"{label_fraction}", f"{window_size}", f"{step_size}"
        )
    else:
        results_save_path = os.path.join(
            RESULTS_PATH, "Transfer_learning", "WESAD", subfolder_name, model_name, classifier_model,
            f"{seed}", f"{label_fraction}", f"{window_size}", f"{step_size}"
        )

    # We will save the embeddings so we can later do some analysis on them
    embedding_save_path = os.path.join(
        DATA_PATH, "embeddings", "WESAD", f"{fs}", model_name, f"{seed}", f"{window_size}", f"{step_size}"
    )

    create_directory(model_save_path)
    create_directory(results_save_path)
    create_directory(embedding_save_path)

    # ── Step 1: Preprocess ───────────────────────────────────────────────────────
    if pretrain_all_conditions:

        label_map = {
            "baseline": 0,
            "mental_stress": 1, #Here not mental stress but other stress, physiological stress
            "transient": 2,
            "amusement": 3,
            "meditation": 4,
            # "other": 5, #should be ignored
            # "other": 6,
            # "other": 7,
        }

    else:
        label_map = {"baseline": 0, "mental_stress": 1}

    # Data path
    window_data_path = os.path.join(
        DATA_PATH, "interim", "WESAD", "ECG", str(fs), str(window_size), str(step_size), 'windowed_data.h5'
    )

    X, y, groups = load_processed_data(window_data_path, label_map=label_map)
    y = y.astype(np.float32)

    # We first get all train idx for the SSL method (label fraction 1.0) as we do not use the labels
    # train_idx_all (represents all training samples as we do not use their labels)
    # Split by participant to get train/test split
    # train_idx to the labeled ones!
    # train_p refers to the labeled training participant!
    # all_train_idx refer to all the training samples (irrespective of labeled or not)
    if loso:
        held_out_id = WESAD_PARTICIPANT_MAP[held_out_participant]
        print(f"LOSO mode: holding out participant {held_out_id} (index {held_out_participant}) as test set")
        test_p = np.array([held_out_id])
        all_train_p = np.array([p for p in WESAD_PARTICIPANT_MAP.values() if p != held_out_id])

        test_idx = np.flatnonzero(groups == held_out_id)
        all_train_idx = np.flatnonzero(np.isin(groups, all_train_p))

        rng = np.random.default_rng(seed)
        if label_fraction < 1.0:
            n_labeled = max(1, int(len(all_train_p) * label_fraction))
            labeled_participants = rng.choice(all_train_p, size=n_labeled, replace=False)
            train_p = labeled_participants.copy()
            train_idx = np.flatnonzero(np.isin(groups, labeled_participants))
        else:
            train_p = all_train_p.copy()
            train_idx = all_train_idx.copy()
    else:
        train_idx, train_p, all_train_p, all_train_idx, test_idx, test_p = split_indices_by_participant_groups(
            groups,
            train_ratio=0.8,
            label_fraction=label_fraction,
            seed=seed,
            return_all_train_p=True
        )

    # This is the dataset we use for training of the encoder!
    groups_train_all_encoder = groups[all_train_idx]

    # Rep is the one that we train the encoder on, for these we do not need the labels, so label fraction is set to 1.0
    train_idx_encoder, train_p_rep, val_idx_encoder, val_p  = split_indices_by_participant_groups(
        groups_train_all_encoder,
        train_ratio=train_ratio_encoder, #This will give a split of 60/20/20
        label_fraction=1.0, # We will discard anyways all labels
        seed=seed,
        return_all_train_p=False,
    )

    # Map back to original indices
    groups_train_idx_encoder = groups_train_all_encoder[train_idx_encoder]  # 60% of original data
    groups_val_idx_encoder = groups_train_all_encoder[val_idx_encoder]  # 20% of original data

    # Test that we have all 15 participants moved in one of the categories
    assert (len(np.unique(groups_train_idx_encoder)) + len(np.unique(groups_val_idx_encoder)) +
            len(np.unique(groups[test_idx])) == 15), \
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

    if (os.path.exists(os.path.join(model_save_path, "tstcc.pt"))) and not (force_retraining):
        print("We found a pretrained model. Load the pretrained weights")
        ckpt_path = os.path.join(model_save_path, "tstcc.pt")

        # rebuild model
        cfg = ECGConfig(fs, window_size)
        cfg.num_epoch = tcc_epochs
        cfg.batch_size = tcc_batch_size
        cfg.TC.timesteps = tc_timesteps
        cfg.TC.hidden_dim = tc_hidden_dim
        cfg.Context_Cont.temperature = cc_temperature
        cfg.Context_Cont.use_cosine_similarity = False if cc_disable_cosine else True
        cfg.use_s3_layers = use_s3_layers
        cfg.initial_num_segments = initial_num_segments
        cfg.num_s3_layers = num_s3_layers
        cfg.segment_multiplier = segment_multiplier

        model = base_Model(cfg).to(device)
        tc_head = TC(cfg, device).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["encoder"])
        tc_head.load_state_dict(state["tc_head"])

    else:
        print("No cached encoder; training TS-TCC from scratch")

        if optimize_hyperparameters:
            # Generate random id for experiment tracking
            run_id = str(uuid.uuid4())
            hyperparameter_file_name = os.path.join(
                results_save_path, "hyperparameter_tuning_results.json"
            )

        cfg = ECGConfig(fs, window_size)
        cfg.num_epoch = tcc_epochs
        cfg.batch_size = tcc_batch_size
        cfg.TC.timesteps = tc_timesteps
        cfg.TC.hidden_dim = tc_hidden_dim
        cfg.Context_Cont.temperature = cc_temperature
        cfg.Context_Cont.use_cosine_similarity = False if cc_disable_cosine else True
        cfg.use_s3_layers = use_s3_layers
        cfg.initial_num_segments = initial_num_segments
        cfg.num_s3_layers = num_s3_layers
        cfg.segment_multiplier = segment_multiplier

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
            results_save_path=results_save_path,
            batch_size=tcc_batch_size,
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
    if fine_tune_encoder:
        # If we fine-tune the encoder we learn a representation right away and save the results
        X_train = X[train_idx][downstream_mask["train"]]
        y_train = y[train_idx][downstream_mask["train"]]
        groups_train = groups[train_idx][downstream_mask["train"]]

        X_test = X[test_idx][downstream_mask["test"]]
        y_test = y[test_idx][downstream_mask["test"]]

        cv_splitter, n_splits = get_participant_cv_splitter(
            groups_train,
            min_participants_for_kfold=min_participants_for_kfold,
            k=k_folds
        )
        feature_names = [f"repr_{i}" for i in range(X_train.shape[1])]

        fine_tune_model = FineTunedNet(encoder=model, tc_head=tc_head).to(device)

        # We follow the lp+ft approach (https://arxiv.org/pdf/2202.10054)
        # So we first freez all the weights
        fine_tune_model = freeze_and_unfreeze_encoder(fine_tune_model, freeze = True)

        # This fine-tunes the encoder and test them right away (we will extract the head here)
        fine_tune_results = run_linear_classifier_with_cv_and_test(
            X_train, y_train, groups_train, X_test, y_test, fine_tune_model,
            feature_names, cv_splitter, device, classifier_epochs=10,
            classifier_batch_size=classifier_batch_size,
            standardize=False, seed=seed
        )

        # Get the updated model (this has the head already fine-tuned)
        fine_tune_model = copy.deepcopy(fine_tune_results["model"])

        # Now unfreeze the weights so the encoder and the head will be trained
        fine_tune_model = freeze_and_unfreeze_encoder(fine_tune_model, freeze=False)

        fine_tuned_results = run_linear_classifier_with_cv_and_test(
            X_train, y_train, groups_train, X_test, y_test, fine_tune_model,
            feature_names, cv_splitter, device, classifier_epochs=15,
            classifier_batch_size=classifier_batch_size,
            standardize=False, seed=seed
        )

        print(f"We finished the fine-tuning stage (LP + FT)")
        print(fine_tuned_results["test_metrics"])

        with open(os.path.join(results_save_path, "test_results_lp_ft.json"), "w") as f:
            json.dump(fine_tuned_results, f, indent=2, default=str)

    else:
        model.eval()
        tc_head.eval()

        with torch.no_grad():
            train_repr, _ = encode_representations(X[train_idx], y[train_idx],
                                                   model, tc_head, tcc_batch_size, device)
            test_repr, _ = encode_representations(X[test_idx], y[test_idx],
                                                  model, tc_head, tcc_batch_size, device)

            if save_embeddings:
                print(f"Saving the embeddings for later analysis ...")
                x_repr_all, _ = encode_representations(X, y, model, tc_head, tcc_batch_size, device)
                np.savez(os.path.join(
                    embedding_save_path, "x_y_groups_embedding.npz"), array1=x_repr_all, array_2=y, array_3=groups
                )
                print(f"We saved the embeddings, y and groups")

        # filter to binary downstream samples
        train_repr = train_repr[downstream_mask["train"]]
        y_train = y[train_idx][downstream_mask["train"]]
        groups_train = groups[train_idx][downstream_mask["train"]]

        test_repr = test_repr[downstream_mask["test"]]
        y_test = y[test_idx][downstream_mask["test"]]

        print(f"train_repr shape = {train_repr.shape}")

        # ── Step 4: Set up Cross-Validation Splitter ───────────────────────────────
        # Important:
        # For LOSO, we do not have variability introduced in the assignment of who is in train & test
        # So we introduce it for LOSO via shufflign in the cv_splitter,
        # so each CV split has across seeds different subjects introducing the subject variability

        if loso:
            cv_splitter, n_splits = get_participant_cv_splitter(
                groups_train,
                min_participants_for_kfold=min_participants_for_kfold,
                k=k_folds, seed=seed
            )
        else:
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
                    test_repr, y_test, feature_names, cv_splitter, False, seed, scoring_metric=scoring_metric
                )


        else:
            results = run_mlp_with_cv_and_test(
                train_repr, y_train, groups_train,
                test_repr, y_test, feature_names, cv_splitter,
                device, classifier_epochs, classifier_batch_size,classifier_lr, False, seed
            )


        # ── Step 7: Save Results ────────────────────────────────────────────────────
        with open(os.path.join(results_save_path, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        if optimize_hyperparameters:
            hyperparameter_save_file = return_track_data_file(hyperparameter_file_name)

            hyperparameter_save_file["runs"][run_id] = {
                "hyperparameters": cfg.to_dict(),
                "CV score": results['best_cv_score'],
                "Test_metrics": results["test_metrics"],
                "timestamp": datetime.now().isoformat(),
            }

            with open(hyperparameter_file_name, "w") as f:
                json.dump(hyperparameter_save_file, f, indent=4)

        # ── Cleanup ────────────────────────────────────────────────────────────────
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"=== Done! Test Acc: {results['test_metrics']['accuracy']:.4f}, "
              f"AUROC: {results['test_metrics']['auroc']:.4f}, "
              f"PR-AUC: {results['test_metrics']['pr_auc']:.4f}, "
              f"F1 (default threshold): {results['test_metrics']['f1']:.4f} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TS-TCC Training Pipeline with CV and Logistic Regression for Stress ID Dataset",
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
    data_group.add_argument("--fs", default=700, type=str,
                           help="Sampling frequency used for training")
    data_group.add_argument("--window_size", type=int, default=10,
                           help="Window size in seconds")
    data_group.add_argument("--step_size", type=int, default=5,
                           help="Step size in seconds for sliding window")
    data_group.add_argument("--label_fraction", type=float, default=1.0,
                           help="Fraction of labeled participants to use (0.0-1.0)")
    data_group.add_argument("--pretrain_all_conditions", action="store_true",
                           help="Pretrain on all conditions (not just baseline/mental_stress)")
    data_group.add_argument("--train_ratio_encoder", default=1.0, type=float,
                            help="If set to 0.75, it will result in 60/20/20 split and have a validation set for TSTCC,"
                                 "Alternatively, set to 1.0 to train on all unlabelled training instances.")
    data_group.add_argument("--save_embeddings", action="store_true",
                            help="If we want to save the embeddings. Note this is computationally expensive.")
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
    tstcc_arch_group.add_argument("--tc_timesteps", type=int, default=50,
                                 help="Number of timesteps for temporal contrasting")
    tstcc_arch_group.add_argument("--tc_hidden_dim", type=int, default=100,
                                 help="Hidden dimension for temporal contrasting")
    tstcc_arch_group.add_argument("--cc_temperature", type=float, default=0.2,
                                 help="Temperature parameter for contrastive learning")
    tstcc_arch_group.add_argument("--cc_disable_cosine", action="store_true",
                                 help="Disable cosine similarity for contrastive learning")
    tstcc_arch_group.add_argument("--use_s3_layers", action="store_true",
                                  help="If set, we use the S3 layer")
    tstcc_arch_group.add_argument("--initial_num_segments", type=int, default=2)
    tstcc_arch_group.add_argument("--num_s3_layers", type=int, default=2)
    tstcc_arch_group.add_argument("--segment_multiplier", type=int, default=1)

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
                                 help="Number of epochs for fine-tuning of the encoder and TC head")
    classifier_group.add_argument("--classifier_lr", type=float, default=1e-4,
                                 help="Learning rate for MLP classifier")
    classifier_group.add_argument("--classifier_batch_size", type=int, default=32,
                                 help="Batch size for fine-tuning")
    classifier_group.add_argument("--use_pretrained_encoder",action="store_true",
                                  help="If set, we use the pre-trained encoder from our dataset")
    classifier_group.add_argument("--fine_tune_encoder", action="store_true",
                                  help="If set, we fine-tune also the encoder and not only the logistic regression.")

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

    # ══════════════════════════════════════════════════════════════════════════════
    # Leave-One-Subject-Out Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    loso_group = parser.add_argument_group('Leave-One-Subject-Out')
    loso_group.add_argument("--loso", action="store_true",
                            help="Leave-one-subject-out: hold out one participant as test")
    loso_group.add_argument("--held_out_participant", type=int, default=0,
                            help="Index (0-14) of participant to hold out under --loso. "
                                 "Mapping: 0=S2, 1=S3, 2=S4, 3=S5, 4=S6, 5=S7, 6=S8, 7=S9, "
                                 "8=S10, 9=S11, 10=S13, 11=S14, 12=S15, 13=S16, 14=S17")

    # Parse arguments and run main function
    args = parser.parse_args()

    #Important:
    args.pretrain_all_conditions = True
    # IMPORTANT:
    # use pretrained encoder -> train a new head saved as pretrained_encoder_new_head (LP)
    # fine_tune_encoder -> fine tune encoder and a new head -> saved as fined_tuned_encoder_new_head (LP+FT)

    main(**vars(args))