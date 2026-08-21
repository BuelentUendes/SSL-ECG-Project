#!/usr/bin/env python
import os
from datetime import datetime
import time
import json
import argparse
import logging
import gc
import uuid

import numpy as np
import torch
import torch.optim as optim

from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, balanced_accuracy_score

from utils.torch_utilities import (
    load_processed_data_with_conditions,
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

from models.simclr import (
    get_simclr_model,
    NTXentLoss,
    simclr_data_loaders,
    pretrain_one_epoch,
    encode_representations,
    build_simclr_fingerprint,
)

# Stressor groups for leave-one-stressor-out analysis: (group_name, [condition_names])
STRESSOR_GROUPS = [
    ("TA", ["TA", "TA_repeat"]),
    ("Pasat", ["Pasat", "Pasat_repeat"]),
    ("Raven", ["Raven"]),
    ("SSST", ["SSST_Sing_countdown"]),
]


def _per_condition_metrics(model, test_repr, y_test, conditions_test):
    """Compute binary (vs baseline) metrics per mental-stress condition using a trained sklearn model."""
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, balanced_accuracy_score
    baseline_mask = y_test == 0
    out = {}
    for cond in np.unique(conditions_test[y_test == 1]):
        cond_mask = (conditions_test == cond) & (y_test == 1)
        mask = baseline_mask | cond_mask
        if mask.sum() < 2 or cond_mask.sum() == 0:
            continue
        y_s = y_test[mask].astype(int)
        r_s = test_repr[mask]
        proba = model.predict_proba(r_s)[:, 1]
        pred = model.predict(r_s)
        out[cond] = {
            "n_stress_samples": int(cond_mask.sum()),
            "auroc": float(roc_auc_score(y_s, proba)),
            "pr_auc": float(average_precision_score(y_s, proba)),
            "accuracy": float(accuracy_score(y_s, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_s, pred)),
            "f1": float(f1_score(y_s, pred)),
        }
    return out


def _pr_auc_ratio_corrected(model, X, y, overall_ratio, seed=42):
    """PR-AUC with baseline subsampled to match overall_ratio prevalence.

    Used alongside the raw (all-baseline) PR-AUC so the two can be compared:
    - raw:            realistic — all baseline samples in the pool
    - ratio-corrected: comparable across conditions — prevalence fixed to the
                       dataset-wide stress rate, matching the reference bootstrap
                       evaluation (get_idx_per_subcategory).
    """
    stress_idx = np.where(np.asarray(y) == 1)[0]
    baseline_idx = np.where(np.asarray(y) == 0)[0]
    n_stress = len(stress_idx)
    if n_stress == 0 or len(baseline_idx) == 0:
        return float("nan"), 0
    n_baseline = int((1 - overall_ratio) * n_stress / overall_ratio)
    n_baseline = min(max(n_baseline, 1), len(baseline_idx))
    rng = np.random.RandomState(seed)
    sampled = rng.choice(baseline_idx, size=n_baseline, replace=False)
    combined = np.concatenate([stress_idx, sampled])
    y_s = np.asarray(y)[combined]
    if len(np.unique(y_s)) < 2:
        return float("nan"), n_baseline
    proba = model.predict_proba(X[combined])[:, 1]
    return float(average_precision_score(y_s, proba)), n_baseline


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
        use_only_inversion_negation: bool,
        use_s3_layers: bool,
        initial_num_segments: int,
        num_s3_layers: int,
        segment_multiplier: int,
        optimize_hyperparameters: bool,
        classifier_model: str,
        classifier_epochs: int,
        classifier_lr: float,
        classifier_batch_size: int,
        label_fraction: float,
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
        verbose: bool = False,
        scoring_metric: str = "roc_auc",
        leave_one_stressor_out: bool = False,
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

    X, y, groups, conditions = load_processed_data_with_conditions(window_data_path, label_map=label_map)
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

    if optimize_hyperparameters:
        if batch_size == 256:
            model_save_name_weights = "simclr_encoder_hyperparameter.pt"
        else:
            model_save_name_weights = f"simclr_encoder_hyperparameter_{batch_size}.pt"
    else:
        model_save_name_weights = "simclr_encoder.pt" if batch_size == 256 else f"simclr_encoder_{batch_size}.pt"

    epoch_train_loss = {}  # populated in the training branch; empty if loading from cache

    # Check for local pretrained model
    if (os.path.exists(os.path.join(model_save_path, model_save_name_weights)) and not force_retraining
            and not optimize_hyperparameters):
        print("Found pretrained model. Loading weights...")
        model_path = os.path.join(model_save_path, model_save_name_weights)
        model.load_state_dict(torch.load(model_path, map_location=device))

    else:
        print(f"No cached encoder; training {model_name} from scratch")

        if optimize_hyperparameters:
            # Generate random id for experiment tracking
            run_id = str(uuid.uuid4())
            if batch_size == 256:
                hyperparameter_file_name = os.path.join(results_save_path, "hyperparameter_tuning_results.json")
            else:
                hyperparameter_file_name = os.path.join(
                    results_save_path, f"hyperparameter_tuning_results_{batch_size}.json"
                )

        # Load data for encoder pretraining
        X_train_encoder = X[train_idx_encoder].astype(np.float32)
        X_val_encoder = X[val_idx_encoder].astype(np.float32)

        loss_fn = NTXentLoss(batch_size, temperature)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        use_all_augmentations = False if use_only_inversion_negation else True
        tr_dl, _ = simclr_data_loaders(
            X_train_encoder, X_val_encoder, batch_size, use_all_augmentations=use_all_augmentations
        )

        print(f"Created {model_name} model on device: {next(model.parameters()).device}")

        logging.info(f"Training parameters: {fp}")

        epoch_peak_memory = {}
        epoch_runtimes = {}
        epoch_train_loss = {}

        # Start recording memory snapshot history
        if torch.cuda.is_available():
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
            epoch_train_loss[f"{ep}"] = tr_loss
            print(f"Epoch {ep}/{epochs}: loss={tr_loss:.4f}")

        if batch_size != 256:
            run_time_save_name = f"runtime_per_epoch_{batch_size}.json"
            peak_memory_save_name = f"peak_memory_consumption_epochs_{batch_size}.json"
        else: # default one:
            run_time_save_name = "runtime_per_epoch.json"
            peak_memory_save_name = "peak_memory_consumption_epochs.json"

        with open(os.path.join(results_save_path, "training_loss_convergence.json"), "w") as f:
            json.dump(epoch_train_loss, f, indent=2)

        with open(os.path.join(results_save_path, run_time_save_name), "w") as f:
            json.dump(epoch_runtimes, f, indent=2)

        if torch.cuda.is_available():
            with open(os.path.join(results_save_path, peak_memory_save_name), "w") as f:
                json.dump(epoch_peak_memory, f, indent=2)

        # Save model locally
        saved_results = os.path.join(model_save_path, model_save_name_weights)
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
    conditions_train = conditions[train_idx][downstream_mask["train"]]

    test_repr = test_repr[downstream_mask["test"]]
    y_test = y[test_idx][downstream_mask["test"]]
    conditions_test = conditions[test_idx][downstream_mask["test"]]

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

    # ── Step 6a: Per-condition metrics ───────────────────────────────────────────
    if classifier_model in ["logistic_regression", "random_forest", "xgboost"]:
        per_condition_results = _per_condition_metrics(
            results["model"], test_repr, y_test, conditions_test
        )
        results["per_condition_metrics"] = per_condition_results
        print("Per-condition test metrics (each stressor vs baseline):")
        for cond, m in per_condition_results.items():
            print(f"  {cond}: AUROC={m['auroc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
                  f"n_stress={m['n_stress_samples']}")

    # ── Step 6b: Leave-one-stressor-out ──────────────────────────────────────────
    if leave_one_stressor_out and classifier_model in ["logistic_regression", "random_forest", "xgboost"]:
        loso_results = {}
        baseline_test_mask = y_test == 0
        for group_name, stressor_conditions in STRESSOR_GROUPS:
            held_out_train = np.isin(conditions_train, stressor_conditions) & (y_train == 1)
            X_tr_loso = train_repr[~held_out_train]
            y_tr_loso = y_train[~held_out_train]
            g_tr_loso = groups_train[~held_out_train]
            held_out_test = np.isin(conditions_test, stressor_conditions) & (y_test == 1)
            loso_test_mask = baseline_test_mask | held_out_test
            X_te_loso = test_repr[loso_test_mask]
            y_te_loso = y_test[loso_test_mask]
            if held_out_test.sum() == 0 or X_tr_loso.shape[0] == 0:
                print(f"LOSO [{group_name}]: skipping — no samples")
                continue
            cv_sp_loso, _ = get_participant_cv_splitter(
                g_tr_loso, min_participants_for_kfold=min_participants_for_kfold, k=k_folds
            )
            feat_names_loso = [f"repr_{i}" for i in range(X_tr_loso.shape[1])]
            res_loso = run_logistic_regression_with_gridsearch(
                X_tr_loso, y_tr_loso, g_tr_loso, X_te_loso, y_te_loso,
                feat_names_loso, cv_sp_loso, False, seed,
                scoring_metric=scoring_metric, classifier_model=classifier_model
            )

            dummy = DummyClassifier(strategy="most_frequent", random_state=seed)
            dummy.fit(X_tr_loso, y_tr_loso)
            dummy_pred = dummy.predict(X_te_loso)
            dummy_proba = dummy.predict_proba(X_te_loso)[:, 1]
            chance_metrics = {
                "auroc": float(roc_auc_score(y_te_loso, dummy_proba)),
                "pr_auc": float(average_precision_score(y_te_loso, dummy_proba)),
                "accuracy": float(accuracy_score(y_te_loso, dummy_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_te_loso, dummy_pred)),
                "f1": float(f1_score(y_te_loso, dummy_pred, zero_division=0)),
            }

            overall_stress_ratio = float((y_test == 1).sum()) / len(y_test)
            pr_auc_corrected, n_baseline_used = _pr_auc_ratio_corrected(
                res_loso["model"], X_te_loso, y_te_loso,
                overall_ratio=overall_stress_ratio, seed=seed,
            )
            loso_results[group_name] = {
                "held_out_stressor": stressor_conditions,
                "n_train_stress": int((y_tr_loso == 1).sum()),
                "n_test_stress": int(held_out_test.sum()),
                "test_metrics": res_loso["test_metrics"],
                "chance_level": chance_metrics,
                "test_metrics_ratio_corrected": {
                    "pr_auc": pr_auc_corrected,
                    "n_baseline_samples_used": n_baseline_used,
                    "overall_stress_ratio_used": round(overall_stress_ratio, 4),
                },
            }
            print(f"LOSO [{group_name}]: AUROC={res_loso['test_metrics']['auroc']:.4f} (chance={chance_metrics['auroc']:.4f}), "
                  f"PR-AUC (raw)={res_loso['test_metrics']['pr_auc']:.4f} (chance={chance_metrics['pr_auc']:.4f}), "
                  f"PR-AUC (ratio-corrected)={pr_auc_corrected:.4f}")

        loso_file = "loso_stressor_results.json" if batch_size == 256 else f"loso_stressor_results_{batch_size}.json"
        with open(os.path.join(results_save_path, loso_file), "w") as f:
            json.dump(loso_results, f, indent=2, default=str)

    # ── Step 6: Save Results ────────────────────────────────────────────────────
    # Different save name for non-default batch size
    test_result_name = "test_results.json" if batch_size == 256 else f"test_results_{batch_size}.json"

    # Track hyperparameter results:
    if optimize_hyperparameters:
        hyperparameter_save_file = return_track_data_file(hyperparameter_file_name)

        hyperparameter_save_file["runs"][run_id] = {
            "hyperparameters": {
                "temperature": temperature,
                "batch_size": batch_size,
                "use_only_inversion_negation": use_only_inversion_negation,
                "tstcc_encoder_used": use_tstcc_encoder,
                "epochs": epochs,
            },
            "Training loss": list(epoch_train_loss.values())[-5:],
            "CV score": results['best_cv_score'], #Criterion for checking the performance and selecting best hyperparameter set
            "Test_metrics": results["test_metrics"],
            "timestamp": datetime.now().isoformat(),
        }

        with open(hyperparameter_file_name, "w") as f:
            json.dump(hyperparameter_save_file, f, indent=4)

    with open(os.path.join(results_save_path, test_result_name), "w") as f:
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
    simclr_group.add_argument("--temperature", type=float, default=0.1, #optimal value
                             help="Temperature parameter for contrastive loss")
    simclr_group.add_argument("--use_tstcc_encoder", action="store_true",
                              help="If set, we use the tstcc encoder (differs from the original architecture.")
    simclr_group.add_argument("--use_only_inversion_negation", action="store_true",
                              help="If set, we use only inversion and negation for the augmentation instead of all.")

    # Use only inversion & negation was used in this paper:
    # Contrastive Self-Supervised Learning for Stress Detection from ECG Data
    # S3 configurations
    simclr_group.add_argument("--use_s3_layers", action="store_true",
                                  help="If set, we use the S3 layer")
    simclr_group.add_argument("--initial_num_segments", type=int, default=2)
    simclr_group.add_argument("--num_s3_layers", type=int, default=2)
    simclr_group.add_argument("--segment_multiplier", type=int, default=1)

    # ══════════════════════════════════════════════════════════════════════════════
    # Hyperparameter Optimization Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    hp_group = parser.add_argument_group('Hyperparameter Optimization')
    hp_group.add_argument("--optimize_hyperparameters", action="store_true",
                         help="Enable hyperparameter optimization for SimCLR augmentation parameters."
                              "It basically saves the run of the CV validation so we can retrieve it later."
                              "We do due to compute costs grid-search based tuning.")

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

    # ══════════════════════════════════════════════════════════════════════════════
    # Leave-one-stressor-out
    # ══════════════════════════════════════════════════════════════════════════════
    loso_group = parser.add_argument_group("Leave-one-stressor-out")
    loso_group.add_argument("--leave_one_stressor_out", action="store_true",
                            help="Run leave-one-stressor-out analysis: for each stressor group "
                                 "(TA+TA_repeat, Pasat+Pasat_repeat, Raven, SSST), train without "
                                 "that stressor and evaluate on it.")

    # Parse arguments and run main function
    args = parser.parse_args()

    # Important:
    args.pretrain_all_conditions = True

    main(**vars(args))