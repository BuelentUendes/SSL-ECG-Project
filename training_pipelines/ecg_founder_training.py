"""
ECGFounder Training Pipeline with CV and Logistic Regression.

Loads the pretrained single-lead ECGFounder weights from HuggingFace
(PKUDigitalHealth/ECGFounder), extracts 1024-dim representations, and
trains a downstream classifier — mirroring tstcc_train_cleaned_cv.py.

Expected input shape stored in the H5 file: (N, L, 1) where L = fs * window_size.
ECGFounder expects (B, 1, 5000) at 500 Hz; signals are resampled on-the-fly.
"""
import os
import json
import argparse
import logging
import gc
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.signal import resample as scipy_resample
from huggingface_hub import hf_hub_download
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, balanced_accuracy_score

from utils.torch_utilities import (
    load_processed_data,
    load_processed_data_with_conditions,
    split_indices_by_participant_groups,
    set_seed,
    create_directory,
    get_participant_cv_splitter,
    run_logistic_regression_with_gridsearch,
    run_logistic_regression_with_gridsearch_verbose,
    run_mlp_with_cv_and_test,
    evaluate_zero_shot_model_performance,
    analyze_subcategory_confusion,
    PhysiologicalDataset,
)
from utils.helper_paths import DATA_PATH, RESULTS_PATH
from models.net1d import Net1D


ECGFOUNDER_REPO = "PKUDigitalHealth/ECGFounder"
ECGFOUNDER_FILENAME = "1_lead_ECGFounder.pth"
# ECGFounder was trained on 10-second windows at 500 Hz
ECGFOUNDER_FS = 500
ECGFOUNDER_LENGTH = 5000
MODEL_NAME = "ECGFounder"

STRESSOR_GROUPS = [
    ("TA", ["TA", "TA_repeat"]),
    ("Pasat", ["Pasat", "Pasat_repeat"]),
    ("Raven", ["Raven"]),
    ("SSST", ["SSST_Sing_countdown"]),
]


def _per_condition_metrics(model, test_repr, y_test, conditions_test):
    """Compute binary (vs baseline) metrics per mental-stress condition using a trained sklearn model."""
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


def _per_condition_metrics_scratch(model, X_test, y_test, conditions_test, device, batch_size=64):
    """Per-condition binary (vs baseline) metrics for a from-scratch ECGFounder (PyTorch) model.

    X_test must already be preprocessed to shape (N, ECGFOUNDER_LENGTH, 1).
    """
    baseline_mask = y_test == 0
    out = {}
    model.eval()
    for cond in np.unique(conditions_test[y_test == 1]):
        cond_mask = (conditions_test == cond) & (y_test == 1)
        mask = baseline_mask | cond_mask
        if mask.sum() < 2 or cond_mask.sum() == 0:
            continue
        X_s = X_test[mask]
        y_s = y_test[mask].astype(int)
        ds = PhysiologicalDataset(X_s, y_s)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        proba_list = []
        with torch.no_grad():
            for X_b, _ in loader:
                logits, _ = model(X_b.to(device).permute(0, 2, 1))
                proba_list.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
        proba = np.array(proba_list)
        pred = (proba > 0.5).astype(int)
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


def _build_ecgfounder(device):
    """Download (once, cached by huggingface_hub) and load the pretrained single-lead encoder."""
    print(f"Downloading / loading {ECGFOUNDER_FILENAME} from {ECGFOUNDER_REPO} ...")
    ckpt_path = hf_hub_download(repo_id=ECGFOUNDER_REPO, filename=ECGFOUNDER_FILENAME)

    model = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=1,       # placeholder — dense head is not used
        use_bn=False,
        use_do=False,
        return_features=True,
        verbose=False,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Skip the classification head; we only use the backbone as a feature extractor
    sd = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith("dense.")}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print("ECGFounder encoder loaded successfully.")
    return model


def _build_ecgfounder_untrained(device):
    """Build the ECGFounder Net1D architecture without loading any pretrained weights."""
    model = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=1,
        use_bn=False,
        use_do=False,
        return_features=True,
        verbose=False,
    )
    model.to(device)
    print("ECGFounder architecture initialised from scratch (no pretrained weights).")
    return model


def encode_representations(X, model, batch_size, device,  use_window_level_normalization=False):
    """Extract ECGFounder representations from raw windowed ECG.

    Args:
        X: np.ndarray of shape (N, L, 1) — windowed ECG at any sampling rate
        model: Net1D with return_features=True
        batch_size: int
        device: torch.device
        use_window_level_normalization: bool = False

    Returns:
        np.ndarray of shape (N, 1024)
    """
    L = X.shape[1]
    model.eval()
    all_reprs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].astype(np.float32)  # (B, L, 1)
            batch = batch.transpose(0, 2, 1)                 # (B, 1, L)

            # Resample to the length ECGFounder was trained on (5000 samples = 10s @ 500 Hz)
            if L != ECGFOUNDER_LENGTH:
                batch = scipy_resample(batch, ECGFOUNDER_LENGTH, axis=-1)

            # Z-score normalise per signal (same as ECGFounder preprocessing)
            if use_window_level_normalization:
                mean = batch.mean(axis=-1, keepdims=True)
                std = batch.std(axis=-1, keepdims=True)
                batch = (batch - mean) / (std + 1e-8)

            batch = np.nan_to_num(batch)

            x = torch.FloatTensor(batch).to(device)
            _, features = model(x)
            all_reprs.append(features.cpu().numpy())

    return np.concatenate(all_reprs, axis=0)


def _preprocess_for_ecgfounder(X: np.ndarray) -> np.ndarray:
    """Resample to ECGFOUNDER_LENGTH and z-score normalise per signal.

    Args:
        X: (N, L, 1) float array at any sampling rate
    Returns:
        (N, ECGFOUNDER_LENGTH, 1) float32 array, z-score normalised
    """
    L = X.shape[1]
    X_t = X.astype(np.float32).transpose(0, 2, 1)  # (N, 1, L)
    if L != ECGFOUNDER_LENGTH:
        X_t = scipy_resample(X_t, ECGFOUNDER_LENGTH, axis=-1)
    mean = X_t.mean(axis=-1, keepdims=True)
    std = X_t.std(axis=-1, keepdims=True)
    X_t = (X_t - mean) / (std + 1e-8)
    X_t = np.nan_to_num(X_t)
    return X_t.transpose(0, 2, 1)  # (N, ECGFOUNDER_LENGTH, 1)


def run_ecgfounder_scratch_with_cv_and_test(
        X_train,
        y_train,
        groups_train,
        X_test,
        y_test,
        cv_splitter,
        device,
        num_epochs: int = 25,
        batch_size: int = 32,
        seed: int = 42,
        scoring_metric: str = "roc_auc",
        results_save_path: str = ".",
):
    """Train ECGFounder from scratch end-to-end with CV hyperparameter tuning.

    Mirrors the protocol of run_supervised_model_with_cv_and_test in
    supervised_training_cleaned_cv.py: grid-searches over learning rates,
    tracks per-epoch validation scores across folds, then retrains the best
    configuration on the full training set and evaluates on the test set.

    Args:
        X_train: (N, ECGFOUNDER_LENGTH, 1) preprocessed training ECG windows
        y_train: (N,) binary labels
        groups_train: (N,) participant IDs for group-based CV splitting
        X_test:  (M, ECGFOUNDER_LENGTH, 1) preprocessed test ECG windows
        y_test:  (M,) binary labels
        cv_splitter: GroupKFold / LOPOCV splitter or None
        device: torch.device
        num_epochs: epochs for CV folds and final model
        batch_size: mini-batch size
        seed: random seed for final model initialisation
        scoring_metric: one of roc_auc, average_precision, f1, balanced_accuracy
        results_save_path: directory for runtime / memory / CV-curve JSON files

    Returns:
        dict with keys: best_params, best_cv_score, test_metrics (dict),
                        total_params, average_epoch_loss, model (Net1D instance)
    """
    lr_rates = [1e-4, 1e-5]
    num_workers = min(8, os.cpu_count() or 2)
    non_blocking_bool = torch.cuda.is_available()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_params = None
    best_cv_score = 0.0
    cv_val_scores_per_epoch: dict = {}

    default_best_params = {"lr": 1e-5}

    print(f"Running CV hyperparameter search for ECGFounder (from scratch) ...")

    if cv_splitter is None:
        print("No CV splitter available (single participant). Using default parameters.")
        best_params = default_best_params
    else:
        for lr in lr_rates:
            print(f"  Testing lr={lr}")
            fold_scores = []
            fold_epoch_scores = []

            for fold, (train_idx, val_idx) in enumerate(
                    cv_splitter.split(X_train, y_train, groups_train), 1):
                X_fold_train = X_train[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]

                model = _build_ecgfounder_untrained(device)
                optimizer = optim.AdamW(model.parameters(), lr=lr)

                tr_ds = PhysiologicalDataset(X_fold_train, y_fold_train)
                val_ds = PhysiologicalDataset(X_fold_val, y_fold_val)
                tr_loader = DataLoader(
                    tr_ds, batch_size=batch_size, shuffle=True,
                    drop_last=True, num_workers=num_workers,
                )
                val_loader = DataLoader(
                    val_ds, batch_size=batch_size, shuffle=False,
                    drop_last=False, num_workers=num_workers,
                )

                this_fold_epoch_scores = []

                for epoch_idx in range(1, num_epochs + 1):
                    print(f"    Fold {fold}: epoch {epoch_idx}/{num_epochs}", end="\r", flush=True)
                    model.train()
                    for X_b, y_b in tr_loader:
                        X_b = X_b.to(device, non_blocking=non_blocking_bool).permute(0, 2, 1)
                        y_b = y_b.to(device, non_blocking=non_blocking_bool).float()
                        optimizer.zero_grad()
                        logits, _ = model(X_b)
                        loss = loss_fn(logits.squeeze(-1), y_b)
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    val_probs, val_labels = [], []
                    with torch.no_grad():
                        for X_b, y_b in val_loader:
                            X_b = X_b.to(device, non_blocking=non_blocking_bool).permute(0, 2, 1)
                            logits, _ = model(X_b)
                            probs = torch.sigmoid(logits.squeeze(-1))
                            val_probs.extend(probs.cpu().numpy())
                            val_labels.extend(y_b.numpy())

                    if scoring_metric == "roc_auc":
                        epoch_val_score = roc_auc_score(val_labels, val_probs)
                    elif scoring_metric == "average_precision":
                        epoch_val_score = average_precision_score(val_labels, val_probs)
                    elif scoring_metric == "f1":
                        val_preds = (np.array(val_probs) > 0.5).astype(int)
                        epoch_val_score = f1_score(val_labels, val_preds)
                    elif scoring_metric == "balanced_accuracy":
                        val_preds = (np.array(val_probs) > 0.5).astype(int)
                        epoch_val_score = balanced_accuracy_score(val_labels, val_preds)
                    else:
                        raise ValueError(f"Unknown scoring metric: {scoring_metric}")

                    this_fold_epoch_scores.append(epoch_val_score)

                fold_scores.append(this_fold_epoch_scores[-1])
                fold_epoch_scores.append(this_fold_epoch_scores)

            mean_cv_score = np.mean(fold_scores)
            mean_epoch_scores = np.mean(fold_epoch_scores, axis=0).tolist()
            std_epoch_scores = np.std(fold_epoch_scores, axis=0).tolist()
            cv_val_scores_per_epoch[lr] = {"mean": mean_epoch_scores, "std": std_epoch_scores}
            print(f"\n  lr={lr}: mean CV {scoring_metric}={mean_cv_score:.4f}")

            if mean_cv_score > best_cv_score:
                best_cv_score = mean_cv_score
                best_params = {"lr": lr}

    print(f"\nBest parameters: {best_params}, Best CV score: {best_cv_score:.4f}")

    # Train final model with best params on full training set
    print("Training final ECGFounder model on full training set ...")
    set_seed(seed)
    final_model = _build_ecgfounder_untrained(device)
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params["lr"])

    total_params = sum(p.numel() for p in final_model.parameters())
    trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    tr_ds = PhysiologicalDataset(X_train, y_train)
    te_ds = PhysiologicalDataset(X_test, y_test)
    tr_loader = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers,
    )
    te_loader = DataLoader(
        te_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
    )

    epoch_runtimes: dict = {}
    epoch_peak_memory: dict = {}
    average_epoch_loss = []

    for epoch_idx in range(1, num_epochs + 1):
        print(f"\nFinal training: Epoch {epoch_idx}/{num_epochs}", end="\r")
        final_model.train()
        epoch_start = time.time()
        epoch_losses = []

        for X_b, y_b in tr_loader:
            X_b = X_b.to(device, non_blocking=non_blocking_bool).permute(0, 2, 1)
            y_b = y_b.to(device, non_blocking=non_blocking_bool).float()
            optimizer.zero_grad()
            logits, _ = final_model(X_b)
            loss = loss_fn(logits.squeeze(-1), y_b)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        average_epoch_loss.append(avg_loss)
        epoch_runtimes[str(epoch_idx)] = time.time() - epoch_start
        print(f"Epoch {epoch_idx}: avg loss={avg_loss:.4f}")

        if torch.cuda.is_available():
            epoch_peak_memory[str(epoch_idx)] = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Peak memory epoch {epoch_idx}: "
                  f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")

    # Evaluate on test set
    final_model.eval()
    test_probs, test_preds, test_labels = [], [], []
    with torch.no_grad():
        for X_b, y_b in te_loader:
            X_b = X_b.to(device, non_blocking=non_blocking_bool).permute(0, 2, 1)
            logits, _ = final_model(X_b)
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).float()
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y_b.numpy())

    test_probs = np.array(test_probs)
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    test_acc = accuracy_score(test_labels, test_preds)
    test_auroc = roc_auc_score(test_labels, test_probs)
    test_f1 = f1_score(test_labels, test_preds)
    test_pr_auc = average_precision_score(test_labels, test_probs)
    test_ba = balanced_accuracy_score(test_labels, test_preds)

    print(f"\n=== Test Set Results ===")
    print(f"Accuracy: {test_acc:.4f}, AUROC: {test_auroc:.4f}, "
          f"F1: {test_f1:.4f}, PR-AUC: {test_pr_auc:.4f}, "
          f"Balanced Acc: {test_ba:.4f}")

    with open(os.path.join(results_save_path, "runtime_per_epoch.json"), "w") as f:
        json.dump(epoch_runtimes, f, indent=2)

    if torch.cuda.is_available():
        with open(os.path.join(results_save_path, "peak_memory_consumption_epochs.json"), "w") as f:
            json.dump(epoch_peak_memory, f, indent=2)

    if cv_val_scores_per_epoch:
        cv_curves = {f"lr={k}": v for k, v in cv_val_scores_per_epoch.items()}
        with open(os.path.join(results_save_path, "cv_val_scores_per_epoch.json"), "w") as f:
            json.dump(cv_curves, f, indent=2)

    return {
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "test_metrics": {
            "accuracy": test_acc,
            "auroc": test_auroc,
            "f1": test_f1,
            "pr_auc": test_pr_auc,
            "balanced_accuracy": test_ba,
        },
        "total_params": total_params,
        "average_epoch_loss": average_epoch_loss,
        "model": final_model,
    }


def main(
        fs: str,
        window_size: int,
        step_size: int,
        gpu: int,
        seed: int,
        encoding_batch_size: int,
        use_window_level_normalization: bool,
        label_fraction: float,
        save_embeddings: bool,
        classification_task: str,
        classifier_model: str,
        classifier_epochs: int,
        classifier_lr: float,
        classifier_batch_size: int,
        k_folds: int = 5,
        min_participants_for_kfold: int = 5,
        verbose: bool = False,
        scoring_metric: str = "roc_auc",
        zero_shot_evaluation: bool = False,
        zero_shot_dataset: str = "wesad",
        leave_one_stressor_out: bool = False,
        train_from_scratch: bool = False,
):
    # ── Step 0: Setup ────────────────────────────────────────────────────────────
    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logging.basicConfig(level=logging.INFO)
    mode_tag = "from_scratch" if train_from_scratch else f"pretrained/{classifier_model}"
    print(f"Starting {MODEL_NAME} training [{mode_tag}], seed={seed}, label_fraction={label_fraction}")
    print(f"Using device: {device}")

    create_directory(RESULTS_PATH)

    if train_from_scratch:
        results_save_path = os.path.join(
            RESULTS_PATH, "ECG", str(fs), MODEL_NAME, "from_scratch", classification_task,
            f"{seed}", f"{label_fraction}", f"{window_size}", f"{step_size}"
        )
    else:
        results_save_path = os.path.join(
            RESULTS_PATH, "ECG", str(fs), MODEL_NAME, classifier_model, classification_task,
            f"{seed}", f"{label_fraction}", f"{window_size}", f"{step_size}"
        )

    embedding_save_path = os.path.join(
        DATA_PATH, "embeddings", "ECG", f"{fs}", MODEL_NAME, classification_task, f"{seed}", f"{window_size}", f"{step_size}"
    )

    X_zero_shot = y_zero_shot = zero_shot_results_path = zero_shot_window_data_path = None

    if zero_shot_evaluation:
        target_domain = "StressID" if zero_shot_dataset == "stressid" else "WESAD"
        zero_shot_results_path = os.path.join(
            RESULTS_PATH, "Transfer_learning", target_domain, "zero_shot_performance",
            MODEL_NAME, classifier_model, f"{seed}", f"{label_fraction}",
        )
        create_directory(zero_shot_results_path)

    create_directory(results_save_path)
    create_directory(embedding_save_path)

    # ── Step 1: Load Data ────────────────────────────────────────────────────────
    if classification_task == "ms_base_lpa_mpa":
        label_map = {"baseline": 0, "low_physical_activity": 0, "moderate_physical_activity": 0, "mental_stress": 1}
    else:
        label_map = {"baseline": 0, "mental_stress": 1}

    window_data_path = os.path.join(
        DATA_PATH, "interim", "ECG", str(fs), str(window_size), str(step_size), "windowed_data.h5"
    )

    if zero_shot_evaluation:
        if zero_shot_dataset == "wesad":
            if int(fs) == 700:
                zero_shot_window_data_path = os.path.join(
                    DATA_PATH, "interim", "WESAD", "ECG", str(fs), str(window_size), str(step_size), "windowed_data.h5"
                )
            else:
                raise ValueError("For zero-shot evaluation for wesad the frequency needs to be 700")
        elif zero_shot_dataset == "stressid":
            if int(fs) == 500:
                zero_shot_window_data_path = os.path.join(
                    DATA_PATH, "interim", "STRESSID", "ECG", str(fs), str(window_size), str(step_size), "windowed_data.h5"
                )
            else:
                raise ValueError("For zero-shot evaluation for stressid the frequency needs to be 500")
        else:
            raise ValueError('Please use a proper dataset: "wesad" or "stressid"')

    X, y, groups, conditions = load_processed_data_with_conditions(window_data_path, label_map=label_map)
    y = y.astype(np.float32)

    if zero_shot_evaluation:
        X_zero_shot, y_zero_shot, groups_shot = load_processed_data(
            zero_shot_window_data_path, label_map={"baseline": 0, "mental_stress": 1}
        )
        y_zero_shot = y_zero_shot.astype(np.float32)

    # ── Step 2: Participant Split ─────────────────────────────────────────────────
    train_idx, train_p, all_train_p, all_train_idx, test_idx, test_p = split_indices_by_participant_groups(
        groups,
        train_ratio=0.8,
        label_fraction=label_fraction,
        seed=seed,
        return_all_train_p=True,
    )

    downstream_mask = {
        "train": np.isin(y[train_idx], [0, 1]),
        "test": np.isin(y[test_idx], [0, 1]),
    }

    # ── Steps 3–8: branch on train_from_scratch ──────────────────────────────────
    if train_from_scratch:
        # ── Step 3: Build ECGFounder architecture from scratch ────────────────
        model = _build_ecgfounder_untrained(device)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total: {total:,}  trainable: {trainable:,}")

        if save_embeddings:
            print("Note: --save_embeddings is not applicable in --train_from_scratch mode; skipping.")

        # ── Step 4: Preprocess raw ECG (resample to 5000 + z-score) ──────────
        print("Preprocessing training data for ECGFounder (resample + z-score) ...")
        X_train_proc = _preprocess_for_ecgfounder(X[train_idx][downstream_mask["train"]])
        print("Preprocessing test data for ECGFounder (resample + z-score) ...")
        X_test_proc = _preprocess_for_ecgfounder(X[test_idx][downstream_mask["test"]])

        y_train = y[train_idx][downstream_mask["train"]]
        groups_train = groups[train_idx][downstream_mask["train"]]
        conditions_train = conditions[train_idx][downstream_mask["train"]]
        y_test = y[test_idx][downstream_mask["test"]]
        conditions_test = conditions[test_idx][downstream_mask["test"]]

        print(f"X_train_proc shape = {X_train_proc.shape}")

        # ── Step 5: Set up Cross-Validation Splitter ──────────────────────────
        cv_splitter, n_splits = get_participant_cv_splitter(
            groups_train,
            min_participants_for_kfold=min_participants_for_kfold,
            k=k_folds,
        )

        # ── Step 6: Train ECGFounder end-to-end with CV hyperparameter search ─
        set_seed(seed)
        scratch_results = run_ecgfounder_scratch_with_cv_and_test(
            X_train_proc, y_train, groups_train, X_test_proc, y_test,
            cv_splitter, device,
            num_epochs=classifier_epochs,
            batch_size=classifier_batch_size,
            seed=seed,
            scoring_metric=scoring_metric,
            results_save_path=results_save_path,
        )
        final_model = scratch_results.pop("model")
        results = scratch_results

        print(f"Best CV score: {results['best_cv_score']:.4f}")
        print(f"Test metrics - Accuracy: {results['test_metrics']['accuracy']:.4f}, "
              f"AUROC: {results['test_metrics']['auroc']:.4f}, "
              f"F1: {results['test_metrics']['f1']:.4f}, "
              f"PR-AUC: {results['test_metrics']['pr_auc']:.4f}")

        # ── Step 7a: Per-condition metrics ────────────────────────────────────
        per_condition_results = _per_condition_metrics_scratch(
            final_model, X_test_proc, y_test, conditions_test, device,
            batch_size=classifier_batch_size,
        )
        results["per_condition_metrics"] = per_condition_results
        print("Per-condition test metrics (each stressor vs baseline):")
        for cond, m in per_condition_results.items():
            print(f"  {cond}: AUROC={m['auroc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
                  f"n_stress={m['n_stress_samples']}")

        if leave_one_stressor_out:
            print("Note: --leave_one_stressor_out is not supported in --train_from_scratch mode; skipping.")

        if zero_shot_evaluation:
            print("Note: --zero_shot_evaluation is not supported in --train_from_scratch mode; skipping.")

        # ── Step 8: Save Results ──────────────────────────────────────────────
        with open(os.path.join(results_save_path, "test_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Additional parameters - Label fraction: {label_fraction}, Seed: {seed}, "
              f"K-folds: {k_folds}, CV splits: {n_splits}")

    else:
        # ── Step 3: Load Pretrained ECGFounder Encoder ───────────────────────
        model = _build_ecgfounder(device)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"total: {total:,}  trainable: {trainable:,}")

        # ── Step 4: Extract Representations ──────────────────────────────────
        print("Extracting train representations ...")
        train_repr = encode_representations(X[train_idx], model, encoding_batch_size, device,
                                            use_window_level_normalization=use_window_level_normalization)
        print("Extracting test representations ...")
        test_repr = encode_representations(X[test_idx], model, encoding_batch_size, device,
                                           use_window_level_normalization=use_window_level_normalization)

        if save_embeddings:
            print("Saving embeddings for later analysis ...")
            x_repr_all = encode_representations(X, model, encoding_batch_size, device,
                                                use_window_level_normalization=use_window_level_normalization)
            np.savez(
                os.path.join(embedding_save_path, "x_y_groups_embedding.npz"),
                array1=x_repr_all, array_2=y, array_3=groups
            )
            print("Embeddings saved.")

        # filter to binary downstream samples
        train_repr = train_repr[downstream_mask["train"]]
        y_train = y[train_idx][downstream_mask["train"]]
        groups_train = groups[train_idx][downstream_mask["train"]]
        conditions_train = conditions[train_idx][downstream_mask["train"]]

        test_repr = test_repr[downstream_mask["test"]]
        y_test = y[test_idx][downstream_mask["test"]]
        conditions_test = conditions[test_idx][downstream_mask["test"]]

        print(f"train_repr shape = {train_repr.shape}")

        # ── Step 5: Set up Cross-Validation Splitter ──────────────────────────
        cv_splitter, n_splits = get_participant_cv_splitter(
            groups_train,
            min_participants_for_kfold=min_participants_for_kfold,
            k=k_folds,
        )

        # ── Step 6: Run CV with Logistic Regression or MLP ───────────────────
        set_seed(seed)
        feature_names = [f"repr_{i}" for i in range(train_repr.shape[1])]

        if classifier_model in ["logistic_regression", "random_forest", "xgboost"]:
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
            print(f"Best CV AUROC: {results['best_cv_score'] if cv_splitter is not None else 0:.4f}")
        else:
            results = run_mlp_with_cv_and_test(
                train_repr, y_train, groups_train,
                test_repr, y_test, feature_names, cv_splitter,
                device, classifier_epochs, classifier_batch_size, classifier_lr, False, seed
            )
            print(f"Best CV AUROC: {results['best_cv_score']:.4f}")

        print(f"Test metrics - Accuracy: {results['test_metrics']['accuracy']:.4f}, "
              f"AUROC: {results['test_metrics']['auroc']:.4f}, F1: {results['test_metrics']['f1']:.4f}, "
              f"PR-AUC: {results['test_metrics']['pr_auc']:.4f}")

        # ── Step 7a: Per-condition metrics ────────────────────────────────────
        if classifier_model in ["logistic_regression", "random_forest", "xgboost"]:
            per_condition_results = _per_condition_metrics(
                results["model"], test_repr, y_test, conditions_test
            )
            results["per_condition_metrics"] = per_condition_results
            print("Per-condition test metrics (each stressor vs baseline):")
            for cond, m in per_condition_results.items():
                print(f"  {cond}: AUROC={m['auroc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
                      f"n_stress={m['n_stress_samples']}")

        # ── Step 7a-ii: Subcategory confusion (LPA/MPA discrimination) ───────
        if classification_task == "ms_base_lpa_mpa" and classifier_model in ["logistic_regression", "random_forest", "xgboost"]:
            y_test_pred = results["model"].predict(test_repr)
            subcategory_confusion = analyze_subcategory_confusion(
                y_true=y_test.astype(int),
                y_pred=y_test_pred.astype(int),
                categories=conditions_test,
                save_path=results_save_path,
                save_name="test_results.json",
            )
            results["subcategory_confusion"] = subcategory_confusion
            coarse = subcategory_confusion.get("summary", {}).get("coarse_grained_categories", {})
            def _fmt(v):
                return f"{v:.4f}" if v is not None else "N/A"

            print("Subcategory confusion (coarse-grained):")
            for cat, m in coarse.items():
                print(f"  {cat}: Sensitivity={_fmt(m['sensitivity_recall'])}, "
                      f"Specificity={_fmt(m['specificity'])}, FPR={_fmt(m['false_positive_rate'])}, "
                      f"n={m['sample_count']}")

        # ── Step 7b: Leave-one-stressor-out ───────────────────────────────────
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
                print(f"LOSO [{group_name}]: AUROC={res_loso['test_metrics']['auroc']:.4f} "
                      f"(chance={chance_metrics['auroc']:.4f}), "
                      f"PR-AUC (raw)={res_loso['test_metrics']['pr_auc']:.4f} "
                      f"(chance={chance_metrics['pr_auc']:.4f}), "
                      f"PR-AUC (ratio-corrected)={pr_auc_corrected:.4f}")

            with open(os.path.join(results_save_path, "loso_stressor_results.json"), "w") as f:
                json.dump(loso_results, f, indent=2, default=str)

        # ── Step 7c: Zero-shot evaluation ─────────────────────────────────────
        if zero_shot_evaluation:
            classifier = results["model"]
            zero_shot_repr = encode_representations(X_zero_shot, model, encoding_batch_size, device)
            zero_shot_results = evaluate_zero_shot_model_performance(classifier, zero_shot_repr, y_zero_shot)
            with open(os.path.join(zero_shot_results_path, "zero_shot_results.json"), "w") as f:
                json.dump(zero_shot_results, f, indent=2, default=str)

        # ── Step 8: Save Results ──────────────────────────────────────────────
        test_result_file_name = "test_results_window_normalization.json" if use_window_level_normalization else "test_results.json"
        with open(os.path.join(results_save_path, test_result_file_name), "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Additional parameters - Classifier: {classifier_model}, Label fraction: {label_fraction}, "
              f"Seed: {seed}, K-folds: {k_folds}, CV splits: {n_splits}")

    # ── Cleanup ────────────────────────────────────────────────────────────────────
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
        description="ECGFounder Training Pipeline with CV and Logistic Regression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ══════════════════════════════════════════════════════════════════════════════
    # General Setup
    # ══════════════════════════════════════════════════════════════════════════════
    general_group = parser.add_argument_group("General Setup")
    general_group.add_argument("--gpu", type=int, default=0,
                               help="GPU device ID to use")
    general_group.add_argument("--seed", type=int, default=42,
                               help="Random seed for reproducibility")
    general_group.add_argument("--verbose", action="store_true",
                               help="Show verbose output of CV for logistic regression")

    # ══════════════════════════════════════════════════════════════════════════════
    # Data Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--fs", default=500, type=str,
                            help="Sampling frequency of the stored ECG windows")
    data_group.add_argument("--window_size", type=int, default=10,
                            help="Window size in seconds")
    data_group.add_argument("--step_size", type=int, default=5,
                            help="Step size in seconds for sliding window")
    data_group.add_argument("--label_fraction", type=float, default=1.0,
                            help="Fraction of labeled train participants to use (0.0-1.0)")
    data_group.add_argument("--save_embeddings", action="store_true",
                            help="Save all representations to disk for later analysis")
    data_group.add_argument("--classification_task", default="ms_base",
                            type=str, choices=("ms_base", "ms_base_lpa_mpa"),
                            help="Downstream classification task: ms_base (binary) or ms_base_lpa_mpa "
                                 "(mental stress vs baseline+lpa+mpa, still binary).")

    # ══════════════════════════════════════════════════════════════════════════════
    # Encoding
    # ══════════════════════════════════════════════════════════════════════════════
    enc_group = parser.add_argument_group("Encoding")
    enc_group.add_argument("--encoding_batch_size", type=int, default=64,
                           help="Batch size used when extracting ECGFounder representations")
    enc_group.add_argument("--use_window_level_normalization", action="store_true",
                           help="If set, we use additional window level normalization, as this is the default setup used.")

    # ══════════════════════════════════════════════════════════════════════════════
    # Downstream Classifier Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    classifier_group = parser.add_argument_group("Downstream Classifier")
    classifier_group.add_argument("--classifier_model", type=str, default="logistic_regression",
                                  choices=("logistic_regression", "mlp", "random_forest", "xgboost"),
                                  help="Type of downstream classifier to use (ignored when --train_from_scratch)")
    classifier_group.add_argument("--classifier_epochs", type=int, default=25,
                                  help="Number of epochs for MLP classifier training or from-scratch end-to-end training")
    classifier_group.add_argument("--classifier_lr", type=float, default=1e-4,
                                  help="Learning rate for MLP classifier")
    classifier_group.add_argument("--classifier_batch_size", type=int, default=32,
                                  help="Batch size for MLP classifier or from-scratch training")

    # ══════════════════════════════════════════════════════════════════════════════
    # Cross-Validation Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    cv_group = parser.add_argument_group("Cross-Validation")
    cv_group.add_argument("--k_folds", type=int, default=5,
                          help="Number of folds for cross-validation")
    cv_group.add_argument("--min_participants_for_kfold", type=int, default=5,
                          help="Minimum participants needed for k-fold (otherwise LOPOCV)")
    cv_group.add_argument("--scoring_metric", type=str, default="roc_auc",
                          choices=["roc_auc", "average_precision", "f1", "balanced_accuracy"],
                          help="Scoring metric for CV hyperparameter selection")

    # ══════════════════════════════════════════════════════════════════════════════
    # Zero-shot evaluation
    # ══════════════════════════════════════════════════════════════════════════════
    zero_shot_group = parser.add_argument_group("Zero-shot evaluation")
    zero_shot_group.add_argument("--zero_shot_evaluation", action="store_true",
                                 help="Run downstream zero-shot evaluation on an external dataset")
    zero_shot_group.add_argument("--zero_shot_dataset", type=str,
                                 choices=("stressid", "wesad"), default="wesad")

    # ══════════════════════════════════════════════════════════════════════════════
    # Leave-one-stressor-out
    # ══════════════════════════════════════════════════════════════════════════════
    loso_group = parser.add_argument_group("Leave-one-stressor-out")
    loso_group.add_argument("--leave_one_stressor_out", action="store_true",
                            help="Run leave-one-stressor-out analysis: for each stressor group "
                                 "(TA+TA_repeat, Pasat+Pasat_repeat, Raven, SSST), train without "
                                 "that stressor and evaluate on it.")

    # ══════════════════════════════════════════════════════════════════════════════
    # From-scratch training
    # ══════════════════════════════════════════════════════════════════════════════
    scratch_group = parser.add_argument_group("From-scratch training")
    scratch_group.add_argument("--train_from_scratch", action="store_true",
                               help="Train the ECGFounder architecture end-to-end from random "
                                    "initialisation instead of using pretrained HuggingFace weights. "
                                    "Uses CV over learning rates [1e-4, 1e-5] and BCEWithLogitsLoss, "
                                    "mirroring the supervised_training_cleaned_cv.py protocol.")

    args = parser.parse_args()
    main(**vars(args))