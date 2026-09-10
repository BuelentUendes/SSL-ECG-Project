import os
import json
import argparse
import gc
import time
import tracemalloc

import numpy as np
import pandas as pd
import torch

from tabpfn import TabPFNClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score,
)

from utils.torch_utilities import (
    load_processed_data,
    split_indices_by_participant_groups,
    set_seed,
    create_directory,
    evaluate_zero_shot_model_performance,
    standardize_features,
)
from utils.helper_paths import DATA_PATH, RESULTS_PATH

TABPFN_MAX_TRAIN_SAMPLES = 1_000_000 # On GPU


def get_device(gpu: int = 0) -> str:
    if torch.cuda.is_available():
        return f"cuda:{gpu}"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def handle_missing_data(data, drop_values=True, verbose=True):
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
        was_numpy = True
    else:
        df = data.copy()
        was_numpy = False

    original_len = len(df)

    inf_mask = df.isin([np.inf, -np.inf])
    if verbose and inf_mask.any().any():
        print(f"Rows with infinity values: {inf_mask.any(axis=1).sum()}")

    nan_mask = df.isna()
    if verbose and nan_mask.any().any():
        print(f"Rows with NaN values: {nan_mask.any(axis=1).sum()}")

    if drop_values:
        clean = df[~df.isin([np.inf, -np.inf]).any(axis=1)].dropna(axis=0)
        dropped_pct = (original_len - len(clean)) / original_len * 100
        if verbose:
            print(f"Dropping rows removed {dropped_pct:.4f}% of original data")
        return clean.values if was_numpy else clean

    return df.values if was_numpy else df


def _valid_mask(X):
    df = pd.DataFrame(X)
    return ~(df.isin([np.inf, -np.inf]).any(axis=1) | df.isna().any(axis=1))


def _subsample_train(X, y, groups, max_samples, seed):
    if len(X) <= max_samples:
        return X, y, groups
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    idx.sort()
    print(f"TabPFN: subsampling training set from {len(X)} to {max_samples} samples")
    return X[idx], y[idx], groups[idx]


def run_tabpfn(
    X_train, y_train, groups_train, X_test, y_test,
    seed,
    max_train_samples=TABPFN_MAX_TRAIN_SAMPLES,
    n_estimators=16,
    balance_probabilities=False,
    device="cpu",
):
    X_tr, y_tr, _ = _subsample_train(X_train, y_train, groups_train, max_train_samples, seed)

    print(f"\n=== Training TabPFN (n_estimators={n_estimators}, device={device}) ===")
    tracemalloc.start()
    fit_start = time.time()

    model = TabPFNClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        device=device,
        ignore_pretraining_limits=len(X_tr) > TABPFN_MAX_TRAIN_SAMPLES,
        balance_probabilities=balance_probabilities,
        show_progress_bar=True,
    )
    model.fit(X_tr, y_tr)

    fit_runtime = time.time() - fit_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_gb = peak / (1024 ** 3)

    print(f"Peak memory: {peak_memory_gb:.4f} GB | Runtime: {fit_runtime:.2f}s")

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = model.predict(X_test)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    test_auroc = roc_auc_score(y_test, y_test_proba)
    test_f1 = f1_score(y_test, y_test_pred)
    test_pr_auc = average_precision_score(y_test, y_test_proba)

    print(f"\n=== Test Set Results ===")
    print(f"Accuracy: {test_acc:.4f}  |  Balanced Acc: {test_bal_acc:.4f}")
    print(f"AUROC: {test_auroc:.4f}  |  PR-AUC: {test_pr_auc:.4f}")
    print(f"F1: {test_f1:.4f}")

    return {
        "params": {"n_estimators": n_estimators},
        "test_metrics": {
            "accuracy": test_acc,
            "balanced_accuracy": test_bal_acc,
            "auroc": test_auroc,
            "f1": test_f1,
            "pr_auc": test_pr_auc,
        },
        "model": model,
        "scaler": None,
        "runtime (seconds)": fit_runtime,
        "peak_memory_gb": peak_memory_gb,
    }


def main(
    fs: str,
    seed: int,
    window_size: int,
    step_size: int,
    label_fraction: float,
    gpu: int = 0,
    classification_task: str = "ms_base",
    max_train_samples: int = TABPFN_MAX_TRAIN_SAMPLES,
    n_estimators: int = 16,
    balance_probabilities: bool = False,
    zero_shot_evaluation: bool = False,
    zero_shot_dataset: str = "wesad",
):
    set_seed(seed)

    device = get_device(gpu)
    print(f"Using device: {device}")

    create_directory(RESULTS_PATH)
    results_save_path = os.path.join(
        RESULTS_PATH, "ECG_features", str(fs), "tabpfn", classification_task, f"{seed}",
        f"{label_fraction}", str(window_size), str(step_size),
    )
    create_directory(results_save_path)

    if zero_shot_evaluation:
        target_domain = "StressID" if zero_shot_dataset == "stressid" else "WESAD"
        zero_shot_results_path = os.path.join(
            RESULTS_PATH, "Transfer_learning", target_domain,
            "zero_shot_performance", "feature_engineered", "tabpfn",
            f"{seed}", f"{label_fraction}", str(window_size), str(step_size),
        )
        create_directory(zero_shot_results_path)

    # ── Load data ──────────────────────────────────────────────────────────────
    if classification_task == "ms_base_lpa_mpa":
        label_map = {"baseline": 0, "low_physical_activity": 0, "moderate_physical_activity": 0, "mental_stress": 1}
    else:
        label_map = {"baseline": 0, "mental_stress": 1}
    window_data_path = os.path.join(
        DATA_PATH, "interim", "ECG_features", str(fs),
        str(window_size), str(step_size), "windowed_data.h5",
    )
    X, y, groups, feature_names = load_processed_data(
        window_data_path, label_map=label_map, domain_features=True
    )
    y = y.astype(np.float32)

    # Load zero-shot dataset
    if zero_shot_evaluation:
        if zero_shot_dataset == "wesad":
            if int(fs) != 700:
                raise ValueError("Zero-shot WESAD evaluation requires fs=700")
            zs_path = os.path.join(
                DATA_PATH, "interim", "WESAD_features", "ECG",
                str(fs), str(window_size), str(step_size), "windowed_data.h5",
            )
        elif zero_shot_dataset == "stressid":
            if int(fs) != 500:
                raise ValueError("Zero-shot StressID evaluation requires fs=500")
            zs_path = os.path.join(
                DATA_PATH, "interim", "STRESSID_features", "ECG",
                str(fs), str(window_size), str(step_size), "windowed_data.h5",
            )
        else:
            raise ValueError('zero_shot_dataset must be "wesad" or "stressid"')

        X_zs, y_zs, groups_zs = load_processed_data(
            zs_path, label_map={"baseline": 0, "mental_stress": 1}
        )
        y_zs = y_zs.astype(np.float32)

        X_zs_clean = handle_missing_data(X_zs, drop_values=True, verbose=True)
        if len(X_zs_clean) != len(X_zs):
            mask = _valid_mask(X_zs)
            y_zs = y_zs[mask]
            groups_zs = groups_zs[mask]
            X_zs = X_zs_clean

    # ── Missing value handling ─────────────────────────────────────────────────
    print("=== Handling missing values ===")
    X_clean = handle_missing_data(X, drop_values=True, verbose=True)
    if len(X_clean) != len(X):
        print(f"Dropped {len(X) - len(X_clean)} samples")
        mask = _valid_mask(X)
        y = y[mask]
        groups = groups[mask]
        X = X_clean

    # ── Train / test split by participant ──────────────────────────────────────
    train_idx, train_p, test_idx, test_p = split_indices_by_participant_groups(
        groups, train_ratio=0.8, label_fraction=label_fraction, seed=seed,
    )

    X_train_all = X[train_idx]
    y_train_all = y[train_idx]
    groups_train_all = groups[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    binary_mask_tr = np.isin(y_train_all, [0, 1])
    binary_mask_te = np.isin(y_test, [0, 1])
    X_train_all = X_train_all[binary_mask_tr]
    y_train_all = y_train_all[binary_mask_tr]
    groups_train_all = groups_train_all[binary_mask_tr]
    X_test = X_test[binary_mask_te]
    y_test = y_test[binary_mask_te]

    print(f"Training data: {X_train_all.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Training participants: {len(np.unique(groups_train_all))}")
    print(f"Test participants: {len(np.unique(groups[test_idx][binary_mask_te]))}")

    # ── Feature standardisation ────────────────────────────────────────────────
    X_train_all, _, X_test, scaler = standardize_features(
        X_train_all, None, X_test, feature_names
    )

    # ── Run TabPFN ─────────────────────────────────────────────────────────────
    results = run_tabpfn(
        X_train_all, y_train_all, groups_train_all,
        X_test, y_test,
        seed,
        max_train_samples=max_train_samples,
        n_estimators=n_estimators,
        balance_probabilities=balance_probabilities,
        device=device,
    )

    # ── Zero-shot evaluation ───────────────────────────────────────────────────
    if zero_shot_evaluation:
        standard_scaler, minmax_scaler = scaler
        X_zs_scaled = X_zs.copy()

        min_max_names = {"nn20", "nn50", "wmax"}
        nn_idx = [i for i, n in enumerate(feature_names) if n.lower() in min_max_names]
        std_idx = [i for i, n in enumerate(feature_names) if n.lower() not in min_max_names]

        if std_idx:
            X_zs_scaled[:, std_idx] = standard_scaler.transform(X_zs[:, std_idx])
        if nn_idx:
            X_zs_scaled[:, nn_idx] = minmax_scaler.transform(X_zs[:, nn_idx])

        zs_results = evaluate_zero_shot_model_performance(
            results["model"], X_zs_scaled, y_zs
        )
        with open(os.path.join(zero_shot_results_path, "zero_shot_results.json"), "w") as f:
            json.dump(zs_results, f, indent=2, default=str)

    # ── Save results ───────────────────────────────────────────────────────────
    results_to_save = {k: v for k, v in results.items() if k not in ("model", "scaler")}
    with open(os.path.join(results_save_path, "test_results.json"), "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)

    gc.collect()
    print(f"=== Done! Results saved to {results_save_path} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabPFN classifier on ECG features")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--fs", default=500, type=str, help="Sample frequency")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window_size", type=int, default=10, help="Window size in seconds")
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--label_fraction", type=float, default=0.01)
    parser.add_argument(
        "--classification_task", default="ms_base",
        type=str, choices=("ms_base", "ms_base_lpa_mpa"),
        help="Downstream classification task: ms_base (binary) or ms_base_lpa_mpa "
             "(mental stress vs baseline+lpa+mpa, still binary).",
    )
    parser.add_argument(
        "--n_estimators", type=int, default=16,
        help="Number of TabPFN estimators (ensemble size)",
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=TABPFN_MAX_TRAIN_SAMPLES,
        help="Max training samples passed to TabPFN (subsamples if exceeded)",
    )
    parser.add_argument(
        "--balance_probabilities", action="store_true",
        help="Ask TabPFN to balance class probabilities",
    )
    parser.add_argument("--zero_shot_evaluation", action="store_true")
    parser.add_argument(
        "--zero_shot_dataset", type=str, choices=("stressid", "wesad"), default="wesad",
    )

    args = parser.parse_args()
    main(**vars(args))