
"""
WildECG Training Pipeline with CV and Logistic Regression.

Loads the pretrained WildECG ResNet1D weights from the GitHub repository
(https://github.com/klean2050/tiles_ecg_model), extracts 256-dim representations,
and trains a downstream classifier — mirroring ecg_founder_training.py.

Expected input shape stored in the H5 file: (N, L, 1) where L = fs * window_size.
WildECG expects (B, 1, 1000) at 100 Hz (10-second windows); signals are
"""
import os
import json
import argparse
import logging
import gc
import urllib.request

import numpy as np
import torch
from scipy.signal import resample as scipy_resample
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
)
from utils.helper_paths import DATA_PATH, RESULTS_PATH
from models.wildecg_resnet import ResNet1D
from models.s4_model import S4Model


# WildECG was pretrained on 10-second windows at 100 Hz
WILDECG_REPO = "https://github.com/klean2050/tiles_ecg_model/raw/master/ckpt"
WILDECG_CHECKPOINT_RESNET = "resnet-trans-small.ckpt"
# The main paper used predicting transformation as an auxilary task
WILDECG_CHECKPOINT_S4 = "s4_trans.ckpt"

WILDECG_FS = 100
WILDECG_LENGTH = 1000          # 10 s × 100 Hz
WILDECG_REPR_DIM = 256         # output_size of the pretrained encoder
MODEL_NAME = "WildECG"

# ResNet1D config matching the pretrained checkpoint
_RESNET_CFG = dict(
    in_channels=1,
    base_filters=16,
    kernel_size=3,
    stride=2,
    groups=1,
    n_block=10,
)

# S4 config matching the pretrained checkpoint (s4_trans.ckpt):
#   d_model=128 from encoder.weight shape [128, 1]
#   n_layers=6  from s4_layers.5.* keys present in checkpoint
#   decoder=True because checkpoint contains decoder.weight / decoder.bias
_S4_CFG = dict(
    d_input=1,
    d_output=128,
    d_model=128,
    n_layers=6,
    dropout=0.2,
    prenorm=True,
    decoder=True,
)

STRESSOR_GROUPS = [
    ("TA", ["TA", "TA_repeat"]),
    ("Pasat", ["Pasat", "Pasat_repeat"]),
    ("Raven", ["Raven"]),
    ("SSST", ["SSST_Sing_countdown"]),
]


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _get_checkpoint_path(checkpoint_name: str) -> str:
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "wildecg")
    os.makedirs(cache_dir, exist_ok=True)
    ckpt_path = os.path.join(cache_dir, checkpoint_name)
    if not os.path.exists(ckpt_path):
        url = f"{WILDECG_REPO}/{checkpoint_name}"
        print(f"Downloading WildECG checkpoint from {url} ...")
        urllib.request.urlretrieve(url, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
    else:
        print(f"Using cached checkpoint: {ckpt_path}")

    # Guard against Git LFS pointer files (binary check)
    with open(ckpt_path, "rb") as fh:
        header = fh.read(200)
    if b"oid sha256:" in header or b"version https://git-lfs" in header:
        os.remove(ckpt_path)
        raise RuntimeError(
            f"The downloaded file is a Git LFS pointer, not the actual checkpoint.\n"
            f"Please manually download '{checkpoint_name}' from\n"
            f"  https://github.com/klean2050/tiles_ecg_model/tree/master/ckpt\n"
            f"and place it at: {ckpt_path}"
        )
    return ckpt_path


def _load_encoder_state_dict(ckpt_path: str, device: torch.device) -> dict:
    """Extract the 'encoder.*' sub-dict from a PyTorch Lightning checkpoint."""
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    pl_sd = raw.get("state_dict", raw)
    encoder_sd = {
        k[len("encoder."):]: v
        for k, v in pl_sd.items()
        if k.startswith("encoder.")
    }
    if not encoder_sd:
        raise RuntimeError(
            "Could not find 'encoder.*' keys in the checkpoint. "
            "Expected a PyTorch Lightning checkpoint from TransformLearning."
        )
    return encoder_sd


def _build_wildecg_resnet(device: torch.device) -> ResNet1D:
    """Download (once, cached) and load the pretrained WildECG ResNet1D encoder."""
    ckpt_path = _get_checkpoint_path(WILDECG_CHECKPOINT_RESNET)
    model = ResNet1D(**_RESNET_CFG)
    encoder_sd = _load_encoder_state_dict(ckpt_path, device)
    model.load_state_dict(encoder_sd, strict=True)
    model.to(device).eval()
    print(f"WildECG ResNet encoder loaded (repr_dim={model.output_size}).")
    return model


def _build_wildecg_s4(device: torch.device) -> S4Model:
    """Download (once, cached) and load the pretrained WildECG S4 encoder."""
    ckpt_path = _get_checkpoint_path(WILDECG_CHECKPOINT_S4)
    model = S4Model(**_S4_CFG)
    encoder_sd = _load_encoder_state_dict(ckpt_path, device)
    model.load_state_dict(encoder_sd, strict=True)
    model.to(device).eval()
    print(f"WildECG S4 encoder loaded (repr_dim={model.output_size}).")
    return model


# ── Encoding ───────────────────────────────────────────────────────────────────
def encode_representations(
    X: np.ndarray,
    model: ResNet1D,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract WildECG representations from raw windowed ECG.

    Args:
        X: (N, L, 1) array at any sampling rate
        model: pretrained ResNet1D with eval() already called
        batch_size: number of windows per forward pass
        device: torch device

    Returns:
        (N, WILDECG_REPR_DIM) float32 array
    """
    L = X.shape[1]
    model.eval()
    all_reprs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size].astype(np.float32)  # (B, L, 1)
            batch = batch.transpose(0, 2, 1)                    # (B, 1, L)

            if L != WILDECG_LENGTH:
                batch = scipy_resample(batch, WILDECG_LENGTH, axis=-1)
            
            x = torch.FloatTensor(batch).to(device)
            reprs = model(x)                  # (B, 256)
            all_reprs.append(reprs.cpu().numpy())

    return np.concatenate(all_reprs, axis=0)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _per_condition_metrics(model, test_repr, y_test, conditions_test):
    """Binary (vs baseline) metrics per stressor using a trained sklearn model."""
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
    """PR-AUC with baseline subsampled to match overall_ratio stress prevalence."""
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


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(
    fs: str,
    window_size: int,
    step_size: int,
    gpu: int,
    seed: int,
    encoding_batch_size: int,
    label_fraction: float,
    save_embeddings: bool,
    classification_task: str,
    classifier_model: str,
    classifier_epochs: int,
    classifier_lr: float,
    classifier_batch_size: int,
    backbone: str = "s4",
    k_folds: int = 5,
    min_participants_for_kfold: int = 5,
    verbose: bool = False,
    scoring_metric: str = "roc_auc",
    zero_shot_evaluation: bool = False,
    zero_shot_dataset: str = "wesad",
    leave_one_stressor_out: bool = False,
):
    # ── Step 0: Setup ────────────────────────────────────────────────────────────
    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    # We do not provide mps here, as with S4 there are issues with some of the computations
    else:
        device = torch.device("cpu")

    logging.basicConfig(level=logging.INFO)
    print(f"Starting {MODEL_NAME} training [{backbone}/{classifier_model}], seed={seed}, "
          f"label_fraction={label_fraction}")
    print(f"Using device: {device}")

    create_directory(RESULTS_PATH)

    results_save_path = os.path.join(
        RESULTS_PATH, "ECG", str(fs), MODEL_NAME, backbone, classifier_model,
        classification_task, f"{seed}", f"{label_fraction}",
        f"{window_size}", f"{step_size}",
    )
    embedding_save_path = os.path.join(
        DATA_PATH, "embeddings", "ECG", f"{fs}", MODEL_NAME, backbone,
        classification_task, f"{seed}", f"{window_size}", f"{step_size}",
    )

    X_zero_shot = y_zero_shot = zero_shot_results_path = None

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
        label_map = {
            "baseline": 0, "low_physical_activity": 0,
            "moderate_physical_activity": 0, "mental_stress": 1,
        }
    else:
        label_map = {"baseline": 0, "mental_stress": 1}

    window_data_path = os.path.join(
        DATA_PATH, "interim", "ECG", str(fs), str(window_size), str(step_size),
        "windowed_data.h5",
    )

    if zero_shot_evaluation:
        if zero_shot_dataset == "wesad":
            if int(fs) == 100:
                zero_shot_window_data_path = os.path.join(
                    DATA_PATH, "interim", "WESAD", "ECG", str(fs),
                    str(window_size), str(step_size), "windowed_data.h5",
                )
            else:
                raise ValueError(
                    "For zero-shot evaluation on WESAD the frequency needs to be 100"
                )
        elif zero_shot_dataset == "stressid":
            if int(fs) == 100:
                zero_shot_window_data_path = os.path.join(
                    DATA_PATH, "interim", "STRESSID", "ECG", str(fs),
                    str(window_size), str(step_size), "windowed_data.h5",
                )
            else:
                raise ValueError(
                    "For zero-shot evaluation on StressID the frequency needs to be 100"
                )
        else:
            raise ValueError('Please use a proper dataset: "wesad" or "stressid"')

    X, y, groups, conditions = load_processed_data_with_conditions(
        window_data_path, label_map=label_map
    )
    y = y.astype(np.float32)

    if zero_shot_evaluation:
        X_zero_shot, y_zero_shot, _ = load_processed_data(
            zero_shot_window_data_path, label_map={"baseline": 0, "mental_stress": 1}
        )
        y_zero_shot = y_zero_shot.astype(np.float32)

    # ── Step 2: Participant Split ─────────────────────────────────────────────────
    train_idx, train_p, all_train_p, all_train_idx, test_idx, test_p = (
        split_indices_by_participant_groups(
            groups,
            train_ratio=0.8,
            label_fraction=label_fraction,
            seed=seed,
            return_all_train_p=True,
        )
    )

    downstream_mask = {
        "train": np.isin(y[train_idx], [0, 1]),
        "test": np.isin(y[test_idx], [0, 1]),
    }

    # ── Step 3: Load Pretrained WildECG Encoder ──────────────────────────────────
    if backbone == "resnet":
        model = _build_wildecg_resnet(device)
    elif backbone == "s4":
        model = _build_wildecg_s4(device)
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose 'resnet' or 's4'.")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total: {total:,}  trainable: {trainable:,}")

    # ── Step 4: Extract Representations ──────────────────────────────────────────
    print("Extracting train representations ...")
    train_repr = encode_representations(X[train_idx], model, encoding_batch_size, device)
    print("Extracting test representations ...")
    test_repr = encode_representations(X[test_idx], model, encoding_batch_size, device)

    if save_embeddings:
        print("Saving embeddings ...")
        x_repr_all = encode_representations(X, model, encoding_batch_size, device)
        np.savez(
            os.path.join(embedding_save_path, "x_y_groups_embedding.npz"),
            array1=x_repr_all, array_2=y, array_3=groups,
        )
        print("Embeddings saved.")

    train_repr = train_repr[downstream_mask["train"]]
    y_train = y[train_idx][downstream_mask["train"]]
    groups_train = groups[train_idx][downstream_mask["train"]]
    conditions_train = conditions[train_idx][downstream_mask["train"]]

    test_repr = test_repr[downstream_mask["test"]]
    y_test = y[test_idx][downstream_mask["test"]]
    conditions_test = conditions[test_idx][downstream_mask["test"]]

    print(f"train_repr shape = {train_repr.shape}")

    # ── Step 5: Cross-Validation Splitter ─────────────────────────────────────────
    cv_splitter, n_splits = get_participant_cv_splitter(
        groups_train,
        min_participants_for_kfold=min_participants_for_kfold,
        k=k_folds,
    )

    # ── Step 6: Run CV with downstream classifier ─────────────────────────────────
    set_seed(seed)
    feature_names = [f"repr_{i}" for i in range(train_repr.shape[1])]

    if classifier_model in ["logistic_regression", "random_forest", "xgboost"]:
        if verbose:
            results = run_logistic_regression_with_gridsearch_verbose(
                train_repr, y_train, groups_train, test_repr, y_test,
                feature_names, cv_splitter, False, seed,
            )
        else:
            results = run_logistic_regression_with_gridsearch(
                train_repr, y_train, groups_train,
                test_repr, y_test, feature_names, cv_splitter, False, seed,
                scoring_metric=scoring_metric, classifier_model=classifier_model,
            )
        print(f"Best CV AUROC: {results['best_cv_score'] if cv_splitter is not None else 0:.4f}")
    else:
        results = run_mlp_with_cv_and_test(
            train_repr, y_train, groups_train,
            test_repr, y_test, feature_names, cv_splitter,
            device, classifier_epochs, classifier_batch_size, classifier_lr, False, seed,
        )
        print(f"Best CV AUROC: {results['best_cv_score']:.4f}")

    print(f"Test metrics - Accuracy: {results['test_metrics']['accuracy']:.4f}, "
          f"AUROC: {results['test_metrics']['auroc']:.4f}, "
          f"F1: {results['test_metrics']['f1']:.4f}, "
          f"PR-AUC: {results['test_metrics']['pr_auc']:.4f}")

    # ── Step 7a: Per-condition metrics ────────────────────────────────────────────
    if classifier_model in ["logistic_regression", "random_forest", "xgboost"]:
        per_condition_results = _per_condition_metrics(
            results["model"], test_repr, y_test, conditions_test
        )
        results["per_condition_metrics"] = per_condition_results
        print("Per-condition test metrics (each stressor vs baseline):")
        for cond, m in per_condition_results.items():
            print(f"  {cond}: AUROC={m['auroc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
                  f"n_stress={m['n_stress_samples']}")

    # ── Step 7a-ii: Subcategory confusion (LPA/MPA discrimination) ───────────────
    if (classification_task == "ms_base_lpa_mpa"
            and classifier_model in ["logistic_regression", "random_forest", "xgboost"]):
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

    # ── Step 7b: Leave-one-stressor-out ──────────────────────────────────────────
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
                g_tr_loso, min_participants_for_kfold=min_participants_for_kfold, k=k_folds,
            )
            feat_names_loso = [f"repr_{i}" for i in range(X_tr_loso.shape[1])]
            res_loso = run_logistic_regression_with_gridsearch(
                X_tr_loso, y_tr_loso, g_tr_loso, X_te_loso, y_te_loso,
                feat_names_loso, cv_sp_loso, False, seed,
                scoring_metric=scoring_metric, classifier_model=classifier_model,
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

    # ── Step 7c: Zero-shot evaluation ─────────────────────────────────────────────
    if zero_shot_evaluation:
        classifier = results["model"]
        zero_shot_repr = encode_representations(X_zero_shot, model, encoding_batch_size, device)
        zero_shot_results = evaluate_zero_shot_model_performance(
            classifier, zero_shot_repr, y_zero_shot
        )
        with open(os.path.join(zero_shot_results_path, "zero_shot_results.json"), "w") as f:
            json.dump(zero_shot_results, f, indent=2, default=str)

    # ── Step 8: Save Results ──────────────────────────────────────────────────────
    with open(os.path.join(results_save_path, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Additional parameters - Classifier: {classifier_model}, "
          f"Label fraction: {label_fraction}, Seed: {seed}, "
          f"K-folds: {k_folds}, CV splits: {n_splits}")

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
        description="WildECG Training Pipeline with CV and Logistic Regression",
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
                               help="Show verbose CV output for logistic regression")

    # ══════════════════════════════════════════════════════════════════════════════
    # Data Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--fs", default=100, type=str,
                            help="Sampling frequency of the stored ECG windows "
                                 "(should be 100 Hz for WildECG)")
    data_group.add_argument("--window_size", type=int, default=10,
                            help="Window size in seconds")
    data_group.add_argument("--step_size", type=int, default=5,
                            help="Step size in seconds for sliding window")
    data_group.add_argument("--label_fraction", type=float, default=1.0,
                            help="Fraction of labeled train participants (0.0-1.0)")
    data_group.add_argument("--save_embeddings", action="store_true",
                            help="Save all representations to disk for later analysis")
    data_group.add_argument("--classification_task", default="ms_base",
                            type=str, choices=("ms_base", "ms_base_lpa_mpa"),
                            help="Downstream classification task")

    # ══════════════════════════════════════════════════════════════════════════════
    # Backbone Model
    # ══════════════════════════════════════════════════════════════════════════════
    backbone_group = parser.add_argument_group("Backbone Model")
    backbone_group.add_argument("--backbone", type=str, default="s4",
                                choices=("resnet", "s4"),
                                help="Pretrained WildECG backbone: 's4' (S4 predicting transformations) "
                                     "or 'resnet' (ResNet1D contrastive)")

    # ══════════════════════════════════════════════════════════════════════════════
    # Encoding
    # ══════════════════════════════════════════════════════════════════════════════
    enc_group = parser.add_argument_group("Encoding")
    enc_group.add_argument("--encoding_batch_size", type=int, default=64,
                           help="Batch size for extracting WildECG representations")

    # ══════════════════════════════════════════════════════════════════════════════
    # Downstream Classifier Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    classifier_group = parser.add_argument_group("Downstream Classifier")
    classifier_group.add_argument("--classifier_model", type=str,
                                  default="logistic_regression",
                                  choices=("logistic_regression", "mlp",
                                           "random_forest", "xgboost"),
                                  help="Type of downstream classifier")
    classifier_group.add_argument("--classifier_epochs", type=int, default=25,
                                  help="Epochs for MLP classifier training")
    classifier_group.add_argument("--classifier_lr", type=float, default=1e-4,
                                  help="Learning rate for MLP classifier")
    classifier_group.add_argument("--classifier_batch_size", type=int, default=32,
                                  help="Batch size for MLP classifier")

    # ══════════════════════════════════════════════════════════════════════════════
    # Cross-Validation Configuration
    # ══════════════════════════════════════════════════════════════════════════════
    cv_group = parser.add_argument_group("Cross-Validation")
    cv_group.add_argument("--k_folds", type=int, default=5,
                          help="Number of folds for cross-validation")
    cv_group.add_argument("--min_participants_for_kfold", type=int, default=5,
                          help="Minimum participants for k-fold (else LOPOCV)")
    cv_group.add_argument("--scoring_metric", type=str, default="roc_auc",
                          choices=["roc_auc", "average_precision", "f1", "balanced_accuracy"],
                          help="Scoring metric for CV hyperparameter selection")

    # ══════════════════════════════════════════════════════════════════════════════
    # Zero-shot evaluation
    # ══════════════════════════════════════════════════════════════════════════════
    zero_shot_group = parser.add_argument_group("Zero-shot evaluation")
    zero_shot_group.add_argument("--zero_shot_evaluation", action="store_true",
                                 help="Run zero-shot evaluation on an external dataset")
    zero_shot_group.add_argument("--zero_shot_dataset", type=str,
                                 choices=("stressid", "wesad"), default="wesad")

    # ══════════════════════════════════════════════════════════════════════════════
    # Leave-one-stressor-out
    # ══════════════════════════════════════════════════════════════════════════════
    loso_group = parser.add_argument_group("Leave-one-stressor-out")
    loso_group.add_argument("--leave_one_stressor_out", action="store_true",
                            help="Run leave-one-stressor-out analysis")

    args = parser.parse_args()
    main(**vars(args))