import tempfile
from pathlib import Path

import os
import random
import numpy as np
import h5py
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_curve
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torcheval.metrics.functional import multiclass_f1_score

# MLflow imports
import mlflow
import mlflow.pytorch
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient

# MLflow helpers
def build_supervised_fingerprint(cfg: dict[str, object]) -> dict[str, str]:
    """
    Create an immutable dictionary (all str) uniquely describing one training.
    """
    keys = (
        "model_name", "seed", "lr", "batch_size", "num_epochs",
        "patience", "scheduler_mode", "scheduler_factor",
        "scheduler_patience", "scheduler_min_lr", "label_fraction"
    )
    return {k: str(cfg[k]) for k in keys}

def search_encoder_fp(fp: dict[str, str], experiment_name: str,
                      tracking_uri: str) -> str | None:
    """Return run_id of an MLflow run whose params exactly match 'fp'."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp    = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    clauses = ["attributes.status = 'FINISHED'"]
    clauses += [f"params.{k} = '{v}'" for k, v in fp.items()]
    hits = mlflow.search_runs([exp.experiment_id],
                              filter_string=" and ".join(clauses),
                              max_results=1)
    return None if hits.empty else hits.iloc[0]["run_id"]

def load_processed_data(hdf5_path, label_map=None):
    """
    Load windowed ECG data from an HDF5 file.
    
    Returns:
        X (np.ndarray): shape (N, window_length, 1)
        y (np.ndarray): shape (N,)
        groups (np.ndarray): shape (N,) - each window's participant ID
    """
    if label_map is None:
        label_map = {"baseline": 0, "mental_stress": 1}

    X_list, y_list, groups_list = [], [], []
    with h5py.File(hdf5_path, "r") as f:
        participants = list(f.keys())
        for participant_key in participants:
            participant_id = participant_key.replace("participant_", "")
            for cat in f[participant_key].keys():
                if cat not in label_map:
                    continue
                cat_group = f[participant_key][cat]
                segment_windows_list = []
                for segment_name in cat_group.keys():
                    windows = cat_group[segment_name][...]
                    segment_windows_list.append(windows)
                if len(segment_windows_list) == 0:
                    continue
                # Concatenate windows from all segments in this category
                windows_all = np.concatenate(segment_windows_list, axis=0)
                n_windows = windows_all.shape[0]
                groups_arr = np.array([participant_id] * n_windows, dtype=object)

                X_list.append(windows_all)
                y_list.append(np.full((n_windows,), label_map[cat], dtype=int))
                groups_list.append(groups_arr)

    if len(X_list) == 0:
        raise ValueError(f"No valid data found in {hdf5_path} with label_map {label_map}.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)

    # Expand dims for CNN: (N, window_length, 1)
    X = np.expand_dims(X, axis=-1)
    return X, y, groups


def split_data_by_participant(X, y, groups, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Split data by unique participant IDs.
    
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    unique_participants = np.unique(groups)
    n_participants = len(unique_participants)
    
    # Set seed and shuffle participants
    np.random.seed(seed)
    shuffled = np.random.permutation(unique_participants)
    
    n_train = int(n_participants * train_ratio)
    n_val = int(n_participants * val_ratio)
    
    train_participants = shuffled[:n_train]
    val_participants = shuffled[n_train:n_train+n_val]
    test_participants = shuffled[n_train+n_val:]
    
    train_mask = np.isin(groups, train_participants)
    val_mask = np.isin(groups, val_participants)
    test_mask = np.isin(groups, test_participants)
    
    return (X[train_mask], y[train_mask]), (X[val_mask], y[val_mask]), (X[test_mask], y[test_mask])

def split_indices_by_participant(groups, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Return index arrays for train / val / test
    """
    uniq = np.unique(groups)
    rng  = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n_train = int(len(uniq) * train_ratio)
    n_val   = int(len(uniq) * val_ratio)

    train_p, val_p, test_p = np.split(uniq, [n_train, n_train + n_val])

    train_idx = np.flatnonzero(np.isin(groups, train_p))
    val_idx   = np.flatnonzero(np.isin(groups, val_p))
    test_idx  = np.flatnonzero(np.isin(groups, test_p))

    return train_idx, val_idx, test_idx

class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG data.
    """
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        # Convert to torch.Tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return sample, label

# -----------------------------------------------
# Helper: threshold sweep
# -----------------------------------------------
def find_best_threshold(
    probs:  np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2,
    average: str = "macro",
    grid: int = 101
):
    """
    Scan `grid` equally–spaced thresholds in (0,1) and return the one that
    maximises macro-F1.  Returns (best_threshold, best_f1).
    """
    ts = np.linspace(0.0, 1.0, grid, endpoint=False)[1:]   # skip 0
    best_t = 0.5
    best_f1 = -1.0
    labels_t = torch.from_numpy(labels.astype(np.int64))

    for t in ts:
        preds = (probs >= t).astype(np.int64)
        f1 = multiclass_f1_score(
                        torch.from_numpy(preds),
                        labels_t,
                        num_classes=num_classes,
                        average=average,
                    ).item()
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

# -------------------------------------------
# TRAIN (one epoch) + VALIDATION
# -------------------------------------------
def train_one_epoch(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    epoch,
    best_threshold_so_far=0.5,
    best_f1_so_far=-1.0,
    log_interval=100,
):
    model.train()
    run_loss = 0.0
    correct = 0
    n_seen = 0

    auroc_tr = BinaryAUROC()
    pr_tr = BinaryAveragePrecision()

    non_blocking_bool = torch.cuda.is_available()

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=non_blocking_bool).permute(0, 2, 1)     # (B,C,L)
        y= y.to(device, non_blocking=non_blocking_bool).float()

        optimizer.zero_grad(set_to_none=True)
        out = model(x).view(-1)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        run_loss += loss.item() * x.size(0)
        n_seen += x.size(0)

        probs = torch.sigmoid(out).detach()
        preds = (probs > 0.5).float()
        correct += preds.eq(y).sum().item()

        auroc_tr.update(probs.cpu(), y.cpu().int())
        pr_tr.update(probs.cpu(), y.cpu().int())

        if batch_idx % log_interval == 0:
            print(f"[Train] Epoch {epoch}  batch {batch_idx}/{len(train_loader)}  "
                  f"loss={run_loss/n_seen:.4f}")

    # train epoch metrics
    train_loss = run_loss / n_seen
    train_acc  = correct  / n_seen
    train_auroc = auroc_tr.compute().item()
    train_pr_auc = pr_tr.compute().item()

    mlflow.log_metrics({
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_auc_roc": train_auroc,
        "train_pr_auc": train_pr_auc,
    }, step=epoch)

    # VALIDATION  
    model.eval()
    val_probs, val_labels = [], []
    auroc_val = BinaryAUROC()
    pr_val = BinaryAveragePrecision()

    val_loss_total = 0.0
    n_val = 0

    non_blocking_bool = torch.cuda.is_available()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=non_blocking_bool).permute(0, 2, 1)
            y = y.to(device, non_blocking=non_blocking_bool).float()
            out = model(x).view(-1)
            probs = torch.sigmoid(out)

            # compute and accumulate BCE loss
            val_loss_total += loss_fn(out, y).item() * x.size(0)
            n_val += x.size(0)

            val_probs.append(probs.cpu().numpy())
            val_labels.append(y.cpu().numpy())

            auroc_val.update(probs.cpu(), y.cpu().int())
            pr_val.update(probs.cpu(), y.cpu().int())

    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels)

    val_loss = val_loss_total / n_val

    # threshold sweep on this epoch
    this_t, this_f1 = find_best_threshold(val_probs, val_labels)

    if this_f1 > best_f1_so_far:
        best_f1_so_far = this_f1
        best_threshold_so_far = this_t

    # metrics at the chosen threshold of THIS epoch
    val_preds = (val_probs >= this_t).astype(float)
    val_acc = (val_preds == val_labels).mean()
    val_f1 = this_f1
    val_auroc = auroc_val.compute().item()
    val_pr_auc= pr_val.compute().item()

    mlflow.log_metrics({
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_auc_roc": val_auroc,
        "val_pr_auc": val_pr_auc,
        "val_f1": val_f1,
        "val_best_threshold": this_t,
    }, step=epoch)

    print(f"[Val ] Epoch {epoch}  acc={val_acc:.4f}  auc={val_auroc:.4f}  "
        f"f1*={val_f1:.4f} @ t={this_t:.2f}  loss={val_loss:.4f}")

    return val_loss, best_threshold_so_far, best_f1_so_far

# ------------------------------------------
# TEST
# ------------------------------------------
def test(
    model,
    test_loader,
    device,
    threshold,
    loss_fn=None  
):
    model.eval()
    probs_all, labels_all = [], []
    auroc_te = BinaryAUROC()
    pr_te = BinaryAveragePrecision()
    total_loss = 0.0
    n_seen = 0

    non_blocking_bool = torch.cuda.is_available()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=non_blocking_bool).permute(0, 2, 1)
            y = y.to(device, non_blocking=non_blocking_bool).float()
            out = model(x).view(-1)
            probs = torch.sigmoid(out)

            probs_all.append(probs.cpu().numpy())
            labels_all.append(y.cpu().numpy())

            auroc_te.update(probs.cpu(), y.cpu().int())
            pr_te.update(probs.cpu(), y.cpu().int())

            if loss_fn is not None:
                total_loss += loss_fn(out, y).item() * x.size(0)
                n_seen += x.size(0)

    probs_all  = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all).astype(np.int64)
    preds_all  = (probs_all >= threshold).astype(np.int64)
    
    acc = (preds_all == labels_all).mean()
    f1 = multiclass_f1_score(
             torch.from_numpy(preds_all),
             torch.from_numpy(labels_all),
             num_classes=2,
             average="macro"
           ).item()
    auroc = auroc_te.compute().item()
    prauc = pr_te.compute().item()
    loss = (total_loss / n_seen) if loss_fn is not None else np.nan

    mlflow.log_metrics({
        "test_loss": loss,
        "test_accuracy": acc,
        "test_auc_roc": auroc,
        "test_pr_auc": prauc,
        "test_f1": f1,
        "test_threshold": threshold,
    })

    print(f"[Test]  acc={acc:.4f}  auc={auroc:.4f}  pr_auc={prauc:.4f}  "
          f"f1*={f1:.4f} @ t={threshold:.2f}")
    return loss, acc, auroc, prauc, f1

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when validation loss doesn't improve for 'patience' epochs.
    """
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
def log_model_summary(model, input_size):
    # Write model summary to a file and log as an artifact
    with tempfile.TemporaryDirectory() as tmp_dir:
        summmary_path = Path(tmp_dir) / "model_summary.txt"
        with open(summmary_path, "w") as f:
            f.write(str(summary(model, input_size=input_size)))
        mlflow.log_artifact(summmary_path)
        
def prepare_model_signature(model, sample_input):
    """Prepare model signature for MLflow registration"""
    model.cpu().eval()
    with torch.no_grad():
        tensor_input = torch.tensor(
            sample_input, 
            dtype=torch.float32
        ).permute(0, 2, 1)
        sample_output = model(tensor_input).numpy()
    
    input_schema = Schema([
        TensorSpec(np.dtype(np.float32), (-1, sample_input.shape[1], 1))
    ])
    output_schema = Schema([
        TensorSpec(np.dtype(np.float32), (-1, 1))
    ])
    
    return ModelSignature(inputs=input_schema, outputs=output_schema)

# def set_seed(seed=42, deterministic=True):
#     """
#     Set seeds for reproducibility.
    
#     Args:
#         seed (int): Seed number
#         deterministic (bool): If True, use deterministic algorithms
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
    
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
#         if deterministic:
#             # Ensure deterministic behavior
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False

def set_seed(seed=42, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)