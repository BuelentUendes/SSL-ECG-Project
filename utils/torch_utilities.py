import os
import tempfile
from pathlib import Path
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import os
import random
import warnings
import numpy as np
import h5py
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_curve, roc_auc_score
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torcheval.metrics.functional import multiclass_f1_score

#Scikit learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut,  GridSearchCV

# MLflow imports
import mlflow
import mlflow.pytorch
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient

from models.supervised import LinearClassifier, MLPClassifier

from models.tstcc import build_linear_loaders


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

def load_processed_data(
        hdf5_path,
        label_map=None,
        ppg_data=False,
        domain_features=False
):
    """
    Load windowed ECG data from an HDF5 file.
    if ppg_data: option to process also stored PPG data
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

        if domain_features:
        # Extract feature names from first participant
            feature_names = f[participants[0]].attrs['feature_names']

        for participant_key in participants:
            if ppg_data:
                participant_id = participant_key.replace("P", "").replace("_empatica", "")
            else:
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
    if not domain_features:
        X = np.expand_dims(X, axis=-1)

    if domain_features:
        return X, y, groups, feature_names

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


def split_indices_by_participant(
        groups,
        train_ratio=0.6,
        val_ratio=0.2,
        label_fraction=0.1,
        self_supervised_method=False,
        seed=42
):
    """
    Return index arrays for train / val / test
    """
    uniq = np.unique(groups)
    rng  = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n_train = int(len(uniq) * train_ratio)
    n_val   = int(len(uniq) * val_ratio)

    train_p, val_p, test_p = np.split(uniq, [n_train, n_train + n_val])

    # For self-supervised method, we need all train_idx (as we do not use the labels anyways) and then a separate set with the labels
    if self_supervised_method:
        train_idx_all = np.flatnonzero(np.isin(groups, train_p))

    if label_fraction < 1.:
        # Select subset of training participants to be labeled
        n_labeled_participants = max(1, int(len(train_p) * label_fraction))

        # Random selection if no labels provided
        labeled_participants = rng.choice(train_p, size=n_labeled_participants, replace=False)
        train_p = labeled_participants.copy()
        train_idx = np.flatnonzero(np.isin(groups, labeled_participants))

    else:
        train_idx = np.flatnonzero(np.isin(groups, train_p))

    val_idx   = np.flatnonzero(np.isin(groups, val_p))
    test_idx  = np.flatnonzero(np.isin(groups, test_p))

    # Verify index splits are non-overlapping and complete
    assert len(train_idx) > 0, "Training indices cannot be empty"
    assert len(val_idx) > 0, "Validation indices cannot be empty"
    assert len(test_idx) > 0, "Test indices cannot be empty"

    # Check no overlap in indices
    assert len(np.intersect1d(train_idx, val_idx)) == 0, "Training and validation indices overlap"
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "Training and test indices overlap"
    assert len(np.intersect1d(val_idx, test_idx)) == 0, "Validation and test indices overlap"

    # Verify participant isolation: check that each split contains only its assigned participants
    train_participants_in_split = np.unique(groups[train_idx])
    val_participants_in_split = np.unique(groups[val_idx])
    test_participants_in_split = np.unique(groups[test_idx])

    assert np.array_equal(np.sort(train_participants_in_split),
                          np.sort(train_p)), "Training split contains wrong participants"
    assert np.array_equal(np.sort(val_participants_in_split),
                          np.sort(val_p)), "Validation split contains wrong participants"
    assert np.array_equal(np.sort(test_participants_in_split),
                          np.sort(test_p)), "Test split contains wrong participants"

    if self_supervised_method:
        return train_idx, train_idx_all, val_idx, test_idx,
    else:
        return train_idx, val_idx, test_idx


def split_indices_by_participant_groups(
        groups,
        train_ratio=0.8,
        label_fraction=0.1,
        seed=42,
        return_all_train_p=False,
):
    """
    Return index arrays for train / val / test
    """
    uniq = np.unique(groups)
    rng  = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n_train = int(len(uniq) * train_ratio)
    n_test   = int(len(uniq) * (1-train_ratio))

    all_train_p, test_p = np.split(uniq, [n_train])
    all_train_idx = np.flatnonzero(np.isin(groups, all_train_p))

    if label_fraction < 1.:
        # Select subset of training participants to be labeled
        n_labeled_participants = max(1, int(len(all_train_p) * label_fraction))

        # Random selection if no labels provided
        labeled_participants = rng.choice(all_train_p, size=n_labeled_participants, replace=False)
        train_p = labeled_participants.copy()
        train_idx = np.flatnonzero(np.isin(groups, labeled_participants))

    else:
        train_p = all_train_p.copy()
        train_idx = np.flatnonzero(np.isin(groups, all_train_p))

    test_idx  = np.flatnonzero(np.isin(groups, test_p))

    # Verify index splits are non-overlapping and complete
    assert len(train_idx) > 0, "Training indices cannot be empty"
    assert len(test_idx) > 0, "Test indices cannot be empty"

    # Check no overlap in indices
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "Training and test indices overlap"

    # Verify participant isolation: check that each split contains only its assigned participants
    train_participants_in_split = np.unique(groups[train_idx])
    test_participants_in_split = np.unique(groups[test_idx])

    assert np.array_equal(np.sort(train_participants_in_split),
                          np.sort(train_p)), "Training split contains wrong participants"
    assert np.array_equal(np.sort(test_participants_in_split),
                          np.sort(test_p)), "Test split contains wrong participants"

    if return_all_train_p:
        return train_idx, train_p, all_train_p, all_train_idx, test_idx, test_p
    else:
        return train_idx, train_p, test_idx, test_p


class PhysiologicalDataset(Dataset):
    """
    PyTorch Dataset for physiological signal data, including ECG and PPG data in our study.
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.use_deterministic_algorithms(True, warn_only=True)


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train_classifier_for_optuna(classifier, train_loader, val_loader, optimizer, loss_fn,
                                num_epochs=50, device='cuda', early_stopping_patience=10):
    """Train classifier and return best validation F1 score for Optuna"""
    best_val_score = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        classifier.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = classifier(batch_x)
            loss = loss_fn(outputs.squeeze(), batch_y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        classifier.eval()
        val_probs = []
        val_targets = []
        val_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = classifier(batch_x)
                loss = loss_fn(outputs.squeeze(), batch_y.float())
                val_loss += loss.item()

                # Get probabilities for AUROC calculation
                probs = torch.sigmoid(outputs.squeeze())
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        # Calculate AUROC score
        val_auroc = roc_auc_score(val_targets, val_probs)

        # Early stopping based on AUROC score
        if val_auroc > best_val_score:
            best_val_score = val_auroc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            break

    return best_val_score


# Utility functions to calculate the output dimensions of Conv Architectures automatically

# Calculate automatically the output sizes of a Conv architecture
def calculate_conv1d_output_size(input_size, kernel_size, stride, padding):
    """Calculate output size for Conv1d layer"""
    return math.floor((input_size + 2 * padding - kernel_size) / stride) + 1

def calculate_maxpool1d_output_size(input_size, kernel_size, stride, padding):
    """Calculate output size for MaxPool1d layer"""
    return math.floor((input_size + 2 * padding - kernel_size) / stride) + 1

def calculate_model_output_dims_flexible(input_length, layer_configs, final_out_channels):
    """
    Calculate output dimensions through variable number of conv blocks

    Args:
        input_length: Input sequence length
        layer_configs: List of layer configurations. Each config is a dict with:
                      - 'type': 'conv' or 'maxpool'
                      - 'kernel_size': kernel size
                      - 'stride': stride
                      - 'padding': padding (or 'auto' for conv layers)
                      - 'name': optional name for tracking
        final_out_channels: Number of output channels from final conv layer

    Returns:
        dict: Dictionary with dimensions after each layer

    Example layer_configs:
    [
        {'type': 'conv', 'kernel_size': 32, 'stride': 4, 'padding': 'auto', 'name': 'conv1'},
        {'type': 'maxpool', 'kernel_size': 2, 'stride': 2, 'padding': 1, 'name': 'maxpool1'},
        {'type': 'conv', 'kernel_size': 8, 'stride': 1, 'padding': 4, 'name': 'conv2'},
        {'type': 'maxpool', 'kernel_size': 2, 'stride': 2, 'padding': 1, 'name': 'maxpool2'},
    ]
    """

    current_length = input_length
    dims_history = {'input': current_length}

    for i, layer_config in enumerate(layer_configs):
        layer_type = layer_config['type']
        kernel_size = layer_config['kernel_size']
        stride = layer_config['stride']
        padding = layer_config['padding']

        # Generate layer name if not provided
        if 'name' in layer_config:
            layer_name = layer_config['name']
        else:
            layer_name = f"{layer_type}_{i + 1}"

        if layer_type == 'conv':
            # Handle auto padding (kernel_size // 2)
            if padding == 'auto':
                padding = kernel_size // 2

            current_length = calculate_conv1d_output_size(
                current_length, kernel_size, stride, padding
            )

        elif layer_type == 'maxpool':
            current_length = calculate_maxpool1d_output_size(
                current_length, kernel_size, stride, padding
            )

        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        dims_history[layer_name] = current_length

    # Final dimensions
    dims_history['final_features_len'] = current_length
    dims_history['final_channels'] = final_out_channels
    dims_history['flattened_size'] = current_length * final_out_channels

    return dims_history


def get_participant_cv_splitter(groups, min_participants_for_kfold=5, k=5):
    """Get appropriate cross-validation splitter based on number of participants."""
    unique_participants = np.unique(groups)
    n_participants = len(unique_participants)

    print(f"Total participants: {n_participants}")

    if n_participants < min_participants_for_kfold:
        if n_participants == 1:
            print(f"We only have 1 participant, return None and fit a default LR!")
            return None, None

        cv_splitter = LeaveOneGroupOut()
        n_splits = n_participants
        print(f"Using Leave-One-Group-Out CV ({n_splits} splits)")
    else:
        actual_k = min(k, n_participants)
        cv_splitter = GroupKFold(n_splits=actual_k)
        n_splits = actual_k
        print(f"Using {actual_k}-Fold Group CV ({n_splits} splits)")

        if n_participants % actual_k != 0:
            print(f"Note: {n_participants} participants don't divide evenly by {actual_k}")
            print("Some folds will have different numbers of participants")

    return cv_splitter, n_splits


def standardize_features(X_train, X_val, X_test, feature_names):
    # Initialize scalers
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    # Create copies to avoid modifying original data
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy() if X_val is not None else None
    X_test_scaled = X_test.copy() if X_test is not None else None

    # Identify nn20 and nn50 indices
    nn_indices = []
    standard_indices = []

    min_max_scaler_names = ["nn20", "nk_pnn20", "nn50", "nk_pnn50"]
    for i, name in enumerate(feature_names):
        if name.lower() in min_max_scaler_names:
            nn_indices.append(i)
        else:
            standard_indices.append(i)

    # Apply StandardScaler to most features (fit only on train)
    if standard_indices:
        standard_scaler.fit(X_train[:, standard_indices])
        X_train_scaled[:, standard_indices] = standard_scaler.transform(X_train[:, standard_indices])
        if X_val is not None:
            X_val_scaled[:, standard_indices] = standard_scaler.transform(X_val[:, standard_indices])
        if X_test is not None:
            X_test_scaled[:, standard_indices] = standard_scaler.transform(X_test[:, standard_indices])

    # Apply MinMaxScaler to nn20/nn50 features (fit only on train)
    if nn_indices:
        minmax_scaler.fit(X_train[:, nn_indices])
        X_train_scaled[:, nn_indices] = minmax_scaler.transform(X_train[:, nn_indices])
        if X_val is not None:
            X_val_scaled[:, nn_indices] = minmax_scaler.transform(X_val[:, nn_indices])
        if X_test is not None:
            X_test_scaled[:, nn_indices] = minmax_scaler.transform(X_test[:, nn_indices])

    return X_train_scaled, X_val_scaled, X_test_scaled


def get_participant_cv_splitter(groups, min_participants_for_kfold=5, k=5):
    """Get appropriate cross-validation splitter based on number of participants."""
    unique_participants = np.unique(groups)
    n_participants = len(unique_participants)

    print(f"Total participants: {n_participants}")

    if n_participants < min_participants_for_kfold:
        if n_participants == 1:
            print(f"We only have 1 participant, return None and fit a default LR!")
            return None, None

        cv_splitter = LeaveOneGroupOut()
        n_splits = n_participants
        print(f"Using Leave-One-Group-Out CV ({n_splits} splits)")
    else:
        actual_k = min(k, n_participants)
        cv_splitter = GroupKFold(n_splits=actual_k)
        n_splits = actual_k
        print(f"Using {actual_k}-Fold Group CV ({n_splits} splits)")

        if n_participants % actual_k != 0:
            print(f"Note: {n_participants} participants don't divide evenly by {actual_k}")
            print("Some folds will have different numbers of participants")

    return cv_splitter, n_splits


def run_logistic_regression_with_gridsearch(
        X_train, y_train, groups_train, X_test, y_test,feature_names, cv_splitter, standardize=False, seed=42
):
    """Run GridSearchCV for Logistic Regression, then evaluate on test set."""

    # Standardize features (fit on train, transform both)
    if standardize:
        X_train, _, X_test = standardize_features(X_train, None, X_test, feature_names)

    # Parameter grid for grid search
    param_grid = {
        'C': [0.0001, 0.001, 0.01],
        'penalty': ['l2', 'l1'],
        'max_iter': [5000],
    }

    default_best_params = {
        'C': 0.001,
        'penalty': 'l2',
        'max_iter': 5000,
        'random_state': seed,
        'n_jobs': -1,
        'solver': 'saga'
    }

    if cv_splitter is not None:
        # Create base model for grid search
        base_model = LogisticRegression(random_state=seed, n_jobs=-1, solver="saga")

        # GridSearchCV with GroupKFold
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_splitter,
            scoring='roc_auc',  # Use AUROC as the scoring metric
            n_jobs=-1,
            verbose=3,
        )

        print("Running GridSearchCV...")
        # Fit grid search (this does 5-fold CV internally)
        grid_search.fit(X_train, y_train, groups=groups_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score (AUROC): {grid_search.best_score_:.4f}")

        # Get the best model (already trained on full training set)
        cv_results = grid_search.cv_results_
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

    else:
        # Use default best parameters when cv_splitter is None
        print("cv_splitter is None (single participant). Using default best parameters...")
        print(f"Default parameters: {default_best_params}")

        best_model = LogisticRegression(**default_best_params)
        best_model.fit(X_train, y_train)

        best_params = default_best_params
        best_cv_score = None
        cv_results = None

    # Evaluate on test set
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = best_model.predict(X_test)

    # Calculate test metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auroc = roc_auc_score(y_test, y_test_proba)
    test_f1 = f1_score(y_test, y_test_pred)
    test_pr_auc = average_precision_score(y_test, y_test_proba)

    print(f"\n=== Test Set Results ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")

    return {
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'cv_results': cv_results,
        'test_metrics': {
            'accuracy': test_acc,
            'auroc': test_auroc,
            'f1': test_f1,
            'pr_auc': test_pr_auc
        },
        'model': best_model
    }


def run_logistic_regression_with_gridsearch_verbose(
        X_train, y_train, groups_train, X_test, y_test,
        feature_names, cv_splitter, standardize=False, seed=42
):
    """Run GridSearchCV for Logistic Regression with detailed logging of splits and fitting."""

    # Standardize features (fit on train, transform both)
    if standardize:
        X_train, _, X_test = standardize_features(X_train, None, X_test, feature_names)

    # Parameter grid for grid search
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0],
        'max_iter': [1000],
    }

    print("=== DETAILED CV PROCESS ===")
    print(f"Total training samples: {len(X_train)}")
    print(f"Unique training participants: {np.unique(groups_train)}")
    print(f"Parameter grid: {param_grid}")
    print(f"Total parameter combinations: {len(param_grid['C']) * len(param_grid['max_iter'])}")

    # First, let's manually inspect the CV splits
    print(f"\n=== CV SPLIT INSPECTION ===")
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train, groups_train)):
        train_participants = np.unique(groups_train[train_idx])
        val_participants = np.unique(groups_train[val_idx])

        print(f"Fold {fold_idx + 1}:")
        print(f"  Train participants: {train_participants} \n({len(train_participants)} participants)")
        print(f"  Val participants: {val_participants} \n({len(val_participants)} participants)")
        print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        print(f"  Train label distribution: {np.bincount(y_train[train_idx].astype(int))}")
        print(f"  Val label distribution: {np.bincount(y_train[val_idx].astype(int))}")
        print()

    # Manual grid search with detailed logging
    print("=== MANUAL GRID SEARCH WITH DETAILED LOGGING ===")

    best_score = -np.inf
    best_params = None
    all_results = []

    total_fits = len(param_grid['C']) * len(param_grid['max_iter']) * cv_splitter.get_n_splits()
    fit_count = 0

    for C in param_grid['C']:
        for max_iter in param_grid['max_iter']:
            print(f"\n--- Testing C={C}, max_iter={max_iter} ---")

            current_params = {'C': C, 'max_iter': max_iter}
            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train, groups_train)):
                fit_count += 1
                print(f"  Fold {fold_idx + 1}/{cv_splitter.get_n_splits()} (Fit {fit_count}/{total_fits})")

                # Get fold data
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train[val_idx]

                # Create and train model for this fold
                model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)

                # Fit model
                print(f"    Fitting model on {len(X_fold_train)} samples...")
                model.fit(X_fold_train, y_fold_train)

                # Evaluate
                y_val_proba = model.predict_proba(X_fold_val)[:, 1]
                auroc = roc_auc_score(y_fold_val, y_val_proba)
                fold_scores.append(auroc)

                # Additional metrics for this fold
                y_val_pred = model.predict(X_fold_val)
                acc = accuracy_score(y_fold_val, y_val_pred)
                f1 = f1_score(y_fold_val, y_val_pred)

                print(f"    Fold {fold_idx + 1} results: AUROC={auroc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
                print(f"    \nParticipants - Train: {np.unique(groups_train[train_idx])}")
                print(f"    \nNumber Participants - Train: {len(np.unique(groups_train[train_idx]))}")
                print(f"    \nParticipants - Val: {np.unique(groups_train[val_idx])}")
                print(f"    \nNumber Participants - Val: {len(np.unique(groups_train[val_idx]))}")

            # Calculate mean CV score for this parameter combination
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            print(f"  Mean CV AUROC: {mean_score:.4f} (±{std_score:.4f})")
            print(f"  Individual fold scores: {[f'{score:.4f}' for score in fold_scores]}")

            # Track results
            result = {
                'params': current_params,
                'mean_test_score': mean_score,
                'std_test_score': std_score,
                'fold_scores': fold_scores
            }
            all_results.append(result)

            # Update best parameters
            if mean_score > best_score:
                best_score = mean_score
                best_params = current_params
                print(f"  *** NEW BEST PARAMETERS! ***")

    print(f"\n=== GRID SEARCH COMPLETE ===")
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {best_score:.4f}")

    # Train final model with best parameters
    print(f"\n=== TRAINING FINAL MODEL ===")
    print(f"Training on all {len(X_train)} training samples with best params: {best_params}")

    final_model = LogisticRegression(**best_params, random_state=seed)
    final_model.fit(X_train, y_train)

    print(f"Final model coefficients shape: {final_model.coef_.shape}")
    print(f"Final model intercept: {final_model.intercept_}")

    # Show feature importance (top 10 most important features)
    if len(feature_names) == len(final_model.coef_[0]):
        feature_importance = np.abs(final_model.coef_[0])
        top_indices = np.argsort(feature_importance)[-10:][::-1]

        print(f"\nTop 10 most important features:")
        for i, idx in enumerate(top_indices):
            print(f"  {i + 1}. {feature_names[idx]}: {final_model.coef_[0][idx]:.4f}")

    # Test evaluation
    print(f"\n=== FINAL TEST EVALUATION ===")
    print(f"Testing on {len(X_test)} test samples")

    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = final_model.predict(X_test)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_auroc = roc_auc_score(y_test, y_test_proba)
    test_f1 = f1_score(y_test, y_test_pred)
    test_pr_auc = average_precision_score(y_test, y_test_proba)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")

    # Summary table of all parameter combinations
    print(f"\n=== COMPLETE RESULTS SUMMARY ===")
    print("Params | Mean AUROC | Std AUROC | Fold Scores")
    print("-" * 60)
    for result in sorted(all_results, key=lambda x: x['mean_test_score'], reverse=True):
        params_str = f"C={result['params']['C']}"
        scores_str = ", ".join([f"{score:.3f}" for score in result['fold_scores']])
        print(
            f"{params_str:8} | {result['mean_test_score']:.4f}     | {result['std_test_score']:.4f}     | [{scores_str}]")

    return {
        'best_params': best_params,
        'best_cv_score': best_score,
        'all_results': all_results,
        'test_metrics': {
            'accuracy': test_acc,
            'auroc': test_auroc,
            'f1': test_f1,
            'pr_auc': test_pr_auc
        },
        'final_model': final_model
    }

def run_mlp_with_cv_and_test(
        X_train, y_train, groups_train, X_test, y_test,
        feature_names, cv_splitter, device, classifier_epochs=25, classifier_batch_size=32,
        classifier_lr=1e-4, standardize=False, seed=42
):
    """Run CV for MLP hyperparameter selection, then train final model and test."""

    # Standardize features
    if standardize:
        X_train, _, X_test = standardize_features(X_train, None, X_test, feature_names)

    # Simple hyperparameter options for MLP
    hidden_dims = [16, 32, 64]
    dropout_rates = [0.1, 0.2, 0.3, 0.5]

    best_params = None
    best_cv_score = 0

    default_best_params = {
        'hidden_dim': 16,
        'dropout': 0.2,
    }

    print("Running manual CV for MLP hyperparameters...")

    if cv_splitter is None:
        print("cv_splitter is None (single participant). Using default best parameters...")
        print(f"Default parameters: {default_best_params}")

        best_params = default_best_params
        best_cv_score = 0.0

    else:
        # Manual grid search (since sklearn GridSearchCV doesn't work with PyTorch)
        for hidden_dim in hidden_dims:
            for dropout_rate in dropout_rates:
                print(f"Testing hidden_dim={hidden_dim}, dropout={dropout_rate}")

                fold_scores = []

                # Run CV for this parameter combination
                for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train, groups_train)):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                    # Create and train model
                    input_dim = X_fold_train.shape[-1]
                    model = MLPClassifier(input_dim, hidden_dim=hidden_dim, dropout=dropout_rate).to(device)

                    tr_loader = build_linear_loaders(X_fold_train, y_fold_train, classifier_batch_size, device)
                    val_loader = build_linear_loaders(X_fold_val, y_fold_val, classifier_batch_size, device, shuffle=False)

                    optimizer = torch.optim.AdamW(model.parameters(), lr=classifier_lr)
                    loss_fn = torch.nn.BCEWithLogitsLoss()

                    # Training loop
                    for epoch in range(classifier_epochs):
                        model.train()
                        for X_batch, y_batch in tr_loader:
                            optimizer.zero_grad()
                            logits = model(X_batch).squeeze(-1)
                            loss = loss_fn(logits, y_batch)
                            loss.backward()
                            optimizer.step()

                    # Validation evaluation
                    model.eval()
                    val_probs = []
                    val_labels = []

                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            logits = model(X_batch).squeeze(-1)
                            probs = torch.sigmoid(logits)
                            val_probs.extend(probs.cpu().numpy())
                            val_labels.extend(y_batch.cpu().numpy())

                    fold_auroc = roc_auc_score(val_labels, val_probs)
                    fold_scores.append(fold_auroc)

                # Average CV score for this parameter combination
                mean_cv_score = np.mean(fold_scores)
                print(f"  Mean CV AUROC: {mean_cv_score:.4f}")

                if mean_cv_score > best_cv_score:
                    best_cv_score = mean_cv_score
                    best_params = {'hidden_dim': hidden_dim, 'dropout': dropout_rate}

    print(f"\nBest parameters: {best_params}")
    print(f"Best CV score: {best_cv_score:.4f}")

    # Train final model with best parameters on full training set
    print("Training final model on full training set...")
    input_dim = X_train.shape[-1]
    final_model = MLPClassifier(
        input_dim,
        hidden_dim=best_params['hidden_dim'],
        dropout=best_params['dropout']
    ).to(device)

    tr_loader = build_linear_loaders(X_train, y_train, 32, device)
    te_loader = build_linear_loaders(X_test, y_test, 32, device, shuffle=False)

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train final model
    for idx, epoch in enumerate(range(classifier_epochs), start=1):
        print(f"Epoch {idx} / {classifier_epochs}")
        final_model.train()
        for X_batch, y_batch in tr_loader:
            optimizer.zero_grad()
            logits = final_model(X_batch).squeeze(-1)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            if epoch % 5 == 0:
                print(f"The loss is {loss}")
            optimizer.step()

    # Test evaluation
    final_model.eval()
    test_probs = []
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for X_batch, y_batch in te_loader:
            logits = final_model(X_batch).squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y_batch.cpu().numpy())

    test_probs = np.array(test_probs)
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    test_acc = accuracy_score(test_labels, test_preds)
    test_auroc = roc_auc_score(test_labels, test_probs)
    test_f1 = f1_score(test_labels, test_preds)
    test_pr_auc = average_precision_score(test_labels, test_probs)

    print(f"\n=== Test Set Results ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")

    return {
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'test_metrics': {
            'accuracy': test_acc,
            'auroc': test_auroc,
            'f1': test_f1,
            'pr_auc': test_pr_auc
        },
        'model': final_model
    }