import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import mlflow
from mlflow.tracking import MlflowClient
from torch.utils.data import Dataset, DataLoader

from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torcheval.metrics.functional import multiclass_f1_score
from typing import Union, Sequence, Tuple, Dict, Any, List

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def to_1d(x: np.ndarray) -> np.ndarray:
    """Remove channel/extra dims so that augmentations see (L,) arrays."""
    return np.asarray(x).squeeze()        # guarantees positive length axis=-1

def same_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Resample (with linear interp) so that len(arr)==target_len."""
    if arr.shape[-1] == target_len:
        return arr
    return cv2.resize(arr.reshape(-1, 1), (1, target_len),
                      interpolation=cv2.INTER_LINEAR).flatten()

def safe_contiguous(x: np.ndarray) -> np.ndarray:
    """Ensure positive stride & contiguous memory."""
    return np.ascontiguousarray(x)

def build_simclr_fingerprint(cfg: dict[str, object]) -> dict[str, str]:
    """Return an *immutable* (str-typed) dict uniquely identifying
       one SimCLR encoder configuration."""
    keys = (
        "model_name", "seed",
        "epochs", "lr", "batch_size", "temperature",
        "window_len",   
    )
    return {k: str(cfg[k]) for k in keys}

def search_encoder_fp(fp: dict[str, str], experiment_name: str,
                      tracking_uri: str) -> str | None:
    """
    Look for a FINISHED MLflow run whose params match *exactly* the fingerprint.
    Returns the run_id (str) or None if not found.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp    = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None

    clauses = ["attributes.status = 'FINISHED'"]
    clauses += [f"params.{k} = '{v}'" for k, v in fp.items()]
    query    = " and ".join(clauses)
    hits = mlflow.search_runs([exp.experiment_id], filter_string=query,
                              max_results=1)
    return None if hits.empty else hits.iloc[0]["run_id"]

# -------------------------------------------------------------
# Tuned augmentations for 10k-sample ECG windows
# -------------------------------------------------------------
def add_noise_with_SNR(x, snr_db=None):
    snr = snr_db if snr_db is not None else np.random.uniform(18, 30)
    p_signal = np.mean(x**2)
    p_noise  = p_signal / (10**(snr/10))
    return x + np.random.normal(0, np.sqrt(p_noise), size=x.shape)

def random_scaling(x):
    return x * np.random.uniform(0.9, 1.1)

def random_crop_shift(x, crop_len=8000):
    if len(x) <= crop_len:
        return x
    start = np.random.randint(0, len(x)-crop_len)
    cropped = x[start:start+crop_len]
    pad_left  = np.random.randint(0, len(x)-crop_len)
    pad_right = len(x)-crop_len-pad_left
    return np.pad(cropped, (pad_left, pad_right))

def local_jitter(x):
    return x + np.random.normal(0, 0.003*np.std(x), size=x.shape)

def baseline_wander(x):
    f = np.random.uniform(0.05, 0.3)
    t = np.arange(len(x)) / 1000   
    amp = np.random.uniform(0.01, 0.05) * np.std(x)
    return x + amp * np.sin(2*np.pi*f*t)

def negate(x):   return -x
def hor_flip(x): return np.ascontiguousarray(np.flip(x))

AUGS = [
    (add_noise_with_SNR, 0.25),
    (random_scaling,     0.20),
    (random_crop_shift,  0.20),
    (local_jitter,       0.15),
    (baseline_wander,    0.10),
    (negate,             0.05),
    (hor_flip,           0.05),
]

funcs, probs = zip(*AUGS)

def sample_augmented(x: np.ndarray) -> np.ndarray:
    idx = np.random.choice(len(funcs), p=probs)
    return funcs[idx](x)

def DataTransform(signal):
    sig = to_1d(signal)
    v1, v2 = sig.copy(), sig.copy()
    v1 = sample_augmented(v1); v1 = sample_augmented(v1)
    v2 = sample_augmented(v2); v2 = sample_augmented(v2)
    # ensure shape & contiguity
    L = sig.shape[-1]
    return safe_contiguous(same_length(v1, L)), safe_contiguous(same_length(v2, L))

# ----------------------------------------------------------------------
# Encoder f(.)
# ----------------------------------------------------------------------
class ECGEncoder(nn.Module):
    """1D CNN encoder with 3 blocks, each block: Conv1d -> ReLU -> MaxPool -> Dropout"""
    def __init__(self, input_channels=1, dropout=0.3, window=10000):
        super(ECGEncoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=32, stride=1, padding=16, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=8, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=4, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        # After conv blocks, flatten and FC to 80 units
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 *  (window // 8), 80) 
        # L2 regularization to be set via optimizer weight_decay

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        h = self.fc(x)
        return h

# ----------------------------------------------------------------------
# Projection head g(.)
# ----------------------------------------------------------------------
class ProjectionHead(nn.Module):
    """MLP → 128‑dim"""
    def __init__(self, in_dim=80, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim))

    def forward(self, h):
        z = self.net(h)
        return F.normalize(z, dim=1)

# ----------------------------------------------------------------------
# SimCLR Model
# ----------------------------------------------------------------------
class SimCLR(nn.Module):
    def __init__(self, encoder:ECGEncoder, projector:ProjectionHead):
        super().__init__()
        self.f, self.g = encoder, projector

    def forward(self, x1, x2):                 # views (B,1,L)
        h1, h2 = self.f(x1), self.f(x2)        # (B,80)
        z1, z2 = self.g(h1), self.g(h2)        # (B,256)
        return h1, h2, z1, z2
    
# ----------------------------------------------------------------------
# NT-Xent Loss
# ----------------------------------------------------------------------
class NTXentLoss(nn.Module):
    def __init__(self, batch_size:int, T:float=0.5):
        super().__init__()
        self.B, self.T = batch_size, T
        self.register_buffer("labels", torch.arange(batch_size).repeat(2))

    def forward(self, z1, z2):                 # (B,D)
        z = torch.cat([z1, z2], dim=0)         # (2B,D)
        sim = torch.mm(z, z.t()) / self.T      # cosine sims after norm
        sim.fill_diagonal_(-1e9)               # mask self‑contrast
        l_pos = torch.diag(sim, self.B)        # positives
        r_pos = torch.diag(sim, -self.B)
        positives = torch.cat([l_pos, r_pos], 0)[:,None]     # (2B,1)
        loss = -torch.mean(torch.log(torch.exp(positives) / torch.exp(sim).sum(dim=1, keepdim=True)))
        return loss

# ----------------------------------------------------------------------
# Dataset wrapper
# ----------------------------------------------------------------------
class SimCLRDataset(Dataset):
    def __init__(self, signals):
        self.signals = signals

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx]
        v1, v2 = DataTransform(x)
        
        # reshape to (C,L)
        v1 = torch.tensor(v1, dtype=torch.float).unsqueeze(0)
        v2 = torch.tensor(v2, dtype=torch.float).unsqueeze(0)
        return v1, v2

def get_simclr_model(window:int=10000, device:str="cpu"):
    enc = ECGEncoder(window=window).to(device)
    proj = ProjectionHead().to(device)
    return SimCLR(enc, proj)

def simclr_data_loaders(X_train, X_val, batch_size:int):
    return (DataLoader(SimCLRDataset(X_train), batch_size, shuffle=True,  drop_last=True),
            DataLoader(SimCLRDataset(X_val),   batch_size, shuffle=False, drop_last=True))

def pretrain_one_epoch(model, loader, loss_fn, opt, device):
    model.train()
    tot = 0
    for v1,v2 in loader:
        v1,v2 = v1.to(device), v2.to(device)
        _,_,z1,z2 = model(v1,v2)
        loss = loss_fn(z1, z2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
    return tot/len(loader)

@torch.no_grad()
def encode_representations(model, X, batch_size:int, device:str):
    """Return L2‑normalised encoder outputs h."""
    loader = DataLoader(torch.from_numpy(X).float(), batch_size, shuffle=False)
    reps = []
    model.eval()
    for xb in loader:
        xb = xb.permute(0, 2, 1).to(device) 
        h = F.normalize(model.f(xb), dim=1)
        reps.append(h.cpu().numpy())
    return np.concatenate(reps,0)

# ----------------------------------------------------------------------        
# linear classifier
# ----------------------------------------------------------------------
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x is expected to be of shape (batch, feature_dim)
        return self.fc(x)
    
# ------------------------------------------------------------------
# threshold sweep helper
# ------------------------------------------------------------------
def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2,
    average: str = "macro",
    grid: int = 101,
):
    """
    Scan `grid` equally-spaced thresholds in (0,1) and return the one
    that maximises Macro-F1 together with that F1.
    """
    ts = np.linspace(0.0, 1.0, grid, endpoint=False)[1:]
    best_t, best_f1 = 0.5, -1.0
    labels_t = torch.from_numpy(labels.astype(np.int64))
    for t in ts:
        preds_t = torch.from_numpy((probs >= t).astype(np.int64))
        f1 = multiclass_f1_score(preds_t, labels_t,
                                 num_classes=num_classes,
                                 average=average).item()
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

# --------------------------------------------------------
# Classifier Training Loop
# --------------------------------------------------------
def train_linear_classifier(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    epochs,
    device,
):
    """
    Train `model`, tune threshold on the validation set each epoch,
    log everything to MLflow and return (`model`, `best_threshold`).
    """
    train_auc_m = BinaryAUROC()
    train_pr_m  = BinaryAveragePrecision()
    val_auc_m   = BinaryAUROC()
    val_pr_m    = BinaryAveragePrecision()

    best_threshold = 0.5
    best_val_f1_overall = -1.0

    for epoch in range(1, epochs + 1):
        # TRAIN
        model.train()
        running_loss = correct = total = 0
        train_auc_m.reset(); train_pr_m.reset()

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X).squeeze()
            loss   = loss_fn(logits, y)
            loss.backward(); optimizer.step()

            running_loss += loss.item() * X.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()
            total   += y.size(0)

            train_auc_m.update(probs.detach().cpu(), y.cpu().int())
            train_pr_m.update(probs.detach().cpu(),  y.cpu().int())

        mlflow.log_metrics({
            "train_loss"  : running_loss / total,
            "train_accuracy": correct / total,
            "train_auroc" : train_auc_m.compute().item(),
            "train_pr_auc": train_pr_m.compute().item(),
        }, step=epoch)

        # VALIDATION + TUNING 
        model.eval()
        val_auc_m.reset(); val_pr_m.reset()
        val_probs, val_labels = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X).squeeze()
                probs  = torch.sigmoid(logits)

                val_probs.append(probs.cpu().numpy())
                val_labels.append(y.cpu().numpy())

                val_auc_m.update(probs.cpu(), y.cpu().int())
                val_pr_m.update(probs.cpu(),  y.cpu().int())

        p = np.concatenate(val_probs)
        l = np.concatenate(val_labels)

        t_star, f1_star = find_best_threshold(p, l)
        if f1_star > best_val_f1_overall:
            best_val_f1_overall = f1_star
            best_threshold      = t_star

        preds_val = (p >= t_star).astype(int)
        val_acc   = (preds_val == l).mean()

        mlflow.log_metrics({
            "val_accuracy"      : val_acc,
            "val_auroc"         : val_auc_m.compute().item(),
            "val_pr_auc"        : val_pr_m.compute().item(),
            "val_best_macro_f1" : f1_star,
            "val_best_threshold": t_star,
        }, step=epoch)

        print(f"[Ep {epoch}] val_acc={val_acc:.4f}  auc={val_auc_m.compute():.4f}  "
              f"f1*={f1_star:.4f} @ t={t_star:.2f}")

    return model, best_threshold

# --------------------------------------------------------
# Classifier Evaluation Loop
# --------------------------------------------------------
def evaluate_classifier(
    model,
    test_loader,
    device,
    threshold: float,
    loss_fn=None,        
):

    model.eval()
    test_auc_m = BinaryAUROC()
    test_pr_m  = BinaryAveragePrecision()

    probs_all, labels_all = [], []
    running_loss = correct = total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X).squeeze(-1)
            probs  = torch.sigmoid(logits)

            if loss_fn is not None:
                running_loss += loss_fn(logits, y).item() * X.size(0)

            probs_all .append(probs.cpu().numpy())
            labels_all.append(y.cpu().numpy().astype(np.int64))  

            test_auc_m.update(probs.cpu(), y.cpu().int())
            test_pr_m .update(probs.cpu(), y.cpu().int())

            correct += ((probs >= threshold).float() == y).sum().item()
            total   += y.size(0)

    p = np.concatenate(probs_all)
    l = np.concatenate(labels_all)              
    preds = (p >= threshold).astype(np.int64)    

    test_metrics = {
        "test_accuracy": correct / total,
        "test_auroc"   : test_auc_m.compute().item(),
        "test_pr_auc"  : test_pr_m.compute().item(),
        "test_f1"      : multiclass_f1_score(
                            torch.from_numpy(preds),
                            torch.from_numpy(l),
                            num_classes=2,
                            average="macro"
                         ).item(),
        "test_threshold": threshold,
    }
    if loss_fn is not None:
        test_metrics["test_loss"] = running_loss / total

    mlflow.log_metrics(test_metrics)
    print(f"TEST ▶ acc={test_metrics['test_accuracy']:.4f} "
          f"auc={test_metrics['test_auroc']:.4f} "
          f"f1={test_metrics['test_f1']:.4f} @ t={threshold:.2f}")

    return (test_metrics["test_accuracy"],
            test_metrics["test_auroc"],
            test_metrics["test_pr_auc"],
            test_metrics["test_f1"])


DEBUG_SHAPES = True # flip to False to mute everything

_printed_once: set[str] = set()  

def _shape(x):
    """convenience – works for tensors / ndarrays / lists"""
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return tuple(x.shape)
    return str(type(x))

def show_shape(label: str, obj: Union[torch.Tensor, np.ndarray, Sequence], *,
               once: bool = True) -> None:
    """
    Print `<label>: <shape>` in a *single* line.
    If `once=True` (default) the same label is never printed again
    """
    if not DEBUG_SHAPES:
        return
    if once and label in _printed_once:
        return
    _printed_once.add(label)

    if isinstance(obj, (list, tuple)):
        shapes = [_shape(o) for o in obj]
    else:
        shapes = _shape(obj)
    print(f"[DBG] {label:<26s} : {shapes}", flush=True)