import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import pandas as pd
import os
import sys
import logging
import mlflow

from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torcheval.metrics.functional import multiclass_f1_score
from typing import Union, Sequence, Tuple, Dict, Any, List
from shutil import copy
from einops import rearrange, repeat
from typing import Union, Sequence
from mlflow.tracking import MlflowClient

# ----------------------------------------------------------------------
# augmentations.py
# ----------------------------------------------------------------------
def DataTransform(sample, config):
    if config.augmentation.use_spectral_aug:
        print(f"We use the spectral augmentation!")
        # Use spectral augmentation instead of default
        weak_aug = frequency_masking(sample, mask_ratio=config.augmentation.freq_mask_ratio)
        strong_aug = frequency_masking(
            permutation(sample, max_segments=config.augmentation.max_seg),
            mask_ratio=config.augmentation.freq_mask_ratio
        )
    else:
        # Default augmentation strategy
        weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
        strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    
    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    x_np = x if isinstance(x, np.ndarray) else x.numpy()
    N, C, L = x_np.shape
    orig_steps = np.arange(L)

    ret = np.empty_like(x_np)                 # (N, C, L)
    for i in range(N):
        # 1) choose how many segments to split into
        n_seg = np.random.randint(1, max_segments + 1)
        if n_seg > 1:
            # 2) split & permute indices
            if seg_mode == "random":
                split_points = np.random.choice(L - 2, n_seg - 1, replace=False) + 1
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, n_seg)
            order = np.random.permutation(len(splits))
            warp = np.concatenate([splits[o] for o in order])
        else:
            warp = orig_steps

        # 3) ***transpose*** to (C, L) before writing
        ret[i] = x_np[i, :, warp].T

    return ret


def frequency_masking(x, mask_ratio=0.1):
    """Apply frequency masking by zeroing out random frequency components"""
    x_np = x if isinstance(x, np.ndarray) else x.numpy()
    N, C, L = x_np.shape
    ret = np.empty_like(x_np)
    
    for i in range(N):
        for c in range(C):
            # FFT to frequency domain
            fft_signal = np.fft.fft(x_np[i, c, :])
            
            # Calculate number of frequencies to mask
            n_freqs = len(fft_signal)
            n_mask = int(n_freqs * mask_ratio)
            
            # Randomly select frequency indices to mask (excluding DC component)
            freq_indices = np.random.choice(range(1, n_freqs), size=n_mask, replace=False)
            
            # Zero out selected frequency components
            fft_masked = fft_signal.copy()
            fft_masked[freq_indices] = 0
            
            # IFFT back to time domain
            ret[i, c, :] = np.real(np.fft.ifft(fft_masked))
    
    return ret



# ----------------------------------------------------------------------
# dataloader.py
# ----------------------------------------------------------------------
class Load_Dataset(Dataset):
    def __init__(self, dataset, config, training_mode):
        super().__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        # ensure a channel-dim exists
        if len(X_train.shape) < 3:           # (N, L)  ->  (N, L, 1)
            if isinstance(X_train, np.ndarray):
                X_train = np.expand_dims(X_train, 2)
            else:  # torch.Tensor
                X_train = X_train.unsqueeze(2)

        # make channel dimension second: (N, L, C) → (N, C, L)
        if X_train.shape.index(min(X_train.shape)) != 1:
            if isinstance(X_train, np.ndarray):
                X_train = np.transpose(X_train, (0, 2, 1))
            else:                             # torch.Tensor
                X_train = X_train.permute(0, 2, 1)

        # final conversion to torch tensors
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = self.x_data.shape[0]

        # Augmentations only for self-supervised mode
        if training_mode == "self_supervised":
            aug1_np, aug2_np = DataTransform(self.x_data.numpy(), config)
            self.aug1 = torch.from_numpy(aug1_np).float()
            self.aug2 = torch.from_numpy(aug2_np).float()

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len

def data_generator_from_arrays(
    X_train, y_train, X_val, y_val, X_test, y_test,
    configs, training_mode
):
    
    train_dict = {"samples": X_train, "labels": y_train}
    val_dict   = {"samples": X_val,   "labels": y_val}
    test_dict  = {"samples": X_test,  "labels": y_test}

    # Create Dataset objects
    train_ds = Load_Dataset(train_dict, configs, training_mode)
    val_ds   = Load_Dataset(val_dict,   configs, training_mode)
    test_ds  = Load_Dataset(test_dict,  configs, training_mode)

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=configs.batch_size,
        shuffle=True,
        drop_last=configs.drop_last,
        num_workers=0
    )
    valid_loader = DataLoader(
        dataset=val_ds,
        batch_size=configs.batch_size,
        shuffle=False,
        drop_last=configs.drop_last,
        num_workers=0
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=configs.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    return train_loader, valid_loader, test_loader

def build_linear_loaders(
        X_repr: np.ndarray, y: np.ndarray,
        batch_size: int, device: str, shuffle: bool = True, drop_last=False,
) -> DataLoader:

    ds = TensorDataset(
        torch.from_numpy(X_repr).float().to(device),
        torch.from_numpy(y).float().to(device)
    )

    num_workers = min(8, os.cpu_count() or 2)
    use_cuda = (device.type == "cuda")
    pin_memory = use_cuda

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory, num_workers=num_workers
    )

# ----------------------------------------------------------------------
# loss.py
# ----------------------------------------------------------------------
class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


# ----------------------------------------------------------------------
# attention.py
# ----------------------------------------------------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()


    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t

# ----------------------------------------------------------------------
# tc.py
# ----------------------------------------------------------------------
class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for _ in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2):
        show_shape("TC IN feats1/2", (features_aug1, features_aug2))
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        c_t = self.seq_transformer(forward_seq)
        show_shape("TC context token", c_t)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)
    
# ----------------------------------------------------------------------
# model.py
# ----------------------------------------------------------------------
class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        show_shape("base_Model in", x_in)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        show_shape("base_Model feat_out(flat)", x_flat)
        logits = self.logits(x_flat)
        return logits, x
    

# ----------------------------------------------------------------------
# utils.py
# ----------------------------------------------------------------------
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


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def copy_Files(destination, data_type):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("trainer/trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("dataloader/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/model.py", os.path.join(destination_dir, f"model.py"))
    copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("models/TC.py", os.path.join(destination_dir, "TC.py"))

def compute_ssl_loss(feat1, feat2, tc_model, nt_xent_loss):
    feat1 = F.normalize(feat1, dim=1)
    feat2 = F.normalize(feat2, dim=1)
    loss1, proj1 = tc_model(feat1, feat2)
    loss2, proj2 = tc_model(feat2, feat1)
    proj1 = F.normalize(proj1, dim=1)
    proj2 = F.normalize(proj2, dim=1)
    ssl_loss = loss1 + loss2 + 0.7 * nt_xent_loss(proj1, proj2)
    return ssl_loss

def build_tstcc_fingerprint(cfg: dict[str, any]) -> dict[str, str]:
    """
    Build immutable fingerprint (all values str) that uniquely
    identifies one TS-TCC encoder configuration.
    """
    keys = (
        "model_name", "seed",
        "tcc_epochs", "tcc_lr", "tcc_batch_size",
        "tc_timesteps", "tc_hidden_dim",
        "cc_temperature", "cc_use_cosine",
    )
    return {k: str(cfg[k]) for k in keys}


def search_encoder_fp(
    fp: Dict[str, str],
    experiment_name: str,
    tracking_uri: str,
    ) -> str | None:
    """
    Look for a finished MLflow run whose params match exactly the fingerprint.
    Returns the run_id or None.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client   = MlflowClient()
    exp      = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None

    # build filter string
    clauses = [ "attributes.status = 'FINISHED'" ]
    clauses += [ f"params.{k} = '{v}'" for k, v in fp.items() ]
    query    = " and ".join(clauses)

    hits = mlflow.search_runs([exp.experiment_id], filter_string=query, max_results=1)
    return None if hits.empty else hits.iloc[0]["run_id"]

# ----------------------------------------------------------------------
# trainer.py
# ----------------------------------------------------------------------
def Trainer(
        model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, config, experiment_log_dir, training_mode
):
    # Start training
    print("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    patience = getattr(config, 'patience', 10)  # Default patience of 10 epochs
    best_model_state = None
    best_tc_model_state = None

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)
        val_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode, config)

        if training_mode == "self_supervised":
            print(f"Epoch {epoch:02d} | ssl_train_loss: {train_loss:.4f} | ssl_val_loss: {val_loss:.4f}")
            mlflow.log_metrics({
                "ssl_train_loss": train_loss,
                "ssl_val_loss": val_loss
            }, step=epoch)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model states
                best_model_state = model.state_dict().copy()
                best_tc_model_state = temporal_contr_model.state_dict().copy()
                print(f"    → New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"    → No improvement. Patience: {patience_counter}/{patience}")

            # Check for early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

    # Save models (either best from early stopping or final)
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    
    if best_model_state is not None:
        # Save best model from early stopping
        chkpoint = {'model_state_dict': best_model_state, 'temporal_contr_model_state_dict': best_tc_model_state}
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_best.pt'))
        print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    # Also save final model
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    print("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model,
                model_opt, tc_opt,
                criterion, train_loader,
                config, device, training_mode):

    model.train()
    temporal_contr_model.train()

    nt_xent = NTXentLoss(device,
                         config.batch_size,
                         config.Context_Cont.temperature,
                         config.Context_Cont.use_cosine_similarity)

    batch_losses = []

    #for data, labels, aug1, aug2 in train_loader:
    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        if batch_idx == 0:
            show_shape("train-loop INPUT   aug1/aug2", (aug1, aug2))
            

        # move to device
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        data, labels = data.float().to(device), labels.long().to(device)

        model_opt.zero_grad()
        tc_opt.zero_grad()

        # SSL vs supervised branch
        if training_mode == "self_supervised":
            _, feat1 = model(aug1)
            _, feat2 = model(aug2)
            loss = compute_ssl_loss(feat1, feat2, temporal_contr_model, nt_xent)
        else:                          
            preds, _ = model(data)
            loss = criterion(preds, labels)

        # backward & step
        loss.backward()
        model_opt.step()
        tc_opt.step()

        batch_losses.append(loss.item())

    return torch.tensor(batch_losses).mean(), torch.nan              

def model_evaluate(model, temporal_contr_model,
                   dl, device, training_mode, config):

    model.eval()
    temporal_contr_model.eval()

    nt_xent = NTXentLoss(device,
                         config.batch_size,
                         config.Context_Cont.temperature,
                         config.Context_Cont.use_cosine_similarity)

    criterion = nn.CrossEntropyLoss()
    val_losses = []

    with torch.no_grad():
        for batch_idx, (data, labels, aug1, aug2) in enumerate(dl):
            if batch_idx == 0:
                show_shape("eval-loop INPUT    aug1/aug2", (aug1, aug2))

            aug1, aug2 = aug1.to(device), aug2.to(device)
            data, labels = data.to(device), labels.to(device)

            if training_mode == "self_supervised":
                _, feat1 = model(aug1)
                _, feat2 = model(aug2)
                loss = compute_ssl_loss(feat1, feat2, temporal_contr_model, nt_xent)
            else:
                preds, _ = model(data)
                loss = criterion(preds, labels)

            val_losses.append(loss.item())

    return torch.tensor(val_losses).mean(), torch.nan, [], []

# ----------------------------------------------------------------------
# config.py
# ----------------------------------------------------------------------
class Config(object):
    """
    Hyper-parameters tuned for windowed ECG segments
    (shape = [N, 10 000, 1])
    """
    def __init__(self, fs:int=1000, window_size:int=10):

        # ─────────────────── CNN encoder ────────────────────
        self.input_channels      = 1         # ECG is univariate
        self.kernel_size         = 32        # same as pFD → good for long series
        self.stride              = 4
        self.final_out_channels  = 128       # feature maps after last conv
        self.dropout             = 0.35

        # Length of the sequence that reaches the projection head
        # 10 000 → 315 after the conv + pool stack

        if int(fs) == 1000:
            if window_size == 30:
                self.features_len = 940
            elif window_size == 10:
                self.features_len = 315

        elif int(fs) == 700:
            if window_size == 30:
                self.features_len = 658
            elif window_size == 10:
                self.features_len = 221

        elif int(fs) == 500:
            if window_size == 30:
                self.features_len = 471
            elif window_size == 10:
                self.features_len = 158

        self.num_classes         = 2

        # ─────────────────── Training ───────────────────────
        self.num_epoch           = 40
        self.patience            = 10        # Early stopping patience

        # Optimiser
        self.beta1, self.beta2   = 0.9, 0.99
        self.lr                  = 3e-4

        # Data loader
        self.batch_size          = 64        # 10000-sample windows use more mem
        self.drop_last           = True      # match original repo

        # ─────────────────── SSL blocks ─────────────────────
        # Contextual contrastive loss
        self.Context_Cont        = Context_Cont_configs()
        # Temporal contrasting
        self.TC                  = TCConfig()      # hidden=100, timesteps=50
        # Data augmentations
        self.augmentation        = augmentations()


class PPGConfig(object):
    """
    Hyper-parameters tuned for windowed ECG segments
    (shape = [N, 640, 1])
    """
    def __init__(self):

        # ─────────────────── CNN encoder ────────────────────
        self.input_channels      = 1         # PPG is univariate
        self.kernel_size         = 8        # same as pFD → good for long series
        self.stride              = 1
        self.final_out_channels  = 128       # feature maps after last conv
        self.dropout             = 0.35

        # Length of the sequence that reaches the projection head
        # 10 000 → 315 after the conv + pool stack
        self.features_len        = 82
        self.num_classes         = 2

        # ─────────────────── Training ───────────────────────
        self.num_epoch           = 40
        self.patience            = 10        # Early stopping patience

        # Optimiser
        self.beta1, self.beta2   = 0.9, 0.99
        self.lr                  = 3e-4

        # Data loader
        self.batch_size          = 64
        self.drop_last           = True      # match original repo

        # ─────────────────── SSL blocks ─────────────────────
        # Contextual contrastive loss
        self.Context_Cont        = Context_Cont_configs()
        # Temporal contrasting
        self.TC                  = TCConfig()      # hidden=100, timesteps=50
        # Data augmentations
        self.augmentation        = augmentations()


class augmentations(object):
    """
    ECG is quite sensitive to amplitude changes; we therefore keep
    small jitter noise and allow permutation across up to 8 segments.
    """
    def __init__(self):
        self.jitter_scale_ratio = 0.001   # very mild scaling
        self.jitter_ratio       = 0.001    # mild additive jitter
        self.max_seg            = 8       # split into ≤8 segments for strong aug
        self.use_spectral_aug   = True   # flag to enable spectral augmentation
        self.freq_mask_ratio    = 0.1     # frequency masking ratio when spectral aug is enabled


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature           = 0.2
        self.use_cosine_similarity = True


class TCConfig(object):
    def __init__(self):
        self.hidden_dim = 100      # same as original paper (here 64 probably, changed from 100)
        self.timesteps  = 50       # 15 % – 30 % of seq_len (here 45 – 95).

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
            logits = model(X).squeeze(-1)
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
            logits = model(X).squeeze()
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

# ----------------------------------------------------------------------
# Encoding function to textract TS-TCC representations
# ----------------------------------------------------------------------
def encode_representations(X, y, model, temporal_contr_model, tcc_batch_size, device):
    show_shape("encode_repr - raw X", X)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()),
        batch_size=tcc_batch_size, shuffle=False
    )
    reprs, labs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if xb.shape.index(min(xb.shape)) != 1:
                xb = xb.permute(0, 2, 1)
                # The xb is now in B, D, T
            # conv encoder
            _, feats = model(xb)
            feats = F.normalize(feats, dim=1)
            # TC projection head
            _, c_proj = temporal_contr_model(feats, feats)
            reprs.append(c_proj.cpu().numpy())
            labs.append(yb.numpy())
    return np.concatenate(reprs, axis=0), np.concatenate(labs, axis=0)