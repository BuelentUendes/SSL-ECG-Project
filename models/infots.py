import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torcheval.metrics.functional import multiclass_f1_score
from typing import Tuple, Dict, Any, List
import numpy as np
import math
import random
from datetime import datetime
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from scipy.special import softmax
from sklearn.metrics import log_loss
from S3 import S3
import os

# --------------------------------------------------------
# utils.py functions needed by InfoTS
# --------------------------------------------------------
def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

# --------------------------------------------------------
# InfoTS losses.py - adapted from InfoTS implementation
# --------------------------------------------------------
def global_infoNCE(z1, z2, pooling='max', temperature=1.0):
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)
    return InfoNCE(z1, z2, temperature)

def local_infoNCE(z1, z2, pooling='max', temperature=1.0, k=16):
    B = z1.size(0)
    T = z1.size(1)
    D = z1.size(2)
    crop_size = int(T/k)
    crop_leng = crop_size*k

    start = random.randint(0, T-crop_leng)
    crop_z1 = z1[:, start:start+crop_leng, :]
    crop_z1 = crop_z1.view(B, k, crop_size, D)

    if pooling == 'max':
        crop_z1 = crop_z1.reshape(B*k, crop_size, D)
        crop_z1_pooling = F.max_pool1d(crop_z1.transpose(1, 2).contiguous(), 
                                       kernel_size=crop_size).transpose(1, 2).reshape(B, k, D)
    elif pooling == 'mean':
        crop_z1_pooling = torch.unsqueeze(torch.mean(z1, 1), 1)

    crop_z1_pooling_T = crop_z1_pooling.transpose(1, 2)
    similarity_matrices = torch.bmm(crop_z1_pooling, crop_z1_pooling_T)

    labels = torch.eye(k-1, dtype=torch.float32)
    labels = torch.cat([labels, torch.zeros(1, k-1)], 0)
    labels = torch.cat([torch.zeros(k, 1), labels], -1)

    pos_labels = labels.to(z1.device)
    pos_labels[k-1, k-2] = 1.0

    neg_labels = labels.T + labels + torch.eye(k)
    neg_labels[0, 2] = 1.0
    neg_labels[-1, -3] = 1.0
    neg_labels = neg_labels.to(z1.device)

    similarity_matrix = similarity_matrices[0]
    positives = similarity_matrix[pos_labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~neg_labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:, 0].mean()
    return loss

def InfoNCE(z1, z2, temperature=1.0):
    batch_size = z1.size(0)
    features = torch.cat([z1, z2], dim=0).squeeze(1)
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:, 0].mean()
    return loss

# --------------------------------------------------------
# InfoTS Augmentation Module - adapted from InfoTS
# --------------------------------------------------------
class InfoTSAugmentation(nn.Module):
    def __init__(self, aug_p1=0.2, aug_p2=0.0, used_augs=None, device=None, dtype=None):
        super(InfoTSAugmentation, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # Define all available augmentations (simplified for ECG data)
        self.all_augs = self._create_augmentations()
        
        if used_augs is not None:
            self.augs = []
            for i in range(len(used_augs)):
                if used_augs[i]:
                    self.augs.append(self.all_augs[i])
        else:
            self.augs = self.all_augs
            
        self.weight = nn.Parameter(torch.empty((2, len(self.augs)), **factory_kwargs))
        self.reset_parameters()
        self.aug_p1 = aug_p1
        self.aug_p2 = aug_p2

    def _create_augmentations(self):
        """Create ECG-specific augmentations"""
        return [
            lambda x: self._jitter(x, 0.001),
            lambda x: self._scaling(x, 0.001), 
            lambda x: self._time_warp(x),
            lambda x: self._window_slice(x),
            # lambda x: self._subsequence(x),
            # lambda x: self._cutout(x),
            # lambda x: self._window_warp(x)
        ]

    def _jitter(self, x, sigma=0.001):
        """Add Gaussian noise"""
        return x + torch.randn_like(x) * sigma

    def _scaling(self, x, sigma=0.001):
        """Scale the time series"""
        factor = torch.normal(1.0, sigma, size=(x.size(0), 1, x.size(2))).to(x.device)
        return x * factor

    def _time_warp(self, x, sigma=0.2):
        """Simple time warping by interpolation"""
        orig_steps = torch.arange(x.size(1), dtype=torch.float32).to(x.device)
        random_warps = torch.normal(1.0, sigma, size=(x.size(0), 1)).to(x.device)
        warped_steps = orig_steps * random_warps
        warped_steps = torch.clamp(warped_steps, 0, x.size(1) - 1)
        
        # Simple linear interpolation
        indices = warped_steps.long()
        weights = warped_steps - indices.float()
        indices_next = torch.clamp(indices + 1, max=x.size(1) - 1)
        
        warped = x[:, indices] * (1 - weights).unsqueeze(-1) + x[:, indices_next] * weights.unsqueeze(-1)
        return warped

    def _window_slice(self, x, reduce_ratio=0.9):
        """Randomly slice a window and pad to original length"""
        original_len = x.size(1)
        target_len = int(original_len * reduce_ratio)
        start = torch.randint(0, original_len - target_len + 1, (1,)).item()
        
        # Extract slice and pad back to original length
        sliced = x[:, start:start + target_len]
        pad_len = original_len - target_len
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
        
        # Pad with zeros or repeat edge values
        padded = F.pad(sliced, (0, 0, pad_left, pad_right), mode='constant', value=0)
        return padded

    def _subsequence(self, x, reduce_ratio=0.95):
        """Extract random subsequence and pad to original length"""
        original_len = x.size(1)
        target_len = int(original_len * reduce_ratio)
        start = torch.randint(0, original_len - target_len + 1, (1,)).item()
        
        # Extract subsequence and pad back to original length
        subseq = x[:, start:start + target_len]
        pad_len = original_len - target_len
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
        
        # Pad with mean values to maintain signal characteristics
        padded = F.pad(subseq, (0, 0, pad_left, pad_right), mode='constant', value=0)
        return padded

    def _cutout(self, x, area_ratio=0.001):
        """Randomly mask part of the signal"""
        x_masked = x.clone()
        seq_len = x.size(1)
        mask_len = int(seq_len * area_ratio)
        
        for i in range(x.size(0)):
            start = torch.randint(0, seq_len - mask_len + 1, (1,)).item()
            x_masked[i, start:start + mask_len] = 0
        return x_masked

    def _window_warp(self, x, window_ratio=0.1, scales=[0.5, 2.]):
        """Warp random windows"""
        x_warped = x.clone()
        seq_len = x.size(1)
        window_len = int(seq_len * window_ratio)
        
        for i in range(x.size(0)):
            start = torch.randint(0, seq_len - window_len + 1, (1,)).item()
            scale = random.choice(scales)
            window = x_warped[i, start:start + window_len]
            
            # Simple scaling as warping
            x_warped[i, start:start + window_len] = window * scale
        return x_warped

    def get_sampling(self, temperature=1.0, bias=0.0):
        if self.training:
            bias = bias + 0.0001
            eps = (bias - (1 - bias)) * torch.rand(self.weight.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(self.weight.device)
            gate_inputs = (gate_inputs + self.weight) / temperature
            para = torch.softmax(gate_inputs, -1)
            return para
        else:
            return torch.softmax(self.weight, -1)

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, xt):
        x, t = xt
        if self.aug_p1 == 0 and self.aug_p2 == 0:
            return x.clone(), x.clone()
        
        para = self.get_sampling()

        if random.random() > self.aug_p1 and self.training:
            aug1 = x.clone()
        else:
            xs1_list = []
            original_shape = x.shape  # Store original shape: [batch, seq, features]
            
            for aug in self.augs:
                try:
                    aug_result = aug(x)
                    # Ensure the augmentation result has the same shape as input
                    if aug_result.shape != original_shape:
                        # If shapes don't match, use fallback
                        xs1_list.append(x.clone())
                    else:
                        xs1_list.append(aug_result)
                except Exception as e:
                    print("We fall back to the original sample ! IMPORTANT")
                    # Fallback to original on any error
                    xs1_list.append(x.clone())
            
            # Verify all tensors have the same shape before stacking
            if all(tensor.shape == original_shape for tensor in xs1_list):
                xs1 = torch.stack(xs1_list, 0)  # [num_augs, batch, seq, features]
                
                # Weighted combination of augmentations
                # para[0] should have shape [num_augs]
                para_expanded = para[0].view(-1, 1, 1, 1)  # [num_augs, 1, 1, 1]
                weighted_augs = xs1 * para_expanded  # Broadcasting
                aug1 = torch.sum(weighted_augs, dim=0)  # Sum over augmentation dimension
            else:
                # If shapes don't match, fall back to original
                aug1 = x.clone()

        # Second augmentation branch (controlled by aug_p2)
        if random.random() > self.aug_p2 and self.training:
            aug2 = x.clone()
        else:
            xs2_list = []
            for aug in self.augs:
                try:
                    aug_result = aug(x)
                    if aug_result.shape != original_shape:
                        xs2_list.append(x.clone())
                    else:
                        xs2_list.append(aug_result)
                except Exception as e:
                    xs2_list.append(x.clone())
            
            # Verify all tensors have the same shape before stacking
            if all(tensor.shape == original_shape for tensor in xs2_list):
                xs2 = torch.stack(xs2_list, 0)  # [num_augs, batch, seq, features]
                
                # Weighted combination of augmentations using para[1] for second view
                para_expanded = para[1].view(-1, 1, 1, 1)  # [num_augs, 1, 1, 1]
                weighted_augs = xs2 * para_expanded  # Broadcasting
                aug2 = torch.sum(weighted_augs, dim=0)  # Sum over augmentation dimension
            else:
                aug2 = x.clone()
        
        return aug1, aug2

# --------------------------------------------------------
# InfoTS Encoder - adapted from InfoTS but using ECG-friendly architecture
# --------------------------------------------------------
def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

# Dilated Conv blocks from existing codebase
class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual
    
class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)

class InfoTSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, 
                 mask_mode='binomial', dropout=0.1, use_s3_layers=False, **kwargs):
        super().__init__()
        
        self.use_s3_layer = use_s3_layers
        if self.use_s3_layer:
            self.s3_layers = S3(
                num_layers=kwargs.get("num_s3_layers", 2),
                initial_num_segments=kwargs.get("initial_num_segments", 2),
                shuffle_vector_dim=kwargs.get("shuffle_vector_dim", 1),
                segment_multiplier=kwargs.get("segment_multiplier", 2),
            )
            
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = None if dropout == 0.0 else nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
        if self.use_s3_layer:
            x_s3 = x.transpose(1, 2)
            x_s3 = self.s3_layers(x_s3)
            x = x_s3.transpose(1, 2)
            
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)
        
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        if self.repr_dropout is not None:
            x = self.repr_dropout(x)
        x = x.transpose(1, 2)
        
        return x

# --------------------------------------------------------
# InfoTS Main Model
# --------------------------------------------------------
class InfoTS:
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=60,
        num_cls=2,
        depth=10,
        device='cuda',
        lr=0.001,
        meta_lr=0.01,
        batch_size=16,
        max_train_length=None,
        mask_mode='binomial',
        dropout=0.1,
        aug_p1=0.2,
        aug_p2=0.0,
        eval_every_epoch=40,
        used_augs=None,
        use_s3_layers=False,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        
        self._net = InfoTSEncoder(
            input_dims=input_dims, 
            output_dims=output_dims,
            hidden_dims=hidden_dims, 
            depth=depth,
            dropout=dropout, 
            mask_mode=mask_mode,
            use_s3_layers=use_s3_layers,
            **kwargs
        ).to(self.device)

        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.n_epochs = 0
        self.n_iters = 0

        self.pred = torch.nn.Linear(output_dims, num_cls).to(self.device)
        self.unsup_pred = torch.nn.Linear(output_dims, batch_size).to(self.device)

        self.aug = InfoTSAugmentation(aug_p1=aug_p1, aug_p2=aug_p2, used_augs=used_augs).to(self.device)
        self.meta_lr = meta_lr

        self.single = (aug_p2 == 0.0)
        self.CE = torch.nn.CrossEntropyLoss()
        self.BCE = torch.nn.BCEWithLogitsLoss()
        self.cls_lr = meta_lr
        self.eval_every_epoch = eval_every_epoch
        self.t0 = 2.0
        self.t1 = 0.1

    def get_dataloader(self, data, shuffle=False, drop_last=False):
        if self.max_train_length is not None:
            sections = data.shape[1] // self.max_train_length
            if sections >= 2:
                data = np.concatenate(split_with_nan(data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            data = centerize_vary_length_series(data)

        data = data[~np.isnan(data).all(axis=2).all(axis=1)]
        data = np.nan_to_num(data)
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), 
                          shuffle=shuffle, drop_last=drop_last)
        return data, dataset, loader

    def get_features(self, x, n_epochs=-1):
        if n_epochs == -1:
            t = 1.0
        else:
            t = float(self.t0 * np.power(self.t1 / self.t0, (self.n_epochs + 1) / n_epochs))

        a1, a2 = self.aug((x, t))
        out1 = self._net(a1)
        out2 = self._net(a2)
        return out1, out2

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=True,
            supervised_meta=True, beta=1.0, valid_dataset=None, miverbose=None, 
            split_number=8, meta_epoch=5, meta_beta=1.0, train_labels=None):
        
        assert train_data.ndim == 3
        
        train_data, train_dataset, train_loader = self.get_dataloader(train_data, shuffle=True, drop_last=True)
        
        cls_optimizer = None
        
        if not supervised_meta:
            train_labels = TensorDataset(torch.arange(train_data.shape[0]).to(torch.long).to(self.device))
            cls_optimizer = torch.optim.AdamW(self.unsup_pred.parameters(), lr=self.cls_lr)
        else:
            train_labels = TensorDataset(torch.from_numpy(train_labels).to(torch.long).to(self.device))
            cls_optimizer = torch.optim.AdamW(self.pred.parameters(), lr=self.cls_lr)

        train_data_label = []
        for i in range(len(train_dataset)):
            train_data_label.append([train_dataset[i], train_labels[i]])

        train_data_label_loader = DataLoader(
            train_data_label, 
            batch_size=min(self.batch_size, len(train_dataset)), 
            shuffle=True, drop_last=True,
            pin_memory=False,  # Disable pin_memory to reduce overhead
            num_workers=0      # Disable multiprocessing for now
        )

        meta_p = self.aug.parameters()
        meta_optimizer = torch.optim.AdamW(meta_p, lr=self.meta_lr)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)

        loss_log = []

        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            if (self.n_epochs + 1) % meta_epoch == 0:
                self.meta_fit(train_data_label_loader, meta_optimizer, meta_beta, 
                            supervised_meta, cls_optimizer)

            cum_loss = 0
            n_epoch_iters = 0
            interrupted = False
            
            self._net.train()
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    print(f"We slice max train length!")
                    print(x.size())
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                optimizer.zero_grad()
                meta_optimizer.zero_grad()

                out1, out2 = self.get_features(x, n_epochs=n_epochs)
                loss = global_infoNCE(out1, out2) + local_infoNCE(out1, out2, k=split_number) * beta

                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1

            self.n_epochs += 1

            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
                mlflow.log_metric("train_loss", cum_loss, step=self.n_epochs)

        return loss_log

    def meta_fit(self, train_loader, meta_optimizer, meta_beta, supervised_meta, cls_optimizer):
        pre_flag = self._net.training
        self._net.eval()
        
        for batch in train_loader:
            x = batch[0][0]
            y = batch[1][0]

            if self.max_train_length is not None and x.size(1) > self.max_train_length:
                window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                x = x[:, window_offset: window_offset + self.max_train_length]
            x = x.to(self.device)
            
            if supervised_meta:
                y = y.to(self.device)
            else:
                y = torch.arange(self.batch_size, dtype=torch.int64).to(self.device)

            meta_optimizer.zero_grad()
            outv, outx = self.get_features(x)
            MI_vx_loss = global_infoNCE(outv, outx)

            zv = F.max_pool1d(outv.transpose(1, 2).contiguous(), kernel_size=outv.size(1)).transpose(1, 2)
            zx = F.max_pool1d(outx.transpose(1, 2).contiguous(), kernel_size=outx.size(1)).transpose(1, 2)

            if supervised_meta:
                pred_yv = torch.squeeze(self.pred(zv), 1)
                pred_yx = torch.squeeze(self.pred(zx), 1)
            else:
                pred_yv = torch.squeeze(self.unsup_pred(zv), 1)
                pred_yx = torch.squeeze(self.unsup_pred(zx), 1)

            MI_vy_loss = self.CE(pred_yv, y)
            MI_xy_loss = self.CE(pred_yx, y)
            
            vx_vy_loss = meta_beta * (MI_vy_loss + MI_xy_loss)
            
            vx_vy_loss.backward()
            meta_optimizer.step()
            cls_optimizer.step()

        if pre_flag:
            self._net.train()

    def encode(self, data, mask=None, batch_size=None):
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                out = self.net(x.to(self.device, non_blocking=True), mask)
                out = F.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1)).transpose(1, 2).cpu()
                out = out.squeeze(1)
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output.numpy()

    def save(self, fn):
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)

# --------------------------------------------------------
# Utility functions for InfoTS integration
# --------------------------------------------------------
def build_infots_fingerprint(cfg: Dict[str, Any]) -> Dict[str, str]:
    keys = (
        "model_name", "seed",
        "infots_epochs", "infots_output_dims", "infots_hidden_dims", "infots_depth",
        "infots_max_train_length", "infots_lr", "infots_meta_lr",
        "infots_aug_p1", "infots_aug_p2", "infots_dropout"
    )
    return {k: str(cfg[k]) for k in keys}

def search_encoder_fp(
    fp: Dict[str, str],
    experiment_name: str,
    tracking_uri: str,
) -> str | None:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None

    clauses = ["attributes.status = 'FINISHED'"]
    clauses += [f"params.{k} = '{v}'" for k, v in fp.items()]
    query = " and ".join(clauses)

    hits = mlflow.search_runs([exp.experiment_id], filter_string=query, max_results=1)
    return None if hits.empty else hits.iloc[0]["run_id"]

def build_linear_loaders(
    X_repr: np.ndarray, y: np.ndarray,
    batch_size: int, device: str, shuffle: bool = True,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X_repr).float(),
        torch.from_numpy(y).float()
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# Classifier classes
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2, num_classes=1):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Threshold finding and training/evaluation functions
def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2,
    average: str = "macro",
    grid: int = 101,
):
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

def train_linear_classifier(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    epochs,
    device,
):
    train_auc_m = BinaryAUROC()
    train_pr_m  = BinaryAveragePrecision()
    val_auc_m   = BinaryAUROC()
    val_pr_m    = BinaryAveragePrecision()

    best_threshold = 0.5
    best_val_f1_overall = -1.0

    for epoch in range(1, epochs + 1):
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