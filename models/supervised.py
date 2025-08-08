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

# -------------------------
# Decoder Model Classes
# -------------------------
# liniar classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x is expected to be of shape (batch, feature_dim)
        return self.fc(x)

# Multi-layer perceptron (MLP) classifier
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


# from: Sarkar, P., Etemad, A.: Self-Supervised ECG Representation Learning for Emotion Recognition. 
# IEEE Transactions on Affective Computing \textbf{13}(3), 1541--1554 (2020). \doi{10.1109/TAFFC.2020.3014842}
class EmotionRecognitionCNN(nn.Module):
    def __init__(self):
        super(EmotionRecognitionCNN, self).__init__()
        self.bn_input = nn.BatchNorm1d(1)
        # Conv block 1
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=32, padding='same')
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=32, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=2)

        # Conv block 2
        self.conv2_1 = nn.Conv1d(32, 64, kernel_size=16, padding='same')
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=16, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=2)

        # Conv block 3
        self.conv3_1 = nn.Conv1d(64, 128, kernel_size=8, padding='same')
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size=8, padding='same')
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Dense layers
        self.fc1 = nn.Linear(128, 512)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # Conv block 1
        x = self.bn_input(x)
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        # Conv block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        # Conv block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.global_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x

# Adapted from: Sarkar, P., Etemad, A.: Self-Supervised ECG Representation Learning for Emotion Recognition. 
# IEEE Transactions on Affective Computing \textbf{13}(3), 1541--1554 (2020). \doi{10.1109/TAFFC.2020.3014842}
class Improved1DCNN_v2(nn.Module):
    """
    A more complex 1D CNN model with 3 convolutional blocks with 
    3-layer classification head with dropout.
    """
    def __init__(self, dropout=None):
        super(Improved1DCNN_v2, self).__init__()
        self.bn_input = nn.BatchNorm1d(1)
        # Block 1
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=5, padding=2, bias=False)
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.1) if dropout is None else nn.Dropout(p=dropout)
        # Block 2
        self.conv2_1 = nn.Conv1d(32, 64, kernel_size=11, padding=5, bias=False)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=11, padding=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.1) if dropout is None else nn.Dropout(p=dropout)
        # Block 3
        self.conv3_1 = nn.Conv1d(64, 128, kernel_size=17, padding=8, bias=False)
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size=17, padding=8)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # Dense layers
        self.fc1 = nn.Linear(128, 128)
        self.dropout3 = nn.Dropout(0.3) if dropout is None else nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.3) if dropout is None else nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Input x: (batch, channels, length)
        x = self.bn_input(x)
        # Block 1
        x = F.gelu(self.conv1_1(x))
        x = F.gelu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        # Block 2
        x = F.gelu(self.conv2_1(x))
        x = F.gelu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        # Block 3
        x = F.gelu(self.conv3_1(x))
        x = F.gelu(self.conv3_2(x))
        x = self.gap(x)  # shape: (batch, 128, 1)
        x = x.view(x.size(0), -1)  # flatten to (batch, 128)
        # Dense layers
        x = F.gelu(self.fc1(x))
        x = self.dropout3(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x
    
# ----------------------
# Transformer Model Class
# ----------------------
# Positional Encoding module (Vaswani et al., 2017)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model: embedding dimension.
            dropout: dropout rate.
            max_len: maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on position and dimension
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            x after adding positional encodings and applying dropout.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# inspired from: Behinaein, B., Bhatti, A., Rodenburg, D., Hungler, P., Etemad, A.: 
# A Transformer Architecture for Stress Detection from ECG. 
# In: International Symposium on Wearable Computers, pp. 1--6 (2021). \doi{10.1145/3460421.3480427}
# Transformer-based classifier for ECG stress detection
class TransformerECGClassifier(nn.Module):
    def __init__(self, dropout=0.4, input_length=10000):
        """
        Args:
            input_length: Length of the input ECG signal.
                         For example, a 10 second window at 1000Hz gives 10000 samples.
        """
        super(TransformerECGClassifier, self).__init__()
        # Convolutional front-end subnetwork
        # For conv1: kernel_size=64, stride=8.
        # Use asymmetric padding: left=31, right=32.
        self.pad_conv1 = nn.ConstantPad1d((31, 32), 0)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=64, stride=8, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # For conv2: kernel_size=32, stride=4.
        # Use asymmetric padding: left=15, right=16.
        self.pad_conv2 = nn.ConstantPad1d((15, 16), 0)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=32, stride=4, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # For input_length=10000:
        # After conv1: output length = ceil(10000/8) = 1250
        # After pool1: output length = floor(1250/2) = 625
        # After conv2: output length = ceil(625/4) = 157
        # After pool2: output length = floor(157/2) = 78
        self.T = 74  # Final sequence length
        
        # Linear projection from conv output (128 channels) to transformer model dimension (1024)
        self.fc_embed = nn.Linear(128, 1024)
        
        # Positional encoder to inject order information into the embeddings
        self.pos_encoder = PositionalEncoding(d_model=1024, dropout=0.1, max_len=self.T)
        
        # Transformer encoder: 4 layers, with model dimension 1024, 4 attention heads, feed-forward dim 512, dropout 0.5
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4, dim_feedforward=512, dropout=0.4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Fully connected (FC) subnetwork for classification
        # Flattened transformer output has dimension T * 1024 (74 * 1024)
        self.fc1 = nn.Linear(self.T * 1024, 512)
        self.dropout_fc1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout_fc2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        """
        Args:
            x: Input ECG signal tensor of shape (batch, 1, input_length)
        Returns:
            Output probabilities of shape (batch, 1) via sigmoid activation.
        """
        # Convolutional front-end
        x = self.conv1(x)      # -> (batch, 64, L1)
        x = self.relu1(x)
        x = self.pool1(x)      # -> (batch, 64, L1/2)
        
        x = self.conv2(x)      # -> (batch, 128, L2)
        x = self.relu2(x)
        x = self.pool2(x)      # -> (batch, 128, T) where T ≈ self.T
        
        # Permute to (batch, T, channels)
        x = x.permute(0, 2, 1)  # -> (batch, T, 128)
        
        # Project to transformer model dimension (1024)
        x = self.fc_embed(x)    # -> (batch, T, 1024)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # -> (batch, T, 1024)
        
        # Transformer expects input shape (seq_len, batch, d_model)
        x = x.permute(1, 0, 2)   # -> (T, batch, 1024)
        x = self.transformer_encoder(x)  # -> (T, batch, 1024)
        x = x.permute(1, 0, 2)   # -> (batch, T, 1024)
        
        # Flatten transformer output
        x = x.flatten(1)       # -> (batch, T * 1024)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        return x

# adapted from: Ingolfsson, T.M., Wang, X., Hersche, M., Burrello, A., Cavigelli, L., Benini, L.: 
# ECG-TCN: Wearable Cardiac Arrhythmia Detection with a Temporal Convolutional Network. 
# arXiv preprint arXiv:2103.13740 (2021). \url{https://arxiv.org/abs/2103.13740}
# ----------------------
# TCN Model Class
# ----------------------
class TCNClassifier(nn.Module):
    def __init__(self, input_length=10000, n_inputs=1, Kt=11, dropout=0.3, Ft=11):
        super(TCNClassifier, self).__init__()
        # Initial 1x1 convolution to expand channels
        self.pad0 = nn.ConstantPad1d(padding=(Kt-1, 0), value=0)
        self.conv0 = nn.Conv1d(in_channels=n_inputs, out_channels=n_inputs + 1, kernel_size=Kt, bias=False)
        self.batchnorm0 = nn.BatchNorm1d(num_features=n_inputs + 1)
        self.act0 = nn.ReLU()
        
        # First residual block (dilation = 1)
        dilation = 1
        self.upsample = nn.Conv1d(in_channels=n_inputs + 1, out_channels=Ft, kernel_size=1, bias=False)
        self.upsamplebn = nn.BatchNorm1d(num_features=Ft)
        self.upsamplerelu = nn.ReLU()
        self.pad1 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv1 = nn.Conv1d(in_channels=n_inputs + 1, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features=Ft)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.pad2 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv2 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(num_features=Ft)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.reluadd1 = nn.ReLU()
        
        # Second residual block (dilation = 2)
        dilation = 2
        self.pad3 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv3 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(num_features=Ft)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)
        self.pad4 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv4 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm4 = nn.BatchNorm1d(num_features=Ft)
        self.act4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=dropout)
        self.reluadd2 = nn.ReLU()
        
        # Third residual block (dilation = 4)
        dilation = 4
        self.pad5 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv5 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm5 = nn.BatchNorm1d(num_features=Ft)
        self.act5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=pt)
        self.pad6 = nn.ConstantPad1d(padding=((Kt-1) * dilation, 0), value=0)
        self.conv6 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm6 = nn.BatchNorm1d(num_features=Ft)
        self.act6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=dropout)
        self.reluadd3 = nn.ReLU()
        
        # Final linear layer: flattened feature map has shape Ft * input_length
        flattened_size = Ft * input_length  
        self.linear = nn.Linear(in_features=flattened_size, out_features=1, bias=False)
        
    def forward(self, x):
        # Input shape: (batch, channels, sequence_length)
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.act0(x)
        
        # First residual block
        res = self.pad1(x)
        res = self.conv1(res)
        res = self.batchnorm1(res)
        res = self.act1(res)
        res = self.dropout1(res)
        res = self.pad2(res)
        res = self.conv2(res)
        res = self.batchnorm2(res)
        res = self.act2(res)
        res = self.dropout2(res)
        
        x = self.upsample(x)
        x = self.upsamplebn(x)
        x = self.upsamplerelu(x)
        x = x + res
        x = self.reluadd1(x)
        
        # Second residual block
        res = self.pad3(x)
        res = self.conv3(res)
        res = self.batchnorm3(res)
        res = self.act3(res)
        res = self.dropout3(res)
        res = self.pad4(res)
        res = self.conv4(res)
        res = self.batchnorm4(res)
        res = self.act4(res)
        res = self.dropout4(res)
        x = x + res
        x = self.reluadd2(x)
        
        # Third residual block
        res = self.pad5(x)
        res = self.conv5(res)
        res = self.batchnorm5(res)
        res = self.act5(res)
        res = self.dropout5(res)
        res = self.pad6(res)
        res = self.conv6(res)
        res = self.batchnorm6(res)
        res = self.act6(res)
        res = self.dropout6(res)
        x = x + res
        x = self.reluadd3(x)
        
        # Flatten and classify
        x = x.flatten(1)
        x = self.linear(x)
        return x
    

class Bottleneck1D(nn.Module):
    """
    A 1D version of the ResNet bottleneck block.
    This block uses a 1x1 conv to reduce channels, a 3x3 conv for processing,
    and a final 1x1 conv to expand channels. If needed, a downsample layer is used.
    """
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

# ----------------------
# XResNet1D Model Class
# ----------------------
class XResNet1D(nn.Module):
    """
    An XResNet1D architecture for 1D signals.
    It uses a modified stem (three 1D convolutional layers with batch norm and ReLU)
    followed by 4 stages of bottleneck blocks. For ResNet101, the layer configuration is [3, 4, 23, 3].
    """
    def __init__(self, block, layers, num_classes=1, in_channels=1):
        super(XResNet1D, self).__init__()
        self.inplanes = 64

        # Stem: Adapted from fastai's xresnet stem but for 1D input.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual layers (each layer downsamples and increases feature channels)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and final classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x is expected to be of shape (batch_size, channels, sequence_length)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # shape: (batch_size, features, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)  # For binary classification
        return x

def xresnet1d101(num_classes=1, in_channels=1):
    """
    Constructs an xresnet1d101 model.
    
    For ResNet101, the block configuration is [3, 4, 23, 3] using Bottleneck1D blocks.
    """
    return XResNet1D(Bottleneck1D, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels)


class DilatedCNN(nn.Module):
    """
    Dilated CNN for ECG-based stress detection as specified in the architecture.
    Architecture based on: https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/wss2.70004

    Architecture:
    - Input: (batch_size, 1, 640) - 1D ECG signal
    - 8 Dilated Conv1D layers with increasing filters and dilation rates
    - Batch normalization and dropout after each conv layer
    - Global max pooling
    - Dense output layer with sigmoid activation
    """

    def __init__(self):
        super(DilatedCNN, self).__init__()

        # Architecture parameters
        self.num_filters = [16, 32, 64, 96, 128, 256, 320, 512]
        self.kernel_size = 8
        self.dilation_rates = [1, 2, 4, 8, 16, 32, 64, 128]
        self.dropout_rate = 0.5
        self.num_classes = 1 #Binary classification

        # Build the convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # Input has 1 channel (single ECG signal)
        in_channels = 1

        for i in range(8):  # 8 convolutional layers
            # Dilated 1D Convolution
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.num_filters[i],
                kernel_size=self.kernel_size,
                dilation=self.dilation_rates[i],
                padding='same'  # Keep same length
            )

            # Batch Normalization
            bn = nn.BatchNorm1d(self.num_filters[i])

            # Dropout
            dropout = nn.Dropout(self.dropout_rate)

            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
            self.dropout_layers.append(dropout)

            # Update input channels for next layer
            in_channels = self.num_filters[i]

        # ReLU activation
        self.relu = nn.ReLU()

        # Global Max Pooling 1D
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Dense output layer
        self.output_layer = nn.Linear(self.num_filters[-1], self.num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, 640)

        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        # Pass through all 8 convolutional blocks
        for i in range(8):
            # Conv1D -> BatchNorm -> ReLU -> Dropout
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.relu(x)
            x = self.dropout_layers[i](x)

        # Global Max Pooling: (batch_size, 512, seq_len) -> (batch_size, 512, 1)
        x = self.global_max_pool(x)

        # Flatten: (batch_size, 512, 1) -> (batch_size, 512)
        x = x.squeeze(-1)

        # Dense output layer - raw logits for BCEWithLogitsLoss
        x = self.output_layer(x)

        return x