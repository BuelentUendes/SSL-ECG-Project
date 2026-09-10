"""
ResNet1D backbone from WildECG (https://github.com/klean2050/tiles_ecg_model).
Adapted from Shenda Hong's resnet1d (https://github.com/hsd1503/resnet1d).

Copied verbatim from src/models/resnet1d.py to avoid the pytorch_lightning
dependency that the original package requires.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
        )

    def forward(self, x):
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        x = F.pad(x, (p // 2, p - p // 2), "constant", 0)
        return self.conv(x)


class MyMaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x):
        in_dim = x.shape[-1]
        p = max(0, (in_dim - 1) * self.stride + self.kernel_size - in_dim)
        x = F.pad(x, (p // 2, p - p // 2), "constant", 0)
        return self.max_pool(x)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        downsample,
        use_bn,
        use_do,
        is_first_block=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if downsample else 1
        self.groups = groups
        self.downsample = downsample
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=groups,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=groups,
        )

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.max_pool(identity)

        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        return out + identity


class ResNet1D(nn.Module):
    """
    1-D ResNet encoder used by WildECG.

    Input:  (B, in_channels, L)
    Output: (B, output_size)  — global-average-pooled feature vector

    With the pretrained config (base_filters=16, n_block=10) output_size = 256.
    """

    def __init__(
        self,
        in_channels,
        base_filters,
        kernel_size,
        stride,
        groups,
        n_block,
        n_classes=None,       # accepted for API compatibility; not used
        downsample_gap=2,
        increasefilter_gap=2,
        use_bn=True,
        use_do=True,
        verbose=False,
    ):
        super().__init__()
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.downsample_gap = downsample_gap
        self.increasefilter_gap = increasefilter_gap

        self.first_block_conv = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=kernel_size,
            stride=1,
        )
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        self.basicblock_list = nn.ModuleList()
        for i_block in range(n_block):
            is_first_block = i_block == 0
            downsample = (i_block % downsample_gap == 1)
            if is_first_block:
                in_ch = base_filters
                out_ch = in_ch
            else:
                in_ch = int(base_filters * 2 ** ((i_block - 1) // increasefilter_gap))
                if (i_block % increasefilter_gap == 0) and (i_block != 0):
                    out_ch = in_ch * 2
                else:
                    out_ch = in_ch
            self.basicblock_list.append(BasicBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups,
                downsample=downsample,
                use_bn=use_bn,
                use_do=use_do,
                is_first_block=is_first_block,
            ))
            out_channels = out_ch

        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.output_size = out_channels

    def forward(self, x):
        out = x.float()
        out = self.first_block_conv(out)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        for block in self.basicblock_list:
            out = block(out)
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        return out.mean(-1)  # global average pool → (B, output_size)