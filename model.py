"""
model.py
========
DVHnet architecture: a ResNet-style 2D CNN that consumes a [B, 2, H, W]
(target mask, OAR mask) input and regresses a 256-length cumulative DVH
vector in [0, 1] via a sigmoid output head.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv-BN-ReLU, optionally downsampling by stride 2."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class DVHNet(nn.Module):
    """
    Input:  [B, 2, H, W]   (channel 0 = target/PTV mask, channel 1 = OAR mask)
    Output: [B, num_bins]  cumulative DVH, sigmoid-bounded to [0, 1]
    """
    def __init__(self, in_channels: int = 2, num_bins: int = 256,
                 base_channels: int = 32, fc_dims=(1024, 512, 256)):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # H/4
        )
        # Progressive downsampling: H/4 -> H/8 -> H/16 -> H/32
        self.stage1 = ConvBlock(base_channels, base_channels * 2, stride=2)      # H/8
        self.stage2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)  # H/16
        self.stage3 = ConvBlock(base_channels * 4, base_channels * 8, stride=2)  # H/32
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling bottleneck
        feat_dim = base_channels * 8
        layers = []
        prev = feat_dim
        for dim in fc_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU(inplace=True), nn.Dropout(0.2)]
            prev = dim
        self.head = nn.Sequential(*layers)
        self.out = nn.Linear(prev, num_bins)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)          # [B, feat_dim]
        x = self.head(x)                    # [B, fc_dims[-1]]
        x = self.out(x)                     # [B, num_bins]
        return self.sigmoid(x)              # cumulative DVH in [0, 1]


def enforce_monotonic(dvh: torch.Tensor) -> torch.Tensor:
    """
    Post-processing fallback: force a non-increasing curve via a running
    cumulative-minimum along the dose axis. Use at inference time if the
    monotonicity loss term hasn't fully eliminated small violations.
    """
    return torch.cummin(dvh, dim=-1).values
