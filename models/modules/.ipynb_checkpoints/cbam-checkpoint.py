# models/modules/cbam.py
"""
CBAM: Convolutional Block Attention Module
通道注意力 + 空间注意力，接在ASPP输出后
针对遥感图像复杂背景，增强前景地物特征表达
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(B, C))  # [B, C]
        max_ = self.fc(self.max_pool(x).view(B, C)) # [B, C]
        attn = self.sigmoid(avg + max_).view(B, C, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)        # [B, 1, H, W]
        max_ = x.max(dim=1, keepdim=True)[0]     # [B, 1, H, W]
        attn = self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """
    CBAM模块
    先通道注意力，再空间注意力，残差连接保留原始特征
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attn(x)
        out = self.spatial_attn(out)
        return out + x   # 残差连接