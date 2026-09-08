# models/multiscale_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDecoder(nn.Module):
    """
    多尺度特征融合解码器
    get_features() 返回解码特征，不做最终分类
    分类由外部的SceneAwareClassifier完成
    """

    def __init__(
        self,
        num_classes: int,
        aspp_channels: int = 256,
        low1_channels: int = 256,
        low2_channels: int = 512,
        decoder_channels: int = 128,
    ):
        super().__init__()

        self.proj1 = nn.Sequential(
            nn.Conv2d(low1_channels, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(low2_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        fused_channels = aspp_channels + 48 + 24

        self.decode = nn.Sequential(
            nn.Conv2d(fused_channels, decoder_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(decoder_channels, decoder_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        # 保留一个默认分类头（给基线模型用，改进模型用SceneAwareClassifier）
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def get_features(
        self,
        aspp_feat: torch.Tensor,
        low1: torch.Tensor,
        low2: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回解码后特征，不做分类
        Returns: [B, decoder_channels, H/4, W/4]
        """
        low1 = self.proj1(low1)
        low2 = self.proj2(low2)

        low2 = F.interpolate(low2, size=low1.shape[2:],
                             mode='bilinear', align_corners=False)
        aspp = F.interpolate(aspp_feat, size=low1.shape[2:],
                             mode='bilinear', align_corners=False)

        fused = torch.cat([aspp, low2, low1], dim=1)
        return self.decode(fused)

    def forward(
        self,
        aspp_feat: torch.Tensor,
        low1: torch.Tensor,
        low2: torch.Tensor,
        output_size: tuple,
    ) -> torch.Tensor:
        """完整前向（特征提取+默认分类头），给基线模型用"""
        feat = self.get_features(aspp_feat, low1, low2)
        out = self.classifier(feat)
        return F.interpolate(out, size=output_size,
                             mode='bilinear', align_corners=False)