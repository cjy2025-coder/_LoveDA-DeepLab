# models/decoder.py
"""
DeepLabV3+ 解码器

将高层语义特征（来自SA-ASPP）与低层细节特征（来自backbone layer1）融合，
恢复空间细节，输出精细的分割预测。

结构（参考 DeepLabV3+）：
  低层特征 [B, C_low, H/4, W/4] → 1×1卷积投影 → [B, 48, H/4, W/4]
  高层特征 [B, 256, H/OS, W/OS] → 双线性上采样到H/4 → [B, 256, H/4, W/4]
  concat → [B, 304, H/4, W/4]
  → 3×3 Conv → BN → ReLU（×2）
  → 1×1 Conv → 类别预测 [B, num_classes, H/4, W/4]
  → 双线性上采样到原始尺寸 [B, num_classes, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabDecoder(nn.Module):
    """
    DeepLabV3+ 风格解码器

    Args:
        num_classes:          分类数量
        high_level_channels:  ASPP输出通道数（高层特征）
        low_level_channels:   骨干layer1输出通道数（低层特征）
        decoder_channels:     解码器内部通道数
        low_level_out:        低层特征投影后通道数
    """

    def __init__(
        self,
        num_classes: int,
        high_level_channels: int = 256,
        low_level_channels: int = 256,
        decoder_channels: int = 256,
        low_level_out: int = 48,
    ):
        super().__init__()

        # 低层特征投影（通道压缩）
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_out),
            nn.ReLU(inplace=True),
        )

        # 融合后的特征处理
        fused_channels = high_level_channels + low_level_out
        self.decode_conv = nn.Sequential(
            nn.Conv2d(fused_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        # 最终分类头
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    
    def forward(
        self,
        high_feat,
        low_feat,
        output_size,
    ):
        decoded = self.get_features(
            high_feat,
            low_feat
        )
    
        logits = self.classifier(decoded)
    
        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
    
        return logits
    """Old Version for baseline"""
    # def forward(
    #     self,
    #     high_feat: torch.Tensor,
    #     low_feat: torch.Tensor,
    #     output_size: tuple,
    # ) -> torch.Tensor:
    #     """
    #     Args:
    #         high_feat:   [B, C_high, H/OS, W/OS]  来自SA-ASPP的高层特征
    #         low_feat:    [B, C_low, H/4, W/4]      来自backbone layer1的低层特征
    #         output_size: (H, W) 最终输出尺寸（原图大小）

    #     Returns:
    #         logits: [B, num_classes, H, W]
    #     """
    #     # 投影低层特征
    #     low_feat = self.low_level_proj(low_feat)              # [B, 48, H/4, W/4]

    #     # 上采样高层特征到低层特征分辨率
    #     high_feat = F.interpolate(
    #         high_feat,
    #         size=low_feat.shape[2:],
    #         mode="bilinear",
    #         align_corners=False,
    #     )                                                      # [B, 256, H/4, W/4]

    #     # 融合
    #     fused = torch.cat([high_feat, low_feat], dim=1)       # [B, 304, H/4, W/4]

    #     # 解码
    #     decoded = self.decode_conv(fused)                     # [B, 256, H/4, W/4]

    #     # 分类预测
    #     logits = self.classifier(decoded)                     # [B, num_classes, H/4, W/4]

    #     # 上采样到原始尺寸
    #     logits = F.interpolate(
    #         logits,
    #         size=output_size,
    #         mode="bilinear",
    #         align_corners=False,
    #     )                                                      # [B, num_classes, H, W]

    #     return logits
    """New Func for new net"""
    def get_features(
        self,
        high_feat,
        low_feat,
    ):
        low_feat = self.low_level_proj(low_feat)
    
        high_feat = F.interpolate(
            high_feat,
            size=low_feat.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
    
        fused = torch.cat(
            [high_feat, low_feat],
            dim=1
        )
    
        decoded = self.decode_conv(fused)
    
        return decoded


class AuxiliaryHead(nn.Module):
    """
    辅助分割头（接在backbone layer3输出上）
    用于提供额外的梯度信号，加速训练收敛
    """

    def __init__(self, in_channels: int, num_classes: int, hidden: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(hidden, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, output_size: tuple) -> torch.Tensor:
        out = self.head(x)
        return F.interpolate(out, size=output_size, mode="bilinear", align_corners=False)
