# models/modules/background_suppression.py
"""
背景抑制模块 (Background Suppression Module, BSM)

针对高分辨率遥感图像复杂背景干扰的问题：
1. 背景判别器：轻量卷积MLP，学习每个像素属于"背景"的概率
2. 自适应特征抑制：feature = feature * (1 - alpha * bg_prob)
3. 辅助监督：用分割mask的背景通道提供二值监督信号

"背景"在LoveDA中定义为类别0（Background）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BackgroundDiscriminator(nn.Module):
    """
    轻量背景判别器

    输入特征图 [B, C, H, W]
    输出背景概率图 [B, 1, H, W]，每像素为属于背景的概率 in [0, 1]

    结构：
      Conv(C→hidden) → BN → ReLU → Conv(hidden→hidden//2) → BN → ReLU
      → Conv(hidden//2→1) → Sigmoid
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        mid = hidden_channels // 2
        self.discriminator = nn.Sequential(
            # 第一层：通道压缩
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            # 第二层：进一步压缩
            nn.Conv2d(hidden_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            # 输出层：预测背景概率
            nn.Conv2d(mid, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns bg_prob: [B, 1, H, W]"""
        return self.discriminator(x)


class BackgroundSuppressionModule(nn.Module):
    """
    背景抑制模块 (BSM)

    训练时：
        1. 判别器预测背景概率 bg_prob
        2. 用背景概率抑制特征：suppressed = feat * (1 - alpha * bg_prob)
        3. 返回抑制后特征 + 背景概率图（用于计算辅助损失）

    推理时：
        与训练相同，不计算辅助损失

    Args:
        in_channels:     输入特征通道数
        hidden_channels: 判别器隐藏通道数
        alpha:           抑制强度，越大背景特征被压制越强
        learnable_alpha: 若True，alpha为可学习参数
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        alpha: float = 0.5,
        learnable_alpha: bool = True,
    ):
        super().__init__()
        self.discriminator = BackgroundDiscriminator(in_channels, hidden_channels)

        if learnable_alpha:
            # 让模型自己学习最优抑制强度，初始化为alpha
            self._alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("_alpha", torch.tensor(alpha))
        self.learnable_alpha = learnable_alpha

    @property
    def alpha(self):
        # 保证alpha在[0, 1]范围内
        return torch.sigmoid(self._alpha) if self.learnable_alpha else self._alpha

    def forward(self, feat: torch.Tensor):
        """
        Args:
            feat: [B, C, H, W] 输入特征

        Returns:
            suppressed_feat: [B, C, H, W] 背景抑制后的特征
            bg_prob:         [B, 1, H, W] 背景概率图（用于辅助损失计算）
        """
        # 预测背景概率
        bg_prob = self.discriminator(feat)  # [B, 1, H, W]

        # 自适应特征抑制：前景区域基本不变，背景区域特征被削弱
        suppression_mask = 1.0 - self.alpha * bg_prob  # [B, 1, H, W], in [1-alpha, 1]
        suppressed_feat = feat * suppression_mask       # broadcast over channels

        return suppressed_feat, bg_prob


class BSMLoss(nn.Module):
    """
    背景抑制模块的辅助损失

    将分割mask中的类别0（背景）作为正样本，其余类别作为负样本，
    训练背景判别器学习准确的背景概率分布

    Args:
        pos_weight: 正样本（背景）的权重，可调整类别不平衡
    """

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        bg_prob: torch.Tensor,
        mask: torch.Tensor,
        ignore_index: int = 255,
    ) -> torch.Tensor:
        """
        Args:
            bg_prob:  [B, 1, H, W] 背景概率图（0-1）
            mask:     [B, H, W]   分割真值（0-indexed，ignore=255）
            ignore_index: 忽略像素值
    
        Returns:
            loss: 标量
        """
        B, _, H, W = bg_prob.shape
    
        # 如果分辨率不匹配，上采样bg_prob到mask分辨率
        mH, mW = mask.shape[1], mask.shape[2]
        if H != mH or W != mW:
            bg_prob = F.interpolate(bg_prob, size=(mH, mW), mode="bilinear", align_corners=False)
    
        # 构造二值背景标签：类别0=1（背景），其余=0（前景）
        bg_target = (mask == 0).float()  # [B, H, W]
        
        # 添加channel维度，使其与bg_logits维度匹配
        bg_target = bg_target.unsqueeze(1)  # [B, 1, H, W]
    
        # 有效像素mask（排除ignore）
        valid = (mask != ignore_index).float()  # [B, H, W]
        valid = valid.unsqueeze(1)  # [B, 1, H, W]
    
        # 将概率转换为logits（对数几率）
        bg_logits = torch.logit(bg_prob, eps=1e-6)  # [B, 1, H, W]
    
        # 使用 with_logits 版本
        bce = F.binary_cross_entropy_with_logits(
            bg_logits,   # [B, 1, H, W]
            bg_target,   # [B, 1, H, W]
            reduction="none",
        )  # [B, 1, H, W]
    
        # 加权：背景像素权重为pos_weight
        weight = torch.ones_like(bg_target)
        weight[bg_target == 1] = self.pos_weight
    
        loss = (bce * weight * valid).sum() / (valid.sum() + 1e-6)
        return loss
