# models/modules/boundary_loss.py
"""
边界感知加权损失
针对LoveDA地物边界模糊问题：
对边界像素赋予更高权重，引导模型精细化边界分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_boundary_mask(mask: torch.Tensor,
                      ignore_index: int = 255,
                      dilation: int = 3) -> torch.Tensor:
    """
    生成边界像素的bool mask

    原理：对mask做最大池化，若池化结果与原始值不同，
    说明该像素周围存在不同类别，即为边界区域

    Args:
        mask:   [B, H, W] 分割真值
        dilation: 边界宽度（像素）

    Returns:
        boundary: [B, H, W] bool，True表示边界像素
    """
    valid = (mask != ignore_index)           # [B, H, W]

    # 临时把ignore区域填0，避免影响池化结果
    mask_f = mask.float()
    mask_f[~valid] = 0.0
    mask_f = mask_f.unsqueeze(1)             # [B, 1, H, W]

    # 最大池化：边界处周围有不同类别，池化结果≠原始值
    kernel = dilation * 2 + 1
    pooled = F.max_pool2d(mask_f, kernel_size=kernel,
                          stride=1, padding=dilation)  # [B, 1, H, W]

    boundary = (pooled.squeeze(1) != mask) & valid     # [B, H, W]
    return boundary


class BoundaryAwareLoss(nn.Module):
    """
    边界感知加权交叉熵损失

    loss = mean( CE(pred, target) * weight_map )
    其中边界像素 weight=boundary_weight，其余=1.0

    Args:
        ignore_index:     忽略像素值
        boundary_weight:  边界像素权重（>1时加强边界监督）
        dilation:         边界宽度
    """

    def __init__(self,
                 ignore_index: int = 255,
                 boundary_weight: float = 2.0,
                 dilation: int = 3):
        super().__init__()
        self.ignore_index = ignore_index
        self.boundary_weight = boundary_weight
        self.dilation = dilation
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction='none',
        )

    def forward(self,
                logits: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W]
            mask:   [B, H, W]
        Returns:
            loss: 标量
        """
        # 逐像素CE损失
        pixel_loss = self.ce(logits, mask)   # [B, H, W]

        # 生成边界权重图
        boundary = get_boundary_mask(mask, self.ignore_index, self.dilation)
        weight = torch.ones_like(mask, dtype=torch.float)
        weight[boundary] = self.boundary_weight
        weight[mask == self.ignore_index] = 0.0

        # 加权平均
        loss = (pixel_loss * weight).sum() / (weight.sum() + 1e-6)
        return loss