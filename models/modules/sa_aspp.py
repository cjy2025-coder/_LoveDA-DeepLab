# models/modules/sa_aspp.py
"""
尺度自适应 ASPP (Scale-Adaptive ASPP, SA-ASPP)

针对 LoveDA 城乡场景物体尺度差异大的问题：
- 多分支空洞卷积（膨胀率 1, 6, 12, 18）覆盖大范围感受野
- 尺度选择注意力机制（SE-like）：动态加权各分支，使模型
  能够根据输入内容自适应调整感受野组合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPBranch(nn.Module):
    """单个ASPP分支：Conv → BN → ReLU"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation if kernel_size > 1 else 0
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class GlobalAvgPoolBranch(nn.Module):
    """全局平均池化分支"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        out = self.relu(self.bn(self.conv(self.gap(x))))
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


class ScaleSelectionAttention(nn.Module):
    """
    尺度选择注意力

    改进点：
    1. 双池化 squeeze：同时使用全局平均池化和全局最大池化，
       捕获各分支的平均激活强度 + 峰值响应（空间分布信息）
    2. 合理的 hidden 维度：通过 reduction 控制瓶颈宽度，
       避免过窄瓶颈限制表达能力

    本质是 SE（Squeeze-and-Excitation）的多分支版本：
    1. 对每个分支特征做 GAP + GMP 双池化（squeeze）
    2. 通过两层 MLP 预测各分支权重（excitation）
    3. 加权求和各分支输出
    """

    def __init__(self, num_branches: int, branch_channels: int, reduction: int = 4):
        super().__init__()
        self.num_branches = num_branches
        self.branch_channels = branch_channels

        # 双池化后每个分支提供 2*C 维描述子
        descriptor_dim = num_branches * branch_channels * 2
        hidden = max(descriptor_dim // reduction, 32)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(descriptor_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_branches),
            nn.Sigmoid(),
        )

    def forward(self, branch_outputs: list) -> torch.Tensor:
        """
        Args:
            branch_outputs: list of [B, C, H, W], 长度=num_branches
        Returns:
            weighted_sum: [B, C, H, W]
        """
        B, C, H, W = branch_outputs[0].shape

        # Squeeze：对每个分支做 GAP + GMP，得到 [B, 2*C] 描述子
        descriptors = []
        for f in branch_outputs:
            avg_desc = self.gap(f).view(B, C)
            max_desc = self.gmp(f).view(B, C)
            descriptors.append(torch.cat([avg_desc, max_desc], dim=1))  # [B, 2*C]
        cat = torch.cat(descriptors, dim=1)  # [B, num_branches * 2*C]

        # Excitation：预测各分支权重 [B, num_branches]
        weights = self.fc(cat).unsqueeze(-1).unsqueeze(-1)  # [B, num_branches, 1, 1]

        # 加权求和
        stacked = torch.stack(branch_outputs, dim=1)          # [B, num_branches, C, H, W]
        weights_expanded = weights.unsqueeze(2)               # [B, num_branches, 1, 1, 1]
        out = (stacked * weights_expanded).sum(dim=1)         # [B, C, H, W]
        return out


class ScaleAdaptiveASPP(nn.Module):
    """尺度自适应ASPP：SE-like注意力动态加权各分支 + 残差连接"""

    def __init__(self, in_channels, out_channels, dilations=[1,6,12,18], dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            if d == 1:
                self.branches.append(ASPPBranch(in_channels, out_channels, 1, 1))
            else:
                self.branches.append(ASPPBranch(in_channels, out_channels, 3, d))
        self.gap_branch = GlobalAvgPoolBranch(in_channels, out_channels)

        num_branches = len(dilations) + 1
        # SE-like 尺度选择注意力：根据输入内容动态预测各分支权重
        self.scale_attention = ScaleSelectionAttention(
            num_branches=num_branches,
            branch_channels=out_channels,
            reduction=4,
        )

        # 加权求和后的后处理
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # 各分支计算
        branch_outs = [branch(x) for branch in self.branches] + [self.gap_branch(x)]
        # SE-like 输入自适应加权融合
        fused = self.scale_attention(branch_outs)          # [B, C, H, W]
        out = self.refine(fused)
        out = out + self.residual(x)
        return out






# # models/modules/sa_aspp.py
# """
# 尺度自适应 ASPP (Scale-Adaptive ASPP, SA-ASPP)

# 针对 LoveDA 城乡场景物体尺度差异大的问题：
# - 多分支空洞卷积（膨胀率 1, 6, 12, 18）覆盖大范围感受野
# - 尺度选择注意力机制（SE-like）：动态加权各分支，使模型
#   能够根据输入内容自适应调整感受野组合
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ASPPBranch(nn.Module):
#     """单个ASPP分支：Conv → BN → ReLU"""

#     def __init__(self, in_channels: int, out_channels: int,
#                  kernel_size: int, dilation: int = 1):
#         super().__init__()
#         padding = dilation if kernel_size > 1 else 0
#         self.conv = nn.Conv2d(
#             in_channels, out_channels,
#             kernel_size=kernel_size,
#             padding=padding,
#             dilation=dilation,
#             bias=False,
#         )
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))


# class GlobalAvgPoolBranch(nn.Module):
#     """全局平均池化分支"""

#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         h, w = x.shape[2], x.shape[3]
#         out = self.relu(self.bn(self.conv(self.gap(x))))
#         return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


# class ScaleSelectionAttention(nn.Module):
#     """
#     尺度选择注意力
#     输入：所有分支的concatenation → 通道压缩 → sigmoid → 各分支权重

#     本质是 SE（Squeeze-and-Excitation）的多分支版本：
#     1. 对每个分支特征做全局平均池化（squeeze）
#     2. 通过共享MLP预测各分支权重（excitation）
#     3. 加权求和各分支输出
#     """

#     def __init__(self, num_branches: int, branch_channels: int, reduction: int = 4):
#         super().__init__()
#         self.num_branches = num_branches
#         self.branch_channels = branch_channels
#         # hidden = max(num_branches * branch_channels // reduction, 16)
#         hidden = max(num_branches * 2, 16)
#         # 以每个分支的全局均值向量拼接后映射到权重
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(num_branches * branch_channels, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, num_branches),
#             nn.Sigmoid(),            
#         )

#     def forward(self, branch_outputs: list) -> torch.Tensor:
#         """
#         Args:
#             branch_outputs: list of [B, C, H, W]，长度=num_branches
#         Returns:
#             weighted_sum: [B, C, H, W]
#         """
#         B, C, H, W = branch_outputs[0].shape

#         # Squeeze：对每个分支做全局平均池化，得到 [B, C] 向量
#         squeezed = [self.gap(f).view(B, -1) for f in branch_outputs]  # 每个: [B, C]
#         cat = torch.cat(squeezed, dim=1)                               # [B, num_branches*C]

#         # Excitation：预测各分支权重 [B, num_branches]
#         weights = self.fc(cat).unsqueeze(-1).unsqueeze(-1)             # [B, num_branches, 1, 1]

#         # 加权求和
#         stacked = torch.stack(branch_outputs, dim=1)                   # [B, num_branches, C, H, W]
#         weights_expanded = weights.unsqueeze(2)                        # [B, num_branches, 1, 1, 1]
#         out = (stacked * weights_expanded).sum(dim=1)                  # [B, C, H, W]
#         return out


# class ScaleAdaptiveASPP(nn.Module):
#     def __init__(self, in_channels, out_channels, dilations=[1,6,12,18], dropout=0.1):
#         super().__init__()
#         self.branches = nn.ModuleList()
#         for d in dilations:
#             if d == 1:
#                 self.branches.append(ASPPBranch(in_channels, out_channels, 1, 1))
#             else:
#                 self.branches.append(ASPPBranch(in_channels, out_channels, 3, d))
#         self.gap_branch = GlobalAvgPoolBranch(in_channels, out_channels)
        
#         self.branch_weights = nn.Parameter(torch.ones(len(dilations)+1) / (len(dilations)+1))
        
#         # Concat + 投影
#         self.project = nn.Sequential(
#             nn.Conv2d((len(dilations)+1)*out_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(dropout)
#         )
#         if in_channels != out_channels:
#             self.residual = nn.Conv2d(in_channels, out_channels, 1)
#         else:
#             self.residual = nn.Identity()
    
#     def forward(self, x):
#         branch_outs = [branch(x) for branch in self.branches] + [self.gap_branch(x)]
#         weights = torch.softmax(self.branch_weights, dim=0)  # [num_branches]
#         # 加权求和（也可以不做求和，而是直接concat）
#         # 为了保留完整信息，这里做concat更安全
#         concat_out = torch.cat(branch_outs, dim=1)
#         out = self.project(concat_out)
#         out = out + self.residual(x)
#         return out
