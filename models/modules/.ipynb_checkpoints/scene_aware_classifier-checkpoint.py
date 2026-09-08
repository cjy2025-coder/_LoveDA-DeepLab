# models/modules/scene_aware_classifier.py
"""
场景感知分类头
针对LoveDA Urban/Rural类别分布不一致问题：
- 两个独立分类头分别处理Urban和Rural场景
- 场景预测器自动识别场景类型
- 训练时用真值场景标签硬选择，推理时用预测概率软加权
- 辅助场景分类损失监督场景预测器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneAwareClassifier(nn.Module):
    """
    场景感知双头分类器

    Args:
        in_channels:  输入特征通道数（解码器输出）
        num_classes:  分割类别数
        num_scenes:   场景数量（默认2：Urban/Rural）
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_scenes: int = 2,
    ):
        super().__init__()

        # 场景专属分类头：每个场景一个独立1×1卷积
        self.heads = nn.ModuleList([
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
            for _ in range(num_scenes)
        ])

        # 轻量场景预测器：全局平均池化 → 线性分类
        self.scene_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, num_scenes),
        )

        self.num_scenes = num_scenes

    def forward(
        self,
        x: torch.Tensor,
        scene_label: torch.Tensor = None,
    ):
        """
        Args:
            x:           [B, C, H, W] 解码后特征
            scene_label: [B] int64，Urban=0, Rural=1
                         训练时传入真值，推理时传None

        Returns:
            logits:       [B, num_classes, H, W] 分割预测
            scene_logits: [B, num_scenes] 场景分类logits（用于辅助损失）
        """
        # 场景预测
        scene_logits = self.scene_predictor(x)   # [B, num_scenes]

        # 每个场景头的输出
        head_outputs = [head(x) for head in self.heads]
        # 每个: [B, num_classes, H, W]

        if scene_label is not None:
            # 训练时：用真值场景标签硬选择对应分类头
            # scene_label: [B]，值为0或1
            head_outputs_stack = torch.stack(head_outputs, dim=1)
            # [B, num_scenes, num_classes, H, W]

            idx = scene_label.view(-1, 1, 1, 1, 1).expand(
                -1, 1,
                head_outputs_stack.shape[2],
                head_outputs_stack.shape[3],
                head_outputs_stack.shape[4],
            )  # [B, 1, num_classes, H, W]

            logits = head_outputs_stack.gather(1, idx).squeeze(1)
            # [B, num_classes, H, W]

        else:
            # 推理时：用预测场景概率软加权两个头的输出
            scene_prob = F.softmax(scene_logits, dim=1)  # [B, num_scenes]

            logits = sum(
                scene_prob[:, i].view(-1, 1, 1, 1) * head_outputs[i]
                for i in range(self.num_scenes)
            )  # [B, num_classes, H, W]

        return logits, scene_logits