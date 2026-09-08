# models/modules/scene_adaptation.py
"""
场景自适应模块 (Scene Adaptation Module, SAM)

针对 LoveDA Urban / Rural 两个域类别分布不一致的问题：
1. 场景分类器：基于全局特征预测场景类型（Urban/Rural），提供辅助监督
2. 可学习场景原型：每个场景一个原型向量，编码场景全局信息
3. 场景条件特征调制：类似 AdaIN，根据场景原型对特征做仿射变换（scale+shift）
   使模型能够自动识别并适应不同场景的特征分布

设计思路：
  - 场景原型通过可学习参数初始化，由分类损失和调制损失共同优化
  - 特征调制采用逐通道仿射变换（类似 AdaIN / CondBatchNorm）
  - 场景分类辅助损失确保原型具有场景判别性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneClassifier(nn.Module):
    """
    场景分类器：Global Average Pooling → MLP → 场景类别

    Args:
        in_channels: 输入特征通道数
        num_scenes:  场景数量（默认2：Urban/Rural）
        hidden_dim:  隐藏层维度
    """

    def __init__(self, in_channels: int, num_scenes: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_scenes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            logits: [B, num_scenes]
        """
        feat = self.gap(x).flatten(1)  # [B, C]
        return self.classifier(feat)


class SceneConditionedNorm(nn.Module):
    """
    场景条件归一化（Scene-Conditioned Normalization）

    对每个样本，根据其场景原型生成通道级 scale 和 shift：
        y = scale * LayerNorm(x) + shift

    其中 scale, shift 由场景原型通过线性映射得到

    Args:
        num_features: 特征通道数
        proto_dim:    场景原型维度
    """

    def __init__(self, num_features: int, proto_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=num_features, affine=False)
        # 从场景原型映射到 scale 和 shift
        self.scale_proj = nn.Linear(proto_dim, num_features)
        self.shift_proj = nn.Linear(proto_dim, num_features)

        # 初始化：scale≈1, shift≈0，保证训练初期稳定
        nn.init.ones_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     [B, C, H, W] 输入特征
            proto: [B, proto_dim] 场景原型（每个样本对应的场景原型向量）
        Returns:
            y: [B, C, H, W] 场景调制后的特征
        """
        # 归一化
        normed = self.norm(x)                          # [B, C, H, W]

        # 生成 scale 和 shift
        scale = self.scale_proj(proto)                 # [B, C]
        shift = self.shift_proj(proto)                 # [B, C]

        # reshape 以支持广播
        scale = scale.view(-1, scale.shape[1], 1, 1)  # [B, C, 1, 1]
        shift = shift.view(-1, shift.shape[1], 1, 1)  # [B, C, 1, 1]

        return scale * normed + shift


class SceneAdaptationModule(nn.Module):
    """
    场景自适应模块 (SAM)

    核心流程：
    1. 场景分类器预测场景概率分布
    2. 用预测概率软加权各场景原型，得到每个样本的"个性化场景原型"
    3. 用场景原型对特征做条件归一化（scale + shift）

    Args:
        in_channels:  输入特征通道数
        proto_dim:    场景原型维度
        num_scenes:   场景数量
        hidden_dim:   场景分类器隐藏层
    """

    def __init__(
        self,
        in_channels: int,
        proto_dim: int = 256,
        num_scenes: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_scenes = num_scenes
        self.proto_dim = proto_dim

        # 可学习场景原型
        self.scene_prototypes = nn.Parameter(
            torch.randn(num_scenes, proto_dim) * 0.01
        )  # [num_scenes, proto_dim]

        # 场景分类器
        self.classifier = SceneClassifier(in_channels, num_scenes, hidden_dim)

        # 特征投影（将in_channels对齐到proto_dim以便调制）
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, proto_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proto_dim),
            nn.ReLU(inplace=True),
        )

        # 场景条件归一化
        self.scene_norm = SceneConditionedNorm(proto_dim, proto_dim)

        # 输出投影（将proto_dim还原到in_channels）
        self.out_proj = nn.Sequential(
            nn.Conv2d(proto_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, scene_label: torch.Tensor = None):
        """
        Args:
            x:           [B, C, H, W] 输入特征
            scene_label: [B]           场景真值标签（训练时用于辅助损失）

        Returns:
            out:          [B, C, H, W] 场景自适应特征
            scene_logits: [B, num_scenes] 场景分类 logits（用于辅助损失）
            proto_used:   [B, proto_dim]  每个样本使用的场景原型
        """
        B = x.shape[0]

        # 1. 场景分类
        scene_logits = self.classifier(x)                  # [B, num_scenes]
        scene_prob = F.softmax(scene_logits, dim=1)        # [B, num_scenes]

        # 2. 软加权场景原型：每个样本得到加权混合的场景原型
        #    prototypes: [num_scenes, proto_dim]
        #    scene_prob: [B, num_scenes]
        #    proto_used: [B, proto_dim]
        proto_used = torch.mm(scene_prob, self.scene_prototypes)  # [B, proto_dim]

        # 3. 特征投影
        feat = self.feat_proj(x)                           # [B, proto_dim, H, W]

        # 4. 场景条件归一化
        feat_adapted = self.scene_norm(feat, proto_used)   # [B, proto_dim, H, W]

        # 5. 输出投影 + 残差
        out = self.out_proj(feat_adapted)                  # [B, C, H, W]
        out = self.relu(out + x)                           # 残差连接

        return out, scene_logits, proto_used

    def get_scene_prototypes(self):
        """返回归一化后的场景原型，用于可视化"""
        return F.normalize(self.scene_prototypes, dim=1)


class SceneClassificationLoss(nn.Module):
    """
    场景分类辅助损失（交叉熵）

    监督场景分类器和场景原型具有场景判别性
    """

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, scene_logits: torch.Tensor, scene_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scene_logits:  [B, num_scenes]
            scene_labels:  [B] int64，Urban=0, Rural=1
        Returns:
            loss: 标量
        """
        return self.criterion(scene_logits, scene_labels)
