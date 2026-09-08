# models/baseline_deeplab.py
"""
标准 DeepLabV3+ 基线模型（无任何改进）
与 LoveDADeepLab 共享相同的骨干网络和解码器，确保对比公平
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.resnet import build_backbone
from models.decoder import DeepLabDecoder, AuxiliaryHead


class StandardASPP(nn.Module):
    """标准 ASPP：固定权重，无尺度注意力"""

    def __init__(self, in_channels=2048, out_channels=128,
                 dilations=(1, 6, 12, 18), dropout=0.1):
        super().__init__()
        branches = []
        for d in dilations:
            if d == 1:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
                ))
            else:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3,
                              padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
                ))
        # 全局平均池化分支
        branches.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        ))
        self.branches = nn.ModuleList(branches)

        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        feats = []
        for i, branch in enumerate(self.branches):
            out = branch(x)
            if out.shape[2] == 1:  # GAP分支需要上采样
                out = F.interpolate(out, size=(H, W),
                                    mode='bilinear', align_corners=False)
            feats.append(out)
        return self.project(torch.cat(feats, dim=1))


class BaselineDeepLab(nn.Module):
    """
    标准 DeepLabV3+ 基线
    """

    def __init__(self, num_classes=7, backbone='resnet50',
                 pretrained=True, output_stride=16,
                 aspp_dilations=None, aspp_channels=256, **kwargs):
        super().__init__()
        if aspp_dilations is None:
            aspp_dilations = [1, 6, 12, 18]

        self.backbone = build_backbone(backbone, pretrained, output_stride)
        high_ch = self.backbone.high_level_channels
        low_ch  = self.backbone.low_level_channels

        self.aspp    = StandardASPP(high_ch, aspp_channels, aspp_dilations)
        self.decoder = DeepLabDecoder(num_classes, aspp_channels, low_ch)
        self.aux_head = AuxiliaryHead(aspp_channels, num_classes)

    def forward(self, image, scene_label=None):
        H, W = image.shape[2], image.shape[3]
        low_feat, _,high_feat = self.backbone(image)
        aspp_feat = self.aspp(high_feat)
        main_logits = self.decoder(aspp_feat, low_feat, (H, W))

        if not self.training:
            return main_logits

        aux_logits = self.aux_head(aspp_feat, (H, W))
        # 为了兼容 LoveDALoss 的接口，补充缺失的键
        return {
            'main_logits':  main_logits,
            'aux_logits':   aux_logits,
            'bg_prob':      None,   # 无BSM
            'scene_logits': None,   # 无SAM
            'bsm_alpha':    None,
        }

    def get_params_groups(self, base_lr, backbone_lr_mult=0.1):
        backbone_ids = set(id(p) for p in self.backbone.parameters())
        new_params   = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {'params': list(self.backbone.parameters()), 'lr': base_lr * backbone_lr_mult},
            {'params': new_params, 'lr': base_lr},
        ]


def build_baseline(config):
    return BaselineDeepLab(
        num_classes=config.NUM_CLASSES,
        backbone=config.BACKBONE,
        pretrained=config.PRETRAINED,
        output_stride=config.OUTPUT_STRIDE,
        aspp_dilations=config.ASPP_DILATIONS,
        aspp_channels=config.ASPP_OUT_CHANNELS,
    )