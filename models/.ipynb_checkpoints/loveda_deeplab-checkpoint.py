# models/loveda_deeplab.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.resnet import build_backbone
from models.modules.cbam import CBAM
from models.modules.multiscale_decoder import MultiScaleDecoder
from models.modules.scene_aware_classifier import SceneAwareClassifier
from models.decoder import AuxiliaryHead,DeepLabDecoder
from models.modules.sa_aspp import ScaleAdaptiveASPP
from models.prototype_classifier import PrototypeClassifier,SceneAwarePrototypeClassifier
class LoveDADeepLab(nn.Module):
    """
    改进的DeepLabV3+ for LoveDA
    """

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "resnet50",
        pretrained: bool = True,
        output_stride: int = 16,
        aspp_channels: int = 256,
        decoder_channels: int = 256,
        num_scenes: int = 2,
        **kwargs,
    ):
        super().__init__()

        # ── 骨干网络 ──────────────────────────────
        self.backbone = build_backbone(backbone, pretrained, output_stride)
        high_ch = self.backbone.high_level_channels   # 2048
        low1_ch = self.backbone.low_level_channels    # 256
        low2_ch = self.backbone.mid_level_channels    # 512
        """ASPP"""
        ######## 使用标准ASPP
        # from models.baseline_deeplab import StandardASPP
        # self.aspp = StandardASPP(high_ch, aspp_channels)
        ######## 使用SA-ASPP
        self.aspp = ScaleAdaptiveASPP(high_ch,aspp_channels)
        """解码器"""
        ######## 标准 DeepLab 解码器
        self.decoder = DeepLabDecoder(num_classes, aspp_channels, low1_ch)
        """残差分类头和原型头一起组合，基线没有"""
        # 残差分类头
        self.linear_classifier = nn.Conv2d(
            256,
            num_classes,
            1
        )
        self.prototype_classifier = PrototypeClassifier(
             feat_dim=decoder_channels,
             num_classes=num_classes,
        )
        """辅助头部，和基线相同"""
        self.aux_head = AuxiliaryHead(
            in_channels=aspp_channels,
            num_classes=num_classes,
        )

        self.num_classes = num_classes
        self.num_scenes = num_scenes


    # def forward(self, image: torch.Tensor, scene_label: torch.Tensor = None): 
    #     H, W = image.shape[2], image.shape[3] # 骨干特征提取 
    #     low1, low2, high = self.backbone(image) 
    #     # ! out of use 
    #     # """使用纹理增强模块""" 
    #     # high = self.te(high) 
    #     """集成CBAM模块""" 
    #     # aspp_feat = self.cbam(self.aspp(high)) 
    #     """ASPP模块""" 
    #     aspp_feat = self.aspp(high) 
    #     """Decoder模块""" 
    #     ##### 使用原版Decoder 
    #     main_logits = self.decoder(aspp_feat, low1, (H, W)) 
    #     ##### 使用改进Decoder 
    #     # main_logits = self.decoder(aspp_feat, low1, low2, (H, W))
    #     if not self.training: 
    #         return main_logits 
    #     aux_logits = self.aux_head(aspp_feat, (H, W))
    #     # 为了兼容 LoveDALoss 的接口，补充缺失的键 
    #     return { 'main_logits': main_logits, 
    #             'aux_logits': aux_logits, 
    #             'bg_prob': None, 
    #             # 无BSM 'scene_logits': None, 
    #             # 无SAM 'bsm_alpha': None, 
    #            }
        
    def forward(self, image: torch.Tensor, scene_label: torch.Tensor = None,mask = None):
        # if scene_label is None:
        #     raise ValueError(
        #         "Scene label is required."
        #     )
        # if mask is None:
        #     raise ValueError(
        #         "Mask is required."
        #     )
        # print(scene_label[:8])
        H, W = image.shape[2], image.shape[3]

        """骨干特征提取"""
        low1, low2, high = self.backbone(image)
        """ASPP模块"""
        aspp_feat = self.aspp(high)
        """Decoder模块"""
        ##### 使用原版Decoder
        main_logits = self.decoder(aspp_feat, low1, (H, W))
        """使用PrototypeClassifier"""
        decoded_feat = self.decoder.get_features(
            aspp_feat,
            low1
        )
        
        logits_linear = self.linear_classifier(
            decoded_feat
        )
        logits_proto = self.prototype_classifier(
            decoded_feat
        )
       
        logits_small = (
            logits_linear
            +
            0.3 * logits_proto
        )
        
        main_logits = F.interpolate(
            logits_small,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        if not self.training:
            return main_logits

        aux_logits = self.aux_head(aspp_feat, (H, W))
        # 为了兼容 LoveDALoss 的接口，补充缺失的键
        return {
            'main_logits':  main_logits,
            'aux_logits':   aux_logits,
            # "consistency_loss":consistency_loss,
            'bg_prob':      None,   # 无BSM
            'scene_logits': None,   # 无SAM
            'bsm_alpha':    None,
        }

    def get_params_groups(self, base_lr, backbone_lr_mult=0.1):
        backbone_ids = set(id(p) for p in self.backbone.parameters())
        new_params = [p for p in self.parameters()
                      if id(p) not in backbone_ids]
        return [
            {"params": list(self.backbone.parameters()),
             "lr": base_lr * backbone_lr_mult},
            {"params": new_params, "lr": base_lr},
        ]


def build_model(config):
    return LoveDADeepLab(
        num_classes=config.NUM_CLASSES,
        backbone=config.BACKBONE,
        pretrained=config.PRETRAINED,
        output_stride=config.OUTPUT_STRIDE,
        aspp_channels=config.ASPP_OUT_CHANNELS,
        decoder_channels=config.DECODER_CHANNELS,
        num_scenes=config.NUM_SCENES,
    )