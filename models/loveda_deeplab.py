# models/loveda_deeplab.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.resnet import build_backbone

from models.decoder import AuxiliaryHead,DeepLabDecoder
from models.baseline_deeplab import StandardASPP
from models.modules.sa_aspp import ScaleAdaptiveASPP
from models.prototype_classifier import PrototypeClassifier
class LoveDADeepLab(nn.Module):
    """
    鏢�硅繘鐨凞eepLabV3+ for LoveDA
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
        use_sa_aspp: bool = True,
        use_prototype: bool = True,
        **kwargs,
    ):
        super().__init__()

        # 鈹€鈹€ 楠ㄥ共缃戠粶 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self.backbone = build_backbone(backbone, pretrained, output_stride)
        high_ch = self.backbone.high_level_channels   # 2048
        low1_ch = self.backbone.low_level_channels    # 256
        self.aspp = (
            ScaleAdaptiveASPP(high_ch, aspp_channels)
            if use_sa_aspp
            else StandardASPP(high_ch, aspp_channels)
        )

        self.decoder = DeepLabDecoder(num_classes, aspp_channels, low1_ch)

        self.linear_classifier = nn.Conv2d(
            decoder_channels,
            num_classes,
            kernel_size=1,
        )

        self.prototype_classifier = (
            PrototypeClassifier(
                feat_dim=decoder_channels,
                num_classes=num_classes,
            )
            if use_prototype else None
        )
        self.aux_head = AuxiliaryHead(
            in_channels=aspp_channels,
            num_classes=num_classes,
        )

        self.num_classes = num_classes
        self.num_scenes = num_scenes

    def forward(self, image: torch.Tensor, scene_label: torch.Tensor = None,mask = None):
        H, W = image.shape[2], image.shape[3]

        """楠ㄥ共鐗瑰緛鎻愬彄1�7"""
        low1, _low2, high = self.backbone(image)
        """ASPP妯��潡"""
        aspp_feat = self.aspp(high)
        decoded_feat = self.decoder.get_features(aspp_feat, low1)
        
        logits_linear = self.linear_classifier(
            decoded_feat
        )
        if self.prototype_classifier is None:
            logits_small = logits_linear
        else:
            logits_proto = self.prototype_classifier(decoded_feat)
            logits_small = logits_linear + 0.3 * logits_proto
        
        main_logits = F.interpolate(
            logits_small,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        if not self.training:
            return main_logits

        aux_logits = self.aux_head(aspp_feat, (H, W))
        # 涓轰簡鍏煎 LoveDALoss 鐨勬帴鍙ｏ紝琛ュ厖缂哄け鐨勯敄1�7
        return {
            'main_logits':  main_logits,
            'aux_logits':   aux_logits,
            # "consistency_loss":consistency_loss,
            'bg_prob':      None,   # 鏃燘SM
            'scene_logits': None,   # 鏃燬AM
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
        use_sa_aspp=getattr(config, "USE_SA_ASPP", True),
        use_prototype=getattr(config, "USE_PROTOTYPE_HEAD", True),
    )






