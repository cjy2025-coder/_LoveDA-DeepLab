# models/backbone/resnet.py
"""
ResNet骨干网络（带空洞卷积支持）
支持 output_stride=8 或 16
使用 torchvision 预训练权重，修改 layer3/layer4 为空洞卷积
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNetBackbone(nn.Module):
    """
    改造后的 ResNet，输出多尺度特征。

    输出:
        low_level_feat:  layer1 输出  [B, C_low, H/4, W/4]
        high_level_feat: layer4 输出  [B, C_high, H/OS, W/OS]
            OS=output_stride (8 or 16)
    """

    def __init__(self, arch: str = "resnet50", pretrained: bool = True, output_stride: int = 16):
        super().__init__()
        assert output_stride in (8, 16), "output_stride 只支持 8 或 16"
        assert arch in ("resnet50", "resnet101"), f"不支持的arch: {arch}"

        # 加载 torchvision 预训练模型
        weights = "IMAGENET1K_V1" if pretrained else None
        if arch == "resnet50":
            base = tv_models.resnet50(weights=weights)
            self.low_level_channels = 256    # layer1 输出
            self.mid_level_channels  = 512   # layer2（新增）
            self.high_level_channels = 2048  # layer4 输出
        else:
            base = tv_models.resnet101(weights=weights)
            self.low_level_channels = 256
            self.mid_level_channels  = 512   # layer2（新增）
            self.high_level_channels = 2048

        # 拆分各层
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1  # stride=4,  channels=256
        self.layer2 = base.layer2  # stride=8,  channels=512
        self.layer3 = base.layer3  # stride=16, channels=1024
        self.layer4 = base.layer4  # stride=32, channels=2048

        # 根据 output_stride 修改空洞卷积
        if output_stride == 16:
            # layer4: dilation=2, stride=1
            self._make_dilated(self.layer4, stride=1, dilation=2)
        elif output_stride == 8:
            # layer3: dilation=2, stride=1
            # layer4: dilation=4, stride=1
            self._make_dilated(self.layer3, stride=1, dilation=2)
            self._make_dilated(self.layer4, stride=1, dilation=4)

    @staticmethod
    def _make_dilated(layer, stride, dilation):
        """将ResNet层改为空洞卷积（修改stride和dilation）"""
        for i, block in enumerate(layer):
            # 修改 block 内的 3×3 卷积
            for name, module in block.named_modules():
                if isinstance(module, nn.Conv2d):
                    if module.kernel_size == (3, 3):
                        module.dilation = (dilation, dilation)
                        module.padding = (dilation, dilation)
                        module.stride = (1, 1)
                    # module.stride = (1, 1)

            # 修改 downsample 中的卷积 stride
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
                
    def forward(self, x):
        x = self.layer0(x)
        low1 = self.layer1(x)    # [B, 256, H/4,  W/4]
        low2 = self.layer2(low1) # [B, 512, H/8,  W/8]
        x    = self.layer3(low2)
        high = self.layer4(x)    # [B, 2048, H/16, W/16]
        return low1, low2, high  # 新增 low2  
        
    # def forward(self, x):
    #     x = self.layer0(x)       # H/4
    #     low = self.layer1(x)     # H/4,  [B, 256, H/4, W/4]
    #     x = self.layer2(low)     # H/8
    #     x = self.layer3(x)       # H/16 (or H/8 if OS=8)
    #     high = self.layer4(x)    # H/16 (or H/8 if OS=8)
    #     return low, high


def build_backbone(arch: str, pretrained: bool, output_stride: int) -> ResNetBackbone:
    return ResNetBackbone(arch=arch, pretrained=pretrained, output_stride=output_stride)
