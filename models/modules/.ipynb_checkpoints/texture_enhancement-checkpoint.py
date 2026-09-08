import torch
import torch.nn as nn
class TextureEnhancement(nn.Module):
    """
    轻量纹理增强模块
    用多个小卷积核并行捕捉不同粒度的纹理特征
    专门针对Forest/Barren这类纹理判别困难的类别
    """
    def __init__(self, channels):
        super().__init__()
        # 三个并行的小卷积核，捕捉不同粒度纹理
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=1,
                      groups=channels//4, bias=False),
            nn.BatchNorm2d(channels//4), nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=2,
                      dilation=2, groups=channels//4, bias=False),
            nn.BatchNorm2d(channels//4), nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=3,
                      dilation=3, groups=channels//4, bias=False),
            nn.BatchNorm2d(channels//4), nn.ReLU(inplace=True),
        )
        # 融合
        self.fuse = nn.Sequential(
            nn.Conv2d(channels//4*3, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        t1 = self.branch1(x)
        t2 = self.branch2(x)
        t3 = self.branch3(x)
        texture = self.fuse(torch.cat([t1, t2, t3], dim=1))
        return x + texture   # 残差