import torch
from models.backbone.resnet import ResNetBackbone

model = ResNetBackbone(arch='resnet50', pretrained=False, output_stride=16)
model.eval()  # 关闭 dropout 等
x = torch.randn(1, 3, 256, 256)  # 减小到 256
with torch.no_grad():
    low1, low2, high = model(x)
print(low1.shape, low2.shape, high.shape)