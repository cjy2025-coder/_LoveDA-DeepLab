# models/backbone/__init__.py
from models.backbone.resnet import ResNetBackbone, build_backbone

__all__ = ["ResNetBackbone", "build_backbone"]
