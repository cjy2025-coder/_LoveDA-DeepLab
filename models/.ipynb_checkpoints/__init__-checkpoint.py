# models/__init__.py
# models/__init__.py
from models.loveda_deeplab  import LoveDADeepLab,    build_model
from models.baseline_deeplab import BaselineDeepLab, build_baseline

def build_model_by_name(config):
    name = getattr(config, 'MODEL', 'loveda').lower()
    if name == 'baseline':
        return build_baseline(config)
    else:
        return build_model(config)

__all__ = ['LoveDADeepLab', 'BaselineDeepLab', 'build_model_by_name']
