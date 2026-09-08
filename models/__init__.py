from models.loveda_deeplab import LoveDADeepLab, build_model
from models.baseline_deeplab import BaselineDeepLab, build_baseline

ABLATION_SETTINGS = {
    "baseline": {"model": "baseline", "sa_aspp": False, "prototype": False, "balance": False},
    "sa_aspp": {"model": "loveda", "sa_aspp": True, "prototype": False, "balance": False},
    "prototype": {"model": "loveda", "sa_aspp": False, "prototype": True, "balance": False},
    "balanced": {"model": "loveda", "sa_aspp": False, "prototype": False, "balance": True},
    "sa_aspp_prototype": {"model": "loveda", "sa_aspp": True, "prototype": True, "balance": False},
    "full": {"model": "loveda", "sa_aspp": True, "prototype": True, "balance": True},
}


def build_model_by_name(config):
    default_ablation = "baseline" if getattr(config, "MODEL", "loveda").lower() == "baseline" else "full"
    ablation = getattr(config, "ABLATION", default_ablation).lower()
    if ablation not in ABLATION_SETTINGS:
        valid = ", ".join(ABLATION_SETTINGS)
        raise ValueError(f"Unknown ABLATION={ablation!r}; choose one of: {valid}")

    setting = ABLATION_SETTINGS[ablation]
    config.USE_SA_ASPP = setting["sa_aspp"]
    config.USE_PROTOTYPE_HEAD = setting["prototype"]
    config.USE_CLASS_BALANCE = setting["balance"]
    config.MODEL = setting["model"]

    return build_baseline(config) if config.MODEL == "baseline" else build_model(config)


__all__ = ["LoveDADeepLab", "BaselineDeepLab", "build_model_by_name"]

