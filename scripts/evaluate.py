#!/usr/bin/env python3
# scripts/evaluate.py
"""
LoveDA-DeepLab 验证/评估脚本

使用方法:
  python evaluate.py --checkpoint important_base_line_outputs/checkpoints/best_model.pth
  python scripts/evaluate.py --checkpoint important_base_line_outputs/checkpoints/best_model.pth --split Val
  python evaluate.py --checkpoint important_base_line_outputs/checkpoints/best_model.pth --split Val --per_scene

  python scripts/evaluate.py --checkpoint loveda_outputs/checkpoints/best_model.pth
  python scripts/evaluate.py --checkpoint loveda_outputs/checkpoints/best_model.pth --split Val
  
  python evaluate.py --checkpoint baseline@256@dropout0.1_outputs/checkpoints/best_model.pth --split Val --per_scene
  python evaluate.py --checkpoint saaspp_loveda_outputs/checkpoints/best_model.pth --split Val --per_scene
  python evaluate.py --checkpoint te_loveda_outputs/checkpoints/best_model.pth --split Val --per_scene
  python evaluate.py --checkpoint ms_decoder_loveda_outputs/checkpoints/best_model.pth --split Val --per_scene
  python evaluate.py --checkpoint new_saaspp_loveda_outputs/checkpoints/best_model.pth --split Val --per_scene
  python evaluate.py --checkpoint PrototypeClassifier_loveda_outputs/checkpoints/best_model.pth --split Val --per_scene

  python evaluate.py --checkpoint scene_aware_PrototypeClassifier_loveda_outputs/checkpoints/best_model.pth --split Val --per_scene
  python evaluate.py --checkpoint sa-aspp_prototype_loveda_outputs/checkpoints/best_model.pth --split Val --per_scene
  python evaluate.py --checkpoint all_loveda_outputs/checkpoints/best_model.pth --split Val --per_scene
"""

import os
import sys
import argparse
import importlib

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models import build_model_by_name
from datasets.loveda_dataset import build_dataloader
from datasets.transforms import get_val_transforms
from utils.metrics import SegmentationMetrics
from torch.utils.data import DataLoader
from datasets.loveda_dataset import collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="LoveDA-DeepLab 评估")
    parser.add_argument("--checkpoint",  required=True, help="模型checkpoint路径")
    parser.add_argument("--config",      default="configs/loveda_config")
    parser.add_argument("--data_root",   default=None)
    parser.add_argument("--split",       default="Val", choices=["Train", "Val"])
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--per_scene",   action="store_true", help="分别输出Urban/Rural指标")
    parser.add_argument("--device",      default="cuda")
    return parser.parse_args()


def load_config(config_path: str):
    config_path = config_path.replace("/", ".").rstrip(".py")
    return importlib.import_module(config_path)


@torch.no_grad()
def evaluate(model, loader, config, device, tag="Val"):
    model.eval()
    metrics = SegmentationMetrics(
        config.NUM_CLASSES, config.IGNORE_INDEX, config.CLASS_NAMES
    )

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask  = batch["mask"].to(device, non_blocking=True)
        scene_label = batch["scene"].to(device, non_blocking=True)
        
        logits = model(image,scene_label,mask)
        pred = logits.argmax(dim=1)
        metrics.update(pred, mask)

    results = metrics.compute()
    print(f"\n{'='*55}")
    print(f"  {tag} 评估结果")
    print(metrics.format_results(results))
    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.data_root:
        config.DATA_ROOT = args.data_root

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    model = build_model_by_name(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    if args.per_scene:
        # 分场景评估
        from pathlib import Path
        from datasets.loveda_dataset import LoveDADataset

        for scene in config.SCENES:
            scene_dir = Path(config.DATA_ROOT) / args.split / scene
            if not scene_dir.exists():
                print(f"跳过: {scene_dir}")
                continue

            ds = LoveDADataset(
                root=str(scene_dir),
                scene=scene,
                split=args.split.lower(),
                transforms=get_val_transforms(config),
            )
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=4, collate_fn=collate_fn
            )
            evaluate(model, loader, config, device, tag=f"{args.split}/{scene}")
    else:
        # 整体评估
        config.BATCH_SIZE = args.batch_size
        from datasets.loveda_dataset import build_dataloader
        loader = build_dataloader(config.DATA_ROOT, args.split, config, shuffle=False)
        evaluate(model, loader, config, device, tag=args.split)


if __name__ == "__main__":
    main()
