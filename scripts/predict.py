#!/usr/bin/env python3
# scripts/predict.py
"""
LoveDA-DeepLab 推理脚本

对 Test 集（无mask）进行预测，保存彩色分割图
支持双模型对比预测
  python predict.py --checkpoint all_loveda_outputs/checkpoints/best_model.pth --baseline_checkpoint baseline@256@dropout0.1_outputs/checkpoints/best_model.pth
"""

import os
import sys
import argparse
import importlib
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision import transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models import build_model_by_name


def parse_args():
    parser = argparse.ArgumentParser(description="LoveDA-DeepLab 推理")
    parser.add_argument("--checkpoint",  required=True, help="当前模型checkpoint")
    parser.add_argument("--baseline_checkpoint",  default=None, help="基线模型checkpoint，用于对比")
    parser.add_argument("--config",      default="configs/loveda_config")
    parser.add_argument("--data_root",   default=None)
    parser.add_argument("--output_dir",  default="outputs/predictions")
    parser.add_argument("--split",       default="Test", choices=["Test", "Val"])
    parser.add_argument("--save_color",  action="store_true", default=False,
                        help="保存彩色分割图")
    parser.add_argument("--save_gray",   action="store_true", default=False,
                        help="保存灰度mask（0-indexed类别值）")
    parser.add_argument("--save_comparison", action="store_true", default=True,
                        help="保存对比图（原图+基线+当前模型）")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--batch_size",  type=int, default=4)
    return parser.parse_args()


def load_config(config_path: str):
    config_path = config_path.replace("/", ".").rstrip(".py")
    return importlib.import_module(config_path)


def colorize_mask(mask: np.ndarray, class_colors: list) -> np.ndarray:
    """将类别mask转为RGB彩色图"""
    h, w = mask.shape
    color_map = np.array(class_colors, dtype=np.uint8)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(class_colors):
        rgb[mask == cls_id] = color
    return rgb


def create_comparison_image(original_img, baseline_pred, current_pred, class_colors):
    """
    创建对比图：原图 + 基线模型预测 + 当前模型预测
    """
    # 确保original_img是numpy数组
    if isinstance(original_img, Image.Image):
        original = np.array(original_img)
    else:
        original = original_img
    
    h, w = original.shape[:2]
    
    # 彩色化预测结果
    baseline_color = colorize_mask(baseline_pred, class_colors)
    current_color = colorize_mask(current_pred, class_colors)
    
    # 创建组合图：横向排列 [原图 | 基线 | 当前]
    comparison = np.hstack([original, baseline_color, current_color])
    
    # 添加文字标签
    comparison_pil = Image.fromarray(comparison)
    draw = ImageDraw.Draw(comparison_pil)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    label_y = 10
    draw.text((10, label_y), "Original", fill=(255, 255, 255), font=font)
    draw.text((w + 10, label_y), "Baseline", fill=(255, 255, 255), font=font)
    draw.text((2*w + 10, label_y), "Current", fill=(255, 255, 255), font=font)
    
    return np.array(comparison_pil)


class SingleImageDataset:
    """简单图像加载器，不需要mask"""

    def __init__(self, img_dir: str, transform):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.paths = sorted(
            list(self.img_dir.glob("*.png")) +
            list(self.img_dir.glob("*.jpg")) +
            list(self.img_dir.glob("*.tif"))
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        orig_size = image.size  # (W, H)
        tensor = self.transform(image)
        return tensor, path.stem, orig_size, image


def predict_directory(
    model, baseline_model, img_dir: str, output_dir: str,
    transform, device, config,
    save_color: bool, save_gray: bool, save_comparison: bool,
    batch_size: int = 4,
):
    """对目录中的图像批量推理，支持双模型对比"""
    ds = SingleImageDataset(img_dir, transform)
    if len(ds) == 0:
        print(f"  未找到图像: {img_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    if save_color:
        os.makedirs(os.path.join(output_dir, "color"), exist_ok=True)
        if baseline_model is not None:
            os.makedirs(os.path.join(output_dir, "baseline_color"), exist_ok=True)
    if save_gray:
        os.makedirs(os.path.join(output_dir, "gray"), exist_ok=True)
        if baseline_model is not None:
            os.makedirs(os.path.join(output_dir, "baseline_gray"), exist_ok=True)
    if save_comparison and baseline_model is not None:
        os.makedirs(os.path.join(output_dir, "comparison"), exist_ok=True)

    from torch.utils.data import DataLoader

    def collate(batch):
        tensors = torch.stack([b[0] for b in batch])
        names = [b[1] for b in batch]
        sizes = [b[2] for b in batch]
        images = [b[3] for b in batch]
        return tensors, names, sizes, images

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate)

    model.eval()
    if baseline_model is not None:
        baseline_model.eval()
    
    total = 0

    with torch.no_grad():
        for images, names, orig_sizes, original_images in loader:
            images = images.to(device)
            logits = model(images)       # [B, C, H, W]
            
            # 如果有基线模型，同时进行预测
            if baseline_model is not None:
                baseline_logits = baseline_model(images)

            for i, (name, (ow, oh)) in enumerate(zip(names, orig_sizes)):
                # 当前模型预测
                logit = logits[i:i+1]
                logit = F.interpolate(logit, size=(oh, ow), mode="bilinear", align_corners=False)
                pred = logit.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                # 保存当前模型的彩色图
                if save_color:
                    color = colorize_mask(pred, config.CLASS_COLORS)
                    Image.fromarray(color).save(
                        os.path.join(output_dir, "color", f"{name}.png")
                    )

                # 保存当前模型的灰度图
                if save_gray:
                    Image.fromarray(pred).save(
                        os.path.join(output_dir, "gray", f"{name}.png")
                    )

                # 基线模型预测
                if baseline_model is not None:
                    baseline_logit = baseline_logits[i:i+1]
                    baseline_logit = F.interpolate(baseline_logit, size=(oh, ow), mode="bilinear", align_corners=False)
                    baseline_pred = baseline_logit.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                    # 保存基线模型的彩色图
                    if save_color:
                        baseline_color = colorize_mask(baseline_pred, config.CLASS_COLORS)
                        Image.fromarray(baseline_color).save(
                            os.path.join(output_dir, "baseline_color", f"{name}.png")
                        )

                    # 保存基线模型的灰度图
                    if save_gray:
                        Image.fromarray(baseline_pred).save(
                            os.path.join(output_dir, "baseline_gray", f"{name}.png")
                        )

                    # 创建对比图
                    if save_comparison:
                        comparison = create_comparison_image(
                            original_images[i], baseline_pred, pred, config.CLASS_COLORS
                        )
                        Image.fromarray(comparison).save(
                            os.path.join(output_dir, "comparison", f"{name}.png")
                        )

                total += 1

    print(f"  处理完成: {total} 张图像 → {output_dir}")


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.data_root:
        config.DATA_ROOT = args.data_root

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 加载当前模型
    print(f"加载当前模型: {args.checkpoint}")
    model = build_model_by_name(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"当前模型加载成功 (mIoU: {ckpt.get('best_miou', 0)*100:.2f}%)")

    # 加载基线模型
    baseline_model = None
    if args.baseline_checkpoint:
        print(f"加载基线模型: {args.baseline_checkpoint}")
        baseline_config = load_config("configs/baseline_config")
        baseline_model = build_model_by_name(baseline_config)
        baseline_ckpt = torch.load(args.baseline_checkpoint, map_location=device)
        baseline_model.load_state_dict(baseline_ckpt["model"])
        baseline_model = baseline_model.to(device)
        baseline_model.eval()
        print(f"基线模型加载成功 (mIoU: {baseline_ckpt.get('best_miou', 0)*100:.2f}%)")

    # 图像预处理
    transform = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=config.VAL_AUGMENT["mean"], std=config.VAL_AUGMENT["std"]),
    ])

    # 对每个场景的Test目录进行推理
    data_root = Path(config.DATA_ROOT)
    split_dir = data_root / args.split

    if not split_dir.exists():
        print(f"错误: 未找到 {split_dir}")
        return

    for scene in config.SCENES:
        img_dir = split_dir / scene / "images"
        if not img_dir.exists():
            print(f"跳过: {img_dir}")
            continue

        out_dir = os.path.join(args.output_dir, args.split, scene)
        print(f"\n推理: {img_dir}")
        predict_directory(
            model, baseline_model, str(img_dir), out_dir,
            transform, device, config,
            save_color=args.save_color,
            save_gray=args.save_gray,
            save_comparison=args.save_comparison,
            batch_size=args.batch_size,
        )

    print(f"\n所有预测结果已保存至: {args.output_dir}")
    if baseline_model is not None:
        print(f"对比图保存在各场景的 comparison/ 目录中")


if __name__ == "__main__":
    main()