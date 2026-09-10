#!/usr/bin/env python3
# scripts/train.py
"""
使用方式：
  python scripts/train.py
  python scripts/train.py --data_root /path/to/LoveDA --epochs 80 --batch_size 8
  python scripts/train.py --resume outputs/checkpoints/last.pth
"""

import os
import sys
import time
import random
import argparse
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from models import build_model_by_name
from datasets.loveda_dataset import build_dataloader
from utils.losses import LoveDALoss
from utils.metrics import SegmentationMetrics, SceneMetrics
from utils.lr_scheduler import build_scheduler
from utils.logger import TrainingLogger, MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description="LoveDA-DeepLab training")
    parser.add_argument("--config", default="configs/loveda_config")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone", default=None, choices=["resnet50", "resnet101"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--ablation", choices=["baseline", "sa_aspp", "prototype", "balanced", "sa_aspp_prototype", "full"], default=None)
    return parser.parse_args()


# 加载配置
def load_config(config_path: str):
    config_path = config_path.replace("/", ".").replace("\\", ".").rstrip(".py")
    config = importlib.import_module(config_path)
    return config

# 设置随机种子
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, scaler,
    device, epoch, config, logger, metric_tracker
):
    model.train()
    metric_tracker.reset()

    n_batches = len(loader)
    t0 = time.time()

    for i, batch in enumerate(loader):
        image       = batch["image"].to(device, non_blocking=True)
        mask        = batch["mask"].to(device, non_blocking=True)
        scene_label = batch["scene"].to(device, non_blocking=True)

        with autocast(enabled=scaler is not None):
            outputs = model(image, scene_label,mask)
            loss, loss_dict = criterion(outputs, mask, scene_label)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        scheduler.step()
        metric_tracker.update(loss_dict)
        current_lr = optimizer.param_groups[-1]["lr"]

        if (i + 1) % config.LOG_FREQ == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_batches - i - 1)
            logger.info(
                f"  Iter [{i+1}/{n_batches}] "
                f"loss={metric_tracker.avg('total'):.4f} "
                f"({metric_tracker.format()}) "
                f"lr={current_lr:.2e} "
                f"ETA={eta/60:.1f}min"
            )

    return metric_tracker.summary()


@torch.no_grad()
def validate(model, loader, criterion_seg, metrics, scene_metrics, device, config):
    model.eval()
    metrics.reset()
    scene_metrics.reset()

    val_loss = 0.0
    n = 0

    seg_criterion = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)

    for batch in loader:
        image       = batch["image"].to(device, non_blocking=True)
        mask        = batch["mask"].to(device, non_blocking=True)
        scene_label = batch["scene"].to(device, non_blocking=True)
       
        logits = model(image,scene_label,mask)

        loss = seg_criterion(logits, mask)
        val_loss += loss.item()
        n += 1

        pred = logits.argmax(dim=1)
        metrics.update(pred, mask)

    results = metrics.compute()
    results["val_loss"] = val_loss / max(n, 1)

    return results


def main():
    args = parse_args()

    args.config = "configs/loveda_config" if args.model == "loveda" else "configs/baseline_config"
    config = load_config(args.config)

    if args.data_root:   config.DATA_ROOT   = args.data_root
    if args.output_dir:  config.OUTPUT_DIR  = args.output_dir
    if args.epochs:      config.EPOCHS      = args.epochs
    if args.batch_size:  config.BATCH_SIZE  = args.batch_size
    if args.lr:          config.LR          = args.lr
    if args.backbone:    config.BACKBONE    = args.backbone
    if args.no_amp:      config.USE_AMP     = False
    if args.seed:        config.SEED        = args.seed
    if args.model:        config.MODEL       = args.model
    if args.ablation:    config.ABLATION    = args.ablation

    set_seed(config.SEED)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)


    logger = TrainingLogger(
        log_dir=config.LOG_DIR,
        exp_name=f"loveda_{config.BACKBONE}",
        use_tb=True,
    )

    logger.info("=" * 60)
    logger.info("LoveDA-DeepLab 训练配置")
    logger.info(f"  设备:    {device}")
    logger.info(f"  模型：  {config.MODEL}")
    logger.info(f"  骨干网络:    {config.BACKBONE}")
    logger.info(f"  Epochs:  {config.EPOCHS}")
    logger.info(f"  BS:      {config.BATCH_SIZE}")
    logger.info(f"  LR:      {config.LR}")
    logger.info(f"  AMP:     {config.USE_AMP}")
    logger.info(f"  数据来源?  {config.DATA_ROOT}")
    logger.info("=" * 60)
    #是否启用类别平衡和过采样
    enable_class_balance = getattr(config, "USE_CLASS_BALANCE", True)
    train_loader = build_dataloader(config.DATA_ROOT, "Train", config, shuffle=True,enable_class_balance=enable_class_balance)
    val_loader   = build_dataloader(config.DATA_ROOT, "Val",   config, shuffle=False)
    logger.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    logger.info(f"启用骨干网络: {config.BACKBONE}")
    # model = build_model(config)
    model = build_model_by_name(config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  总参数量 {n_params/1e6:.2f}M")

  
    param_groups = model.get_params_groups(config.LR, backbone_lr_mult=0.1)
    if config.OPTIMIZER.lower() == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(
            param_groups,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
            nesterov=True,
        )

   
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    criterion = LoveDALoss(
        num_classes=config.NUM_CLASSES,
        ignore_index=config.IGNORE_INDEX,
        seg_weight=config.LOSS_SEG_WEIGHT,
        aux_weight=config.LOSS_AUX_WEIGHT,
        bsm_weight=config.LOSS_BSM_WEIGHT,
        scene_weight=config.LOSS_SCENE_WEIGHT,
        use_dice=True,
        dice_weight=0.5,
        enable_class_balance=enable_class_balance
    )

    if enable_class_balance:
        criterion.seg_criterion.criterion.weight = criterion.seg_criterion.criterion.weight.to(device)
        criterion.aux_criterion.weight = criterion.aux_criterion.weight.to(device)

    val_metrics   = SegmentationMetrics(
        config.NUM_CLASSES, config.IGNORE_INDEX, config.CLASS_NAMES
    )
    scene_metrics = SceneMetrics(config.NUM_SCENES)
    metric_tracker = MetricTracker()

    scaler = GradScaler() if (config.USE_AMP and device.type == "cuda") else None

  
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        logger.info(f"resume: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        logger.best_miou = ckpt.get("best_miou", 0.0)
        logger.best_epoch = ckpt.get("best_epoch", 0)
        logger.info(f"  Epoch {start_epoch} IoU={logger.best_miou*100:.2f}%")

    if torch.cuda.device_count() > 1:

        model = nn.DataParallel(model)

    best_miou = logger.best_miou

    
    for epoch in range(start_epoch, config.EPOCHS + 1):
        logger.info(f"\nEpoch [{epoch}/{config.EPOCHS}]")


        train_summary = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            scaler, device, epoch, config, logger, metric_tracker
        )


        val_results = validate(
            model, val_loader, criterion, val_metrics, scene_metrics, device, config
        )


        logger.log_epoch(epoch, train_summary, {
            "mIoU":    val_results["mIoU"],
            "mPA":     val_results["mPA"],
            "val_loss": val_results["val_loss"],
        })
        logger.info("\n" + val_metrics.format_results(val_results))


        current_miou = val_results["mIoU"]
        is_best = current_miou > best_miou
        if is_best:
            best_miou = current_miou

        raw_model = model.module if hasattr(model, "module") else model
        state = {
            "epoch":      epoch,
            "model":      raw_model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "best_miou":  best_miou,
            "best_epoch": logger.best_epoch,
            "val_results": val_results,
        }

        if is_best:
            best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(state, best_path)
            logger.info(f"  最佳模型已保存在 {best_path}")

        # 隔段时间保存模型
        if epoch % config.SAVE_FREQ == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth")
            save_checkpoint(state, ckpt_path)


        last_path = os.path.join(config.CHECKPOINT_DIR, "last.pth")
        save_checkpoint(state, last_path)

    logger.close()


if __name__ == "__main__":
    main()

