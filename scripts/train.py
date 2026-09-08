#!/usr/bin/env python3
# scripts/train.py
"""
LoveDA-DeepLab 璁粌鑴氭湰

浣跨敤鏂规硶:
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

# 纭繚椤圭洰鏍圭洰褰曞湪sys.path涓?PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# from models.loveda_deeplab import build_model
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

def load_config(config_path: str):
    """鍔犺浇閰嶇疆妯″潡"""
    config_path = config_path.replace("/", ".").replace("\\", ".").rstrip(".py")
    config = importlib.import_module(config_path)
    return config


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

        # if i < 5:
        #         print(f"scene_label: {scene_label}, dtype: {scene_label.dtype}")
        #         assert scene_label.dtype == torch.long, "scene_label蹇呴』鏄痩ong绫诲瀷锛?
        #         assert scene_label.max() <= 1 and scene_label.min() >= 0, "scene鍊艰秴鍑鸿寖鍥达紒"
        
        # 鍓嶅悜浼犳挱锛堟贩鍚堢簿搴︼級
        with autocast(enabled=scaler is not None):
            outputs = model(image, scene_label,mask)
            loss, loss_dict = criterion(outputs, mask, scene_label)

        # 鍙嶅悜浼犳挱
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

        # 璁板綍
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

        # 鎺ㄧ悊妯″紡锛坢odel.eval()鏃惰嚜鍔ㄨ繑鍥瀖ain_logits锛?        logits = model(image,scene_label,mask)

        # 璁＄畻楠岃瘉鎹熷け
        loss = seg_criterion(logits, mask)
        val_loss += loss.item()
        n += 1

        # 鏇存柊鎸囨爣
        pred = logits.argmax(dim=1)
        metrics.update(pred, mask)

    results = metrics.compute()
    results["val_loss"] = val_loss / max(n, 1)

    return results


def main():
    args = parse_args()

    # 鍔犺浇閰嶇疆
    args.config = "configs/loveda_config" if args.model == "loveda" else "configs/baseline_config"
    config = load_config(args.config)

    # 鍛戒护琛屽弬鏁拌鐩朿onfig
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
    # 璁剧疆闅忔満绉嶅瓙
    set_seed(config.SEED)

    # 璁惧
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # 鍒涘缓杈撳嚭鐩綍
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # 鏃ュ織
    logger = TrainingLogger(
        log_dir=config.LOG_DIR,
        exp_name=f"loveda_{config.BACKBONE}",
        use_tb=True,
    )

    logger.info("=" * 60)
    logger.info("LoveDA-DeepLab 璁粌鍚姩")
    logger.info(f"  璁惧:    {device}")
    logger.info(f"  妯″瀷锛?  {config.MODEL}")
    logger.info(f"  楠ㄥ共:    {config.BACKBONE}")
    logger.info(f"  Epochs:  {config.EPOCHS}")
    logger.info(f"  BS:      {config.BATCH_SIZE}")
    logger.info(f"  LR:      {config.LR}")
    logger.info(f"  AMP:     {config.USE_AMP}")
    logger.info(f"  鏁版嵁闆?  {config.DATA_ROOT}")
    logger.info("=" * 60)

    # 鏁版嵁闆?    logger.info("鍔犺浇鏁版嵁闆?..")
    train_loader = build_dataloader(config.DATA_ROOT, "Train", config, shuffle=True)
    val_loader   = build_dataloader(config.DATA_ROOT, "Val",   config, shuffle=False)
    logger.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 妯″瀷
    logger.info(f"鏋勫缓妯″瀷: {config.BACKBONE}")
    # model = build_model(config)
    model = build_model_by_name(config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  鍙缁冨弬鏁? {n_params/1e6:.2f}M")

    # 浼樺寲鍣紙楠ㄥ共浣跨敤鏇村皬鐨勫涔犵巼锛?    param_groups = model.get_params_groups(config.LR, backbone_lr_mult=0.1)
    if config.OPTIMIZER.lower() == "adamw":
        optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(
            param_groups,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
            nesterov=True,
        )

    # 瀛︿範鐜囪皟搴?    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    # 鎹熷け鍑芥暟
    criterion = LoveDALoss(
        num_classes=config.NUM_CLASSES,
        ignore_index=config.IGNORE_INDEX,
        seg_weight=config.LOSS_SEG_WEIGHT,
        aux_weight=config.LOSS_AUX_WEIGHT,
        bsm_weight=config.LOSS_BSM_WEIGHT,
        scene_weight=config.LOSS_SCENE_WEIGHT,
        use_dice=True,
        dice_weight=0.5,
    )
    # 鏄惁鍚敤绫诲埆骞宠　
    enable_class_balance = getattr(config, "USE_CLASS_BALANCE", True)
    if enable_class_balance:
        criterion.seg_criterion.criterion.weight = criterion.seg_criterion.criterion.weight.to(device)
        criterion.aux_criterion.weight = criterion.aux_criterion.weight.to(device)
    # 璇勪及鎸囨爣
    val_metrics   = SegmentationMetrics(
        config.NUM_CLASSES, config.IGNORE_INDEX, config.CLASS_NAMES
    )
    scene_metrics = SceneMetrics(config.NUM_SCENES)
    metric_tracker = MetricTracker()

    # 娣峰悎绮惧害
    scaler = GradScaler() if (config.USE_AMP and device.type == "cuda") else None

    # 鎭㈠璁粌
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        logger.info(f"鎭㈠璁粌: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        logger.best_miou = ckpt.get("best_miou", 0.0)
        logger.best_epoch = ckpt.get("best_epoch", 0)
        logger.info(f"  浠?Epoch {start_epoch} 缁х画锛屾渶浣砿IoU={logger.best_miou*100:.2f}%")

    # 澶欸PU鏀寔
    if torch.cuda.device_count() > 1:
        logger.info(f"浣跨敤 {torch.cuda.device_count()} 涓狦PU")
        model = nn.DataParallel(model)

    best_miou = logger.best_miou

    # 鈹€鈹€ 璁粌寰幆 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    for epoch in range(start_epoch, config.EPOCHS + 1):
        logger.info(f"\nEpoch [{epoch}/{config.EPOCHS}]")

        # 璁粌
        train_summary = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            scaler, device, epoch, config, logger, metric_tracker
        )

        # 楠岃瘉
        val_results = validate(
            model, val_loader, criterion, val_metrics, scene_metrics, device, config
        )

        # 鏃ュ織
        logger.log_epoch(epoch, train_summary, {
            "mIoU":    val_results["mIoU"],
            "mPA":     val_results["mPA"],
            "val_loss": val_results["val_loss"],
        })
        logger.info("\n" + val_metrics.format_results(val_results))

        # 淇濆瓨鏈€浣砪heckpoint
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
            logger.info(f"  鉁?宸蹭繚瀛樻渶浣虫ā鍨? {best_path}")

        # 瀹氭湡淇濆瓨
        if epoch % config.SAVE_FREQ == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth")
            save_checkpoint(state, ckpt_path)

        # 淇濆瓨鏈€鏂癱heckpoint锛堢敤浜庢仮澶嶏級
        last_path = os.path.join(config.CHECKPOINT_DIR, "last.pth")
        save_checkpoint(state, last_path)

    logger.close()


if __name__ == "__main__":
    main()

