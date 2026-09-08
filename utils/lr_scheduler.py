# utils/lr_scheduler.py
"""
学习率调度器

支持：
  - Poly LR（DeepLab标准调度）
  - Cosine Annealing with Warmup
  - Step LR with Warmup
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    """
    多项式衰减学习率调度（DeepLab标准配置）

    lr = base_lr * (1 - iter/max_iter)^power

    Args:
        optimizer:  优化器
        max_iters:  总迭代步数
        power:      衰减幂次（通常0.9）
        warmup_iters: 线性warmup步数
        last_epoch: 上次epoch（用于恢复训练）
    """

    def __init__(self, optimizer, max_iters: int, power: float = 0.9,
                 warmup_iters: int = 0, last_epoch: int = -1):
        self.max_iters = max_iters
        self.power = power
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current = self.last_epoch
        if current < self.warmup_iters:
            # 线性warmup
            alpha = current / max(self.warmup_iters, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (current - self.warmup_iters) / max(self.max_iters - self.warmup_iters, 1)
            factor = (1.0 - min(progress, 1.0)) ** self.power
            return [base_lr * factor for base_lr in self.base_lrs]


class CosineAnnealingWarmup(_LRScheduler):
    """
    余弦退火 + 线性warmup

    Args:
        optimizer:    优化器
        max_iters:    总步数
        warmup_iters: warmup步数
        min_lr_ratio: 最小学习率比例（相对base_lr）
    """

    def __init__(self, optimizer, max_iters: int, warmup_iters: int = 0,
                 min_lr_ratio: float = 0.01, last_epoch: int = -1):
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current = self.last_epoch
        if current < self.warmup_iters:
            alpha = current / max(self.warmup_iters, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (current - self.warmup_iters) / max(self.max_iters - self.warmup_iters, 1)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_factor
            return [base_lr * factor for base_lr in self.base_lrs]


def build_scheduler(optimizer, config, steps_per_epoch: int):
    """
    根据配置构建学习率调度器（以step为单位）

    Args:
        optimizer:        优化器
        config:           配置模块
        steps_per_epoch:  每个epoch的step数

    Returns:
        scheduler, 调度单位 ('step' or 'epoch')
    """
    total_steps = config.EPOCHS * steps_per_epoch
    warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch

    sched_type = config.LR_SCHEDULER.lower()

    if sched_type == "poly":
        scheduler = PolyLR(
            optimizer,
            max_iters=total_steps,
            power=config.LR_POWER,
            warmup_iters=warmup_steps,
        )
    elif sched_type == "cosine":
        scheduler = CosineAnnealingWarmup(
            optimizer,
            max_iters=total_steps,
            warmup_iters=warmup_steps,
            min_lr_ratio=0.01,
        )
    else:
        raise ValueError(f"不支持的调度器类型: {sched_type}")

    return scheduler
