# utils/logger.py
"""
训练日志工具

支持：
  - 控制台输出（带颜色）
  - 文件日志
  - TensorBoard（可选）
  - 训练状态追踪
"""

import os
import sys
import time
import logging
from collections import defaultdict, deque
from datetime import datetime

import numpy as np


class AverageMeter:
    """滑动窗口平均值计算器"""

    def __init__(self, window_size: int = 50):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.deque.append(val)
        self.total += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return np.mean(self.deque) if self.deque else 0.0

    @property
    def global_avg(self) -> float:
        return self.total / max(self.count, 1)

    def __str__(self):
        return f"{self.avg:.4f}"


class MetricTracker:
    """多指标跟踪器"""

    def __init__(self):
        self._meters = defaultdict(AverageMeter)

    def update(self, metrics: dict, n: int = 1):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._meters[k].update(float(v), n)

    def reset(self):
        self._meters.clear()

    def avg(self, key: str) -> float:
        return self._meters[key].avg

    def global_avg(self, key: str) -> float:
        return self._meters[key].global_avg

    def summary(self) -> dict:
        return {k: m.global_avg for k, m in self._meters.items()}

    def format(self) -> str:
        parts = [f"{k}: {m.avg:.4f}" for k, m in self._meters.items()]
        return " | ".join(parts)


class TrainingLogger:
    """
    训练日志器

    Args:
        log_dir:    日志目录
        exp_name:   实验名称
        use_tb:     是否使用TensorBoard
    """

    def __init__(self, log_dir: str, exp_name: str = "exp", use_tb: bool = True):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.exp_name = exp_name

        # 设置文件日志
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{exp_name}_{timestamp}.log")

        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        self.logger.addHandler(console_handler)

        # 文件handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        self.logger.addHandler(file_handler)

        # TensorBoard
        self.writer = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tensorboard", f"{exp_name}_{timestamp}")
                self.writer = SummaryWriter(tb_dir)
                self.info(f"TensorBoard 日志目录: {tb_dir}")
            except ImportError:
                self.info("TensorBoard 未安装，跳过")

        # 最佳结果跟踪
        self.best_miou = 0.0
        self.best_epoch = 0

        # 训练时间
        self.start_time = time.time()

    def info(self, msg: str):
        self.logger.info(msg)

    def log_metrics(self, metrics: dict, step: int, prefix: str = "train"):
        """记录指标到TensorBoard"""
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict = None):
        """记录epoch结果"""
        elapsed = time.time() - self.start_time
        elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}min"

        self.info(f"{'='*60}")
        self.info(f"Epoch [{epoch}] | 已用时: {elapsed_str}")

        # 训练指标
        train_str = " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
        self.info(f"  Train: {train_str}")

        if self.writer:
            self.log_metrics(train_metrics, epoch, "train")

        # 验证指标
        if val_metrics:
            val_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()
                                  if isinstance(v, float))
            self.info(f"  Val:   {val_str}")

            if self.writer:
                self.log_metrics(val_metrics, epoch, "val")

            # 更新最佳结果
            miou = val_metrics.get("mIoU", 0.0)
            if miou > self.best_miou:
                self.best_miou = miou
                self.best_epoch = epoch
                self.info(f"  新最佳 mIoU: {self.best_miou*100:.2f}% @ Epoch {epoch}")

        self.info(f"  当前最佳: mIoU={self.best_miou*100:.2f}% @ Epoch {self.best_epoch}")

    def close(self):
        if self.writer:
            self.writer.close()
        elapsed = time.time() - self.start_time
        self.info(f"训练完成！总用时: {elapsed/3600:.2f}h")
        self.info(f"最终最佳 mIoU: {self.best_miou*100:.2f}% @ Epoch {self.best_epoch}")
