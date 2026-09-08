# # utils/metrics.py
# """
# 分割评估指标

# 主要指标：
#   - mIoU（平均交并比）：分割任务的核心指标
#   - IoU per class：每类的交并比
#   - Pixel Accuracy (PA)
#   - Mean Pixel Accuracy (mPA)

# 支持累积模式（每个batch更新混淆矩阵，epoch结束统一计算）
# """

# import numpy as np
# import torch


# class SegmentationMetrics:
#     """
#     分割评估指标计算器（基于混淆矩阵）

#     使用方法：
#         metrics = SegmentationMetrics(num_classes=7, ignore_index=255)
#         for batch in loader:
#             pred = model(image).argmax(1)
#             metrics.update(pred, mask)
#         results = metrics.compute()
#         metrics.reset()

#     Args:
#         num_classes:  类别数
#         ignore_index: 忽略像素值
#         class_names:  类别名称列表（用于输出）
#     """

#     def __init__(self, num_classes: int, ignore_index: int = 255, class_names: list = None):
#         self.num_classes = num_classes
#         self.ignore_index = ignore_index
#         self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
#         self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

#     def reset(self):
#         self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

#     def update(self, pred: torch.Tensor, target: torch.Tensor):
#         """
#         更新混淆矩阵

#         Args:
#             pred:   [B, H, W] or [H, W] 预测类别
#             target: [B, H, W] or [H, W] 真值类别
#         """
#         if isinstance(pred, torch.Tensor):
#             pred = pred.cpu().numpy()
#         if isinstance(target, torch.Tensor):
#             target = target.cpu().numpy()

#         pred = pred.flatten().astype(np.int64)
#         target = target.flatten().astype(np.int64)

#         # 过滤忽略像素
#         valid = (target >= 0) & (target < self.num_classes) & (target != self.ignore_index)
#         pred = pred[valid]
#         target = target[valid]

#         # 更新混淆矩阵
#         cm = np.bincount(
#             self.num_classes * target + pred,
#             minlength=self.num_classes ** 2,
#         ).reshape(self.num_classes, self.num_classes)
#         self.confusion_matrix += cm

#     def compute(self) -> dict:
#         """
#         计算所有指标

#         Returns:
#             dict with keys:
#               'mIoU':      float  平均IoU
#               'iou':       list   每类IoU
#               'mPA':       float  平均像素精度
#               'PA':        float  全局像素精度
#               'per_class': dict   每类详细指标
#         """
#         cm = self.confusion_matrix.astype(np.float64)

#         # 交集：对角线
#         intersection = np.diag(cm)
#         # 并集：行和 + 列和 - 对角线
#         union = cm.sum(axis=1) + cm.sum(axis=0) - intersection

#         # IoU（跳过没有出现的类别）
#         iou = np.where(union > 0, intersection / (union + 1e-10), np.nan)

#         # 有效类别（在GT中出现的类别）
#         valid_classes = ~np.isnan(iou)
#         miou = np.nanmean(iou)

#         # 像素精度
#         pa = intersection.sum() / (cm.sum() + 1e-10)

#         # 每类像素精度
#         class_sum = cm.sum(axis=1)
#         per_class_pa = np.where(class_sum > 0, intersection / (class_sum + 1e-10), np.nan)
#         mpa = np.nanmean(per_class_pa)

#         per_class = {}
#         for i, name in enumerate(self.class_names):
#             per_class[name] = {
#                 "IoU": float(iou[i]) if not np.isnan(iou[i]) else None,
#                 "PA":  float(per_class_pa[i]) if not np.isnan(per_class_pa[i]) else None,
#             }

#         return {
#             "mIoU": float(miou),
#             "iou":  [float(v) if not np.isnan(v) else None for v in iou],
#             "mPA":  float(mpa),
#             "PA":   float(pa),
#             "per_class": per_class,
#         }

#     def format_results(self, results: dict = None) -> str:
#         """格式化输出结果"""
#         if results is None:
#             results = self.compute()

#         lines = [
#             f"{'─'*55}",
#             f"{'Class':<20} {'IoU':>10} {'PA':>10}",
#             f"{'─'*55}",
#         ]
#         for name, vals in results["per_class"].items():
#             iou_str = f"{vals['IoU']*100:.2f}%" if vals['IoU'] is not None else "  N/A  "
#             pa_str  = f"{vals['PA']*100:.2f}%"  if vals['PA']  is not None else "  N/A  "
#             lines.append(f"{name:<20} {iou_str:>10} {pa_str:>10}")

#         lines += [
#             f"{'─'*55}",
#             f"{'mIoU':<20} {results['mIoU']*100:>10.2f}%",
#             f"{'mPA':<20} {results['mPA']*100:>10.2f}%",
#             f"{'PA':<20} {results['PA']*100:>10.2f}%",
#             f"{'─'*55}",
#         ]
#         return "\n".join(lines)


# class SceneMetrics:
#     """场景分类指标（配合场景自适应模块）"""

#     def __init__(self, num_scenes: int = 2, scene_names: list = None):
#         self.num_scenes = num_scenes
#         self.scene_names = scene_names or ["Urban", "Rural"]
#         self.correct = 0
#         self.total = 0

#     def reset(self):
#         self.correct = 0
#         self.total = 0

#     def update(self, logits: torch.Tensor, labels: torch.Tensor):
#         pred = logits.argmax(dim=1)
#         self.correct += (pred == labels).sum().item()
#         self.total += labels.numel()

#     def compute(self) -> dict:
#         acc = self.correct / (self.total + 1e-10)
#         return {"scene_accuracy": float(acc)}

# utils/metrics.py
"""
分割评估指标

主要指标：
  - mIoU（平均交并比）：分割任务的核心指标
  - IoU per class：每类的交并比
  - F1 per class：每类的F1分数
  - mF1：平均F1分数
  - Pixel Accuracy (PA)
  - Mean Pixel Accuracy (mPA)

支持累积模式（每个batch更新混淆矩阵，epoch结束统一计算）
"""

import numpy as np
import torch


class SegmentationMetrics:
    """
    分割评估指标计算器（基于混淆矩阵）

    使用方法：
        metrics = SegmentationMetrics(num_classes=7, ignore_index=255)
        for batch in loader:
            pred = model(image).argmax(1)
            metrics.update(pred, mask)
        results = metrics.compute()
        metrics.reset()

    Args:
        num_classes:  类别数
        ignore_index: 忽略像素值
        class_names:  类别名称列表（用于输出）
    """

    def __init__(self, num_classes: int, ignore_index: int = 255, class_names: list = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        更新混淆矩阵

        Args:
            pred:   [B, H, W] or [H, W] 预测类别
            target: [B, H, W] or [H, W] 真值类别
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        pred = pred.flatten().astype(np.int64)
        target = target.flatten().astype(np.int64)

        # 过滤忽略像素
        valid = (target >= 0) & (target < self.num_classes) & (target != self.ignore_index)
        pred = pred[valid]
        target = target[valid]

        # 更新混淆矩阵
        cm = np.bincount(
            self.num_classes * target + pred,
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += cm

    def compute(self) -> dict:
        """
        计算所有指标

        Returns:
            dict with keys:
              'mIoU':      float  平均IoU
              'iou':       list   每类IoU
              'mF1':       float  平均F1分数
              'f1':        list   每类F1分数
              'mPA':       float  平均像素精度
              'PA':        float  全局像素精度
              'per_class': dict   每类详细指标
        """
        cm = self.confusion_matrix.astype(np.float64)

        # 交集：对角线
        TP = np.diag(cm)
        # 并集：行和 + 列和 - 对角线
        union = cm.sum(axis=1) + cm.sum(axis=0) - TP

        # 每类的 FP 和 FN
        FP = cm.sum(axis=0) - TP  # 列和 - TP（预测为该类但实际不是）
        FN = cm.sum(axis=1) - TP  # 行和 - TP（实际为该类但预测不是）

        # IoU（跳过没有出现的类别）
        iou = np.where(union > 0, TP / (union + 1e-10), np.nan)

        # F1 = 2 * Precision * Recall / (Precision + Recall)
        # Precision = TP / (TP + FP)
        # Recall = TP / (TP + FN)
        precision = np.where((TP + FP) > 0, TP / (TP + FP + 1e-10), np.nan)
        recall = np.where((TP + FN) > 0, TP / (TP + FN + 1e-10), np.nan)
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall + 1e-10),
            np.nan
        )

        # 有效类别（在GT中出现的类别）
        valid_classes = ~np.isnan(iou)
        miou = np.nanmean(iou)
        mf1 = np.nanmean(f1)

        # 像素精度
        pa = TP.sum() / (cm.sum() + 1e-10)

        # 每类像素精度
        class_sum = cm.sum(axis=1)
        per_class_pa = np.where(class_sum > 0, TP / (class_sum + 1e-10), np.nan)
        mpa = np.nanmean(per_class_pa)

        per_class = {}
        for i, name in enumerate(self.class_names):
            per_class[name] = {
                "IoU": float(iou[i]) if not np.isnan(iou[i]) else None,
                "F1":  float(f1[i])  if not np.isnan(f1[i])  else None,
                "PA":  float(per_class_pa[i]) if not np.isnan(per_class_pa[i]) else None,
            }

        return {
            "mIoU": float(miou),
            "iou":  [float(v) if not np.isnan(v) else None for v in iou],
            "mF1":  float(mf1),
            "f1":   [float(v) if not np.isnan(v) else None for v in f1],
            "mPA":  float(mpa),
            "PA":   float(pa),
            "per_class": per_class,
        }

    def format_results(self, results: dict = None) -> str:
        """格式化输出结果"""
        if results is None:
            results = self.compute()

        lines = [
            f"{'─'*65}",
            f"{'Class':<20} {'IoU':>10} {'F1':>10} {'PA':>10}",
            f"{'─'*65}",
        ]
        for name, vals in results["per_class"].items():
            iou_str = f"{vals['IoU']*100:.2f}%" if vals['IoU'] is not None else "  N/A  "
            f1_str  = f"{vals['F1']*100:.2f}%"  if vals['F1']  is not None else "  N/A  "
            pa_str  = f"{vals['PA']*100:.2f}%"  if vals['PA']  is not None else "  N/A  "
            lines.append(f"{name:<20} {iou_str:>10} {f1_str:>10} {pa_str:>10}")

        lines += [
            f"{'─'*65}",
            f"{'mIoU':<20} {results['mIoU']*100:>10.2f}%",
            f"{'mF1':<20} {results['mF1']*100:>10.2f}%",
            f"{'mPA':<20} {results['mPA']*100:>10.2f}%",
            f"{'PA':<20} {results['PA']*100:>10.2f}%",
            f"{'─'*65}",
        ]
        return "\n".join(lines)


class SceneMetrics:
    """场景分类指标（配合场景自适应模块）"""

    def __init__(self, num_scenes: int = 2, scene_names: list = None):
        self.num_scenes = num_scenes
        self.scene_names = scene_names or ["Urban", "Rural"]
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        pred = logits.argmax(dim=1)
        self.correct += (pred == labels).sum().item()
        self.total += labels.numel()

    def compute(self) -> dict:
        acc = self.correct / (self.total + 1e-10)
        return {"scene_accuracy": float(acc)}
