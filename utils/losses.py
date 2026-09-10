# utils/losses.py



import torch
import torch.nn as nn
import torch.nn.functional as F


from models.modules.scene_adaptation import SceneClassificationLoss


class OhemCrossEntropyLoss(nn.Module):
    """
    在线难样本挖掘 (OHEM) 交叉熵损失

    选取损失值最大的 top-k 像素参与反向传播，
    避免简单样本主导梯度，提升对难分样本的学习效果

    Args:
        ignore_index:  忽略像素值
        thresh:        难样本损失阈值（低于此值的像素被视为简单样本）
        min_kept:      至少保留的像素数
        weight:        类别权重（可选）
    """

    def __init__(
        self,
        ignore_index: int = 255,
        thresh: float = 0.7,
        min_kept: int = 100000,
        weight: torch.Tensor = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = thresh
        self.min_kept = min_kept
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction="none",
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W]
            target: [B, H, W] int64

        Returns:
            loss: 标量
        """
        # 逐像素交叉熵
        pixel_loss = self.criterion(logits, target)  # [B, H, W]

        # 有效像素mask
        valid_mask = (target != self.ignore_index)   # [B, H, W]
        pixel_loss = pixel_loss[valid_mask]           # [N_valid]

        if pixel_loss.numel() == 0:
            return pixel_loss.sum()

        # 排序，选取难样本（损失大的像素）
        n_keep = max(self.min_kept, int(pixel_loss.numel() * 0.3))
        n_keep = min(n_keep, pixel_loss.numel())

        sorted_loss, _ = pixel_loss.sort(descending=True)
        thresh_loss = sorted_loss[n_keep - 1]

        # 阈值：难样本要求满足：损失 > thresh 或 在top-k中
        hard_mask = pixel_loss >= min(self.thresh, thresh_loss.item())

        if hard_mask.sum() < self.min_kept:
            # 保底：至少选min_kept个
            hard_mask = torch.zeros_like(pixel_loss, dtype=torch.bool)
            hard_mask[:n_keep] = True  # 按排序选前n_keep个

        return pixel_loss[hard_mask].mean()


class DiceLoss(nn.Module):
    """
    Dice Loss：对类别不平衡有较好的鲁棒性

    Args:
        num_classes:  类别数
        ignore_index: 忽略像素
        smooth:       平滑项防止除零
    """

    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)          # [B, C, H, W]
        valid = (target != self.ignore_index)      # [B, H, W]

        # one-hot编码target（忽略区域用0填充）
        target_clamp = target.clone()
        target_clamp[~valid] = 0
        target_oh = F.one_hot(target_clamp, self.num_classes).permute(0, 3, 1, 2).float()
        # [B, C, H, W]

        valid_expand = valid.unsqueeze(1).float()  # [B, 1, H, W]
        probs = probs * valid_expand
        target_oh = target_oh * valid_expand

        dice_losses = []
        for c in range(self.num_classes):
            p = probs[:, c].reshape(-1)
            g = target_oh[:, c].reshape(-1)
            inter = (p * g).sum()
            union = p.sum() + g.sum()
            dice_losses.append(1.0 - (2.0 * inter + self.smooth) / (union + self.smooth))

        return torch.stack(dice_losses).mean()


class LoveDALoss(nn.Module):
    """
    LoveDA 组合损失函数

    total_loss = w_seg * seg_loss
               + w_aux * aux_loss
               + w_bsm * bsm_loss
               + w_scene * scene_loss

    Args:
        num_classes:      类别数
        ignore_index:     忽略像素值
        seg_weight:       主分割损失权重
        aux_weight:       辅助分割损失权重
        bsm_weight:       BSM辅助损失权重
        scene_weight:     场景分类辅助损失权重
        use_dice:         是否添加Dice损失
        dice_weight:      Dice损失权重
    """

    def __init__(
        self,
        num_classes: int = 7,
        ignore_index: int = 255,
        seg_weight: float = 1.0,
        aux_weight: float = 0.4,
        bsm_weight: float = 0.2,
        scene_weight: float = 0.1,
        use_dice: bool = True,
        dice_weight: float = 0.5,
        enable_class_balance = True
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.aux_weight = aux_weight
        self.bsm_weight = bsm_weight
        self.scene_weight = scene_weight
        self.use_dice = use_dice
        self.dice_weight = dice_weight


        # 启用类别平衡
        if enable_class_balance: 
            print("启用类别平衡！")
            class_weights = torch.tensor([
                1.0,   # Background  
                1.0,   # Building
                1.0,   # Road
                1.0,   # Water
                2.0,   # Barren      - 样本极少，大幅加权
                1.5,   # Forest      - 样本少，加权
                1.0,   # Agriculture
            ])
            self.seg_criterion = OhemCrossEntropyLoss(
                ignore_index=ignore_index,
                thresh=0.7,
                min_kept=100000,
                weight=class_weights,
            )
            self.aux_criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_index,
                weight=class_weights,
            )
        else:
            # 主分割损失
            self.seg_criterion = OhemCrossEntropyLoss(ignore_index=ignore_index)
            self.aux_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)


        
        
        # Dice损失
        if use_dice:
            self.dice_criterion = DiceLoss(num_classes, ignore_index)

        self.scene_criterion = SceneClassificationLoss(label_smoothing=0.1)

    def forward(self, outputs: dict, mask: torch.Tensor, scene_label: torch.Tensor):
        main_logits  = outputs["main_logits"]
        aux_logits   = outputs["aux_logits"]
        scene_logits = outputs["scene_logits"]
        # consistency_loss = outputs.get(
        #     "consistency_loss",
        #     0.0
        # )
        # 主分割损失（OHEM CrossEntropy）
        seg_loss = self.seg_criterion(main_logits, mask)
    
        # Dice损失
        if self.use_dice:
            dice_loss = self.dice_criterion(main_logits, mask)
            seg_loss = seg_loss + self.dice_weight * dice_loss
    
        # 辅助分割损失
        if aux_logits is not None:
            aux_loss = self.aux_criterion(aux_logits, mask)
        else:
            aux_loss = torch.tensor(0.0, device=main_logits.device)
    
        # 场景分类辅助损失
        if scene_logits is not None:
            scene_loss = self.scene_criterion(scene_logits, scene_label)
        else:
            scene_loss = torch.tensor(0.0, device=main_logits.device)
    
        # 加权求和
        total_loss = (
            self.seg_weight   * seg_loss
            + self.aux_weight * aux_loss
            + self.scene_weight * scene_loss
            # + 0.05 * consistency_loss
        )
    
        loss_dict = {
            "total": total_loss.item(),
            "seg":   seg_loss.item(),
            # "consistency": consistency_loss.item(),
            "aux":   aux_loss.item(),
            "scene": scene_loss.item(),
        }
        return total_loss, loss_dict
    
    # def forward(self, outputs: dict, mask: torch.Tensor, scene_label: torch.Tensor):
    #     """
    #     Args:
    #         outputs:      模型输出字典（训练时）
    #         mask:         [B, H, W] 分割真值
    #         scene_label:  [B] 场景标签

    #     Returns:
    #         total_loss: 标量
    #         loss_dict:  各项损失的字典（用于日志）
    #     """
    #     main_logits  = outputs["main_logits"]
    #     aux_logits   = outputs["aux_logits"]
    #     bg_prob      = outputs["bg_prob"]
    #     scene_logits = outputs["scene_logits"]
    
    #     # 主分割损失（OHEM CrossEntropy）
    #     seg_loss = self.seg_criterion(main_logits, mask)
    
    #     # Dice损失
    #     if self.use_dice:
    #         dice_loss = self.dice_criterion(main_logits, mask)
    #         seg_loss = seg_loss + self.dice_weight * dice_loss
    
    #     # 辅助分割损失
    #     aux_loss = self.aux_criterion(aux_logits, mask)
    
    #     # # BSM背景抑制辅助损失（基线模型无此模块，bg_prob=None时跳过）
    #     # if bg_prob is not None:
    #     #     bsm_loss = self.bsm_criterion(bg_prob, mask)
    #     # else:
    #     #     bsm_loss = torch.tensor(0.0, device=main_logits.device)
    #     bsm_loss = torch.tensor(0.0, device=main_logits.device)
    #     # 场景分类辅助损失（基线模型无此模块，scene_logits=None时跳过）
    #     if scene_logits is not None:
    #         scene_loss = self.scene_criterion(scene_logits, scene_label)
    #     else:
    #         scene_loss = torch.tensor(0.0, device=main_logits.device)
    
    #     # 加权求和
    #     total_loss = (
    #         self.seg_weight   * seg_loss
    #         + self.aux_weight   * aux_loss
    #         + self.bsm_weight   * bsm_loss
    #         + self.scene_weight * scene_loss
    #     )
    
    #     loss_dict = {
    #         "total": total_loss.item(),
    #         "seg":   seg_loss.item(),
    #         "aux":   aux_loss.item(),
    #         "bsm":   bsm_loss.item(),
    #         "scene": scene_loss.item(),
    #     }
    #     return total_loss, loss_dict
        # main_logits   = outputs["main_logits"]
        # aux_logits    = outputs["aux_logits"]
        # bg_prob       = outputs["bg_prob"]
        # scene_logits  = outputs["scene_logits"]

        # # 主分割损失（OHEM CrossEntropy）
        # seg_loss = self.seg_criterion(main_logits, mask)

        # # Dice损失
        # if self.use_dice:
        #     dice_loss = self.dice_criterion(main_logits, mask)
        #     seg_loss = seg_loss + self.dice_weight * dice_loss

        # # 辅助分割损失
        # aux_loss = self.aux_criterion(aux_logits, mask)

        # # BSM背景抑制辅助损失
        # bsm_loss = self.bsm_criterion(bg_prob, mask)

        # # 场景分类辅助损失
        # scene_loss = self.scene_criterion(scene_logits, scene_label)

        # # 加权求和
        # total_loss = (
        #     self.seg_weight   * seg_loss
        #     + self.aux_weight   * aux_loss
        #     + self.bsm_weight   * bsm_loss
        #     + self.scene_weight * scene_loss
        # )

        # loss_dict = {
        #     "total": total_loss.item(),
        #     "seg":   seg_loss.item(),
        #     "aux":   aux_loss.item(),
        #     "bsm":   bsm_loss.item(),
        #     "scene": scene_loss.item(),
        # }

        # return total_loss, loss_dict
