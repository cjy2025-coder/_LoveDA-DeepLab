# datasets/transforms.py
"""
数据增强变换
支持图像和mask的同步变换，以及场景标签的传递
"""

import random
import numpy as np
from PIL import Image, ImageFilter

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class Compose:
    """组合多个变换，同步作用于sample字典"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomResizeCrop:
    """随机缩放+随机裁剪，同步处理图像和mask"""

    def __init__(self, scale_range=(0.5, 2.0), crop_size=512):
        self.scale_range = scale_range
        self.crop_size = crop_size

    def __call__(self, sample):
        image = sample["image"]
        w, h = image.size

        # 随机缩放
        scale = random.uniform(*self.scale_range)
        new_w, new_h = int(w * scale), int(h * scale)
        new_w = max(new_w, self.crop_size)
        new_h = max(new_h, self.crop_size)

        image = image.resize((new_w, new_h), Image.BILINEAR)
        sample["image"] = image

        if "mask" in sample:
            sample["mask"] = sample["mask"].resize((new_w, new_h), Image.NEAREST)

        # 随机裁剪
        i = random.randint(0, new_h - self.crop_size)
        j = random.randint(0, new_w - self.crop_size)

        sample["image"] = TF.crop(sample["image"], i, j, self.crop_size, self.crop_size)
        if "mask" in sample:
            sample["mask"] = TF.crop(sample["mask"], i, j, self.crop_size, self.crop_size)
        return sample


class Resize:
    """固定尺寸缩放"""

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def __call__(self, sample):
        h, w = self.size
        sample["image"] = sample["image"].resize((w, h), Image.BILINEAR)
        if "mask" in sample:
            sample["mask"] = sample["mask"].resize((w, h), Image.NEAREST)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample["image"] = TF.hflip(sample["image"])
            if "mask" in sample:
                sample["mask"] = TF.hflip(sample["mask"])
        return sample


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample["image"] = TF.vflip(sample["image"])
            if "mask" in sample:
                sample["mask"] = TF.vflip(sample["mask"])
        return sample


class RandomRotate90:
    """随机旋转0/90/180/270度"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            angle = k * 90
            sample["image"] = sample["image"].rotate(angle, expand=False)
            if "mask" in sample:
                sample["mask"] = sample["mask"].rotate(angle, expand=False)
        return sample


class ColorJitter:
    """颜色抖动（仅作用于图像）"""

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, sample):
        sample["image"] = self.jitter(sample["image"])
        return sample


class RandomGaussianBlur:
    """随机高斯模糊（仅作用于图像）"""

    def __init__(self, p=0.3, radius_range=(1, 3)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, sample):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            sample["image"] = sample["image"].filter(ImageFilter.GaussianBlur(radius))
        return sample


class ToTensor:
    """PIL Image → Tensor，mask转为LongTensor"""

    def __call__(self, sample):
        # 图像: [H,W,3] → [3,H,W], float, /255
        sample["image"] = TF.to_tensor(sample["image"])

        if "mask" in sample:
            mask_arr = np.array(sample["mask"], dtype=np.int64)
            sample["mask"] = torch.from_numpy(mask_arr).long()

        return sample


class Normalize:
    """图像归一化"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample["image"] = TF.normalize(sample["image"], self.mean, self.std)
        return sample


# ─────────────────────────────────────────────
#  对外接口
# ─────────────────────────────────────────────

def get_train_transforms(config):
    transforms_list = []

    aug = config.TRAIN_AUGMENT

    if aug.get("random_resize_crop"):
        transforms_list.append(
            RandomResizeCrop(
                scale_range=aug["scale_range"],
                crop_size=aug["crop_size"],
            )
        )

    if aug.get("horizontal_flip"):
        transforms_list.append(RandomHorizontalFlip(p=0.5))

    if aug.get("vertical_flip"):
        transforms_list.append(RandomVerticalFlip(p=0.5))

    if aug.get("random_rotate"):
        transforms_list.append(RandomRotate90(p=0.5))

    if aug.get("color_jitter"):
        transforms_list.append(ColorJitter(**aug.get("color_jitter_params", {})))

    transforms_list.append(RandomGaussianBlur(p=0.2))
    transforms_list.append(ToTensor())

    if aug.get("normalize"):
        transforms_list.append(Normalize(aug["mean"], aug["std"]))

    return Compose(transforms_list)


def get_val_transforms(config):
    transforms_list = []

    aug = config.VAL_AUGMENT

    if aug.get("resize"):
        transforms_list.append(Resize(aug["resize_size"]))

    transforms_list.append(ToTensor())

    if aug.get("normalize"):
        transforms_list.append(Normalize(aug["mean"], aug["std"]))

    return Compose(transforms_list)
