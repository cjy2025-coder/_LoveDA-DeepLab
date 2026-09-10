# datasets/loveda_dataset.py
"""
LoveDA 数据集加载器
支持 Train / Val / Test 三种模式，自动处理 Urban / Rural 场景标签
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import WeightedRandomSampler
from datasets.transforms import get_train_transforms, get_val_transforms


# LoveDA mask 原始值：1-7（7=ignore），转换为 0-indexed：0-6，ignore→255
def remap_mask(mask: np.ndarray) -> np.ndarray:
    """将原始mask（1-7）映射为 0-indexed（0-6），ignore类(7)→255"""
    out = mask.astype(np.int32) - 1      # 1-7 → 0-6, 0→-1(背景里没有0，但以防万一)
    out[out == 254] = 255                   # 原始7 → 255 (ignore)
    out[out < 0] = 255                    # 非法值也忽略
    return out.astype(np.uint8)


class LoveDADataset(Dataset):
    """
    LoveDA 单场景数据集

    Args:
        root (str): 数据集根目录（如 /path/to/LoveDA/Train/Urban）
        scene (str): 场景类型 'Urban' 或 'Rural'
        scene_id (int): 场景ID（0=Urban, 1=Rural）
        split (str): 'train', 'val', 'test'
        transforms: 数据变换
    """

    SCENE2ID = {"Urban": 0, "Rural": 1}

    def __init__(self, root: str, scene: str, split: str = "train", transforms=None):
        super().__init__()
        self.root = Path(root)
        self.scene = scene
        self.scene_id = self.SCENE2ID[scene]
        self.split = split
        self.transforms = transforms
        self.has_mask = (split != "test")

        # 图像目录
        self.img_dir = self.root / "images"
        assert self.img_dir.exists(), f"图像目录不存在: {self.img_dir}"

        # 收集所有图像文件
        self.img_paths = sorted(
            list(self.img_dir.glob("*.png")) +
            list(self.img_dir.glob("*.jpg")) +
            list(self.img_dir.glob("*.tif"))
        )
        assert len(self.img_paths) > 0, f"未找到图像: {self.img_dir}"

        if self.has_mask:
            self.mask_dir = self.root / "masks"
            assert self.mask_dir.exists(), f"Mask目录不存在: {self.mask_dir}"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")

        sample = {
            "image": image,
            "scene": self.scene_id,
            "image_name": img_path.stem,
        }

        if self.has_mask:
            mask_path = self.mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                # 尝试其他扩展名
                for ext in [".tif", ".jpg"]:
                    mask_path = self.mask_dir / (img_path.stem + ext)
                    if mask_path.exists():
                        break

            mask_raw = np.array(Image.open(mask_path))
            mask = remap_mask(mask_raw)
            sample["mask"] = Image.fromarray(mask)

        if self.transforms:
            sample = self.transforms(sample)

        return sample


def build_dataset(data_root: str, split: str, config) -> Dataset:
    """
    构建合并了Urban和Rural的完整数据集

    Args:
        data_root: LoveDA根目录
        split: 'Train', 'Val', 'Test'
        config: 配置模块

    Returns:
        ConcatDataset 包含Urban和Rural
    """
    split_dir = Path(data_root) / split
    assert split_dir.exists(), f"数据集分割目录不存在: {split_dir}"

    if split.lower() == "train":
        transforms = get_train_transforms(config)
    else:
        transforms = get_val_transforms(config)

    datasets = []
    for scene in config.SCENES:
        scene_dir = split_dir / scene
        if scene_dir.exists():
            ds = LoveDADataset(
                root=str(scene_dir),
                scene=scene,
                split=split.lower(),
                transforms=transforms,
            )
            datasets.append(ds)
            print(f"  [{split}/{scene}] 共 {len(ds)} 张图像")
        else:
            print(f"  警告: 未找到 {scene_dir}")

    return ConcatDataset(datasets)

# def build_dataset(data_root: str, split: str, config) -> Dataset:
#     """构建合并了Urban和Rural的完整数据集"""
#     split_dir = Path(data_root) / split
#     assert split_dir.exists(), f"数据集分割目录不存在: {split_dir}"

#     if split.lower() == "train":
#         transforms = get_train_transforms(config)
#     else:
#         transforms = get_val_transforms(config)

#     datasets = []
#     for scene in config.SCENES:
#         scene_dir = split_dir / scene
#         if scene_dir.exists():
#             ds = LoveDADataset(
#                 root=str(scene_dir),
#                 scene=scene,
#                 split=split.lower(),
#                 transforms=transforms,
#             )
#             datasets.append(ds)
#             print(f"  [{split}/{scene}] 共 {len(ds)} 张图像")
#         else:
#             print(f"  警告: 未找到 {scene_dir}")

#     # 合并数据集
#     if len(datasets) > 1:
#         from torch.utils.data import ConcatDataset
#         combined = ConcatDataset(datasets)
        
#         # 对于验证集，可以选择采样策略
#         if split.lower() == "val":
#             # 可选：创建索引映射，让样本交错
#             return InterleavedDataset(datasets, shuffle=False)  # 使用交错采样
#         return combined
#     elif len(datasets) == 1:
#         return datasets[0]
#     else:
#         raise RuntimeError(f"未找到任何有效数据: {split_dir}")


# class InterleavedDataset(torch.utils.data.Dataset):
#     """交错采样多个数据集，确保每个batch都包含各类别"""
    
#     def __init__(self, datasets, shuffle=False):
#         self.datasets = datasets
#         self.shuffle = shuffle
#         self.lengths = [len(ds) for ds in datasets]
#         self.total_length = sum(self.lengths)
        
#         # 创建索引映射
#         self.indices = []
#         for i, ds in enumerate(datasets):
#             for j in range(len(ds)):
#                 self.indices.append((i, j))
        
#         if shuffle:
#             import random
#             random.shuffle(self.indices)
    
#     def __len__(self):
#         return self.total_length
    
#     def __getitem__(self, idx):
#         ds_idx, sample_idx = self.indices[idx]
#         return self.datasets[ds_idx][sample_idx]


"""原始DeepLab不使用加权采样和类别平衡"""
# def build_dataloader(data_root: str, split: str, config, shuffle: bool = None) -> DataLoader:
#     """构建 DataLoader"""
#     dataset = build_dataset(data_root, split, config)

#     if shuffle is None:
#         shuffle = (split.lower() == "train")

#     loader = DataLoader(
#         dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=shuffle,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=config.PIN_MEMORY,
#         drop_last=(split.lower() == "train"),
#         collate_fn=collate_fn,
#     )
#     return loader



def build_dataloader(data_root: str, split: str, config, shuffle: bool = None,enable_class_balance:bool =True) -> DataLoader:
    dataset = build_dataset(data_root, split, config)

    """使用加权采样和类别平衡"""
    if enable_class_balance:
        if split.lower() == "train":
            # Rural样本权重是Urban的2倍，解决场景不平衡
            weights = []
            for ds in dataset.datasets:
                scene_weight = 2.0 if ds.scene == "Rural" else 1.0
                weights.extend([scene_weight] * len(ds))

            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            return DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                sampler=sampler,          # 用sampler替换shuffle=True
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
                drop_last=True,
                collate_fn=collate_fn,
            )
        else:
            # 验证和测试不过采样
            return DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY,
                drop_last=False,
                collate_fn=collate_fn,
            )
    else:
        dataset = build_dataset(data_root, split, config)

        if shuffle is None:
            shuffle = (split.lower() == "train")

        loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=(split.lower() == "train"),
            collate_fn=collate_fn,
        )
        return loader     

# def build_dataloader(data_root, split, config, shuffle=None):
#     dataset = build_dataset(data_root, split, config)
    
#     if split.lower() == "train":
#         # 给Rural样本更高权重
#         weights = []
#         for ds in dataset.datasets:
#             scene_weight = 2.0 if ds.scene == "Rural" else 1.0
#             weights.extend([scene_weight] * len(ds))
        
#         sampler = WeightedRandomSampler(
#             weights=weights,
#             num_samples=len(weights),
#             replacement=True,
#         )
#         return DataLoader(
#             dataset,
#             batch_size=config.BATCH_SIZE,
#             sampler=sampler,          # 用sampler替换shuffle
#             num_workers=config.NUM_WORKERS,
#             pin_memory=config.PIN_MEMORY,
#             drop_last=True,
#             collate_fn=collate_fn,
#         )
#     else:
#         # 验证和测试不需要过采样
#         return DataLoader(
#             dataset,
#             batch_size=config.BATCH_SIZE,
#             shuffle=False,
#             num_workers=config.NUM_WORKERS,
#             pin_memory=config.PIN_MEMORY,
#             drop_last=False,
#             collate_fn=collate_fn,
#         )


def collate_fn(batch):
    """自定义collate，确保batch中所有字段正确堆叠"""
    images = torch.stack([b["image"] for b in batch])
    scenes = torch.tensor([b["scene"] for b in batch], dtype=torch.long)
    image_names = [b["image_name"] for b in batch]

    result = {
        "image": images,
        "scene": scenes,
        "image_name": image_names,
    }

    if "mask" in batch[0]:
        masks = torch.stack([b["mask"] for b in batch])
        result["mask"] = masks

    return result
