# quick_check.py
import sys
sys.path.insert(0, '.')

from datasets.loveda_dataset import build_dataset
import configs.loveda_config as config

# 检查验证集组成
print("检查验证集目录结构:")
import os
val_dir = os.path.join(config.DATA_ROOT, "Val")
if os.path.exists(val_dir):
    print(f"  {val_dir}/")
    for scene in os.listdir(val_dir):
        print(f"    └─ {scene}/")
        mask_dir = os.path.join(val_dir, scene, "masks")
        if os.path.exists(mask_dir):
            masks = os.listdir(mask_dir)[:3]
            print(f"        masks: {masks}")
else:
    print(f"  验证集目录不存在: {val_dir}")

# 检查实际加载的数据集
print("\n加载验证集...")
val_dataset = build_dataset(config.DATA_ROOT, "Val", config)
print(f"验证集总样本数: {len(val_dataset)}")

# 检查场景分布
scene_counts = {0: 0, 1: 0}  # 0=Urban, 1=Rural
for i in range(min(100, len(val_dataset))):
    sample = val_dataset[i]
    scene_counts[sample['scene']] += 1

print(f"场景分布 (前100个样本):")
print(f"  Urban: {scene_counts[0]} 样本")
print(f"  Rural: {scene_counts[1]} 样本")