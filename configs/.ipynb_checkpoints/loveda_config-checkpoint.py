# configs/loveda_config.py
# LoveDA-DeepLab 全局配置文件

import os

# ─────────────────────────────────────────────
#  路径配置
# ─────────────────────────────────────────────
DATA_ROOT = "/workspace/LoveDA"          # ← 修改为你的数据集路径
OUTPUT_DIR = "sa-aspp_prototype_loveda_outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ─────────────────────────────────────────────
#  数据集配置
# ─────────────────────────────────────────────
NUM_CLASSES = 7          # LoveDA: 背景(0)+6类地物
IGNORE_INDEX = 255       # mask中值为7的像素映射到255后忽略
IMAGE_SIZE = 512         # 训练时裁剪尺寸
SCENES = ["Urban", "Rural"]

# LoveDA类别名称（0-indexed）
CLASS_NAMES = [
    "Background",   # 0
    "Building",     # 1
    "Road",         # 2
    "Water",        # 3
    "Barren",       # 4
    "Forest",       # 5
    "Agriculture",  # 6
]

# 类别颜色（用于可视化）
CLASS_COLORS = [
    (255, 255, 255),  # Background - 白
    (255, 0,   0  ),  # Building   - 红
    (255, 255, 0  ),  # Road       - 黄
    (0,   0,   255),  # Water      - 蓝
    (159, 129, 183),  # Barren     - 紫
    (0,   255, 0  ),  # Forest     - 绿
    (255, 195, 128),  # Agriculture- 橙
]

# ─────────────────────────────────────────────
#  模型配置
# ─────────────────────────────────────────────
MODEL = "loveda"
BACKBONE = "resnet50"        # resnet50 | resnet101
PRETRAINED = True            # 使用ImageNet预训练权重
OUTPUT_STRIDE = 16           # 骨干网络输出步长 (8 or 16)

# SA-ASPP配置
ASPP_DILATIONS = [1, 6, 12, 18]   # 空洞率列表（1对应1×1卷积）
# ASPP_OUT_CHANNELS = 256
ASPP_OUT_CHANNELS = 256

# 背景抑制模块配置
BSM_ALPHA = 0.5              # 背景抑制强度
BSM_HIDDEN = 64             # 判别器隐藏层通道数

# 场景自适应模块配置
# SAM_PROTO_DIM = 256
SAM_PROTO_DIM = 128          # 场景原型维度
NUM_SCENES = 2               # 场景数量 (Urban / Rural)

# 解码器配置
# DECODER_CHANNELS = 256
DECODER_CHANNELS = 256
LOW_LEVEL_CHANNELS = 48      # 低层特征压缩后通道数

# ─────────────────────────────────────────────
#  训练配置
# ─────────────────────────────────────────────
EPOCHS = 100
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True

# 优化器
OPTIMIZER = "adamw"          # sgd | adamw
LR = 6e-4
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9               # SGD only

# 学习率调度
LR_SCHEDULER = "poly"        # poly | cosine | step
LR_POWER = 0.9               # poly调度的幂次
WARMUP_EPOCHS = 5

# 损失权重
LOSS_SEG_WEIGHT = 1.0        # 主分割损失权重
LOSS_AUX_WEIGHT = 0.4        # 辅助分割损失权重
LOSS_BSM_WEIGHT = 0.2        # 背景抑制辅助损失权重
LOSS_SCENE_WEIGHT = 0.1      # 场景分类辅助损失权重

# ─────────────────────────────────────────────
#  数据增强配置
# ─────────────────────────────────────────────
TRAIN_AUGMENT = dict(
    random_resize_crop=True,
    scale_range=(0.5, 2.0),
    crop_size=IMAGE_SIZE,
    horizontal_flip=True,
    vertical_flip=True,
    random_rotate=True,
    rotate_degrees=90,
    color_jitter=True,
    color_jitter_params=dict(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    normalize=True,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

VAL_AUGMENT = dict(
    resize=True,
    resize_size=(IMAGE_SIZE, IMAGE_SIZE),
    normalize=True,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# ─────────────────────────────────────────────
#  其他配置
# ─────────────────────────────────────────────
SEED = 42
SAVE_FREQ = 50               # 每N个epoch保存一次checkpoint
LOG_FREQ = 50               # 每N个iteration打印一次log
USE_AMP = True              # 混合精度训练
DEVICE = "cuda"             # cuda | cpu
