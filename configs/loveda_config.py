import os

DATA_ROOT = "/workspace/LoveDA"
OUTPUT_DIR = "sa-aspp_prototype_loveda_outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

NUM_CLASSES = 7
IGNORE_INDEX = 255
IMAGE_SIZE = 512
SCENES = ["Urban", "Rural"]
CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Forest", "Agriculture"]
CLASS_COLORS = [
    (255, 255, 255), (255, 0, 0), (255, 255, 0),
    (0, 0, 255), (159, 129, 183), (0, 255, 0), (255, 195, 128),
]

MODEL = "loveda"
# baseline | sa_aspp | prototype | balanced | sa_aspp_prototype | full
ABLATION = "full"
USE_SA_ASPP = True
USE_PROTOTYPE_HEAD = True
USE_CLASS_BALANCE = True

BACKBONE = "resnet50"
PRETRAINED = True
OUTPUT_STRIDE = 16
ASPP_DILATIONS = [1, 6, 12, 18]
ASPP_OUT_CHANNELS = 256
DECODER_CHANNELS = 256
LOW_LEVEL_CHANNELS = 48
NUM_SCENES = 2

EPOCHS = 100
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
OPTIMIZER = "adamw"
LR = 6e-4
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
LR_SCHEDULER = "poly"
LR_POWER = 0.9
WARMUP_EPOCHS = 5

LOSS_SEG_WEIGHT = 1.0
LOSS_AUX_WEIGHT = 0.4
LOSS_BSM_WEIGHT = 0.2
LOSS_SCENE_WEIGHT = 0.1

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

SEED = 42
SAVE_FREQ = 50
LOG_FREQ = 50
USE_AMP = True
DEVICE = "cuda"
