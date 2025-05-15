import os
import torch
from datetime import datetime


class Config:
    # Dataset paths
    DATA_ROOT = '/NewRaidData/ghazal/graft_mscoco/MS-COCO'
    TRAIN_ANN = os.path.join(DATA_ROOT, 'annotations/instances_train2017.json')
    VAL_ANN = os.path.join(DATA_ROOT, 'annotations/instances_val2017.json')
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'train2017')
    VAL_IMG_DIR = os.path.join(DATA_ROOT, 'val2017')

    # Model parameters
    PRETRAINED_WEIGHTS = "/NewRaidData/ghazal/graft_mscoco/graft_mscoco/weights/vit-base_X-rays_0.5M_mae.pth"
    NUM_CLASSES = 80

    # Graph components parameters
    COOCCURRENCE_WEIGHT = 0.3
    SPATIAL_WEIGHT = 0.3
    VISUAL_WEIGHT = 0.4
    SPATIAL_SCALES = [10, 20]  # Simplified from [5, 15, 25]
    SIMILARITY_THRESHOLD = 0.5  # For visual relationship graph

    # Distributed training parameters
    NUM_GPUS = 8
    NUM_WORKERS = 4  # Per GPU
    BATCH_SIZE = 32  # Per GPU, effective batch size = BATCH_SIZE * NUM_GPUS

    # Training parameters
    NUM_EPOCHS = 3
    BASE_LR = 2.5e-4
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-5
    DROP_PATH = 0.2

    # Multi-component loss weights
    WBCE_WEIGHT = 0.5
    FL_WEIGHT = 0.25
    ASL_WEIGHT = 0.25

    # Focal loss parameters
    FL_GAMMA = 2.0

    # Asymmetric loss parameters
    ASL_BETA = 4.0

    # Logging and checkpointing
    OUTPUT_DIR = os.path.join("outputs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    LOG_INTERVAL = 50
    EVAL_INTERVAL = 1
    CKPT_INTERVAL = 5

    # Random seed for reproducibility
    SEED = 42

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # Adapt batch size if not enough GPU memory
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 20e9:  # < 20GB
            print(
                f"Warning: Reducing batch size from {self.BATCH_SIZE} to {self.BATCH_SIZE // 2} due to GPU memory constraints")
            self.BATCH_SIZE = self.BATCH_SIZE // 2

