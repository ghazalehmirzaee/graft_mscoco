import os
import torch
from datetime import datetime
import json


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
    NUM_WORKERS = 1  # Reduced to minimize memory pressure
    BATCH_SIZE = 4  # Reduced from 16 to prevent memory issues

    # NCCL parameters
    NCCL_DEBUG = "INFO"
    NCCL_SOCKET_IFNAME = "^lo,docker"
    NCCL_IB_DISABLE = "1"  # Disable InfiniBand
    NCCL_P2P_DISABLE = "1"  # Disable P2P explicitly

    # Training parameters
    NUM_EPOCHS = 1
    BASE_LR = 1e-4  # Reduced learning rate
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 1
    MIN_LR = 1e-6
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
    LOG_INTERVAL = 10  # More frequent logging
    EVAL_INTERVAL = 1
    CKPT_INTERVAL = 1

    # Random seed for reproducibility
    SEED = 42

    # Graph complexity management
    ENABLE_GRAPH_COMPONENTS = True  # Toggle to disable graph components during debugging
    SIMPLIFIED_GRAPH_MODE = True  # Use simplified graph operations
    TIMEOUT_MINUTES = 20  # Increase timeout for operations

    # Dataset subset parameters
    USE_SUBSET = True  # Whether to use a subset of MS-COCO
    SUBSET_SIZE = 5000  # Number of images in train subset
    SUBSET_VAL_SIZE = 1000  # Number of images in validation subset
    SUBSET_BALANCED = True  # Whether to ensure class balance in subset

    # Computational complexity tracking
    TRACK_COMPLEXITY = True  # Whether to track computational complexity
    COMPLEXITY_LOG_INTERVAL = 50  # How often to log complexity metrics

    # Metrics saving
    SAVE_METRICS = True  # Whether to save metrics to JSON
    METRICS_DIR = None  # Will be set in __init__

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.METRICS_DIR = os.path.join(self.OUTPUT_DIR, 'metrics')
        os.makedirs(self.METRICS_DIR, exist_ok=True)

        # Save config to JSON for reproducibility
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('__') and not callable(v)}
        with open(os.path.join(self.OUTPUT_DIR, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)

        # Adapt batch size if not enough GPU memory
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 20e9:  # < 20GB
            print(
                f"Warning: Reducing batch size from {self.BATCH_SIZE} to {self.BATCH_SIZE // 2} due to GPU memory constraints")
            self.BATCH_SIZE = self.BATCH_SIZE // 2

        # Set NCCL environment variables
        os.environ["NCCL_DEBUG"] = self.NCCL_DEBUG
        os.environ["NCCL_SOCKET_IFNAME"] = self.NCCL_SOCKET_IFNAME
        os.environ["NCCL_IB_DISABLE"] = self.NCCL_IB_DISABLE
        os.environ["NCCL_P2P_DISABLE"] = self.NCCL_P2P_DISABLE
        os.environ["TORCH_DISTRIBUTED_DETAIL"] = "DEBUG"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Enable async error handling

        