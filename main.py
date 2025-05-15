import os
import argparse
import torch
from datetime import datetime, timedelta  # Import timedelta

from config import Config
from train import train_and_eval
from distributed_utils import spawn_workers
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='GRAFT for MS-COCO')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_path', type=str, help='Path to MS-COCO dataset')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained weights')
    parser.add_argument('--batch_size', type=int, help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--single_gpu', action='store_true', help='Force single GPU training')
    parser.add_argument('--no_graph', action='store_true', help='Disable graph components')
    args = parser.parse_args()

    # Create config
    config = Config()

    # Update config from command line arguments
    if args.data_path:
        config.DATA_ROOT = args.data_path
        config.TRAIN_ANN = os.path.join(config.DATA_ROOT, 'annotations/instances_train2017.json')
        config.VAL_ANN = os.path.join(config.DATA_ROOT, 'annotations/instances_val2017.json')
        config.TRAIN_IMG_DIR = os.path.join(config.DATA_ROOT, 'train2017')
        config.VAL_IMG_DIR = os.path.join(config.DATA_ROOT, 'val2017')

    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir

    if args.pretrained_weights:
        config.PRETRAINED_WEIGHTS = args.pretrained_weights

    if args.batch_size:
        config.BATCH_SIZE = args.batch_size

    if args.num_epochs:
        config.NUM_EPOCHS = args.num_epochs

    if args.no_graph:
        config.ENABLE_GRAPH_COMPONENTS = False
        print("Graph components disabled for debugging")

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Set random seed for reproducibility
    set_seed(config.SEED)

    # Set NCCL timeout - using timedelta directly
    torch.distributed.init_process_group_timeout = timedelta(minutes=config.TIMEOUT_MINUTES)

    # Check if we have multiple GPUs
    if torch.cuda.is_available() and not args.single_gpu:
        if torch.cuda.device_count() > 1:
            config.NUM_GPUS = min(torch.cuda.device_count(), config.NUM_GPUS)
            print(f"Using {config.NUM_GPUS} GPUs for distributed training")

            # Spawn multiple processes
            try:
                spawn_workers(train_and_eval, config.NUM_GPUS, config)
            except Exception as e:
                print(f"Distributed training failed with error: {e}")
                print("Falling back to single GPU training...")
                config.NUM_GPUS = 1
                train_and_eval(0, 1, config)
        else:
            # Single GPU training
            config.NUM_GPUS = 1
            train_and_eval(0, 1, config)
    else:
        # CPU training or forced single GPU
        config.NUM_GPUS = 1 if torch.cuda.is_available() else 0
        train_and_eval(0, 1, config)


if __name__ == '__main__':
    main()

