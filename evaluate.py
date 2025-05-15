import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import Config
from datasets import CocoMultiLabelDataset, get_val_transform, custom_collate_fn
from models.graft import GRAFT
from utils import compute_metrics, visualize_predictions, visualize_attention, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GRAFT model on MS-COCO')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, help='Path to MS-COCO dataset')
    parser.add_argument('--output_dir', type=str, help='Output directory for visualizations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    return parser.parse_args()


def evaluate(config, checkpoint_path, visualize=False):
    """
    Evaluate model on validation set.

    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        visualize: Whether to create visualizations
    """
    # Set random seed for reproducibility
    set_seed(config.SEED)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset
    val_dataset = CocoMultiLabelDataset(
        ann_file=config.VAL_ANN,
        img_dir=config.VAL_IMG_DIR,
        transform=get_val_transform(),
        is_train=False
    )

    # Get class names
    class_names = val_dataset.coco.loadCats(val_dataset.cat_ids)
    class_names = [cat['name'] for cat in class_names]

    # Create dataloader with custom collate function
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=custom_collate_fn  # Use custom collate function
    )

    # Create model
    model = GRAFT(config)
    model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")

    # FIXED: Load checkpoint with weights_only=False
    try:
        import argparse
        from torch.serialization import add_safe_globals

        # Add argparse.Namespace to the safelist
        with add_safe_globals([argparse.Namespace]):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"First loading attempt failed with error: {str(e)}")
        print("Trying with weights_only=False for backward compatibility...")
        # Fall back to the less secure option if needed
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load model weights
    model.load_state_dict(checkpoint['model'])

    # Evaluate
    model.eval()

    # Collect all predictions and targets
    all_logits = []
    all_targets = []
    all_images = []

    print("Evaluating model...")
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            # Get inputs and labels
            images = sample['image'].to(device, non_blocking=True)
            labels = sample['labels'].to(device, non_blocking=True)
            bboxes = sample['bboxes']  # Already a list of tensors
            bbox_cat_idxs = sample['bbox_cat_idxs']  # Already a list of tensors

            # Move bbox data to device
            bboxes = [b.to(device, non_blocking=True) for b in bboxes]
            bbox_cat_idxs = [b.to(device, non_blocking=True) for b in bbox_cat_idxs]

            # Forward pass
            outputs = model(images, labels, bboxes, bbox_cat_idxs)
            logits = outputs['logits']

            # Collect predictions and targets
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

            # Save images for visualization
            if visualize and i < 5:
                all_images.append(images.cpu())

            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(val_loader)} batches")

    # Concatenate all logits and targets
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_metrics(all_logits, all_targets)

    # Print metrics
    print("\nEvaluation Results:")
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"OP: {metrics['OP']:.4f}, OR: {metrics['OR']:.4f}, OF1: {metrics['OF1']:.4f}")
    print(f"CP: {metrics['CP']:.4f}, CR: {metrics['CR']:.4f}, CF1: {metrics['CF1']:.4f}")

    # Print top 5 and bottom 5 categories by AP
    APs = metrics['APs']
    top_indices = np.argsort(APs)[-5:][::-1]
    bottom_indices = np.argsort(APs)[:5]

    print("\nTop 5 Categories:")
    for idx in top_indices:
        print(f"{class_names[idx]}: AP = {APs[idx]:.4f}")

    print("\nBottom 5 Categories:")
    for idx in bottom_indices:
        print(f"{class_names[idx]}: AP = {APs[idx]:.4f}")

    # Create visualizations
    if visualize:
        print("\nCreating visualizations...")

        # Create output directory
        os.makedirs(os.path.join(config.OUTPUT_DIR, 'visualizations'), exist_ok=True)

        # Visualize predictions
        all_images = torch.cat(all_images, dim=0)
        sample_indices = np.random.choice(len(all_images), min(10, len(all_images)), replace=False)

        for i, idx in enumerate(sample_indices):
            fig = visualize_predictions(
                all_images[idx:idx + 1],
                all_targets[idx:idx + 1],
                all_logits[idx:idx + 1],
                class_names,
                # Use generic class names if actual ones not available
                num_samples=1
            )
            fig.savefig(os.path.join(config.OUTPUT_DIR, 'visualizations', f'prediction_{i}.png'))
            plt.close(fig)

        # Try to visualize attention weights from the model's graph components
        try:
            # Get attention weights from the model
            if hasattr(model, 'cooccurrence_graph'):
                # Visualize co-occurrence matrix
                comatrix = model.cooccurrence_graph.cooccurrence.cpu()
                fig = plt.figure(figsize=(10, 8))
                plt.imshow(comatrix, cmap='viridis')
                plt.colorbar()
                plt.title('Co-occurrence Matrix')
                plt.savefig(os.path.join(config.OUTPUT_DIR, 'visualizations', 'cooccurrence_matrix.png'))
                plt.close(fig)

                # Visualize top co-occurrences
                fig = visualize_attention(comatrix, class_names, num_classes=15)
                fig.savefig(os.path.join(config.OUTPUT_DIR, 'visualizations', 'cooccurrence_attention.png'))
                plt.close(fig)
        except Exception as e:
            print(f"Error visualizing attention weights: {e}")

    return metrics


def main():
    # Parse arguments
    args = parse_args()

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

    if args.batch_size:
        config.BATCH_SIZE = args.batch_size

    if args.num_workers:
        config.NUM_WORKERS = args.num_workers

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Evaluate
    evaluate(config, args.checkpoint, args.visualize)


if __name__ == '__main__':
    main()

