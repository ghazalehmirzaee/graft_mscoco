import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(all_logits, all_targets):
    """
    Compute evaluation metrics for multi-label classification.

    Args:
        all_logits: Concatenated logits from all batches
        all_targets: Concatenated targets from all batches

    Returns:
        Dict containing metrics
    """
    all_probs = torch.sigmoid(all_logits).cpu().numpy()
    all_targets = all_targets.cpu().numpy()

    # Class-wise metrics
    APs = []
    precisions = []
    recalls = []
    f1_scores = []

    # Calculate per-class metrics
    for c in range(all_targets.shape[1]):
        AP = average_precision_score(all_targets[:, c], all_probs[:, c])
        APs.append(AP)

        # Calculate precision, recall, f1 at threshold 0.5
        y_true = all_targets[:, c]
        y_pred = (all_probs[:, c] >= 0.5).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Calculate mAP
    mAP = np.mean(APs)

    # Calculate Overall Precision (OP), Recall (OR), F1 (OF1)
    OP = np.mean(precisions)
    OR = np.mean(recalls)
    OF1 = 2 * OP * OR / (OP + OR) if (OP + OR) > 0 else 0

    # Calculate Class-wise Precision (CP), Recall (CR), F1 (CF1)
    y_true = all_targets
    y_pred = (all_probs >= 0.5).astype(int)

    CP = 0
    CR = 0
    N = 0

    for i in range(y_true.shape[0]):
        TP_i = np.sum((y_true[i, :] == 1) & (y_pred[i, :] == 1))
        FP_i = np.sum((y_true[i, :] == 0) & (y_pred[i, :] == 1))
        FN_i = np.sum((y_true[i, :] == 1) & (y_pred[i, :] == 0))

        if (TP_i + FP_i) > 0:
            CP += TP_i / (TP_i + FP_i)
            N += 1

        if (TP_i + FN_i) > 0:
            CR += TP_i / (TP_i + FN_i)

    CP = CP / N if N > 0 else 0
    CR = CR / N if N > 0 else 0
    CF1 = 2 * CP * CR / (CP + CR) if (CP + CR) > 0 else 0

    return {
        'mAP': mAP,
        'APs': APs,
        'OP': OP,
        'OR': OR,
        'OF1': OF1,
        'CP': CP,
        'CR': CR,
        'CF1': CF1
    }


def load_checkpoint(model, optimizer, scheduler, filename):
    """
    Load model checkpoint.

    Args:
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        filename: Checkpoint filename

    Returns:
        Epoch number and best mAP value
    """
    if not os.path.isfile(filename):
        return 0, 0.0

    print(f"Loading checkpoint from {filename}")
    # Fixed: Set weights_only=False to allow loading of argparse.Namespace objects
    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if 'scheduler' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return checkpoint.get('epoch', 0), checkpoint.get('best_map', 0.0)


def save_checkpoint(state, is_best, output_dir):
    """
    Save model checkpoint.

    Args:
        state: Checkpoint state dict
        is_best: Whether this is the best model so far
        output_dir: Directory to save checkpoints
    """
    filename = os.path.join(output_dir, 'checkpoint.pth')
    # Save using backward-compatible format
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        best_filename = os.path.join(output_dir, 'model_best.pth')
        torch.save(state, best_filename, _use_new_zipfile_serialization=False)


def visualize_predictions(images, targets, predictions, class_names, num_samples=5):
    """
    Visualize model predictions for a few samples.

    Args:
        images: Batch of images
        targets: Ground-truth labels
        predictions: Model predictions
        class_names: List of class names
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, images.size(0))
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        # Get the image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        # Get predictions and targets
        pred_probs = torch.sigmoid(predictions[i]).cpu().numpy()
        tgt = targets[i].cpu().numpy()

        # Find top predictions and ground truth classes
        top_pred_idx = np.where(pred_probs > 0.5)[0]
        gt_idx = np.where(tgt > 0.5)[0]

        # Sort predictions by confidence
        top_pred_idx = top_pred_idx[np.argsort(-pred_probs[top_pred_idx])]

        # Get class names
        top_pred_names = [f"{class_names[idx]} ({pred_probs[idx]:.2f})" for idx in top_pred_idx]
        gt_names = [class_names[idx] for idx in gt_idx]

        # Display the image
        axes[i].imshow(img)
        axes[i].axis('off')

        # Display predictions and ground truth
        pred_text = "Predictions: " + ", ".join(top_pred_names)
        gt_text = "Ground Truth: " + ", ".join(gt_names)

        axes[i].set_title(f"{pred_text}\n{gt_text}")

    plt.tight_layout()
    return fig


def visualize_attention(attention_weights, class_names, num_classes=10):
    """
    Visualize attention weights between label nodes.

    Args:
        attention_weights: Attention weights matrix (num_classes, num_classes)
        class_names: List of class names
        num_classes: Number of top classes to visualize
    """
    # Use only top num_classes for visualization
    top_classes = min(num_classes, len(class_names))

    # Select top classes based on average attention weight
    avg_weights = attention_weights.mean(dim=1)
    top_indices = torch.argsort(avg_weights, descending=True)[:top_classes]

    # Extract submatrix for visualization
    vis_weights = attention_weights[top_indices][:, top_indices]
    vis_names = [class_names[i] for i in top_indices]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(vis_weights.cpu().numpy(), cmap='viridis')

    # Add labels
    ax.set_xticks(np.arange(len(vis_names)))
    ax.set_yticks(np.arange(len(vis_names)))
    ax.set_xticklabels(vis_names, rotation=45, ha='right')
    ax.set_yticklabels(vis_names)

    # Add colorbar
    plt.colorbar(im)

    # Add title
    plt.title('Attention Weights Between Label Nodes')

    # Adjust layout
    plt.tight_layout()

    return fig

