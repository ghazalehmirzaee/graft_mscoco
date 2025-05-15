import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches


def denormalize_image(img_tensor):
    """
    Denormalize image tensor for visualization.

    Args:
        img_tensor: Image tensor (C, H, W) with ImageNet normalization

    Returns:
        Denormalized numpy array (H, W, C)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img_tensor * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    return img


def visualize_graph_attention(attention_weights, class_names, save_path=None, top_k=15):
    """
    Visualize graph attention weights.

    Args:
        attention_weights: Attention weights tensor (num_classes, num_classes)
        class_names: List of class names
        save_path: Path to save the visualization
        top_k: Number of top classes to visualize

    Returns:
        Matplotlib figure
    """
    # Get top classes based on attention
    num_classes = attention_weights.shape[0]
    avg_attn = attention_weights.mean(dim=1)
    top_indices = torch.argsort(avg_attn, descending=True)[:top_k].cpu().numpy()

    # Get class names for top classes
    top_classes = [class_names[i] for i in top_indices]

    # Create attention matrix for visualization
    sub_attn = attention_weights[top_indices][:, top_indices].cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(sub_attn, cmap='viridis')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Add ticks and labels
    ax.set_xticks(np.arange(len(top_classes)))
    ax.set_yticks(np.arange(len(top_classes)))
    ax.set_xticklabels(top_classes, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(top_classes)

    # Add title
    plt.title("Graph Attention Weights")

    # Add grid lines
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(sub_attn.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(sub_attn.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Loop over data dimensions and create text annotations
    for i in range(len(top_classes)):
        for j in range(len(top_classes)):
            text = ax.text(j, i, f"{sub_attn[i, j]:.2f}",
                           ha="center", va="center", color="w" if sub_attn[i, j] > 0.5 else "black")

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_prediction_with_bbox(image, target, prediction, bboxes, bbox_cat_idxs, class_names, save_path=None):
    """
    Visualize model prediction with bounding boxes.

    Args:
        image: Image tensor (C, H, W)
        target: Target labels (num_classes)
        prediction: Predicted logits (num_classes)
        bboxes: Bounding boxes (N, 4) in format [x, y, width, height]
        bbox_cat_idxs: Category indices for bounding boxes (N)
        class_names: List of class names
        save_path: Path to save the visualization

    Returns:
        Matplotlib figure
    """
    # Denormalize image
    img = denormalize_image(image)

    # Get prediction probabilities
    probs = torch.sigmoid(prediction).cpu().numpy()

    # Get ground truth and prediction classes
    gt_classes = torch.where(target > 0.5)[0].cpu().numpy()
    pred_classes = torch.where(probs > 0.5)[0]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot image with ground truth boxes
    ax1.imshow(img)
    ax1.set_title("Ground Truth")
    ax1.axis('off')

    # Add ground truth boxes
    for i, bbox in enumerate(bboxes):
        if i < len(bbox_cat_idxs):
            box = bbox.cpu().numpy()
            category_idx = bbox_cat_idxs[i].item()

            if category_idx < len(class_names):
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2], box[3],
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax1.add_patch(rect)
                ax1.text(
                    box[0], box[1] - 5,
                    class_names[category_idx],
                    color='white', fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.5)
                )

    # Plot image with predictions
    ax2.imshow(img)
    ax2.set_title("Prediction")
    ax2.axis('off')

    # Add text with predictions
    y_pos = 10
    for cls_idx in pred_classes:
        if cls_idx < len(class_names):
            ax2.text(
                10, y_pos,
                f"{class_names[cls_idx]}: {probs[cls_idx]:.2f}",
                color='white', fontsize=12,
                bbox=dict(facecolor='blue', alpha=0.5)
            )
            y_pos += 20

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_cooccurrence_matrix(cooccurrence_matrix, class_names, save_path=None, top_k=20):
    """
    Visualize co-occurrence matrix.

    Args:
        cooccurrence_matrix: Co-occurrence matrix (num_classes, num_classes)
        class_names: List of class names
        save_path: Path to save the visualization
        top_k: Number of top classes to visualize

    Returns:
        Matplotlib figure
    """
    # Get top classes based on co-occurrence
    num_classes = cooccurrence_matrix.shape[0]
    total_cooc = cooccurrence_matrix.sum(dim=1)
    top_indices = torch.argsort(total_cooc, descending=True)[:top_k].cpu().numpy()

    # Get class names for top classes
    top_classes = [class_names[i] for i in top_indices]

    # Create co-occurrence matrix for visualization
    sub_matrix = cooccurrence_matrix[top_indices][:, top_indices].cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(sub_matrix, cmap='viridis')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Add ticks and labels
    ax.set_xticks(np.arange(len(top_classes)))
    ax.set_yticks(np.arange(len(top_classes)))
    ax.set_xticklabels(top_classes, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(top_classes)

    # Add title
    plt.title("Label Co-occurrence Matrix")

    # Add grid lines
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(sub_matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(sub_matrix.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_per_class_performance_plot(APs, class_names, save_path=None):
    """
    Create bar plot of per-class Average Precision.

    Args:
        APs: List of Average Precision values for each class
        class_names: List of class names
        save_path: Path to save the visualization

    Returns:
        Matplotlib figure
    """
    # Sort classes by AP
    sorted_indices = np.argsort(APs)
    sorted_APs = [APs[i] for i in sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))

    # Create bar plot
    bars = ax.barh(range(len(sorted_names)), sorted_APs, align='center')

    # Add class labels
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)

    # Add values to bars
    for i, v in enumerate(sorted_APs):
        ax.text(v + 0.01, i, f"{v:.2f}", va='center')

    # Add labels and title
    ax.set_xlabel('Average Precision')
    ax.set_title('Per-class Performance')

    # Adjust layout
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_confusion_matrix(all_targets, all_preds, class_names, save_path=None, threshold=0.5, top_k=15):
    """
    Create confusion matrix for multi-label classification.

    Args:
        all_targets: Target labels (num_samples, num_classes)
        all_preds: Predicted probabilities (num_samples, num_classes)
        class_names: List of class names
        save_path: Path to save the visualization
        threshold: Threshold for positive prediction
        top_k: Number of top classes to visualize

    Returns:
        Matplotlib figure
    """
    # Convert predictions to binary
    binary_preds = (all_preds > threshold).astype(np.int32)

    # Calculate class frequencies
    class_freq = all_targets.sum(axis=0)
    top_indices = np.argsort(class_freq)[-top_k:]

    # Get top class names
    top_classes = [class_names[i] for i in top_indices]

    # Initialize confusion matrix (TP, FP, TN, FN)
    confusion = np.zeros((4, len(top_indices)))

    for i, idx in enumerate(top_indices):
        targets = all_targets[:, idx]
        preds = binary_preds[:, idx]

        # True Positives
        confusion[0, i] = np.sum((targets == 1) & (preds == 1))
        # False Positives
        confusion[1, i] = np.sum((targets == 0) & (preds == 1))
        # True Negatives
        confusion[2, i] = np.sum((targets == 0) & (preds == 0))
        # False Negatives
        confusion[3, i] = np.sum((targets == 1) & (preds == 0))

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))

    # Define colors
    colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e']
    labels = ['True Positives', 'False Positives', 'True Negatives', 'False Negatives']

    # Create stacked bar plot
    bottoms = np.zeros(len(top_indices))
    for i in range(4):
        ax.bar(top_classes, confusion[i], bottom=bottoms, label=labels[i], color=colors[i])
        bottoms += confusion[i]

    # Add legend and labels
    ax.set_ylabel('Count')
    ax.set_title('Confusion Matrix per Class')
    ax.legend()

    # Rotate class labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig



