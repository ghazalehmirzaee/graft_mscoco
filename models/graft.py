import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vit_backbone import ViTBackbone
from models.graph_modules import CooccurrenceGraph, SpatialGraph, VisualGraph, GraphFusionModule


class MultiLabelLoss(nn.Module):
    """
    Multi-component loss for multi-label classification.
    Combines Weighted BCE, Focal Loss, and Asymmetric Loss.
    """

    def __init__(self, num_classes, class_weights=None, gamma=2.0, beta=4.0,
                 wbce_weight=0.5, fl_weight=0.25, asl_weight=0.25):
        """
        Args:
            num_classes: Number of classes
            class_weights: Class weights for WBCE
            gamma: Focal loss gamma parameter
            beta: Asymmetric loss beta parameter
            wbce_weight: Weight for WBCE component
            fl_weight: Weight for Focal Loss component
            asl_weight: Weight for Asymmetric Loss component
        """
        super().__init__()

        self.num_classes = num_classes
        self.gamma = gamma
        self.beta = beta

        # Component weights
        self.wbce_weight = wbce_weight
        self.fl_weight = fl_weight
        self.asl_weight = asl_weight

        # Register class weights or use default
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', torch.ones(num_classes))

    def forward(self, logits, targets):
        """
        Forward pass computing the multi-component loss.

        Args:
            logits: Predicted logits (B, num_classes)
            targets: Target labels (B, num_classes)

        Returns:
            Combined loss value
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Weighted Binary Cross Entropy
        wbce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, weight=self.class_weights, reduction='none'
        )

        # Focal Loss
        pt = targets * probs + (1 - targets) * (1 - probs)
        fl_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        ) * (1 - pt).pow(self.gamma)

        # Asymmetric Loss
        asl_pos = targets * F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        asl_neg = (1 - targets) * F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        ) * probs.pow(self.beta)
        asl_loss = asl_pos + asl_neg

        # Combine losses
        combined_loss = (
                self.wbce_weight * wbce_loss +
                self.fl_weight * fl_loss +
                self.asl_weight * asl_loss
        )

        # Mean reduction
        return combined_loss.mean()


class GRAFT(nn.Module):
    """
    Graph-Augmented Framework with Vision Transformers for multi-label classification.
    """

    def __init__(self, config, cooccurrence_matrix=None):
        """
        Args:
            config: Configuration object
            cooccurrence_matrix: Pre-computed co-occurrence matrix
        """
        super().__init__()

        # Model components
        self.config = config
        self.num_classes = config.NUM_CLASSES
        self.hidden_dim = 256  # Hidden dimension for graph components

        # Vision Transformer backbone
        self.backbone = ViTBackbone(
            pretrained_path=config.PRETRAINED_WEIGHTS,
            num_classes=self.num_classes,
            freeze_layers=8  # Freeze early layers
        )

        # Get feature dimension from backbone
        self.feature_dim = self.backbone.embed_dim

        # Feature projector
        self.feature_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # Graph components
        self.cooccurrence_graph = CooccurrenceGraph(
            num_classes=self.num_classes,
            cooccurrence_matrix=cooccurrence_matrix,
            hidden_dim=self.hidden_dim
        )

        self.spatial_graph = SpatialGraph(
            num_classes=self.num_classes,
            scales=config.SPATIAL_SCALES,
            hidden_dim=self.hidden_dim
        )

        self.visual_graph = VisualGraph(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            similarity_threshold=config.SIMILARITY_THRESHOLD
        )

        # Graph fusion module
        self.fusion_module = GraphFusionModule(
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim
        )

        # Final classifier - FIXED: Output 1 value per class instead of num_classes
        self.classifier = nn.Linear(self.hidden_dim + self.feature_dim, 1)

        # Loss function
        self.criterion = MultiLabelLoss(
            num_classes=self.num_classes,
            gamma=config.FL_GAMMA,
            beta=config.ASL_BETA,
            wbce_weight=config.WBCE_WEIGHT,
            fl_weight=config.FL_WEIGHT,
            asl_weight=config.ASL_WEIGHT
        )

    def update_cooccurrence(self, cooccurrence_matrix):
        """Update the co-occurrence graph with a new matrix."""
        self.cooccurrence_graph.update_cooccurrence(cooccurrence_matrix)

    def forward(self, images, labels=None, bboxes=None, bbox_classes=None):
        """
        Forward pass through the GRAFT model.

        Args:
            images: Input images (B, C, H, W)
            labels: Ground-truth labels (B, num_classes) for loss computation
            bboxes: List of bounding boxes tensors for each image, where each tensor has shape (num_boxes_i, 4)
                    and format [x, y, width, height]
            bbox_classes: List of class indices tensors for each image, where each tensor has shape (num_boxes_i)

        Returns:
            Dict containing model outputs and loss if labels are provided
        """
        # Get features from backbone
        backbone_out = self.backbone(images)
        logits = backbone_out['logits']
        features = backbone_out['features']

        # Project features to hidden dimension
        node_features = self.feature_projector(features).unsqueeze(1).expand(-1, self.num_classes, -1)

        # Process graph components
        cooccurrence_feat = self.cooccurrence_graph(node_features, labels)
        spatial_feat = self.spatial_graph(node_features, bboxes, bbox_classes)
        visual_feat = self.visual_graph(node_features, features, labels)

        # Fuse graph features
        fused_graph_feat = self.fusion_module(cooccurrence_feat, spatial_feat, visual_feat)

        # Reshape graph features
        batch_size = features.shape[0]
        graph_feat_flat = fused_graph_feat.view(batch_size, self.num_classes, self.hidden_dim)

        # Combine with visual features for final prediction
        combined_features = torch.cat([
            features.unsqueeze(1).expand(-1, self.num_classes, -1),
            graph_feat_flat
        ], dim=2)

        # Apply final classifier - FIXED: Handle the single output value per label
        refined_logits = self.classifier(combined_features.view(batch_size * self.num_classes, -1))
        # Squeeze the last dimension and reshape to [batch_size, num_classes]
        refined_logits = refined_logits.squeeze(-1).view(batch_size, self.num_classes)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Combine losses from initial and refined predictions
            initial_loss = self.criterion(logits, labels)
            refined_loss = self.criterion(refined_logits, labels)
            loss = 0.4 * initial_loss + 0.6 * refined_loss

        return {
            'logits': refined_logits,
            'features': features,
            'loss': loss
        }

    