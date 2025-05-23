import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CooccurrenceGraph(nn.Module):
    """
    Statistical Co-occurrence Graph Module.
    Captures statistical relationships between labels.
    """

    def __init__(self, num_classes, cooccurrence_matrix=None, hidden_dim=256, simplified=False):
        """
        Args:
            num_classes: Number of label classes
            cooccurrence_matrix: Pre-computed co-occurrence matrix (can be None during initialization)
            hidden_dim: Dimension of hidden node embeddings
            simplified: Whether to use simplified operations for less memory usage
        """
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.simplified = simplified

        # Label embeddings
        self.label_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))

        # Graph projection layers
        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

        # Register co-occurrence matrix as buffer (non-trainable)
        if cooccurrence_matrix is not None:
            self.register_buffer('cooccurrence', torch.from_numpy(cooccurrence_matrix).float())
            # Apply adaptive class balancing
            self._apply_class_balancing()
        else:
            self.register_buffer('cooccurrence', torch.eye(num_classes))

    def _apply_class_balancing(self):
        """Apply adaptive class balancing to the co-occurrence matrix."""
        # Get class counts from diagonal of co-occurrence
        class_counts = torch.diag(self.cooccurrence)
        class_counts = torch.clamp(class_counts, min=1e-10)  # Prevent division by zero

        # Calculate balancing weights
        max_count = class_counts.max()
        avg_count = class_counts.mean()

        # Apply adaptive balancing formula - simplified version
        alpha = 0.3
        beta = 0.5

        # Create multiplier for each pair of labels
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:  # Skip diagonal elements
                    # Calculate balance factor based on frequency of rare class
                    rare_idx = i if class_counts[i] < class_counts[j] else j
                    ratio = min(class_counts[i], class_counts[j]) / max(class_counts[i], class_counts[j])
                    rare_weight = ((max_count / avg_count) ** alpha) * (ratio ** beta)

                    # Apply correction weight
                    self.cooccurrence[i, j] *= rare_weight

        # Normalize to keep values in reasonable range
        self.cooccurrence = F.normalize(self.cooccurrence, p=1, dim=1)

    def update_cooccurrence(self, new_cooccurrence):
        """Update the co-occurrence matrix."""
        self.cooccurrence.copy_(torch.from_numpy(new_cooccurrence).float())
        self._apply_class_balancing()

    def forward(self, x, labels=None):
        """
        Forward pass through the co-occurrence graph.

        Args:
            x: Node features (batch_size, num_classes, hidden_dim)
            labels: Ground-truth labels for guidance (batch_size, num_classes)

        Returns:
            Updated node features
        """
        batch_size = x.shape[0]
        device = x.device  # Get device from input tensor

        # Use label embeddings as node features if x is None
        if x is None:
            x = self.label_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Simplified processing to reduce memory usage
        if self.simplified:
            # Generate queries from input
            q = self.proj_q(x)  # (batch_size, num_classes, hidden_dim)

            # Direct application of co-occurrence weights
            cooccurrence = self.cooccurrence.to(device)
            # Process batches with less memory by splitting
            outputs = []

            chunk_size = max(1, batch_size // 2)  # Process half batch at a time
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk_x = x[i:end_idx]
                chunk_q = q[i:end_idx]

                # Apply co-occurrence directly
                chunk_out = torch.bmm(cooccurrence.unsqueeze(0).expand(end_idx - i, -1, -1), chunk_x)
                chunk_out = self.proj_out(chunk_out)
                outputs.append(chunk_out)

            return torch.cat(outputs, dim=0)

        # Full processing if not simplified
        # Generate queries, keys, values
        q = self.proj_q(x)  # (batch_size, num_classes, hidden_dim)
        k = self.proj_k(x)  # (batch_size, num_classes, hidden_dim)
        v = self.proj_v(x)  # (batch_size, num_classes, hidden_dim)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (
                    self.hidden_dim ** 0.5)  # (batch_size, num_classes, num_classes)

        # Apply co-occurrence as edge weights
        cooccurrence = self.cooccurrence.to(device)
        scores = scores * cooccurrence.unsqueeze(0)

        # If labels are provided, use them to guide attention
        if labels is not None:
            # Create a mask based on labels (1 for positive labels, small value for others)
            label_mask = labels.unsqueeze(1).float() * 0.8 + 0.2  # (batch_size, 1, num_classes)
            scores = scores * label_mask

        # Apply attention
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Project output
        out = self.proj_out(out)

        return out


# Update only the SpatialGraph class in the graph_modules.py file

class SpatialGraph(nn.Module):
    """
    Spatial Relationship Graph Module.
    Captures spatial relationships between objects.
    """

    def __init__(self, num_classes, scales=(10, 20), hidden_dim=256, simplified=False):
        """
        Args:
            num_classes: Number of label classes
            scales: Tuple of grid scales (default: (10, 20))
            hidden_dim: Dimension of hidden node embeddings
            simplified: Whether to use simplified operations for less memory usage
        """
        super().__init__()

        self.num_classes = num_classes
        self.scales = scales
        self.hidden_dim = hidden_dim
        self.simplified = simplified

        # Label embeddings
        self.label_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))

        # Spatial embedding layers for each scale
        self.spatial_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(scale * scale, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ) for scale in scales
        ])

        # Graph projection layers
        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

        # Default spatial affinity matrix (identity)
        self.register_buffer('spatial_affinity', torch.eye(num_classes))

    def _compute_spatial_distributions(self, bboxes, bbox_classes, img_size=224):
        """
        Compute spatial distribution grids for each class.

        Args:
            bboxes: List of bounding boxes tensors [num_boxes, 4] in format [x, y, width, height] for each image
            bbox_classes: List of class indices tensors [num_boxes] for each image
            img_size: Size of the input image

        Returns:
            Dict of spatial distribution grids for each scale
        """
        batch_size = len(bboxes)
        device = bboxes[0].device if len(bboxes) > 0 and len(bboxes[0]) > 0 else torch.device('cpu')
        distributions = {}

        for scale in self.scales:
            # Initialize distribution grid for each class
            grid = torch.zeros(batch_size, self.num_classes, scale, scale, device=device)

            # For each batch
            for b in range(batch_size):
                if len(bboxes[b]) > 0:
                    batch_boxes = bboxes[b]  # [num_boxes, 4]
                    batch_classes = bbox_classes[b]  # [num_boxes]

                    # For each box
                    for box_idx in range(len(batch_boxes)):
                        box = batch_boxes[box_idx]
                        cls = batch_classes[box_idx].item() if isinstance(batch_classes[box_idx], torch.Tensor) else \
                            batch_classes[box_idx]

                        if cls >= 0 and cls < self.num_classes:
                            # Normalize box coordinates
                            x, y, w, h = box
                            x1 = max(0, min(img_size - 1, x))
                            y1 = max(0, min(img_size - 1, y))
                            x2 = max(0, min(img_size - 1, x + w))
                            y2 = max(0, min(img_size - 1, y + h))

                            # Convert to grid indices
                            grid_x1 = int(x1.item() * scale / img_size) if isinstance(x1, torch.Tensor) else int(
                                x1 * scale / img_size)
                            grid_y1 = int(y1.item() * scale / img_size) if isinstance(y1, torch.Tensor) else int(
                                y1 * scale / img_size)
                            grid_x2 = int(x2.item() * scale / img_size) if isinstance(x2, torch.Tensor) else int(
                                x2 * scale / img_size)
                            grid_y2 = int(y2.item() * scale / img_size) if isinstance(y2, torch.Tensor) else int(
                                y2 * scale / img_size)

                            # Update grid (simple presence, not density)
                            grid[b, cls, grid_y1:grid_y2 + 1, grid_x1:grid_x2 + 1] = 1

            # Flatten grid to (batch_size, num_classes, scale*scale)
            distributions[scale] = grid.view(batch_size, self.num_classes, scale * scale)

        return distributions

    def _compute_spatial_affinity(self, spatial_dists):
        """
        Compute spatial affinity matrix from spatial distributions.

        Args:
            spatial_dists: Dict of spatial distribution grids

        Returns:
            Spatial affinity matrix
        """
        batch_size = spatial_dists[self.scales[0]].shape[0]
        device = spatial_dists[self.scales[0]].device

        # Initialize affinity matrix
        affinity = torch.zeros(batch_size, self.num_classes, self.num_classes, device=device)

        # For each scale
        for scale_idx, scale in enumerate(self.scales):
            dist = spatial_dists[scale]  # (batch_size, num_classes, scale*scale)

            # Calculate L1 distance between spatial distributions
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    # Calculate L1 distance (simpler than EMD)
                    l1_dist = torch.abs(dist[:, i, :] - dist[:, j, :]).sum(dim=1)  # (batch_size)

                    # Convert distance to similarity (1 - normalized distance)
                    similarity = 1.0 - l1_dist / (scale * scale)

                    # Add to affinity matrix (equal weights for all scales)
                    affinity[:, i, j] += similarity / len(self.scales)

        # Normalize
        affinity = F.normalize(affinity, p=1, dim=2)

        return affinity

    def forward(self, x, bboxes=None, bbox_classes=None):
        """
        Forward pass through the spatial graph.

        Args:
            x: Node features (batch_size, num_classes, hidden_dim)
            bboxes: List of bounding boxes for each image
            bbox_classes: List of class indices for each box

        Returns:
            Updated node features
        """
        batch_size = x.shape[0] if x is not None else (len(bboxes) if bboxes is not None else 1)
        device = x.device if x is not None else (
            bboxes[0].device if bboxes is not None and len(bboxes) > 0 and len(bboxes[0]) > 0 else torch.device('cpu'))

        # Use label embeddings as node features if x is None
        if x is None:
            x = self.label_embeddings.unsqueeze(0).expand(batch_size, -1, -1).to(device)

        # Simplified processing for memory efficiency if enabled
        if self.simplified:
            # Basic spatial processing without full attention
            out = x.clone()  # Start with identity mapping

            # If bounding boxes are provided, add simple spatial features
            if bboxes is not None and bbox_classes is not None and all(len(b) > 0 for b in bboxes):
                try:
                    # Simple spatial features based on grid positions
                    for scale in self.scales:
                        grid_features = torch.zeros(batch_size, self.num_classes, scale * scale, device=device)

                        # Process in chunks for memory efficiency
                        chunk_size = max(1, batch_size // 2)
                        for i in range(0, batch_size, chunk_size):
                            end_idx = min(i + chunk_size, batch_size)
                            # Just use simple grid positions
                            for b in range(i, end_idx):
                                if len(bboxes[b]) > 0:
                                    for box_idx, box in enumerate(bboxes[b]):
                                        if box_idx < len(bbox_classes[b]):
                                            cls = bbox_classes[b][box_idx].item()
                                            if 0 <= cls < self.num_classes:
                                                x, y, w, h = box
                                                center_x = int((x + w / 2) * scale / 224)
                                                center_y = int((y + h / 2) * scale / 224)
                                                pos = center_y * scale + center_x
                                                if 0 <= pos < scale * scale:
                                                    grid_features[b, cls, pos] = 1.0

                        # Project to feature space for each scale
                        scale_idx = self.scales.index(scale)
                        scale_features = grid_features.view(batch_size * self.num_classes, -1)
                        scale_embed = self.spatial_embeddings[scale_idx](scale_features)
                        scale_embed = scale_embed.view(batch_size, self.num_classes, -1)

                        # Add to output with scale weight
                        out = out + scale_embed * (0.2 + 0.1 * scale_idx)

                except Exception as e:
                    print(f"Error in simplified spatial processing: {e}")

            return self.proj_out(out)

        # Full processing if not simplified
        # Generate queries, keys, values
        q = self.proj_q(x)  # (batch_size, num_classes, hidden_dim)
        k = self.proj_k(x)  # (batch_size, num_classes, hidden_dim)
        v = self.proj_v(x)  # (batch_size, num_classes, hidden_dim)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (
                self.hidden_dim ** 0.5)  # (batch_size, num_classes, num_classes)

        # If bounding boxes are provided, compute spatial affinity
        if bboxes is not None and bbox_classes is not None and all(len(b) > 0 for b in bboxes):
            try:
                spatial_dists = self._compute_spatial_distributions(bboxes, bbox_classes)
                spatial_affinity = self._compute_spatial_affinity(spatial_dists)
                # Apply spatial affinity as edge weights
                scores = scores * spatial_affinity
            except Exception as e:
                print(f"Error computing spatial distributions: {e}")
                # Use default affinity (identity)
                scores = scores * self.spatial_affinity.to(device).unsqueeze(0)
        else:
            # Use default affinity (identity)
            scores = scores * self.spatial_affinity.to(device).unsqueeze(0)

        # Apply attention
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Project output
        out = self.proj_out(out)

        return out


class VisualGraph(nn.Module):
    """
    Visual Relationship Graph Module.
    Captures visual feature similarities between objects.
    """

    def __init__(self, num_classes, feature_dim=768, hidden_dim=256, similarity_threshold=0.5, simplified=False):
        """
        Args:
            num_classes: Number of label classes
            feature_dim: Dimension of input visual features
            hidden_dim: Dimension of hidden node embeddings
            similarity_threshold: Threshold for visual similarity
            simplified: Whether to use simplified operations for less memory usage
        """
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.similarity_threshold = similarity_threshold
        self.simplified = simplified

        # Label embeddings
        self.label_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))

        # Visual feature projection
        self.visual_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Class-specific feature extractors
        self.class_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ) for _ in range(num_classes)
        ])

        # Graph projection layers
        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, visual_features=None, labels=None):
        """
        Forward pass through the visual graph.

        Args:
            x: Node features (batch_size, num_classes, hidden_dim)
            visual_features: Global visual features from backbone (batch_size, feature_dim)
            labels: Ground-truth labels for guidance (batch_size, num_classes)

        Returns:
            Updated node features
        """
        device = x.device if x is not None else visual_features.device
        batch_size = x.shape[0] if x is not None else visual_features.shape[0]

        # Use label embeddings as node features if x is None
        if x is None:
            x = self.label_embeddings.unsqueeze(0).expand(batch_size, -1, -1).to(device)

        # Simplified processing if enabled
        if self.simplified and visual_features is not None:
            # Project visual features
            vis_emb = self.visual_proj(visual_features)  # (batch_size, hidden_dim)

            # More efficient implementation with fewer operations
            chunk_size = max(1, self.num_classes // 4)  # Process in smaller chunks
            results = []

            for i in range(0, self.num_classes, chunk_size):
                end_idx = min(i + chunk_size, self.num_classes)
                chunk_features = []

                # Extract features for this chunk of classes
                for class_idx in range(i, end_idx):
                    # Move extractor to correct device if needed
                    extractor = self.class_extractors[class_idx].to(device)
                    class_feat = extractor(vis_emb)
                    chunk_features.append(class_feat)

                # Stack features for this chunk
                chunk_result = torch.stack(chunk_features, dim=1)  # (batch_size, chunk_size, hidden_dim)
                results.append(chunk_result)

            # Combine all chunks
            class_feats = torch.cat(results, dim=1)  # (batch_size, num_classes, hidden_dim)

            # Simple feature combination without full attention
            if labels is not None:
                # Use labels to emphasize important classes
                label_weights = labels.unsqueeze(-1).float() * 0.5 + 0.5  # (batch_size, num_classes, 1)
                class_feats = class_feats * label_weights

            # Combine with input features
            out = x + class_feats
            return self.proj_out(out)

        # Project visual features if provided
        if visual_features is not None:
            vis_emb = self.visual_proj(visual_features)  # (batch_size, hidden_dim)

            # Extract class-specific features
            class_feats = []
            for i in range(self.num_classes):
                # Move extractor to correct device if needed
                extractor = self.class_extractors[i].to(device)
                class_feats.append(extractor(vis_emb))

            class_feats = torch.stack(class_feats, dim=1)  # (batch_size, num_classes, hidden_dim)

            # Combine with existing features
            x = x + class_feats

        # Generate queries, keys, values
        q = self.proj_q(x)  # (batch_size, num_classes, hidden_dim)
        k = self.proj_k(x)  # (batch_size, num_classes, hidden_dim)
        v = self.proj_v(x)  # (batch_size, num_classes, hidden_dim)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (
                self.hidden_dim ** 0.5)  # (batch_size, num_classes, num_classes)

        # Apply visual similarity threshold
        similarity_mask = (scores > self.similarity_threshold).float()
        scores = scores * similarity_mask

        # If labels are provided, use them to guide attention
        if labels is not None:
            # Create a mask based on labels (1 for positive labels, small value for others)
            label_mask = labels.unsqueeze(1).float() * 0.8 + 0.2  # (batch_size, 1, num_classes)
            scores = scores * label_mask

        # Apply attention
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Project output
        out = self.proj_out(out)

        return out


class GraphFusionModule(nn.Module):
    """
    Graph Fusion Module.
    Combines different graph components using uncertainty-weighted attention.
    """

    def __init__(self, num_classes, hidden_dim=256):
        """
        Args:
            num_classes: Number of label classes
            hidden_dim: Dimension of hidden node embeddings
        """
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Graph uncertainty parameters (learnable)
        self.cooccurrence_uncertainty = nn.Parameter(torch.tensor(1.0))
        self.spatial_uncertainty = nn.Parameter(torch.tensor(1.0))
        self.visual_uncertainty = nn.Parameter(torch.tensor(1.0))

        # MLP for final projection
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, cooccurrence_feat, spatial_feat, visual_feat):
        """
        Forward pass through the fusion module.

        Args:
            cooccurrence_feat: Features from co-occurrence graph
            spatial_feat: Features from spatial graph
            visual_feat: Features from visual graph

        Returns:
            Fused features
        """
        # Make sure all features are on the same device
        device = cooccurrence_feat.device

        # Move uncertainty parameters to the correct device
        cooccurrence_uncertainty = self.cooccurrence_uncertainty.to(device)
        spatial_uncertainty = self.spatial_uncertainty.to(device)
        visual_uncertainty = self.visual_uncertainty.to(device)

        # Calculate uncertainty weights
        total_uncertainty = (
                1.0 / cooccurrence_uncertainty +
                1.0 / spatial_uncertainty +
                1.0 / visual_uncertainty
        )

        cooccurrence_weight = (1.0 / cooccurrence_uncertainty) / total_uncertainty
        spatial_weight = (1.0 / spatial_uncertainty) / total_uncertainty
        visual_weight = (1.0 / visual_uncertainty) / total_uncertainty

        # Weighted combination
        fused_feat = (
                cooccurrence_weight * cooccurrence_feat +
                spatial_weight * spatial_feat +
                visual_weight * visual_feat
        )

        # Apply fusion MLP
        fused_feat = self.fusion_mlp(fused_feat)

        return fused_feat

