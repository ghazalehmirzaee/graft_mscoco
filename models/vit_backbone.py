import torch
import torch.nn as nn
import timm
from functools import partial
from models.models_vit import VisionTransformer


class ViTBackbone(nn.Module):
    """Vision Transformer backbone adapted for multi-label classification on MSCOCO."""

    def __init__(self, pretrained_path, num_classes=80, freeze_layers=8):
        """
        Args:
            pretrained_path: Path to the pretrained ViT-B/16 model
            num_classes: Number of output classes (80 for MSCOCO)
            freeze_layers: Number of transformer layers to freeze
        """
        super().__init__()

        # Create the ViT model
        self.model = VisionTransformer(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        # Remove the last layer (head)
        self.embed_dim = self.model.embed_dim

        # Create a new multi-label classification head
        self.head = nn.Linear(self.embed_dim, num_classes)

        # Load pretrained weights
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)

        # Freeze early layers
        if freeze_layers > 0:
            for i in range(min(freeze_layers, len(self.model.blocks))):
                for param in self.model.blocks[i].parameters():
                    param.requires_grad = False

    def _load_pretrained_weights(self, pretrained_path):
        """Load weights from the pretrained model."""
        print(f"Loading pretrained weights from {pretrained_path}")

        # FIXED: Explicitly set weights_only=False to handle argparse.Namespace objects
        try:
            # First try a safer approach with a custom safelist (for PyTorch 2.6+)
            import argparse
            try:
                from torch.serialization import add_safe_globals
                with add_safe_globals([argparse.Namespace]):
                    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
            except (ImportError, AttributeError):
                # Fallback for older PyTorch versions
                pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        except Exception as e:
            print(f"First loading attempt failed with error: {str(e)}")
            print("Trying with weights_only=False for backward compatibility...")
            # Fall back to the less secure option if needed
            pretrained_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        # Print some info about the loaded weights for debugging
        if isinstance(pretrained_dict, dict):
            print(f"Pretrained dict keys: {list(pretrained_dict.keys())[:10] if len(pretrained_dict) > 0 else 'empty'}")

            # Check if weights are in a nested dictionary
            if 'model' in pretrained_dict:
                print("Found 'model' key in pretrained weights")
                pretrained_dict = pretrained_dict['model']
                print(
                    f"Updated pretrained dict keys: {list(pretrained_dict.keys())[:10] if len(pretrained_dict) > 0 else 'empty'}")
            elif 'state_dict' in pretrained_dict:
                print("Found 'state_dict' key in pretrained weights")
                pretrained_dict = pretrained_dict['state_dict']
                print(
                    f"Updated pretrained dict keys: {list(pretrained_dict.keys())[:10] if len(pretrained_dict) > 0 else 'empty'}")

            # Create a new dict with modified keys
            new_state_dict = {}

            # Get the model state dict
            model_dict = self.model.state_dict()

            # Try to match keys between pretrained and model
            for k, v in model_dict.items():
                # Try direct match
                if k in pretrained_dict and pretrained_dict[k].shape == v.shape:
                    new_state_dict[k] = pretrained_dict[k]
                    continue

                # Try with different prefixes
                for prefix in ['', 'encoder.', 'backbone.', 'model.', 'vit.']:
                    prefixed_key = prefix + k
                    if prefixed_key in pretrained_dict and pretrained_dict[prefixed_key].shape == v.shape:
                        new_state_dict[k] = pretrained_dict[prefixed_key]
                        break

                # Try with different suffixes
                if 'blocks.' in k:
                    for alt_pattern in ['transformer.encoderblock.', 'blocks.', 'encoder.layer.']:
                        alt_key = k.replace('blocks.', alt_pattern)
                        if alt_key in pretrained_dict and pretrained_dict[alt_key].shape == v.shape:
                            new_state_dict[k] = pretrained_dict[alt_key]
                            break

            # Load the matched weights
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded {len(new_state_dict)} / {len(model_dict)} parameters")
        else:
            print(f"Warning: Pretrained weights not in expected format. Type: {type(pretrained_dict)}")

    def forward(self, x):
        """Forward pass through the backbone and head."""
        # Get embeddings from the ViT backbone
        features = self.model.forward_features(x)

        # Apply the classification head
        logits = self.head(features)

        return {
            'logits': logits,
            'features': features
        }

