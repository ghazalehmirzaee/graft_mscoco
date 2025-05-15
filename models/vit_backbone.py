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

        # Set weights_only=False to allow loading of argparse.Namespace objects
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

            # Extract encoder weights if they have 'encoder.' prefix
            encoder_state_dict = {}
            for k, v in pretrained_dict.items():
                new_key = k
                # Remove prefixes if present
                if k.startswith('encoder.'):
                    new_key = k[8:]  # Remove 'encoder.' prefix
                elif k.startswith('module.'):
                    new_key = k[7:]  # Remove 'module.' prefix

                encoder_state_dict[new_key] = v

            # Get the model state dict
            model_dict = self.model.state_dict()

            # Print some keys for debugging
            print(f"Model keys: {list(model_dict.keys())[:5]}")
            print(f"Encoder keys: {list(encoder_state_dict.keys())[:5]}")

            # Try to match keys
            matched_dict = {}
            for model_key in model_dict.keys():
                # Try direct match
                if model_key in encoder_state_dict:
                    if model_dict[model_key].shape == encoder_state_dict[model_key].shape:
                        matched_dict[model_key] = encoder_state_dict[model_key]

                # Try without 'blocks.' prefix
                elif model_key.startswith('blocks.'):
                    potential_key = model_key.replace('blocks.', 'transformer.encoderblock.')
                    if potential_key in encoder_state_dict and model_dict[model_key].shape == encoder_state_dict[
                        potential_key].shape:
                        matched_dict[model_key] = encoder_state_dict[potential_key]

            # If no matches, try more aggressive matching by parameter shape and name suffix
            if len(matched_dict) == 0:
                print("No direct key matches, trying shape-based matching...")
                for model_key, model_param in model_dict.items():
                    for enc_key, enc_param in encoder_state_dict.items():
                        # Check if shapes match and the parameter names have similar endings
                        if model_param.shape == enc_param.shape and model_key.split('.')[-1] == enc_key.split('.')[-1]:
                            matched_dict[model_key] = enc_param
                            break

            # Update the model state dict with matched weights
            model_dict.update(matched_dict)
            self.model.load_state_dict(model_dict)

            print(f"Loaded {len(matched_dict)} / {len(model_dict)} parameters")
        else:
            print(f"Pretrained weights not in expected dictionary format. Type: {type(pretrained_dict)}")


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

