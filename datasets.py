import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from config import Config


class CocoMultiLabelDataset(Dataset):
    """MS-COCO Dataset for multi-label classification."""

    def __init__(self, ann_file, img_dir, transform=None, is_train=True):
        """
        Args:
            ann_file (string): Path to the annotation file.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_train (bool): Whether this is training set or validation set.
        """
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.transform = transform
        self.is_train = is_train

        # Get all image ids
        self.ids = list(sorted(self.coco.imgs.keys()))

        # Map category id to continuous index
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        # Create index to image id mapping
        self.idx_to_img_id = {i: img_id for i, img_id in enumerate(self.ids)}

        # Pre-compute co-occurrence matrix during initialization
        if is_train:
            self.cooccurrence_matrix = self._compute_cooccurrence_matrix()

    def _compute_cooccurrence_matrix(self):
        """Compute co-occurrence matrix for all categories."""
        num_classes = len(self.cat_ids)
        cooccurrence = np.zeros((num_classes, num_classes), dtype=np.float32)

        # Count co-occurrences
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            # Get all categories in this image
            cats = set()
            for ann in anns:
                cat_idx = self.cat_id_to_idx.get(ann['category_id'], -1)
                if cat_idx >= 0:
                    cats.add(cat_idx)

            # Update co-occurrence matrix
            cats = list(cats)
            for i in range(len(cats)):
                for j in range(len(cats)):
                    cooccurrence[cats[i], cats[j]] += 1

        # Normalize
        class_counts = np.diag(cooccurrence)
        for i in range(num_classes):
            for j in range(num_classes):
                if class_counts[i] > 0 and class_counts[j] > 0:
                    cooccurrence[i, j] /= np.sqrt(class_counts[i] * class_counts[j])

        return cooccurrence

    def get_cooccurrence_matrix(self):
        """Return the pre-computed co-occurrence matrix."""
        if hasattr(self, 'cooccurrence_matrix'):
            return self.cooccurrence_matrix
        else:
            return self._compute_cooccurrence_matrix()

    def get_bbox_info(self, img_id):
        """Get bounding box information for an image."""
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        bboxes = []
        cat_idxs = []

        for ann in anns:
            cat_idx = self.cat_id_to_idx.get(ann['category_id'], -1)
            if cat_idx >= 0 and 'bbox' in ann:
                bboxes.append(ann['bbox'])  # [x, y, width, height]
                cat_idxs.append(cat_idx)

        return np.array(bboxes) if bboxes else np.zeros((0, 4)), np.array(cat_idxs)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.idx_to_img_id[idx]

        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # Create multi-label target
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        labels = torch.zeros(len(self.cat_ids), dtype=torch.float32)
        for ann in anns:
            cat_idx = self.cat_id_to_idx.get(ann['category_id'], -1)
            if cat_idx >= 0:
                labels[cat_idx] = 1

        # Get bbox information for spatial graph
        bboxes, bbox_cat_idxs = self.get_bbox_info(img_id)

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        # Create a sample dict
        sample = {
            'image': img,
            'labels': labels,
            'bboxes': torch.from_numpy(bboxes).float() if len(bboxes) > 0 else torch.zeros(0, 4),
            'bbox_cat_idxs': torch.from_numpy(bbox_cat_idxs).long() if len(bbox_cat_idxs) > 0 else torch.zeros(0,
                                                                                                               dtype=torch.long),
            'img_id': img_id
        }

        return sample


def get_train_transform():
    """Gets transform for training images."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transform():
    """Gets transform for validation images."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def custom_collate_fn(batch):
    """
    Custom collate function that handles variable-sized bounding box data.

    Args:
        batch: List of samples from the dataset

    Returns:
        Properly collated batch
    """
    images = torch.stack([sample['image'] for sample in batch])
    labels = torch.stack([sample['labels'] for sample in batch])
    img_ids = [sample['img_id'] for sample in batch]

    # Handle bounding boxes (don't stack, keep as a list)
    bboxes = [sample['bboxes'] for sample in batch]
    bbox_cat_idxs = [sample['bbox_cat_idxs'] for sample in batch]

    return {
        'image': images,
        'labels': labels,
        'bboxes': bboxes,
        'bbox_cat_idxs': bbox_cat_idxs,
        'img_id': img_ids
    }

def create_dataloaders(config, world_size=None, rank=None):
    """Creates dataloaders for training and validation."""
    # Create datasets
    train_dataset = CocoMultiLabelDataset(
        ann_file=config.TRAIN_ANN,
        img_dir=config.TRAIN_IMG_DIR,
        transform=get_train_transform(),
        is_train=True
    )

    val_dataset = CocoMultiLabelDataset(
        ann_file=config.VAL_ANN,
        img_dir=config.VAL_IMG_DIR,
        transform=get_val_transform(),
        is_train=False
    )

    # Create samplers for distributed training
    if world_size is not None and rank is not None:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn  # Use custom collate function
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn  # Use custom collate function
    )

    return train_loader, val_loader, train_dataset, val_dataset

