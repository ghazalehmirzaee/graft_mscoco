import os
import time
from collections import deque, defaultdict
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.graft import GRAFT
from datasets import create_dataloaders
from utils import compute_metrics, save_checkpoint, visualize_predictions, load_checkpoint
from distributed_utils import reduce_tensor, save_on_master, is_main_process, setup, cleanup, DDP


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a window."""

    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = '{median:.4f} ({global_avg:.4f})' if fmt is None else fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg
        )


class MetricLogger:
    """Log metrics during training and evaluation."""

    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}MB'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])

        MB = 1024.0 * 1024.0
        i = 0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time)
                    ))

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')


def train_one_epoch(model, optimizer, data_loader, device, epoch, config, writer, scaler):
    """
    Train model for one epoch.

    Args:
        model: Model instance
        optimizer: Optimizer
        data_loader: DataLoader for training data
        device: Device to train on
        epoch: Current epoch number
        config: Configuration object
        writer: TensorBoard writer
        scaler: Gradient scaler for mixed precision training

    Returns:
        Average loss for the epoch
    """
    model.train()

    metric_logger = MetricLogger()
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f'Epoch: [{epoch}]'

    # Set learning rate for this epoch
    for param_group in optimizer.param_groups:
        if writer is not None:
            writer.add_scalar('train/lr', param_group['lr'], epoch)

    for i, sample in enumerate(metric_logger.log_every(data_loader, config.LOG_INTERVAL, header)):
        # Get inputs and labels
        images = sample['image'].to(device, non_blocking=True)
        labels = sample['labels'].to(device, non_blocking=True)
        bboxes = sample['bboxes']  # Already a list of tensors
        bbox_cat_idxs = sample['bbox_cat_idxs']  # Already a list of tensors

        # Move bbox data to device
        bboxes = [b.to(device, non_blocking=True) for b in bboxes]
        bbox_cat_idxs = [b.to(device, non_blocking=True) for b in bbox_cat_idxs]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(images, labels, bboxes, bbox_cat_idxs)
            loss = outputs['loss']

        # Scale loss and compute gradients
        scaler.scale(loss).backward()

        # Update weights
        scaler.step(optimizer)
        scaler.update()

        # Reduce loss across all processes
        if config.NUM_GPUS > 1:
            reduced_loss = reduce_tensor(loss.detach(), config.NUM_GPUS)
        else:
            reduced_loss = loss.detach()

        # Log metrics
        metric_logger.update(loss=reduced_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

        # Log to TensorBoard
        global_step = epoch * len(data_loader) + i
        if is_main_process(device) and writer is not None:
            writer.add_scalar('train/loss', reduced_loss.item(), global_step)

            # Visualize predictions for first batch of each epoch
            if i == 0 and epoch % 5 == 0:
                # Create visualization
                with torch.no_grad():
                    fig = visualize_predictions(
                        images[:5].detach().cpu(),
                        labels[:5].detach().cpu(),
                        outputs['logits'][:5].detach().cpu(),
                        [f"Class_{i}" for i in range(config.NUM_CLASSES)],
                        # Use generic class names if actual ones not available
                        num_samples=5
                    )
                    writer.add_figure(f'train/predictions_epoch_{epoch}', fig, global_step)

    # Return average loss
    return metric_logger.meters['loss'].global_avg


def evaluate(model, data_loader, device, epoch, config, writer):
    """
    Evaluate model on validation data.

    Args:
        model: Model instance
        data_loader: DataLoader for validation data
        device: Device to evaluate on
        epoch: Current epoch number
        config: Configuration object
        writer: TensorBoard writer

    Returns:
        Dict containing evaluation metrics
    """
    model.eval()

    metric_logger = MetricLogger()
    header = 'Eval:'

    # Collect all predictions and targets
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for sample in metric_logger.log_every(data_loader, config.LOG_INTERVAL, header):
            # Get inputs and labels
            images = sample['image'].to(device, non_blocking=True)
            labels = sample['labels'].to(device, non_blocking=True)
            bboxes = sample['bboxes']
            bbox_cat_idxs = sample['bbox_cat_idxs']

            # Move bbox data to device
            if all(b is not None for b in bboxes):
                bboxes = [b.to(device, non_blocking=True) for b in bboxes]
                bbox_cat_idxs = [b.to(device, non_blocking=True) for b in bbox_cat_idxs]

            # Forward pass
            outputs = model(images, labels, bboxes, bbox_cat_idxs)
            logits = outputs['logits']
            loss = outputs['loss']

            # Reduce loss across all processes
            if config.NUM_GPUS > 1:
                reduced_loss = reduce_tensor(loss.detach(), config.NUM_GPUS)
            else:
                reduced_loss = loss.detach()

            # Collect predictions and targets
            all_logits.append(logits)
            all_targets.append(labels)

            # Log metrics
            metric_logger.update(loss=reduced_loss.item())

    # Concatenate all logits and targets
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Gather predictions from all processes
    if config.NUM_GPUS > 1:
        all_logits_list = [torch.zeros_like(all_logits) for _ in range(config.NUM_GPUS)]
        all_targets_list = [torch.zeros_like(all_targets) for _ in range(config.NUM_GPUS)]

        torch.distributed.all_gather(all_logits_list, all_logits)
        torch.distributed.all_gather(all_targets_list, all_targets)

        all_logits = torch.cat(all_logits_list, dim=0)
        all_targets = torch.cat(all_targets_list, dim=0)

    # Compute metrics
    metrics = compute_metrics(all_logits, all_targets)

    # Log metrics
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            metric_logger.update(**{k: v})

            if is_main_process(device) and writer is not None:
                writer.add_scalar(f'val/{k}', v, epoch)

    return metrics


def train_and_eval(rank, world_size, config):
    """
    Main training and evaluation function.

    Args:
        rank: Current process rank
        world_size: Total number of processes
        config: Configuration object
    """
    # Important: Wrap the entire function in try-finally to ensure cleanup
    try:
        # Set up distributed training
        if world_size > 1:
            setup(rank, world_size)

        # Create output directory
        if is_main_process(rank):
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        # Set device
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

        # Debug: Print config values
        print(f"Rank {rank}: NUM_EPOCHS={config.NUM_EPOCHS}, BATCH_SIZE={config.BATCH_SIZE}")

        # Create TensorBoard writer
        writer = SummaryWriter(log_dir=os.path.join(config.OUTPUT_DIR, 'tensorboard')) if is_main_process(
            rank) else None

        # Create dataloaders
        train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
            config, world_size if world_size > 1 else None, rank if world_size > 1 else None
        )

        # Get co-occurrence matrix
        cooccurrence_matrix = train_dataset.get_cooccurrence_matrix()

        # Create model
        model = GRAFT(config, cooccurrence_matrix)
        model.to(device)

        # Wrap model with DDP
        if world_size > 1:
            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        # Define optimizer
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.BASE_LR,
            weight_decay=config.WEIGHT_DECAY
        )

        # Define learning rate scheduler - FIX: Ensure T_0 is at least 1
        T_0 = max(1, config.NUM_EPOCHS // 3)
        print(f"Rank {rank}: Creating scheduler with T_0={T_0}, NUM_EPOCHS={config.NUM_EPOCHS}")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=1,
            eta_min=config.MIN_LR
        )

        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler()

        # Resume from checkpoint if available
        start_epoch = 0
        best_map = 0.0
        if os.path.exists(os.path.join(config.OUTPUT_DIR, 'checkpoint.pth')):
            ckpt_path = os.path.join(config.OUTPUT_DIR, 'checkpoint.pth')
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            # Load model weights
            if isinstance(model, DDP):
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])

            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer'])

            # Load scheduler state
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])

            # Get resume epoch and best mAP
            start_epoch = checkpoint.get('epoch', 0)
            best_map = checkpoint.get('best_map', 0.0)

            print(f"Resumed from epoch {start_epoch} with best mAP: {best_map:.4f}")

        # Start training
        print(f"Starting training from epoch {start_epoch} to {config.NUM_EPOCHS}")

        for epoch in range(start_epoch, config.NUM_EPOCHS):
            # Set sampler epoch for distributed training
            if world_size > 1 and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # Train one epoch
            train_loss = train_one_epoch(
                model, optimizer, train_loader, device, epoch, config, writer, scaler
            )

            # Update scheduler
            scheduler.step()

            # Evaluate model
            if epoch % config.EVAL_INTERVAL == 0 or epoch == config.NUM_EPOCHS - 1:
                metrics = evaluate(model, val_loader, device, epoch, config, writer)

                # Check if best model so far
                is_best = metrics['mAP'] > best_map
                best_map = max(metrics['mAP'], best_map)

                # Save checkpoint
                if is_main_process(rank):
                    model_state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'model': model_state_dict,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'best_map': best_map,
                            'config': {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
                        },
                        is_best,
                        config.OUTPUT_DIR
                    )

                    # Log best result so far
                    if is_best and writer is not None:
                        writer.add_scalar('val/best_mAP', best_map, epoch)

        return best_map

    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # Critical change: Make sure to clean up distributed resources even if there's an error
        if world_size > 1:
            cleanup()

        if writer is not None:
            writer.close()

