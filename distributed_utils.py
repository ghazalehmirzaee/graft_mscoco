import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    """
    Setup distributed training environment.

    Args:
        rank: Current process rank
        world_size: Total number of processes
    """
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set GPU device
    torch.cuda.set_device(rank)


def cleanup():
    """Cleanup distributed training environment."""
    dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """
    Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        world_size: Total number of processes

    Returns:
        Reduced tensor
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def is_main_process(rank):
    """Check if current process is the main process."""
    if isinstance(rank, torch.device):
        # If rank is a device, extract the device index
        if rank.type == 'cuda':
            rank = rank.index if rank.index is not None else 0
        else:
            return True  # CPU is always main process

    return rank == 0


def save_on_master(state, filename):
    """Save checkpoint only on master process."""
    if dist.get_rank() == 0:
        torch.save(state, filename)


def spawn_workers(fn, world_size, *args, **kwargs):
    """
    Spawn multiple processes for distributed training.

    Args:
        fn: Function to run in each process
        world_size: Number of processes to spawn
        args, kwargs: Arguments to pass to the function
    """
    mp.spawn(fn, args=(world_size, *args), nprocs=world_size, join=True)

