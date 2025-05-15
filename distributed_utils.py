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

    # Some environments need a backend switch
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Try multiple times to initialize the process group
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Initialize process group with timeout
            dist.init_process_group(
                backend,
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.init_process_group_timeout
            )
            break
        except Exception as e:
            retry_count += 1
            print(f"Process group initialization failed (attempt {retry_count}/{max_retries}): {e}")
            if retry_count == max_retries:
                raise

    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        # Add synchronization point
        torch.cuda.synchronize()

    # Set NCCL options for better performance
    if backend == "nccl":
        torch.backends.cudnn.benchmark = True


def cleanup():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """
    Reduce tensor across all processes with error handling.

    Args:
        tensor: Tensor to reduce
        world_size: Total number of processes

    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor

    rt = tensor.clone()

    try:
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= world_size
    except Exception as e:
        print(f"Warning: Failed to reduce tensor: {e}")
        # Return unmodified tensor on error
        return tensor

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
    try:
        mp.spawn(fn, args=(world_size, *args), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Error in worker processes: {e}")
        # Clean up in case of error
        cleanup()
        raise e

