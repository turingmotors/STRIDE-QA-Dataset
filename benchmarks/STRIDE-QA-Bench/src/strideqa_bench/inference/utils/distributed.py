import math
import os
from typing import Any

import torch
import torch.distributed as dist


def init_distributed() -> tuple[torch.device, int, int]:
    """Initialize torch.distributed if launched with torchrun.

    Returns:
        device: cuda device associated with this process (or cpu)
        rank: global rank id
        world_size: total number of processes
    """
    if torch.cuda.is_available() and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size


def split_by_rank(data: list[Any], rank: int, world_size: int) -> list[Any]:
    """Return the slice of `data` assigned to this rank."""
    per_rank = math.ceil(len(data) / world_size)
    start = rank * per_rank
    end = min(start + per_rank, len(data))
    return data[start:end]
