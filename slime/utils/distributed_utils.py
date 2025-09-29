from datetime import timedelta
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


GLOO_GROUP = None


def init_gloo_group():
    """Initialize Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        GLOO_GROUP = dist.new_group(backend="gloo")
    return GLOO_GROUP


def get_gloo_group():
    """Get the Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        raise RuntimeError("Gloo group has not been initialized. Call _init_gloo_group() first.")
    return GLOO_GROUP


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def distributed_masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    process_group: Optional[dist.ProcessGroup] = None,
    shift_mean: bool = True,
    epsilon: float = 1e-8,
):
    """
    Performs whitening on a tensor using global statistics from all participating GPUs.

    It calculates the global mean and variance across all ranks in the default
    process group (the WORLD) and uses these global statistics to normalize the
    local data on each rank.

    Args:
        values (torch.Tensor): The local tensor of values to whiten.
        mask (torch.Tensor): The local mask corresponding to the values.
        process_group: The process group for all_reduce. 
                      If None, uses the default world group.
        shift_mean (bool): If True, the output is zero-mean. Defaults to True.
        epsilon (float): A small value for numerical stability.

    Returns:
        torch.Tensor: The locally whitened tensor using global statistics.
    """
    # Calculate local intermediate statistics
    local_sum = (values * mask).sum()
    local_sum_sq = ((values**2) * mask).sum()
    local_mask_sum = mask.sum()

    stats_tensor = torch.tensor(
        [local_sum, local_sum_sq, local_mask_sum],
        device=values.device,
        dtype=torch.float32,
    )

    # Aggregate via all_reduce within the DP group
    dist.all_reduce(stats_tensor, group=process_group)

    # Calculate global stats from aggregated results
    global_sum, global_sum_sq, global_mask_sum = stats_tensor

    if global_mask_sum.item() == 0:
        raise ValueError("The global mask sum across all participating GPUs is zero.")

    global_mean = global_sum / global_mask_sum
    global_mean_sq = global_sum_sq / global_mask_sum
    global_var = global_mean_sq - global_mean**2

    # Bessel's correction for unbiased estimate
    if global_mask_sum.item() >= 2:
        bessel_correction = global_mask_sum / (global_mask_sum - 1)
        global_var = global_var * bessel_correction

    # Whiten local data using global stats
    whitened_values = (values - global_mean) * torch.rsqrt(global_var + epsilon)

    if not shift_mean:
        whitened_values += global_mean

    return whitened_values
