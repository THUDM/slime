from __future__ import annotations

import sys
import types

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from slime.utils import reloadable_process_group as rpg


NUM_GPUS = 0


def _run_rebinding_worker(result_queue) -> None:
    # Megatron binds these at import time, before monkey_patch_torch_dist runs in the train actor.
    original_reduce_scatter = dist.reduce_scatter_tensor
    original_all_gather = dist.all_gather_into_tensor
    original_coalescing = dist._coalescing_manager
    megatron_module = types.ModuleType("megatron.core.distributed.param_and_grad_buffer")
    megatron_module.dist_reduce_scatter_func = original_reduce_scatter
    megatron_module.dist_all_gather_func = original_all_gather
    megatron_module._coalescing_manager = original_coalescing
    megatron_module.unrelated = original_reduce_scatter.__name__
    other_module = types.ModuleType("not_megatron.bindings")
    other_module.dist_reduce_scatter_func = original_reduce_scatter
    sys.modules[megatron_module.__name__] = megatron_module
    sys.modules[other_module.__name__] = other_module

    rpg.monkey_patch_torch_dist()

    result_queue.put(
        {
            "reduce_scatter_rebound": megatron_module.dist_reduce_scatter_func is dist.reduce_scatter_tensor,
            "all_gather_rebound": megatron_module.dist_all_gather_func is dist.all_gather_into_tensor,
            "coalescing_rebound": megatron_module._coalescing_manager is dist._coalescing_manager,
            "patched": dist.reduce_scatter_tensor is not original_reduce_scatter,
            "unrelated_untouched": megatron_module.unrelated == original_reduce_scatter.__name__,
            "other_module_untouched": other_module.dist_reduce_scatter_func is original_reduce_scatter,
        }
    )


def test_monkey_patch_rebinds_megatron_import_time_collectives():
    # The patch mutates torch.distributed for the whole process, so run it in a fresh one.
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_run_rebinding_worker, args=(result_queue,))
    process.start()
    result = result_queue.get(timeout=120)
    process.join(timeout=30)
    assert process.exitcode == 0
    assert result == {
        "reduce_scatter_rebound": True,
        "all_gather_rebound": True,
        "coalescing_rebound": True,
        "patched": True,
        "unrelated_untouched": True,
        "other_module_untouched": True,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
