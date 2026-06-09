from __future__ import annotations

from argparse import Namespace

import _cp_dist_helpers
import pytest
import torch
from _cp_dist_helpers import (
    cp_chunk_response_tensor,
    free_port,
    init_worker_process_group,
    stub_megatron_in_worker,
)


def _gspo_cp_worker(rank: int, world_size: int, cp_size: int, master_port: int, result_path: str) -> None:
    import torch.distributed as _dist

    cp_rank = rank % cp_size
    stub_megatron_in_worker(cp_size, cp_rank)
    cp_group = init_worker_process_group(rank, world_size, master_port)
    try:
        from slime.backends.megatron_utils.cp_utils import get_local_response_mask_with_cp
        from slime.utils.ppo_utils import compute_gspo_kl, compute_opsm_mask

        total_length = 14
        response_length = 10
        full_log_probs = torch.tensor([-0.3, -0.4, -0.6, -0.8, -1.0, -0.2, -0.5, -0.7, -0.9, -1.1])
        full_old_log_probs = torch.tensor([-0.5, -0.2, -0.7, -0.4, -1.3, -0.3, -0.1, -1.0, -0.8, -1.4])
        full_loss_mask = torch.tensor([1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=torch.float32)

        local_log_probs = cp_chunk_response_tensor(full_log_probs, total_length, response_length).requires_grad_(True)
        local_old_log_probs = cp_chunk_response_tensor(full_old_log_probs, total_length, response_length)
        local_loss_mask = get_local_response_mask_with_cp(full_loss_mask, total_length, response_length)

        local_kl = compute_gspo_kl(
            local_log_probs=[local_log_probs],
            local_old_log_probs=[local_old_log_probs],
            local_loss_masks=[local_loss_mask],
            loss_masks=[full_loss_mask],
            cp_group=cp_group,
        )

        expected = ((full_old_log_probs - full_log_probs) * full_loss_mask).sum() / full_loss_mask.sum()
        assert torch.allclose(local_kl, torch.full_like(local_kl, expected))

        local_loss = (local_kl * local_loss_mask).sum()
        local_loss.backward()
        assert local_log_probs.grad is not None

        local_advantages = torch.full_like(local_log_probs.detach(), -1.0)
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=Namespace(opsm_delta=-1.0),
            local_log_probs=[local_log_probs.detach()],
            local_old_log_probs=[local_old_log_probs],
            advantages=[local_advantages],
            local_loss_masks=[local_loss_mask],
            loss_masks=[full_loss_mask],
            cp_group=cp_group,
        )
        expected_mask = torch.zeros_like(local_log_probs)
        assert torch.equal(opsm_mask, expected_mask)
        assert torch.allclose(opsm_clipfrac, torch.tensor(1.0))

        if rank == 0:
            with open(result_path, "w") as f:
                f.write("ok")
    finally:
        _dist.destroy_process_group()


def _run_gspo_cp_case(cp_size: int, tmp_path) -> str:
    import torch.multiprocessing as mp

    result_path = str(tmp_path / f"gspo_cp{cp_size}.txt")
    mp.spawn(
        _gspo_cp_worker,
        args=(cp_size, cp_size, free_port(), result_path),
        nprocs=cp_size,
        join=True,
    )
    with open(result_path) as f:
        return f.read()


@pytest.mark.unit
@pytest.mark.parametrize("cp_size", [2, 4])
def test_gspo_sequence_kl_uses_local_cp_chunks(cp_size, tmp_path):
    assert _run_gspo_cp_case(cp_size, tmp_path) == "ok"


_ = _cp_dist_helpers
