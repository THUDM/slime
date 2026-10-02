"""Advantage-whitening CP-invariance check on CPU.

``compute_advantages_and_returns`` whitens advantages with
``distributed_masked_whiten``, which all-reduces ``(sum, sum_sq, mask_sum)``
over the process group it is handed. Under context parallelism each rank only
holds its zigzag slice of every sequence, so those statistics are only the
*global* statistics if the group spans the CP dimension as well as DP.

If the CP-excluding group is used instead, every CP rank normalizes with the
mean/variance of its own slice: the two halves of one sequence come out with
different affine transforms, and none of them matches the correct whitening.

The contract pinned here: for a fixed set of samples, the whitened advantage of
a given sample is the same number for every ``(dp_size, cp_size)`` factorization
of the world — and in particular the same as the single-rank baseline.

Two of the cases use a prompt-heavy sequence whose response tokens all land on
one CP rank, so some ranks contribute an entirely empty local mask. Those ranks
must still take part in the all-reduce; a rank-dependent "skip whitening when I
hold nothing" shortcut desyncs the collective and hangs. The spawn join below
is bounded so that regression surfaces as a failure rather than a stuck job.
"""

from __future__ import annotations

import json
import os
import time

# Megatron stub must land in sys.modules before anything imports
# slime.backends.megatron_utils. pytest's prepend importmode puts ``tests/`` on
# sys.path, so the bare-name import works without an ``__init__.py``.
import _cp_dist_helpers  # noqa: F401
import pytest
from _cp_dist_helpers import free_port, stub_megatron_in_worker


NUM_GPUS = 0

# (total_length, response_length) per sample, plus its rollout reward. Eight
# samples so dp_size in {1, 2, 4} divides evenly. Sample 3 is deliberately
# prompt-heavy: at cp_size >= 2 its response sits entirely inside one rank's
# chunk, leaving the other ranks with an empty mask for it.
SEQS = [(64, 48), (100, 90), (40, 12), (256, 16), (72, 30), (90, 84), (50, 20), (110, 96)]
REWARDS = [1.5, -0.5, -1.0, 2.0, 0.25, -1.75, 0.75, -0.25]

WHITEN_CASES = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 4), (4, 1)]


class _Args:
    advantage_estimator = "grpo"
    normalize_advantages = True
    kl_coef = 0.0
    use_kl_loss = False
    use_rollout_logprobs = False
    use_opd = False
    custom_advantage_function_path = None
    kl_loss_type = "low_var_kl"
    gamma = 1.0
    lambd = 1.0


def _whiten_worker(rank, world_size, cp_size, dp_size, master_port, result_dir):
    """One spawned rank: whiten its DP shard's CP slice, dump the results."""
    import torch
    import torch.distributed as _dist

    cp_rank = rank % cp_size
    dp_rank = rank // cp_size
    stub_megatron_in_worker(cp_size, cp_rank)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    _dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from megatron.core import mpu

        # No TP/PP here, so DP-with-CP is the whole world. The DP-only group is
        # the set of ranks sharing this rank's cp_rank -- exactly the group that
        # Megatron's ``get_data_parallel_group(with_context_parallel=False)``
        # returns. Every rank must build every subgroup, in the same order.
        dp_cp_group = _dist.new_group(ranks=list(range(world_size)))
        dp_only_groups = [
            _dist.new_group(ranks=[r for r in range(world_size) if r % cp_size == c]) for c in range(cp_size)
        ]

        mpu.is_pipeline_last_stage = lambda: True
        mpu.get_data_parallel_group = lambda with_context_parallel=False, **kw: (
            dp_cp_group if with_context_parallel else dp_only_groups[cp_rank]
        )

        from slime.backends.megatron_utils.cp_utils import get_logits_and_tokens_offset_with_cp
        from slime.backends.megatron_utils.loss import compute_advantages_and_returns

        def cp_slice(x, total_len, response_len):
            """Keep only the response positions this CP rank owns."""
            if cp_size == 1:
                return x
            prompt_len = total_len - response_len
            _, _, _, offsets = get_logits_and_tokens_offset_with_cp(total_len, response_len)
            parts = []
            for start, end in offsets:
                lo, hi = max(0, start - prompt_len), max(0, end - prompt_len)
                if hi > lo:
                    parts.append(x[lo:hi])
            return torch.cat(parts) if parts else x[:0]

        # Round-robin DP shard, mirroring how samples spread over DP ranks.
        my_samples = [i for i in range(len(SEQS)) if i % dp_size == dp_rank]

        rollout_data = {
            "log_probs": [cp_slice(torch.zeros(SEQS[i][1]), *SEQS[i]) for i in my_samples],
            "loss_masks": [torch.ones(SEQS[i][1]) for i in my_samples],
            "rewards": [REWARDS[i] for i in my_samples],
            "response_lengths": [SEQS[i][1] for i in my_samples],
            "total_lengths": [SEQS[i][0] for i in my_samples],
        }

        compute_advantages_and_returns(_Args(), rollout_data)

        # grpo gives every token of a sample the same advantage, and whitening is
        # affine, so one value per sample fully describes the result. Ranks that
        # own no tokens of a sample simply report nothing for it.
        out = {
            str(i): round(adv[0].item(), 6)
            for i, adv in zip(my_samples, rollout_data["advantages"], strict=True)
            if adv.numel() > 0
        }
        with open(os.path.join(result_dir, f"rank{rank}.json"), "w") as f:
            json.dump(out, f)
    finally:
        _dist.destroy_process_group()


def _run_case(dp_size, cp_size, tmp_path):
    """Spawn the world, then merge every rank's per-sample whitened values."""
    import torch.multiprocessing as mp

    world_size = dp_size * cp_size
    result_dir = tmp_path / f"dp{dp_size}_cp{cp_size}"
    result_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.spawn(
        _whiten_worker,
        args=(world_size, cp_size, dp_size, free_port(), str(result_dir)),
        nprocs=world_size,
        join=False,
    )
    deadline = time.time() + 180
    while not ctx.join(timeout=5):
        if time.time() > deadline:
            for p in ctx.processes:
                if p.is_alive():
                    p.terminate()
            pytest.fail(
                f"dp={dp_size} cp={cp_size} workers did not finish in 180s; "
                "a rank-dependent branch around the whitening all_reduce desyncs the collective"
            )

    merged: dict[str, float] = {}
    for rank in range(world_size):
        with open(result_dir / f"rank{rank}.json") as f:
            for sample_idx, value in json.load(f).items():
                if sample_idx in merged:
                    # Every rank holding part of a sample must agree on it --
                    # this is the assertion that fails on the CP-excluding group.
                    assert merged[sample_idx] == pytest.approx(value, abs=1e-5), (
                        f"dp={dp_size} cp={cp_size}: CP ranks disagree on sample {sample_idx}: "
                        f"{merged[sample_idx]} vs {value}"
                    )
                merged[sample_idx] = value
    return merged


@pytest.mark.unit
@pytest.mark.parametrize("dp_size,cp_size", WHITEN_CASES)
def test_whitened_advantages_are_cp_invariant(dp_size, cp_size, tmp_path):
    baseline = _run_case(1, 1, tmp_path)
    assert len(baseline) == len(SEQS), "baseline should cover every sample"

    got = _run_case(dp_size, cp_size, tmp_path)

    assert sorted(got) == sorted(baseline)
    for sample_idx, expected in baseline.items():
        assert got[sample_idx] == pytest.approx(expected, abs=1e-5), (
            f"dp={dp_size} cp={cp_size}: sample {sample_idx} whitened to {got[sample_idx]}, "
            f"single-rank baseline is {expected}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
