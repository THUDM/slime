"""End-to-end backward / gradient-norm CP-invariance check on CPU.

This is the closest thing to a real training-step backward we can run on
the CPU CI image without standing up Megatron, FlashAttention, or NCCL.
The goal: prove that for the same training samples, *the gradient norm
after the optimizer-side all-reduce is identical regardless of CP size*.

Why this matters
----------------
Slime's loss prescaling + Megatron's per-mb scaling + DDP's grad
averaging compose into one big formula. Any time we touch any one of
those three layers the numbers should land in the same place. Until
this test existed we only had end-to-end report-formula checks
(`test_metric_report_dist.py`); none of them ran a real ``backward()``,
so a sign or factor error in the prescaling would slip through.

Mapping to Megatron source
--------------------------
We reproduce, for each spawned rank, the exact sequence Megatron applies
when a 3-tuple ``(loss, num_tokens, log)`` comes back from the loss
function with ``calculate_per_token_loss=False`` — slime's per-rollout-
mean path:

  1. Loss function pre-scales::
         loss *= num_microbatches / step_global_batch_size * (dp * cp)
     See ``slime/backends/megatron_utils/loss.py:1209-1215``.
  2. Megatron divides by ``clamp(num_tokens, 1)`` then by
     ``num_microbatches``::
         output_tensor /= torch.clamp(num_tokens, min=1)  # num_tokens=1 → no-op
         output_tensor /= num_microbatches
     See ``Megatron-LM/megatron/core/pipeline_parallel/schedules.py:258-264``
     (the ``len(outputs) == 3`` branch with ``not calculate_per_token_loss``).
  3. Backward fills grad buffers; per-mb contributions sum on each rank.
  4. DDP grad sync averages across the DP-with-CP group::
         grad_sum_across_dp_cp_world / (dp * cp)
     See ``Megatron-LM/megatron/core/distributed/distributed_data_parallel.py:283-290``
     (``average_in_collective=False``, ``gradient_scaling_factor = 1.0 / dp_cp_group.size()``).

Composing 1-4 collapses to
    final_grad = total_sum_of_rollout_means / step_global_batch_size,
i.e. the gradient of ``mean_of_per_rollout_means(x)``. That doesn't
contain ``cp`` anywhere, so the grad norm must be identical for any
(dp, cp) factorization of the same world size.

What this test does NOT exercise: the actual Megatron model classes, the
real DDP buffer code, fused optimizers, mixed-precision. We use a plain
``nn.Linear`` with manual all-reduce-average to simulate steps 1-4 above.
The contract here is on *our* scaling math (steps 1 + 4 are slime's;
step 2 is what Megatron does to our 3-tuple). If Megatron later changes
step 2 — e.g. drops the ``/= num_microbatches`` — this test won't catch
it, but the real GPU integration suite (``test_qwen2.5_0.5B_short.py``)
will.
"""

from __future__ import annotations

# Megatron stub must land in sys.modules first; the slime imports inside
# the worker pick it up via this same module. pytest's prepend importmode
# puts ``tests/`` on sys.path so the bare-name import works without an
# ``__init__.py``; mp.spawn children inherit the parent's sys.path.
import _cp_dist_helpers
import pytest
import torch
from _cp_dist_helpers import (
    FOUR_ROLLOUT_EXPECTED_REPORT,
    FOUR_ROLLOUT_RESPONSE_LENGTHS,
    FOUR_ROLLOUT_TOTAL_LENGTHS,
    FOUR_ROLLOUT_X_VALUES,
    cp_chunk_response_tensor,
    free_port,
    init_worker_process_group,
    stub_megatron_in_worker,
)


NUM_GPUS = 0


def _grad_norm_worker(
    rank: int,
    world_size: int,
    cp_size: int,
    dp_size: int,
    seed: int,
    master_port: int,
    result_path: str,
) -> None:
    """One spawned rank.

    Builds a tiny ``nn.Linear`` model (deterministic init via ``seed``),
    runs slime's per-rollout-mean loss reducer with the rank's share of
    the four-rollout fixture, applies the slime-side prescaling, then
    Megatron's per-mb scaling, then ``.backward()``, then a manual
    all-reduce-average across the dp-with-cp group (mirroring DDP's
    ``average_in_collective=False`` path with
    ``gradient_scaling_factor = 1 / dp_cp_world_size``). Rank 0 writes the
    final ``grad_norm`` to ``result_path``.
    """
    import torch.distributed as _dist

    cp_rank = rank % cp_size
    dp_rank = rank // cp_size
    stub_megatron_in_worker(cp_size, cp_rank)

    dp_cp_group = init_worker_process_group(rank, world_size, master_port)
    try:
        from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean

        # Same init across all (dp, cp) configs so the grad we backprop
        # into is comparable. ``manual_seed`` is enough on CPU because we
        # only do one forward/backward and no dropout.
        torch.manual_seed(seed)
        model = torch.nn.Linear(1, 1, bias=False)
        # Force a known weight value to keep the math hand-checkable: with
        # weight = 1.0 and input = x, the linear output equals x, and the
        # grad of (output * x).sum() wrt weight equals (x*x).sum(). That
        # makes the gradient a pure function of the fixture's x values,
        # independent of the random init draw.
        with torch.no_grad():
            model.weight.fill_(1.0)

        all_total_lengths = FOUR_ROLLOUT_TOTAL_LENGTHS
        all_response_lengths = FOUR_ROLLOUT_RESPONSE_LENGTHS
        all_loss_masks = [torch.ones(r, dtype=torch.float32) for r in all_response_lengths]
        all_x = [torch.tensor(v) for v in FOUR_ROLLOUT_X_VALUES]
        step_global_batch_size = 4  # 4 rollouts in the step
        num_microbatches = 1  # this CPU model does the whole rank-share in one mb

        my_indices = [i for i in range(4) if i % dp_size == dp_rank]
        my_tl = [all_total_lengths[i] for i in my_indices]
        my_rl = [all_response_lengths[i] for i in my_indices]
        my_masks = [all_loss_masks[i] for i in my_indices]
        my_x_full = [all_x[i] for i in my_indices]
        # Pre-computed per-rollout denoms = each sample's own mask sum
        # (each rollout in the fixture has exactly one sample, so the
        # per-rollout denom collapses to the per-sample denom).
        my_denoms = torch.tensor([float(m.sum().item()) for m in my_masks], dtype=torch.float32)

        if cp_size == 1:
            x_for_rank = torch.cat(my_x_full)
        else:
            x_for_rank = torch.cat(
                [cp_chunk_response_tensor(x, tl, rl) for tl, rl, x in zip(my_tl, my_rl, my_x_full, strict=True)]
            )

        # === Forward path =====================================================
        # Tiny "model": output[i] = x[i] * weight. We treat the linear
        # output as the per-token quantity the loss is computed over —
        # this stands in for the (logits @ token_emb) the policy loss
        # consumes in real training.
        x_input = x_for_rank.unsqueeze(-1)  # shape [T, 1]
        output = model(x_input).squeeze(-1)  # shape [T]

        reducer = get_sum_of_sample_mean(my_tl, my_rl, my_masks, my_denoms)
        loss = reducer(output)

        # === Step 1: slime's per-rollout-mean prescaling ======================
        # loss.py:1209-1215. ``mpu.get_data_parallel_world_size(with_context_parallel=True)``
        # is the dp-with-cp world size, which is ``world_size`` in this setup.
        loss = loss * num_microbatches / step_global_batch_size * world_size

        # === Step 2: Megatron's forward_step_calc_loss scaling ================
        # schedules.py:258-264 — for the 3-tuple, not-per-token-loss path:
        #     output_tensor /= torch.clamp(num_tokens, min=1)
        #     output_tensor /= num_microbatches
        # slime passes num_tokens=1 in this path (loss.py:1221), so the
        # first divide is a no-op; we keep it explicit to mirror the
        # source faithfully.
        num_tokens_for_scaling = torch.tensor(1.0)  # slime's placeholder
        loss = loss / torch.clamp(num_tokens_for_scaling, min=1.0)
        loss = loss / num_microbatches

        # === Step 3: backward fills per-rank grad =============================
        loss.backward()

        # === Step 4: DDP all-reduce-average across dp-with-cp world ===========
        # distributed_data_parallel.py:283-290, ``average_in_collective=False``
        # case: ``gradient_scaling_factor = 1.0 / dp_cp_group.size()`` is
        # baked into the buffer, so the all-reduce is a SUM and the
        # 1/world_size scaling pre-applies. We do the equivalent here by
        # all-reducing then dividing.
        grad = model.weight.grad.detach()
        _dist.all_reduce(grad, group=dp_cp_group)
        grad = grad / world_size

        # The norm of a 1-element gradient is its absolute value. We
        # report ``grad.item()`` directly so the assertion side can also
        # eyeball the sign, which is more useful than a strict norm when
        # debugging a regression.
        grad_value = grad.item()

        if rank == 0:
            with open(result_path, "w") as f:
                f.write(repr(grad_value))
    finally:
        _dist.destroy_process_group()


def _run_grad_norm_worker(dp_size: int, cp_size: int, tmp_path) -> float:
    """Spawn ``dp_size * cp_size`` workers and return rank-0's final grad."""
    import torch.multiprocessing as mp

    world_size = dp_size * cp_size
    result_path = str(tmp_path / f"grad_dp{dp_size}_cp{cp_size}.txt")
    mp.spawn(
        _grad_norm_worker,
        args=(world_size, cp_size, dp_size, 0, free_port(), result_path),
        nprocs=world_size,
        join=True,
    )
    with open(result_path) as f:
        return float(f.read())


# Subset of (dp, cp) configs to keep runtime down; covers the four
# qualitatively distinct cases:
#   - (1, 1) baseline (no parallelism)
#   - (2, 1) DP-only
#   - (1, 2) CP-only
#   - (2, 2) DP + CP combined
#   - (1, 4) deeper CP-only
#   - (4, 1) deeper DP-only
# The full 3*3 matrix lives in test_metric_report_dist.py — here we just
# want enough coverage to catch a sign/factor regression in the slime
# prescaling math.
_PARALLELISM_CASES = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 4), (4, 1)]


@pytest.mark.unit
@pytest.mark.parametrize("dp_size,cp_size", _PARALLELISM_CASES)
def test_backward_grad_is_cp_invariant(dp_size, cp_size, tmp_path):
    """The post-DDP-average gradient must be identical across all
    (dp, cp) configurations of the same global batch.

    Hand-derivable expectation: with weight = 1.0 and the fixture above,
    the gradient of ``mean_of_per_rollout_means(model(x))`` wrt weight is
    the same quantity the rollout-report tests pin
    (FOUR_ROLLOUT_EXPECTED_REPORT = 1249.875), because for each rollout
    the per-token mean of ``x * weight`` differentiates to the per-token
    mean of ``x``.
    """
    grad = _run_grad_norm_worker(dp_size=dp_size, cp_size=cp_size, tmp_path=tmp_path)
    # Tolerance: float32 / multi-rank sums introduce ~1e-3 relative error
    # on numbers up to ~1250; that's still 5+ digits of agreement. Each
    # (dp, cp) case is pinned to the same hand-derived value, so a sign
    # or factor regression in the prescaling math will fail the whole
    # matrix uniformly — easy to spot in CI logs.
    assert grad == pytest.approx(FOUR_ROLLOUT_EXPECTED_REPORT, rel=1e-4)


# Keep the helpers import load-bearing (it installs the megatron stub).
_ = _cp_dist_helpers


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
