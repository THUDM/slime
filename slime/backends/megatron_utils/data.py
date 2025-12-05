import logging
from argparse import Namespace
from collections.abc import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

from slime.utils import train_metric_utils
from slime.utils.data import get_minimum_num_micro_batch_size
from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.types import RolloutBatch

from ...utils import tracking_utils
from .cp_utils import get_sum_of_sample_mean, slice_with_cp

logger = logging.getLogger(__name__)


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
) -> dict[str, torch.Tensor | PackedSeqParams | list[torch.Tensor] | None]:
    """
    Generate a CP-ready micro-batch with packed sequence parameters.

    Steps:
    - Fetch raw fields via iterator.
    - Save original token tensors under "unconcat_tokens".
    - Slice tokens into two chunks for Context Parallelism (CP), concatenate, and pad to a configurable multiple.
    - Build cu_seqlens and `PackedSeqParams` with T-H-D layout (T: sequence length, H: attention heads, D: head dimension).

    Args:
        data_iterator: Iterator providing micro-batch data.
        keys: List of keys to fetch from the iterator.
        pad_multiplier: Multiplier for padding size calculation (default: 128).

    Returns a dict including:
    - "tokens": torch.LongTensor of shape [1, T_padded] on the current CUDA device
    - "unconcat_tokens": list[torch.LongTensor] for the micro-batch before CP slicing/concat
    - "packed_seq_params": PackedSeqParams with T-H-D settings (cu_seqlens on CUDA, dtype=int)
    Plus any other requested keys forwarded from the iterator.
    """

    assert "tokens" in keys
    batch = data_iterator.get_next(keys)

    packed_seq_params = None
    tokens = batch["tokens"]
    # use 0 as the pad token id should be fine?
    pad_token_id = 0

    # for cp, we need all tokens to calculate logprob
    batch["unconcat_tokens"] = tokens

    cp_size = mpu.get_context_parallel_world_size()
    tokens = [slice_with_cp(t, pad_token_id) for t in tokens]

    cu_seqlens = [0]
    for t in tokens:
        cu_seqlens.append(cu_seqlens[-1] + t.size(0))

    tokens = torch.cat(tokens)

    # Always pad to reduce memory fragmentation and maybe make the computation faster
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier
    pad = (pad_size - tokens.size(0) % pad_size) % pad_size
    if pad != 0:
        tokens = F.pad(tokens, (0, pad), value=pad_token_id)
        cu_seqlens.append(cu_seqlens[-1] + pad)

    # thd requires the cu_seqlens to be of the origin length
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int).cuda() * cp_size
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )

    tokens = tokens.unsqueeze(0)
    batch["tokens"] = tokens
    batch["packed_seq_params"] = packed_seq_params
    return batch


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, float],
) -> dict[str, float] | None:
    """
    Gather per-rank metrics, reduce by mean on the DP source rank, and log.

    Expects `log_dict` to contain plain scalars. The DP source rank prints and
    optionally logs to WandB/TensorBoard with a step derived from `rollout_id` and
    batch sizes. Returns the reduced dict on the DP source rank; returns None on others.
    """

    if mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)

        gathered_log_dict = [None] * dp_size
        # Not sure if this will be a performance bottleneck.
        dist.gather_object(
            log_dict,
            gathered_log_dict,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )

        reduced_log_dict = {
            f"{metric_name}/{key}": sum([d[key] for d in gathered_log_dict]) / dp_size for key in log_dict
        }
        logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")

        # Calculate step once to avoid duplication
        step = compute_rollout_step(args, rollout_id)
        reduced_log_dict["rollout/step"] = step
        tracking_utils.log(args, reduced_log_dict, step_key="rollout/step")

        return reduced_log_dict
    else:
        dist.gather_object(
            log_dict,
            None,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )
        return None


class DataIterator:
    """Micro-batch iterator over rollout dicts.

    Supports either fixed contiguous micro-batches or an explicit per-step
    index schedule (for dynamic batch sizing / sequence-length balancing).
    """

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_size: int | None = None,
        micro_batch_indices: list[list[int]] | None = None,
    ) -> None:
        """Initialize an iterator over `rollout_data`.

        Args:
            rollout_data: Dict of per-sample fields for the local step.
            micro_batch_size: Fixed contiguous slice size when not using dynamic scheduling.
            micro_batch_indices: Explicit indices per micro-batch when using dynamic balancing.
                Must be mutually exclusive with `micro_batch_size`.
        """
        self.rollout_data = rollout_data
        self.micro_batch_size = micro_batch_size
        self.micro_batch_indices = micro_batch_indices
        assert micro_batch_size is None or micro_batch_indices is None
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        """Return the next micro-batch for the requested keys.

        - If `micro_batch_indices` is provided, selects rows according to the current
          index list for each requested key.
        - Otherwise, slices a contiguous window of size `micro_batch_size` starting
          at the current offset.

        Returns a dict mapping each key to a list subset (or None if absent).
        """
        batch = {}
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                if self.micro_batch_indices is not None:
                    indices = self.micro_batch_indices[self.offset]
                    batch[key] = [vals[i] for i in indices]
                else:
                    assert self.offset + self.micro_batch_size <= len(
                        vals
                    ), f"offset: {self.offset}, micro_batch_size: {self.micro_batch_size}, len(vals): {len(vals)}"
                    batch[key] = vals[self.offset : self.offset + self.micro_batch_size]

        if self.micro_batch_indices is not None:
            self.offset += 1
        else:
            self.offset += self.micro_batch_size
        return batch

    def reset(self) -> "DataIterator":
        """Reset internal offset to the start and return self."""
        self.offset = 0
        return self


def get_data_iterator(
    args: Namespace,
    model: torch.nn.Module | Sequence[torch.nn.Module],
    rollout_data: RolloutBatch,
) -> tuple[list[DataIterator], list[int]]:
    """
    Create iterators and a micro-batch schedule for a rollout step.

    - If `use_dynamic_batch_size` is False, splits into fixed-size contiguous
      micro-batches of `micro_batch_size`.
    - If True, computes the number of micro-batches per local step based on
      `max_tokens_per_gpu` and per-sample lengths, all-reduces to a DP-wide
      maximum, optionally enforces divisibility for Virtual Pipeline Parallelism (VPP), and builds a balanced
      index schedule to equalize token counts across micro-batches.

    Returns `(data_iterators, num_microbatches)` where:
    - `data_iterators`: list of `DataIterator`, one per VPP stage (size 1 if VPP disabled)
    - `num_microbatches`: list[int], one per local step in the rollout (length = steps)
    """
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    dp_group = mpu.get_data_parallel_group()
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is None:
        vpp_size = 1
    if vpp_size > 1:
        from megatron.core.utils import get_model_config

        config = get_model_config(model[0])
        microbatch_group_size_per_vp_stage = config.microbatch_group_size_per_vp_stage
    cp_size = mpu.get_context_parallel_world_size()

    num_local_samples = len(rollout_data["total_lengths"])
    num_local_gbs = args.global_batch_size // dp_size
    num_steps_per_rollout = num_local_samples // num_local_gbs

    def _generate_data_iterator(rollout_data, micro_batch_size, micro_batch_indices=None):
        data_iterator = []
        for _ in range(vpp_size):
            data_iterator.append(DataIterator(rollout_data, micro_batch_size, micro_batch_indices))
        return data_iterator

    if not args.use_dynamic_batch_size:
        num_microbatches = [num_local_gbs // args.micro_batch_size for _ in range(num_steps_per_rollout)]
        data_iterator = _generate_data_iterator(rollout_data, args.micro_batch_size)
    else:
        assert args.max_tokens_per_gpu is not None
        # calculate the number of mirobatches for each step
        samples = rollout_data["total_lengths"]
        assert len(samples) == num_local_samples
        num_microbatches = []
        for i in range(num_steps_per_rollout):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            num_microbatches.append(
                get_minimum_num_micro_batch_size(samples[start:end], args.max_tokens_per_gpu * cp_size)
            )

        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)

        if vpp_size > 1:
            # vpp requies the number of microbatches to be divisible by vpp_size
            num_microbatches = torch.clamp(
                num_microbatches // microbatch_group_size_per_vp_stage * microbatch_group_size_per_vp_stage,
                min=1,
            )

        num_microbatches = num_microbatches.tolist()

        # balance the each micro batch
        samples = rollout_data["total_lengths"]
        # balance the number of mirobatches across steps
        micro_batch_indices = []
        for i, num_mbs in enumerate(num_microbatches):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            samples = rollout_data["total_lengths"][start:end]
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
            micro_batch_indices.extend(partitions)

        assert len(set(sum(micro_batch_indices, []))) == num_local_samples

        data_iterator = _generate_data_iterator(rollout_data, None, micro_batch_indices)

    return (
        data_iterator,
        num_microbatches,
    )


def log_rollout_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Summarize rollout fields and log reduced metrics on PP last stage, TP rank 0.

    - Tensor-valued lists are concatenated and averaged. For token-level metrics
      like log-probs/returns/advantages/values, computes a CP-correct sample mean
      using `loss_masks` and total/response lengths.
    - Non-tensor lists are averaged elementwise.
    - Scalars are converted to Python numbers.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        cp_size = mpu.get_context_parallel_world_size()
        log_dict = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "loss_masks",
                "sample_indices",
                "rollout_routed_experts",
            ]:
                continue
            # Upload per sample mean for each rollout value
            # There are the following assumptions:
            # - Each dp rank has the same number of samples
            if isinstance(val, (list, tuple)):
                if isinstance(val[0], torch.Tensor):
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    val = torch.cat(val).clone().detach()
                    if key in ["log_probs", "ref_log_probs", "rollout_log_probs", "returns", "advantages", "values"]:
                        sum_of_sample_mean = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks)
                        val = cp_size * sum_of_sample_mean(val) / len(loss_masks)
                    else:
                        val = val.mean() * cp_size
                else:
                    val = sum(val) / len(val)
            elif isinstance(val, torch.Tensor):
                val = val.float().mean()
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")
            log_dict[key] = val.item() if isinstance(val, torch.Tensor) else val

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict)
        if args.ci_test and reduced_log_dict is not None:
            if (
                rollout_id == 0
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                assert reduced_log_dict["rollout/log_probs"] == reduced_log_dict["rollout/ref_log_probs"]
            if "rollout/log_probs" in reduced_log_dict:
                assert -0.5 < reduced_log_dict["rollout/log_probs"] < 0
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 0.5

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data)


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Log multi-turn auxiliary metrics such as raw/observed response lengths and rounds.

    Operates only on PP last stage and TP rank 0. Uses GPU tensors when available
    to compute statistics without host transfers.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key == "loss_masks":
                if val:  # Check if val is not empty
                    device = val[0].device  # Get device from first tensor

                    # Vectorized length calculation using torch
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )

                    # Vectorized sum calculation using torch - stay on GPU
                    wo_obs_response_lengths = torch.tensor(
                        [v.sum().item() for v in val], dtype=torch.float32, device=device
                    )
                    log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
                    log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
                    log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
            if key == "round_number":
                # Use numpy for vectorized round number statistics
                round_number_array = np.array(val)
                log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
                log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
                log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
        gather_log_data("multi_turn", args, rollout_id, log_dict)


def log_passrate(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Compute pass@k metrics from `raw_reward` groups and log the results.

    `raw_reward` is reshaped to `[group_number, group_size]`, then pass@k is
    estimated per problem and averaged.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key != "raw_reward":
                continue

            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )

        gather_log_data("passrate", args, rollout_id, log_dict)


def log_perf_data(rollout_id: int, args: Namespace) -> None:
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.is_pipeline_last_stage()
            and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        ),
        compute_total_fwd_flops=lambda seq_lens: calculate_fwd_flops(seqlens=seq_lens, args=args)
        / dist.get_world_size()
        / 1e12,
    )


def sync_actor_critic_data(
    args: Namespace,
    rollout_data: RolloutBatch | None = None,
    group: dist.ProcessGroup | None = None,
) -> None:
    """
    Broadcast `values` (from critic) and optionally `log_probs`/`ref_log_probs`
    (from actor) across PP ranks to align data dependencies.

    - Values are broadcast from src=1.
    - Log-probs and ref-log-probs are broadcast from src=0 when KL is used.
    Updates `rollout_data` in place with the synchronized tensors.
    """
    log_probs_key = "log_probs" if not args.use_rollout_logprobs else "rollout_log_probs"
    values, log_probs, ref_log_probs = map(rollout_data.get, ("values", log_probs_key, "ref_log_probs"))

    # return when not the pp last stage
    if not values and not log_probs:
        return

    handles = []

    if not values:
        values = [torch.empty_like(log_prob) for log_prob in log_probs]
    for value in values:
        handles.append(dist.broadcast(value, src=1, group=group, async_op=True))

    if args.kl_coef != 0 or args.use_kl_loss:
        if not log_probs:
            log_probs = [torch.empty_like(value) for value in values]
        if not ref_log_probs:
            ref_log_probs = [torch.empty_like(value) for value in values]
        for ref_log_prob, log_prob in zip(ref_log_probs, log_probs, strict=False):
            handles.append(dist.broadcast(log_prob, src=0, group=group, async_op=True))
            handles.append(dist.broadcast(ref_log_prob, src=0, group=group, async_op=True))

    for handle in handles:
        handle.wait()

    rollout_data.update(
        {
            k: v
            for k, v in {
                "values": values,
                log_probs_key: log_probs,
                "ref_log_probs": ref_log_probs,
            }.items()
            if v is not None
        }
    )
