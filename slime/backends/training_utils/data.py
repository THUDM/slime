import logging
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from slime.utils.data import get_minimum_num_micro_batch_size
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.types import RolloutBatch

from ...utils.data import process_rollout_data
from ...utils.misc import Box
from .cp_utils import slice_log_prob_with_cp, slice_with_cp
from .parallel import ParallelState

logger = logging.getLogger(__name__)


def get_rollout_data(args: Namespace, rollout_data_ref: Box, parallel_state: ParallelState) -> RolloutBatch:
    # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
    # Both first pp stage and the last pp stage will receive the data.
    rollout_data = process_rollout_data(
        args,
        rollout_data_ref,
        parallel_state.dp_rank,
        parallel_state.dp_size,
    )
    # move tokens to GPU in advance
    rollout_data["tokens"] = [
        torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in rollout_data["tokens"]
    ]
    rollout_data["loss_masks"] = [
        torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device()) for t in rollout_data["loss_masks"]
    ]
    if "multimodal_train_inputs" in rollout_data:
        # Move multimodal training tensors to GPU in advance
        rollout_data["multimodal_train_inputs"] = [
            (
                {key: tensor.to(device=torch.cuda.current_device()) for key, tensor in mm_dict.items()}
                if mm_dict is not None
                else None
            )
            for mm_dict in rollout_data["multimodal_train_inputs"]
        ]

    if args.qkv_format == "bshd":
        # TODO: micro-batch wise dynamic, possibly move to @data.py:get_data_iterator
        max_seq_len = max(rollout_data["total_lengths"])

        # pad to reduce memory fragmentation and maybe make the computation faster
        pad_size = parallel_state.tp_size * args.data_pad_size_multiplier
        max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size

        rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])

    if "rollout_log_probs" in rollout_data:
        rollout_data["rollout_log_probs"] = [
            torch.tensor(
                slice_log_prob_with_cp(
                    log_prob,
                    total_length,
                    response_length,
                    parallel_state,
                    args.qkv_format,
                    rollout_data["max_seq_lens"][i] if args.qkv_format == "bshd" else None,
                ),
                device=torch.cuda.current_device(),
                dtype=torch.float32,
            )
            for i, (log_prob, total_length, response_length) in enumerate(
                zip(
                    rollout_data["rollout_log_probs"],
                    rollout_data["total_lengths"],
                    rollout_data["response_lengths"],
                    strict=False,
                )
            )
        ]
    if "rollout_routed_experts" in rollout_data:
        rollout_data["rollout_routed_experts"] = [torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]]
    return rollout_data


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    parallel_state: ParallelState,
    pad_multiplier: int = 128,
    qkv_format: str = "thd",
    get_position_ids: bool = False,
) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
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

    if "dynamic_global_batch_size" in data_iterator.rollout_data:
        batch["dynamic_global_batch_size"] = data_iterator.rollout_data["dynamic_global_batch_size"]

    tokens = batch["tokens"]
    # use 0 as the pad token id should be fine?
    pad_token_id = 0
    pad_size = parallel_state.dp_size * pad_multiplier

    # for cp, we need all tokens to calculate logprob
    batch["unconcat_tokens"] = tokens

    cp_size = parallel_state.cp_size

    if qkv_format == "bshd":
        max_seqlen = batch["max_seq_lens"][0]
        assert max([t.size(0) for t in tokens]) <= max_seqlen
        tokens = [slice_with_cp(t, pad_token_id, parallel_state, qkv_format, max_seqlen) for t in tokens]
        tokens = torch.stack(tokens)

    elif qkv_format == "thd":
        tokens = [slice_with_cp(t, pad_token_id, parallel_state, qkv_format) for t in tokens]

        cu_seqlens = [0]
        for t in tokens:
            cu_seqlens.append(cu_seqlens[-1] + t.size(0))

        tokens = torch.cat(tokens)

        # Always pad to reduce memory fragmentation and maybe make the computation faster
        pad = (pad_size - tokens.size(0) % pad_size) % pad_size
        if pad != 0:
            tokens = F.pad(tokens, (0, pad), value=pad_token_id)
            cu_seqlens.append(cu_seqlens[-1] + pad)

        # thd requires the cu_seqlens to be of the origin length
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int).cuda() * cp_size
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        tokens = tokens.unsqueeze(0)

        batch["cu_seqlens"] = cu_seqlens
        batch["max_seqlen"] = max_seqlen
    else:
        raise ValueError(f"Unsupported qkv_format: {qkv_format}")

    batch["tokens"] = tokens

    if get_position_ids:
        position_ids_list = []
        for t in batch["unconcat_tokens"]:
            seq_len = t.size(0)
            pos_ids = torch.arange(seq_len, device=t.device, dtype=torch.long)
            position_ids_list.append(pos_ids)

        if qkv_format == "bshd":
            position_ids = [slice_with_cp(p, 0, parallel_state, qkv_format, max_seqlen) for p in position_ids_list]
            position_ids = torch.stack(position_ids)
        elif qkv_format == "thd":
            position_ids = [slice_with_cp(p, 0, parallel_state, qkv_format) for p in position_ids_list]
            position_ids = torch.cat(position_ids)
            if pad != 0:
                position_ids = F.pad(position_ids, (0, pad), value=0)
            position_ids = position_ids.unsqueeze(0)

        batch["position_ids"] = position_ids

    # loss masks
    loss_masks = []
    for loss_mask, total_length, response_length in zip(
        batch["loss_masks"],
        batch["total_lengths"],
        batch["response_lengths"],
        strict=True,
    ):
        prompt_length = total_length - response_length
        loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
        loss_mask = slice_with_cp(loss_mask, 0, parallel_state, qkv_format, max_seqlen)
        loss_masks.append(loss_mask)

    if qkv_format == "bshd":
        loss_masks = torch.stack(loss_masks)
    elif qkv_format == "thd":
        loss_masks = torch.cat(loss_masks)
        loss_masks = F.pad(loss_masks, (0, pad), value=0).unsqueeze(0)

    assert loss_masks.shape == tokens.shape, f"loss_masks.shape: {loss_masks.shape}, tokens.shape: {tokens.shape}"
    batch["full_loss_masks"] = loss_masks

    # Process multimodal training tensors if present
    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None:
        multimodal_data = {}  # key -> concatenated tensor
        multimodal_num_items = {}  # key -> list of item counts per sequence
        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is not None:
                for key, mm_tensor in mm_input_dict.items():
                    if key not in multimodal_data:
                        multimodal_data[key] = mm_tensor
                        multimodal_num_items[key] = [mm_tensor.size(0)]
                    else:
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
                        multimodal_num_items[key].append(mm_tensor.size(0))
        batch["multimodal_train_inputs"] = multimodal_data
        batch["multimodal_num_items"] = multimodal_num_items

    return batch


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
    parallel_state: ParallelState,
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
    dp_size = parallel_state.dp_size
    dp_group = parallel_state.dp_group
    vpp_size = parallel_state.vpp_size
    microbatch_group_size_per_vp_stage = parallel_state.microbatch_group_size_per_vp_stage

    cp_size = parallel_state.cp_size

    num_local_samples = len(rollout_data["total_lengths"])
    global_batch_size = rollout_data.get("dynamic_global_batch_size", args.global_batch_size)
    num_local_gbs = global_batch_size // dp_size
    num_steps_per_rollout = num_local_samples // num_local_gbs

    if global_batch_size != args.global_batch_size:
        logger.info(
            f"Using dynamic global_batch_size={global_batch_size} (original={args.global_batch_size}), "
            f"num_local_samples={num_local_samples}, num_steps_per_rollout={num_steps_per_rollout}"
        )

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
