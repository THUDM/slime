from collections.abc import Sequence

import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

from slime.utils import accelerator
from slime.utils.types import RolloutBatch

from .cp_utils import slice_with_cp

# One-shot latch to suppress repeated empty-microbatch warnings (see get_batch).
_empty_microbatch_warned: bool = False


def _fill_empty_microbatch_placeholder(batch: dict, keys, pad_token_id: int) -> list:
    """Populate ``batch`` with a self-consistent 1-token placeholder sample
    and return the new ``tokens`` list. Invoked by ``get_batch`` only when
    the local DP rank has no real samples for this micro-batch.

    Invariants after this function returns (matching the 1-token placeholder):
      batch["tokens"]           == [pad_token_tensor]
      batch["total_lengths"]    == [1]
      batch["response_lengths"] == [0]
      batch["loss_masks"]       == [zero-size int tensor]
      batch["max_seq_lens"]     == [1]  (bshd-only, if requested)
      every other list-valued key in ``keys`` has exactly one entry

    The downstream effect of response_length=0 is that CP-chunk sizes are
    0 on every CP rank, so per-sample log_probs / rollout_log_probs / etc.
    contribute 0-size tensors and ``sum_of_sample_mean`` splits a 0-sized
    slice with ``split_sizes=[0]``.

    If a new rollout-schema field lands in ``keys`` without a corresponding
    entry in the ``placeholder_for_key`` map below or matching the
    per-token fp32 default, the post-fill invariant assertion fails here
    — not 400 lines downstream inside ``torch.split``.
    """
    device = torch.cuda.current_device()
    placeholder_for_key = {
        "total_lengths": 1,
        "response_lengths": 0,
        "max_seq_lens": 1,
        # length-0 mask; the post-padding step in get_batch aligns it
        # with the size-1 placeholder token.
        "loss_masks": torch.zeros(0, dtype=torch.int, device=device),
        "multimodal_train_inputs": None,
    }
    # Per-token tensor-valued keys (log_probs, ref_log_probs, advantages,
    # etc.) default to an empty fp32 tensor so torch.cat downstream works.
    # New per-token fields of a different dtype must add themselves to
    # placeholder_for_key instead of relying on this default.
    per_token_default = torch.zeros(0, dtype=torch.float32, device=device)

    placeholder = torch.tensor([pad_token_id], dtype=torch.long, device=device)
    batch["tokens"] = [placeholder]
    for key in keys:
        if key == "tokens":
            continue
        v = batch.get(key)
        if v is None:
            continue
        if isinstance(v, list) and len(v) == 0:
            if key in placeholder_for_key:
                batch[key] = [placeholder_for_key[key]]
            else:
                batch[key] = [per_token_default]

    # Post-fill invariant: every list-valued key now has exactly one entry.
    # If this fails, a new schema field was added to `keys` without a
    # placeholder rule above.
    for key in keys:
        if key == "tokens":
            continue
        v = batch.get(key)
        if isinstance(v, list):
            assert len(v) == 1, (
                f"empty-microbatch placeholder did not fill key={key!r}; "
                f"got len(batch[{key!r}])={len(v)}, expected 1. Add a "
                f"placeholder_for_key entry in _fill_empty_microbatch_placeholder."
            )

    return batch["tokens"]


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
    allgather_cp: bool = False,
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

    tokens = batch["tokens"]
    # use 0 as the pad token id should be fine?
    pad_token_id = 0
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier

    # DP-imbalance guard: when DP ranks need different microbatch counts, the
    # pipeline schedule loops for `max(num_mbs)` steps and ranks with surplus
    # see empty micro-batches. Before, this raised a confusing
    # ``torch.cat(): expected a non-empty list of Tensors``; replace with a
    # self-consistent single-token placeholder so downstream operations
    # (loss_mask align, CP slicing, per-sample log-prob extraction,
    # sum_of_sample_mean split) all agree on 0 response tokens.
    #
    # Invariants of the placeholder:
    #   tokens          = [pad]              (size 1 — prompt only)
    #   total_lengths   = [1]
    #   response_lengths= [0]
    #   loss_masks      = [ [] ]             (0 response tokens → empty mask)
    #   max_seq_lens    = [1]                (bshd-only, if requested)
    # With response_length=0 the CP-chunk sizes are 0 on every CP rank, so
    # the fake sample contributes 0-size tensors to per-sample log_probs /
    # rollout_log_probs etc. `split_sizes=[0]` then splits a 0-sized slice.
    if not tokens:
        # Log once per rank — on an unbalanced multi-hour training run, every
        # empty microbatch fires this path and per-instance logging would
        # drown the logs.
        global _empty_microbatch_warned
        if not _empty_microbatch_warned:
            logger.warning(
                "get_batch: empty micro-batch (DP rank has fewer partitions "
                "than the collective max); inserting 1-token placeholder. "
                "Further occurrences on this rank will not be re-logged."
            )
            _empty_microbatch_warned = True
        tokens = _fill_empty_microbatch_placeholder(batch, keys, pad_token_id)

    # for cp, we need all tokens to calculate logprob
    batch["unconcat_tokens"] = tokens

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if allgather_cp:
        # DSA mode: concatenate all sequences first, then slice once with CP.
        # We also pad the *global* concatenated stream to make per-rank chunks equal.
        cu_seqlens_list: list[int] = [0]
        for t in tokens:
            cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))

        tokens = torch.cat(tokens, dim=0)

        # Pad global stream so (1) divisible by cp_size (equal chunks),
        # (2) divisible by pad_size (reduce fragmentation).
        global_pad_size = cp_size * pad_size
        pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
        if pad != 0:
            tokens = F.pad(tokens, (0, pad), value=pad_token_id)
            cu_seqlens_list.append(cu_seqlens_list[-1] + pad)

        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int, device=accelerator.current_device())
        tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
    else:
        tokens = [slice_with_cp(t, pad_token_id) for t in tokens]

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
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int, device=accelerator.device()) * cp_size

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

    # loss masks
    loss_masks = []
    for loss_mask, total_length, response_length in zip(
        batch["loss_masks"],
        batch["total_lengths"],
        batch["response_lengths"],
        strict=True,
    ):
        prompt_length = total_length - response_length
        # Align mask to token stream positions (prompt_length-1 left pad, 1 right pad)
        loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
        if allgather_cp:
            loss_masks.append(loss_mask)
            continue
        loss_mask = slice_with_cp(loss_mask, 0)
        loss_masks.append(loss_mask)

    if allgather_cp:
        # DSA: concatenate first (same as tokens), pad globally (same pad as above), then slice once.
        loss_masks = torch.cat(loss_masks, dim=0)
        if pad != 0:
            loss_masks = F.pad(loss_masks, (0, pad), value=0)
        loss_masks = loss_masks.chunk(cp_size, dim=0)[cp_rank].unsqueeze(0)
    else:
        loss_masks = torch.cat(loss_masks)
        loss_masks = F.pad(loss_masks, (0, pad), value=0).unsqueeze(0)

    assert loss_masks.shape == tokens.shape, f"loss_masks.shape: {loss_masks.shape}, tokens.shape: {tokens.shape}"
    batch["full_loss_masks"] = loss_masks

    # Process multimodal training tensors if present
    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None:
        multimodal_data = {}  # key -> concatenated tensor
        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is not None:
                for key, mm_tensor in mm_input_dict.items():
                    if key not in multimodal_data:
                        multimodal_data[key] = mm_tensor
                    else:
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
        batch["multimodal_train_inputs"] = multimodal_data

    return batch


class DataIterator:
    """Iterator over a rollout dict following an explicit micro-batch index schedule."""

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_indices: list[list[int]],
    ) -> None:
        """Initialize an iterator over ``rollout_data``.

        Args:
            rollout_data: Dict of per-sample fields for this DP rank.
            micro_batch_indices: List of mbs, each mbs being the local sample indices to select.
        """
        self.rollout_data = rollout_data
        self.micro_batch_indices = micro_batch_indices
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        """Return the next micro-batch for the requested keys.

        Returns a dict mapping each key to a list subset (or None if absent).
        """
        batch = {}
        indices = self.micro_batch_indices[self.offset]
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                batch[key] = [vals[i] for i in indices]
        self.offset += 1
        return batch

    def reset(self) -> "DataIterator":
        """Reset internal offset to the start and return self."""
        self.offset = 0
        return self


def get_data_iterator(rollout_data: RolloutBatch) -> list[DataIterator]:
    """Build one ``DataIterator`` per VPP stage from the pre-computed schedule in ``rollout_data``."""
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    micro_batch_indices = rollout_data["micro_batch_indices"]
    return [DataIterator(rollout_data, micro_batch_indices) for _ in range(vpp_size)]


def tensors_to_cpu(tensor_list):
    """Move a list of GPU tensors to CPU for Ray object store transfer.

    Args:
        tensor_list: List of GPU tensors, or None.

    Returns:
        List of CPU tensors (detached), or None if input is None.
    """
    if tensor_list is None:
        return None
    return [t.detach().cpu() for t in tensor_list]


def tensors_to_gpu(tensor_list, device=None):
    """Move a list of CPU tensors back to GPU.

    Args:
        tensor_list: List of CPU tensors, or None.
        device: Target CUDA device. If None, uses current device.

    Returns:
        List of GPU tensors, or None if input is None.
    """
    if tensor_list is None:
        return None
    if device is None:
        device = accelerator.current_device()
    return [t.to(device=device, dtype=torch.float32) for t in tensor_list]
