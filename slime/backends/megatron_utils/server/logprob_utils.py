import logging
from functools import partial
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, get_logits_and_tokens_offset_with_cp
from slime.backends.megatron_utils.data import get_data_iterator
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, get_responses
from slime.backends.megatron_utils.model import forward_only

logging.getLogger().setLevel(logging.WARNING)


@torch.no_grad()
def sample_from_vocab_parallel_logits_without_full_gather(
    vocab_parallel_logits: torch.Tensor,
    *,
    sample_n: int,
    tp_group: dist.ProcessGroup | None = None,
    reduction_chunk_size: int = 4096,
    global_vocab_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample token ids/log-probs from TP-sharded logits without full-vocab gather.

    This performs a two-stage sampling:
    1) Sample which TP rank owns each sample slot from per-rank probability mass.
    2) Each TP rank samples local vocab ids only for the slots assigned to it.
    Final `(token_id, log_prob)` tensors are merged with TP all-reduce(max), so
    no rank materializes full-vocab probabilities.

    Args:
        vocab_parallel_logits: Shape `[num_tokens, vocab_per_tp]` logits shard.
        sample_n: Number of samples per token position, with replacement.
        tp_group: Tensor-parallel process group. If None, uses Megatron TP group.
        reduction_chunk_size: Row chunk size for denominator computation.
        global_vocab_size: Optional unpadded vocab size. If set, padded logits
            (outside `[0, global_vocab_size)`) are excluded from sampling.

    Returns:
        sampled_token_ids: `[num_tokens, sample_n]` global token ids.
        sampled_log_probs: `[num_tokens, sample_n]` log-probabilities.
    """
    if vocab_parallel_logits.dim() != 2:
        raise ValueError(f"Expected 2D logits, got shape={tuple(vocab_parallel_logits.shape)}")
    if sample_n < 0:
        raise ValueError(f"sample_n must be >= 0, got {sample_n}")
    if reduction_chunk_size <= 0:
        raise ValueError(f"reduction_chunk_size must be > 0, got {reduction_chunk_size}")

    num_tokens, vocab_per_tp = vocab_parallel_logits.shape
    device = vocab_parallel_logits.device
    logits_dtype = vocab_parallel_logits.dtype

    if tp_group is None:
        tp_group = mpu.get_tensor_model_parallel_group()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    reduction_dtype = torch.float32 if logits_dtype in (torch.float16, torch.bfloat16) else logits_dtype

    vocab_start = tp_rank * vocab_per_tp
    valid_vocab_per_tp = vocab_per_tp
    if global_vocab_size is not None:
        if global_vocab_size < 0:
            raise ValueError(f"global_vocab_size must be >= 0, got {global_vocab_size}")
        valid_vocab_per_tp = max(min(global_vocab_size - vocab_start, vocab_per_tp), 0)

    if valid_vocab_per_tp == 0:
        local_max = torch.full((num_tokens, 1), -torch.inf, dtype=logits_dtype, device=device)
    elif valid_vocab_per_tp == vocab_per_tp:
        local_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
    else:
        local_max = vocab_parallel_logits[:, :valid_vocab_per_tp].max(dim=-1, keepdim=True).values

    global_max = local_max.clone()
    if tp_size > 1:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

    local_exp_sums = torch.zeros((num_tokens, 1), dtype=reduction_dtype, device=device)
    if valid_vocab_per_tp > 0:
        for start in range(0, num_tokens, reduction_chunk_size):
            end = min(start + reduction_chunk_size, num_tokens)
            logits_part = vocab_parallel_logits[start:end, :valid_vocab_per_tp]
            centered = logits_part.to(reduction_dtype) - global_max[start:end].to(reduction_dtype)
            local_exp_sums[start:end] = centered.exp_().sum(dim=-1, keepdim=True)

    global_exp_sums = local_exp_sums.clone()
    if tp_size > 1:
        dist.all_reduce(global_exp_sums, op=dist.ReduceOp.SUM, group=tp_group)
    denom = global_exp_sums.clamp_min(torch.finfo(global_exp_sums.dtype).tiny)

    local_tp_mass = (local_exp_sums / denom).squeeze(-1)
    if tp_size > 1:
        gathered_tp_masses = [torch.empty_like(local_tp_mass) for _ in range(tp_size)]
        dist.all_gather(gathered_tp_masses, local_tp_mass.contiguous(), group=tp_group)
        tp_masses = torch.stack(gathered_tp_masses, dim=-1)
    else:
        tp_masses = local_tp_mass.unsqueeze(-1)

    if tp_rank == 0:
        tp_probs = tp_masses / tp_masses.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(tp_masses.dtype).tiny)
        tp_assignments = torch.multinomial(tp_probs, num_samples=sample_n, replacement=True)
    else:
        tp_assignments = torch.empty((num_tokens, sample_n), dtype=torch.long, device=device)
    if tp_size > 1:
        dist.broadcast(tp_assignments, src=mpu.get_tensor_model_parallel_src_rank(), group=tp_group)

    owner_slots = tp_assignments.eq(tp_rank)
    if valid_vocab_per_tp == 0 and owner_slots.any():
        raise RuntimeError("Received sample slots on a TP rank with zero valid vocab.")

    sampled_token_ids = torch.full((num_tokens, sample_n), -1, dtype=torch.long, device=device)
    sampled_log_probs = torch.full((num_tokens, sample_n), -torch.inf, dtype=logits_dtype, device=device)
    log_denom = denom.log()

    if valid_vocab_per_tp > 0:
        local_slot_count = owner_slots.sum(dim=-1)
        max_slot_count = int(local_slot_count.max().item())
        for slot_count in range(1, max_slot_count + 1):
            row_ids = torch.nonzero(local_slot_count == slot_count, as_tuple=False).flatten()
            if row_ids.numel() == 0:
                continue

            row_logits = vocab_parallel_logits.index_select(0, row_ids)[:, :valid_vocab_per_tp]
            row_max = global_max.index_select(0, row_ids).to(reduction_dtype)
            local_row_max = row_logits.max(dim=-1, keepdim=True).values.to(reduction_dtype)
            row_weights = (row_logits.to(reduction_dtype) - local_row_max).exp_()
            local_ids = torch.multinomial(row_weights, num_samples=slot_count, replacement=True)

            slot_cols = torch.nonzero(owner_slots.index_select(0, row_ids), as_tuple=False)[:, 1].view(-1, slot_count)

            selected_logits = torch.gather(row_logits, dim=-1, index=local_ids)
            selected_log_probs = (
                selected_logits.to(reduction_dtype) - row_max - log_denom.index_select(0, row_ids)
            ).to(logits_dtype)
            global_ids = local_ids + vocab_start

            sampled_token_ids[row_ids.unsqueeze(-1), slot_cols] = global_ids
            sampled_log_probs[row_ids.unsqueeze(-1), slot_cols] = selected_log_probs

    if tp_size > 1:
        dist.all_reduce(sampled_token_ids, op=dist.ReduceOp.MAX, group=tp_group)
        dist.all_reduce(sampled_log_probs, op=dist.ReduceOp.MAX, group=tp_group)

    if (sampled_token_ids < 0).any():
        raise RuntimeError("Distributed TP sampling produced incomplete sample slots.")

    return sampled_token_ids, sampled_log_probs


@torch.no_grad()
def get_label_token_log_probs_from_vocab_parallel_logits(
    vocab_parallel_logits: torch.Tensor,
    label_token_ids: torch.Tensor,
    *,
    tp_group: dist.ProcessGroup | None = None,
    reduction_chunk_size: int = 4096,
) -> torch.Tensor:
    if vocab_parallel_logits.dim() != 2:
        raise ValueError(f"Expected 2D logits, got shape={tuple(vocab_parallel_logits.shape)}")
    if label_token_ids.dim() != 2:
        raise ValueError(f"Expected 2D label_token_ids, got shape={tuple(label_token_ids.shape)}")
    if vocab_parallel_logits.size(0) != label_token_ids.size(0):
        raise ValueError(
            "label_token_ids must align with response positions: "
            f"{vocab_parallel_logits.size(0)} vs {label_token_ids.size(0)}"
        )
    if reduction_chunk_size <= 0:
        raise ValueError(f"reduction_chunk_size must be > 0, got {reduction_chunk_size}")

    num_tokens, partition_vocab_size = vocab_parallel_logits.shape
    num_labels = label_token_ids.size(1)
    if num_labels == 0:
        return torch.empty((num_tokens, 0), dtype=vocab_parallel_logits.dtype, device=vocab_parallel_logits.device)

    device = vocab_parallel_logits.device
    logits_dtype = vocab_parallel_logits.dtype
    reduction_dtype = torch.float32 if logits_dtype in (torch.float16, torch.bfloat16) else logits_dtype

    if tp_group is None:
        tp_group = mpu.get_tensor_model_parallel_group()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    from megatron.core.tensor_parallel.utils import VocabUtility  # type: ignore

    vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
        partition_vocab_size, tp_rank, tp_size
    )

    local_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
    global_max = local_max.clone()
    if tp_size > 1:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

    local_exp_sums = torch.zeros((num_tokens, 1), dtype=reduction_dtype, device=device)
    for start in range(0, num_tokens, reduction_chunk_size):
        end = min(start + reduction_chunk_size, num_tokens)
        logits_part = vocab_parallel_logits[start:end]
        centered = logits_part.to(reduction_dtype) - global_max[start:end].to(reduction_dtype)
        local_exp_sums[start:end] = centered.exp_().sum(dim=-1, keepdim=True)

    global_exp_sums = local_exp_sums.clone()
    if tp_size > 1:
        dist.all_reduce(global_exp_sums, op=dist.ReduceOp.SUM, group=tp_group)
    log_denom = global_exp_sums.clamp_min(torch.finfo(global_exp_sums.dtype).tiny).log()

    local_mask = (label_token_ids >= vocab_start_index) & (label_token_ids < vocab_end_index)
    local_label_ids = (label_token_ids - vocab_start_index).masked_fill(~local_mask, 0)
    local_label_ids = local_label_ids.clamp_(0, max(partition_vocab_size - 1, 0))

    local_selected_logits = torch.gather(vocab_parallel_logits, dim=-1, index=local_label_ids)
    local_selected_logits = local_selected_logits.masked_fill(~local_mask, 0.0).to(reduction_dtype)
    if tp_size > 1:
        dist.all_reduce(local_selected_logits, op=dist.ReduceOp.SUM, group=tp_group)

    return (local_selected_logits - global_max.to(reduction_dtype) - log_denom).to(logits_dtype)


def _to_cuda_tensors(values, dtype: torch.dtype) -> list[torch.Tensor]:
    return [torch.as_tensor(value, dtype=dtype, device=torch.cuda.current_device()) for value in values]


def _prepare_rollout_data(rollout_data_ref):
    rollout_data = ray.get(rollout_data_ref[0].inner)

    rollout_data["tokens"] = _to_cuda_tensors(rollout_data["tokens"], torch.long)
    rollout_data["loss_masks"] = _to_cuda_tensors(rollout_data["loss_masks"], torch.int)
    if rollout_data.get("label_token_ids") is not None:
        rollout_data["label_token_ids"] = _to_cuda_tensors(rollout_data["label_token_ids"], torch.long)
        for idx, tensor in enumerate(rollout_data["label_token_ids"]):
            if tensor.dim() == 1 and tensor.numel() == 0:
                rollout_data["label_token_ids"][idx] = tensor.reshape(0, 0)

    micro_batch_size = len(rollout_data["tokens"])
    rollout_data["micro_batch_indices"] = [list(range(micro_batch_size))]
    rollout_data["num_microbatches"] = [1]
    rollout_data["global_batch_sizes"] = [micro_batch_size]

    return rollout_data


def _merge_tensors_with_cp(
    tensors: list[torch.Tensor] | None,
    total_lengths: list[int],
    response_lengths: list[int],
    args,
) -> list[torch.Tensor] | None:
    if not tensors:
        return tensors

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return tensors

    merged = []
    for tensor, total_length, response_length in zip(tensors, total_lengths, response_lengths, strict=False):
        merged.append(all_gather_with_cp(tensor, total_length, response_length))
    return merged


def _slice_response_rows_for_current_cp_rank(
    rows: torch.Tensor,
    sample_idx: int,
    *,
    logits_local_len: int,
    args,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None,
) -> torch.Tensor:
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return rows

    total_length = total_lengths[sample_idx]
    response_length = response_lengths[sample_idx]
    prompt_length = total_length - response_length

    if getattr(args, "allgather_cp", False):
        seq_start = sum(total_lengths[:sample_idx])
        chunk_start = mpu.get_context_parallel_rank() * logits_local_len
        chunk_end = chunk_start + logits_local_len
        logit_global_start = seq_start + prompt_length - 1
        logit_global_end = seq_start + total_length - 1
        start = max(logit_global_start, chunk_start)
        end = min(logit_global_end, chunk_end)
        if end <= start:
            return rows[:0]
        return rows[start - logit_global_start : end - logit_global_start]

    _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(total_length, response_length)
    rows_0 = rows[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
    rows_1 = rows[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
    return torch.cat([rows_0, rows_1], dim=0)


def _merge_allgather_cp_tensors(
    outputs: dict[str, list[torch.Tensor]],
    keys: tuple[str, ...],
    *,
    logits_local_len: int,
    total_lengths: list[int],
    response_lengths: list[int],
) -> None:
    if mpu.get_context_parallel_world_size() == 1:
        return

    cp_rank = mpu.get_context_parallel_rank()
    cp_group = mpu.get_context_parallel_group()
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len

    for key in keys:
        values = outputs.get(key)
        if values is None:
            continue

        full_values = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1
            start = max(logit_global_start, chunk_start)
            end = min(logit_global_end, chunk_end)

            if end <= start:
                full_value = value.new_zeros((response_length, *value.shape[1:]))
            else:
                expected_len = end - start
                if value.size(0) != expected_len:
                    raise ValueError(f"{key} length mismatch: got {value.size(0)}, expected {expected_len}")
                response_start = start - logit_global_start
                response_end = end - logit_global_start
                left = value.new_zeros((response_start, *value.shape[1:]))
                right = value.new_zeros((response_length - response_end, *value.shape[1:]))
                full_value = torch.cat([left, value, right], dim=0)

            full_values.append(full_value)
            seq_start += total_length

        gathered = dist.nn.all_reduce(torch.cat(full_values, dim=0), group=cp_group)
        outputs[key] = list(gathered.split(response_lengths, dim=0))


def _get_log_probs_and_optional_samples(
    logits: torch.Tensor,
    *,
    args,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    sample_n: int = 0,
    label_token_ids: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    _, outputs = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=with_entropy,
        non_loss_data=non_loss_data,
        max_seq_lens=max_seq_lens,
    )
    logits_local_len = logits.size(1) if args.qkv_format == "thd" else logits.view(-1, logits.size(-1)).size(0)

    if label_token_ids is not None:
        if len(label_token_ids) != len(unconcat_tokens):
            raise ValueError(f"label_token_ids batch size mismatch: {len(label_token_ids)} vs {len(unconcat_tokens)}")
        label_token_log_probs = []
        label_reduction_chunk_size = args.teacher_label_reduction_chunk_size
        for sample_idx, ((logits_chunk, _), sample_label_token_ids) in enumerate(
            zip(
                get_responses(
                    logits,
                    args=args,
                    unconcat_tokens=unconcat_tokens,
                    total_lengths=total_lengths,
                    response_lengths=response_lengths,
                    max_seq_lens=max_seq_lens,
                ),
                label_token_ids,
                strict=True,
            )
        ):
            local_label_token_ids = _slice_response_rows_for_current_cp_rank(
                sample_label_token_ids,
                sample_idx,
                logits_local_len=logits_local_len,
                args=args,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                max_seq_lens=max_seq_lens,
            )
            label_token_log_probs.append(
                get_label_token_log_probs_from_vocab_parallel_logits(
                    logits_chunk,
                    local_label_token_ids,
                    reduction_chunk_size=label_reduction_chunk_size,
                )
            )
        outputs["label_token_log_probs"] = label_token_log_probs

    if sample_n > 0:
        sampled_token_ids = []
        sampled_log_probs = []
        reduction_chunk_size = args.teacher_sample_reduction_chunk_size

        for logits_chunk, _ in get_responses(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        ):
            if logits_chunk.size(0) == 0:
                sampled_token_ids.append(torch.empty((0, sample_n), dtype=torch.long, device=logits_chunk.device))
                sampled_log_probs.append(
                    torch.empty((0, sample_n), dtype=logits_chunk.dtype, device=logits_chunk.device)
                )
                continue

            sampled_ids, sampled_logp = sample_from_vocab_parallel_logits_without_full_gather(
                logits_chunk,
                sample_n=sample_n,
                reduction_chunk_size=reduction_chunk_size,
            )

            sampled_token_ids.append(sampled_ids)
            sampled_log_probs.append(sampled_logp)

        outputs["sampled_token_ids"] = sampled_token_ids
        outputs["sampled_log_probs"] = sampled_log_probs

    if getattr(args, "allgather_cp", False):
        if "sampled_token_ids" in outputs:
            outputs["sampled_token_ids"] = [x.to(torch.float32) for x in outputs["sampled_token_ids"]]
        _merge_allgather_cp_tensors(
            outputs,
            ("label_token_log_probs", "sampled_token_ids", "sampled_log_probs"),
            logits_local_len=logits_local_len,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
        )
    return torch.empty((0,), device=logits.device), outputs


class TeacherLogpRayActor(MegatronTrainRayActor):
    """A Megatron actor subclass that exposes a log-prob computation RPC."""

    def get_parallel_infos(self) -> dict[str, int]:
        return {
            "dp_rank": mpu.get_data_parallel_rank(),
            "pp_rank": mpu.get_pipeline_model_parallel_rank(),
            "tp_rank": mpu.get_tensor_model_parallel_rank(),
            "cp_rank": mpu.get_context_parallel_rank(),
            "dp_size": mpu.get_data_parallel_world_size(),
            "cp_size": mpu.get_context_parallel_world_size(),
            "tp_size": mpu.get_tensor_model_parallel_world_size(),
            "pp_size": mpu.get_pipeline_model_parallel_world_size(),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.getLogger().setLevel(logging.WARNING)

    def compute_logp(self, rollout_data_ref) -> dict[str, Any]:
        rollout_data = _prepare_rollout_data(rollout_data_ref)
        sample_ns = rollout_data.get("sample_ns", [0])
        sample_n = int(sample_ns[0]) if sample_ns else 0
        label_token_ids = rollout_data.get("label_token_ids")
        if sample_n < 0:
            raise ValueError(f"sample_n must be >= 0, got {sample_n}")

        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]

        if sample_n > 0 or label_token_ids is not None:
            forward_outputs = forward_only(
                partial(
                    _get_log_probs_and_optional_samples,
                    sample_n=sample_n,
                    label_token_ids=label_token_ids,
                ),
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix="",
            )
        else:
            forward_outputs = self.compute_log_prob(
                data_iterator,
                num_microbatches,
                store_prefix="",
            )

        log_probs = forward_outputs.get("log_probs", None)
        sampled_token_ids = forward_outputs.get("sampled_token_ids", None)
        sampled_log_probs = forward_outputs.get("sampled_log_probs", None)
        label_token_log_probs = forward_outputs.get("label_token_log_probs", None)
        if mpu.is_pipeline_last_stage():
            log_probs = _merge_tensors_with_cp(
                log_probs,
                rollout_data["total_lengths"],
                rollout_data["response_lengths"],
                self.args,
            )
            if sampled_log_probs is not None:
                if not self.args.allgather_cp:
                    sampled_log_probs = _merge_tensors_with_cp(
                        sampled_log_probs,
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        self.args,
                    )
            if sampled_token_ids is not None:
                sampled_token_ids = [x.to(torch.float32) for x in sampled_token_ids]
                if not self.args.allgather_cp:
                    sampled_token_ids = _merge_tensors_with_cp(
                        sampled_token_ids,
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        self.args,
                    )
            if label_token_log_probs is not None:
                if not self.args.allgather_cp:
                    label_token_log_probs = _merge_tensors_with_cp(
                        label_token_log_probs,
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        self.args,
                    )
            if mpu.get_context_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
                log_prob = log_probs[0].tolist()
                sampled_log_prob = sampled_log_probs[0].tolist() if sampled_log_probs else None
                sampled_token_id = sampled_token_ids[0].round().to(torch.long).tolist() if sampled_token_ids else None
                label_token_log_prob = label_token_log_probs[0].tolist() if label_token_log_probs else None
            else:
                log_prob = None
                sampled_log_prob = None
                sampled_token_id = None
                label_token_log_prob = None
        else:
            log_prob = None
            sampled_log_prob = None
            sampled_token_id = None
            label_token_log_prob = None

        return {
            "log_prob": log_prob,
            "sampled_log_probs": sampled_log_prob,
            "sampled_token_ids": sampled_token_id,
            "label_token_log_probs": label_token_log_prob,
            "dp_rank": mpu.get_data_parallel_rank(with_context_parallel=False),
            "pp_rank": mpu.get_pipeline_model_parallel_rank(),
            "cp_rank": mpu.get_context_parallel_rank(),
            "tp_rank": mpu.get_tensor_model_parallel_rank(),
        }

    def update_from_disk(self, model_path: str) -> dict[str, Any]:
        self.load_other_checkpoint("actor", model_path)
        return {
            "rank": self.args.rank,
            "model_path": model_path,
        }
