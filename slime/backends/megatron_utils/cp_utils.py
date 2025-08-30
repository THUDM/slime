import torch
import torch.nn.functional as F
import torch.distributed as dist
from megatron.core import mpu


def get_logits_and_tokens_offset_with_cp(
    total_length: int,
    response_length: int,
):
    """
    All offsets start from the begining of the prompt.
    """
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size > 1

    prompt_length = total_length - response_length
    chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)

    # the offset of 2 chunks
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size)

    # the offset of 2 logits, note that the logits need a "-1".
    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1), min(chunk_1[1], total_length - 1))

    # when the sequence is empty, make an empty slice to continue the gradient flow.
    if logits_0[0] < logits_0[1]:
        token_0 = (logits_0[0] + 1, logits_0[1] + 1)
    else:
        logits_0 = (0, 0)
        token_0 = (0, 0)

    if logits_1[0] < logits_1[1]:
        token_1 = (logits_1[0] + 1, logits_1[1] + 1)
    else:
        logits_1 = (0, 0)
        token_1 = (0, 0)

    return chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1)


def get_sum_of_sample_mean(
    total_lengths,
    response_lengths,
    loss_masks,
    calculate_per_token_loss: bool = False,
):
    """
    Calculate correct sample mean for CP
    """
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:

        def sum_of_sample_mean(x: torch.Tensor):
            return sum(
                [
                    (x_i * loss_mask_i).sum() / torch.clamp_min(loss_mask_i.sum(), 1)
                    for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks)
                ]
            )

        def sum_of_token(x: torch.Tensor):
            return sum(
                [(x_i * loss_mask_i).sum() for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks)]
            )

    else:
        cp_chunk_lengths = []
        chunked_loss_masks = []
        for i, (total_length, response_length, loss_mask) in enumerate(
            zip(total_lengths, response_lengths, loss_masks)
        ):
            prompt_length = total_length - response_length
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(total_length, response_length)
            loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            chunked_loss_masks.append(torch.cat([loss_mask_0, loss_mask_1], dim=0))
            cp_chunk_lengths.append(chunked_loss_masks[i].size(0))

        def sum_of_sample_mean(x):
            return sum(
                [
                    (x_i * chunked_loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
                    for x_i, chunked_loss_mask, loss_mask in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, loss_masks
                    )
                ]
            )

        def sum_of_token(x: torch.Tensor):
            return sum(
                [
                    (x_i * chunked_loss_mask).sum()
                    for x_i, chunked_loss_mask in zip(x.split(cp_chunk_lengths, dim=0), chunked_loss_masks)
                ]
            )

    return sum_of_sample_mean if not calculate_per_token_loss else sum_of_token


def all_gather_with_cp(tensor: torch.Tensor, total_length: int, response_length: int):
    """
    Gather tensors across all ranks in the context parallel group.
    The first dimension of the output tensor will be the `response_length`.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_size = mpu.get_context_parallel_world_size()

    if cp_size == 1:
        return tensor

    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length)

    prompt_length = total_length - response_length

    chunk_0 = tensor[: logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0] :]
    assert chunk_1.shape[0] == logits_offset[1][1] - logits_offset[1][0]

    def zero(len):
        return torch.zeros(
            [len] + list(tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=True,
        )

    # logprob should be within the range of [prompt_length - 1, total_length - 1]
    if chunk_0.shape[0] == 0 and chunk_1.shape[0] == 0:
        # all empty
        full_tensor = zero(response_length)
    elif chunk_0.shape[0] != 0 and chunk_1.shape[0] == 0:
        # only first chunk
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[0][1])
        full_tensor = torch.cat([left, chunk_0, right], dim=0)
    elif chunk_0.shape[0] == 0 and chunk_1.shape[0] != 0:
        # only second chunk
        left = zero(logits_offset[1][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_1, right], dim=0)
    else:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        mid = zero(logits_offset[1][0] - logits_offset[0][1])
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)

    assert full_tensor.shape[0] == response_length, f"Expected {response_length}, got {full_tensor.shape}"
    full_tensor = dist.nn.all_reduce(full_tensor, group=cp_group)
    return full_tensor


def slice_with_cp(tokens: torch.Tensor, pad_value):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()

    if cp_size == 1:
        return tokens

    # pad
    chunk_size = (len(tokens) + 2 * cp_size - 1) // (2 * cp_size)
    pad = 2 * cp_size * chunk_size - len(tokens)
    tokens = F.pad(tokens, (0, pad), value=pad_value)
    # get 2 chunk for thd cp
    start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])


def slice_log_prob_with_cp(log_prob: list[float], total_length: int, response_length: int):
    assert len(log_prob) == response_length

    cp_size = mpu.get_context_parallel_world_size()

    if cp_size == 1:
        return log_prob

    prompt_length = total_length - response_length
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length)

    chunk_1 = log_prob[logits_offset[0][0] - (prompt_length - 1) : logits_offset[0][1] - (prompt_length - 1)]
    chunk_2 = log_prob[logits_offset[1][0] - (prompt_length - 1) : logits_offset[1][1] - (prompt_length - 1)]
    return chunk_1 + chunk_2
