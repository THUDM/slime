import torch
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

    # the offset of 2 logits, not that the logits need a "-1".
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
    xs: list[torch.Tensor],
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
            if loss_masks is not None:
                res = sum(
                    [
                        (x_i * loss_mask_i).sum() / loss_mask_i.sum()
                        for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks)
                    ]
                )
            else:
                res = sum([torch.mean(x_i, dim=0) for x_i in x.split(response_lengths, dim=0)])
            return res

        def sum_of_token(x: torch.Tensor):
            if loss_masks is not None:
                res = sum(
                    [
                        (x_i * loss_mask_i).sum()
                        for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks)
                    ]
                )
            else:
                res = sum([x_i.sum() for x_i in x.split(response_lengths, dim=0)])
            return res

    else:
        cp_chunk_lengths = [len(l) for l in xs]
        chunked_loss_masks = []
        if loss_masks is not None:
            for i, (total_length, response_length, loss_mask) in enumerate(
                zip(total_lengths, response_lengths, loss_masks)
            ):
                prompt_length = total_length - response_length
                _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(total_length, response_length)
                loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
                loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
                chunked_loss_masks.append(torch.cat([loss_mask_0, loss_mask_1], dim=0))
                assert (
                    chunked_loss_masks[i].size(0) == cp_chunk_lengths[i]
                ), f"{chunked_loss_masks[i].size(0)} vs {cp_chunk_lengths[i]}"

        def sum_of_sample_mean(x):
            if loss_masks is not None:
                x = sum(
                    [
                        (x_i * chunked_loss_mask).sum() / loss_mask.sum()
                        for x_i, chunked_loss_mask, loss_mask in zip(
                            x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, loss_masks
                        )
                    ]
                )
            else:
                x = sum(
                    [
                        torch.sum(x_i, dim=0) / l_i
                        for x_i, l_i in zip(x.split(cp_chunk_lengths, dim=0), response_lengths)
                        if l_i > 0
                    ]
                )
            return x

        def sum_of_token(x: torch.Tensor):
            if loss_masks is not None:
                x = sum(
                    [
                        (x_i * chunked_loss_mask).sum()
                        for x_i, chunked_loss_mask in zip(x.split(cp_chunk_lengths, dim=0), chunked_loss_masks)
                    ]
                )
            else:
                x = sum([torch.sum(x_i, dim=0) for x_i in x.split(cp_chunk_lengths, dim=0)])
            return x

    return sum_of_sample_mean if not calculate_per_token_loss else sum_of_token
