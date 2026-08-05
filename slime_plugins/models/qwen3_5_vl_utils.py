from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist


def get_packed_cp_local_indices(
    cu_seqlens: Sequence[int] | torch.Tensor,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> torch.Tensor:
    """Map a THD CP rank's local tokens back to the full packed token stream."""

    boundaries = [int(value) for value in cu_seqlens]
    indices = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        sequence_length = end - start
        if sequence_length % (2 * cp_size) != 0:
            raise ValueError(f"Packed sequence length {sequence_length} must be divisible by 2 * CP size {cp_size}")
        chunk_size = sequence_length // (2 * cp_size)
        first = start + cp_rank * chunk_size
        second = start + (2 * cp_size - cp_rank - 1) * chunk_size
        indices.extend(
            (
                torch.arange(first, first + chunk_size, device=device),
                torch.arange(second, second + chunk_size, device=device),
            )
        )
    return torch.cat(indices) if indices else torch.empty(0, dtype=torch.long, device=device)


def gather_packed_input_ids(
    input_ids: torch.Tensor,
    cu_seqlens: Sequence[int] | torch.Tensor,
    cp_group,
) -> torch.Tensor:
    """Reconstruct the full THD token stream from Megatron's two-chunk CP layout."""

    if cp_group is None or cp_group.size() == 1:
        return input_ids

    local_tokens = input_ids.flatten()
    gathered_tokens = [torch.empty_like(local_tokens) for _ in range(cp_group.size())]
    dist.all_gather(gathered_tokens, local_tokens, group=cp_group)

    full_tokens = torch.empty(int(cu_seqlens[-1]), dtype=input_ids.dtype, device=input_ids.device)
    for rank, rank_tokens in enumerate(gathered_tokens):
        indices = get_packed_cp_local_indices(cu_seqlens, cp_group.size(), rank, input_ids.device)
        if indices.numel() != rank_tokens.numel():
            raise ValueError(
                f"CP rank {rank} has {rank_tokens.numel()} tokens, expected {indices.numel()} from cu_seqlens"
            )
        full_tokens[indices] = rank_tokens
    return full_tokens.unsqueeze(0)


def _vision_positions(
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
    grid_h //= spatial_merge_size
    grid_w //= spatial_merge_size

    temporal = torch.arange(grid_t, device=device).repeat_interleave(grid_h * grid_w) + start_position
    height = torch.arange(grid_h, device=device).repeat_interleave(grid_w).repeat(grid_t) + start_position
    width = torch.arange(grid_w, device=device).repeat(grid_h * grid_t) + start_position
    return torch.stack((temporal, height, width))


def build_packed_mrope_position_ids(
    input_ids: torch.Tensor,
    cu_seqlens: Sequence[int] | torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    *,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Build Qwen3.5 MRoPE positions while resetting every packed sequence."""

    if input_ids.shape[0] != 1:
        raise ValueError(f"Packed Qwen3.5-VL input_ids must have batch size 1, got {tuple(input_ids.shape)}")

    device = input_ids.device
    positions = torch.zeros((3, 1, input_ids.shape[1]), dtype=input_ids.dtype, device=device)
    boundaries = [int(value) for value in cu_seqlens]
    image_grids = iter(()) if image_grid_thw is None else iter(image_grid_thw)

    if video_grid_thw is None:
        video_grids = iter(())
    else:
        split_video_grids = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
        split_video_grids[:, 0] = 1
        video_grids = iter(split_video_grids)

    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        tokens = input_ids[0, start:end]
        token_list = tokens.tolist()
        pieces = []
        cursor = 0
        current_position = 0

        vision_starts = (tokens == vision_start_token_id).nonzero(as_tuple=False).flatten()
        modality_tokens = tokens[vision_starts + 1] if vision_starts.numel() else ()

        for modality_token in modality_tokens:
            modality_token = int(modality_token)
            if modality_token not in (image_token_id, video_token_id):
                continue
            try:
                vision_start = token_list.index(modality_token, cursor)
                grid = next(image_grids if modality_token == image_token_id else video_grids)
            except (StopIteration, ValueError) as exc:
                raise ValueError("Qwen3.5-VL tokens and vision grids do not match") from exc

            text_length = vision_start - cursor
            if text_length:
                pieces.append(torch.arange(text_length, device=device).view(1, -1).expand(3, -1) + current_position)
                current_position += text_length

            vision_position_ids = _vision_positions(current_position, grid, spatial_merge_size, device)
            pieces.append(vision_position_ids)
            current_position += max(int(grid[1]), int(grid[2])) // spatial_merge_size
            cursor = vision_start + vision_position_ids.shape[1]

        if cursor < len(token_list):
            text_length = len(token_list) - cursor
            pieces.append(torch.arange(text_length, device=device).view(1, -1).expand(3, -1) + current_position)

        if pieces:
            sequence_positions = torch.cat(pieces, dim=1)
            if sequence_positions.shape[1] != len(token_list):
                raise ValueError(
                    "Qwen3.5-VL vision token count does not match its grid: "
                    f"expected {len(token_list)} positions, built {sequence_positions.shape[1]}"
                )
            positions[:, 0, start:end] = sequence_positions

    try:
        next(image_grids)
        raise ValueError("Unused Qwen3.5-VL image grids")
    except StopIteration:
        pass
    try:
        next(video_grids)
        raise ValueError("Unused Qwen3.5-VL video grids")
    except StopIteration:
        pass
    return positions
