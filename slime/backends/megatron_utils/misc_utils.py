import torch


def collate_multimodal_train_inputs(multimodal_train_inputs) -> dict:
    """Merge the per-sample multimodal tensors of one micro-batch, key by key.

    Concatenating along dim 0 requires every other dimension to agree. It does
    for a tower whose processor emits ragged-packed patches -- Qwen-VL's
    pixel_values is (total_patches, patch_dim), where only dim 0 varies. It does
    not for a tower that keeps images as pictures at their own resolution:
    Nemotron's pixel_values is (tiles, 3, H, W) with H and W set per image by
    dynamic tiling, so two samples in one micro-batch can produce 416 and 384
    and torch.cat raises.

    Those keys are handed through as the per-sample list instead. The model is
    what knows how to consume its own tower's layout, and Nemotron's projector
    iterates a list. Keys that do concatenate still do: num_patches, num_tokens
    and imgs_sizes are all (1, ...) per sample. A key present on a single sample
    is passed through as that tensor.
    """
    grouped: dict[str, list] = {}
    for mm_input_dict in multimodal_train_inputs:
        if mm_input_dict is not None:
            for key, mm_tensor in mm_input_dict.items():
                grouped.setdefault(key, []).append(mm_tensor)

    multimodal_data = {}  # key -> concatenated tensor, or a per-sample list
    for key, tensors in grouped.items():
        if len(tensors) == 1:
            multimodal_data[key] = tensors[0]
            continue
        first_trailing = tensors[0].shape[1:] if hasattr(tensors[0], "shape") else None
        if first_trailing is not None and all(hasattr(t, "shape") and t.shape[1:] == first_trailing for t in tensors):
            multimodal_data[key] = torch.cat(tensors, dim=0)
        else:
            multimodal_data[key] = tensors
    return multimodal_data


def strip_param_name_prefix(name: str):
    prefix = "module."
    while name.startswith(prefix):
        name = name.removeprefix(prefix)
    return name


def has_gated_linear_unit(args) -> bool:
    """Whether ``linear_fc1`` holds the fused ``cat(gate, up)`` rather than one projection.

    There is no ``--gated-linear-unit`` flag to read: megatron derives
    ``config.gated_linear_unit`` from the activation in
    ``training/argument_utils.py``, and only ``--swiglu`` and ``--quick-geglu``
    set it. Models such as nemotron_h are ``--squared-relu`` and ungated -- their
    ``linear_fc1`` is ``up_proj`` alone, half the width, with no gate half to
    split off or to put back.

    This has to be asked of the model, because the parameter cannot answer it:

    * the NAME cannot -- ``linear_fc1`` is the same name either way, and halving
      an ungated projection yields a tensor of exactly the right shape holding
      the wrong rows;
    * ``partition_stride`` cannot either, tempting as it looks. ``mlp.py``
      sets ``fc1_stride = 2`` for a gated dense MLP, but ``TEGroupedMLP``
      (``moe/experts.py``) doubles the width for gating and leaves stride at the
      TE default of 1, so a gated grouped expert reports stride 1.

    Both halves of the HF <-> Megatron round trip must ask this same question and
    get the same answer: hf_to_megatron/common.py (shard on load) and
    update_weight/common.py (gather on weight sync).
    """
    return bool(
        getattr(args, "gated_linear_unit", False)
        or getattr(args, "swiglu", False)
        or getattr(args, "quick_geglu", False)
    )
