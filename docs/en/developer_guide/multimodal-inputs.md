# Multimodal Inputs

A VL sample carries more than a token stream. Everything else the HF processor returned for that sample — `pixel_values`, `image_grid_thw`, `num_patches`, … — travels alongside it as `Sample.multimodal_train_inputs`, a plain dict that the model provider eventually receives as forward kwargs.

This page documents the shape contract on that dict, because it is weaker than it looks and every stage that walks it has to respect the same rule.

## The values are not uniformly tensors

`sft_rollout_vl.encode_multimodal_sample` keeps whatever the processor produced, minus `input_ids` and `attention_mask`:

```python
multimodal_train_inputs = {
    key: value for key, value in processor_output.items() if key not in ("input_ids", "attention_mask")
} or None
```

What that dict contains depends on the processor:

- **Fixed-resolution processors** (Qwen3.5-VL) put every image on the same patch grid, so each field is one stacked tensor.
- **Dynamic-resolution processors** (Nemotron 3.5 Super VL) resize every image to its own aspect-preserving grid. When two images in a sample differ in shape they cannot be stacked, so `pixel_values` comes back as a **list of `(3, H, W)` tensors**. `num_patches` / `num_tokens` come back as plain Python lists of ints.

With an aspect-preserving resize the list form is the normal case, not the corner one.

It is also not a defect to normalize away. The model's own vision stack consumes the list: `slime_plugins/models/nemotron_35_super_vl.py::project_images` iterates it and projects each image separately, because per-image feature counts differ and `torch.cat(..., dim=0)` on them would raise. Flattening the list into one tensor upstream would just move the problem.

So the contract is: **a field is a tensor, or a (possibly nested) list/tuple of tensors, or a list of plain Python values.**

## Three stages walk these dicts

| Stage | Location | What it does |
| --- | --- | --- |
| CPU tensorization | `slime/observability/rollout_data_utils.py::tensorize_rollout_data_for_training` | Normalizes tensor-likes to contiguous CPU tensors before the Ray handoff |
| Host→device copy | `slime/backends/megatron_utils/actor.py::_get_rollout_data` | Moves them to the training device in advance |
| Micro-batch join | `slime/backends/megatron_utils/data.py::get_batch` | Merges the per-sample dicts into one dict for the micro-batch |

The first two are **maps** and share `slime/utils/multimodal.py::map_multimodal_fields`. The third is a **join** and uses `_concat_multimodal_field` in `data.py`. Both dispatch on type rather than assume one.

### Maps: guard at the leaf, not at the field

The failure mode this replaced was a top-level type test:

```python
# wrong: skips a list of per-image tensors whole
key: value.to(device=device) if isinstance(value, torch.Tensor) else value
```

A dynamic-resolution `pixel_values` is a `list`, so it fell into the `else` branch and stayed on the host. The projector then ran CPU inputs against CUDA weights.

`map_multimodal_fields` walks into lists and tuples, preserves the container type, and applies the caller's function only to leaves:

```python
rollout_data["multimodal_train_inputs"] = [
    map_multimodal_fields(
        mm_dict,
        lambda value: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value,
    )
    for mm_dict in rollout_data["multimodal_train_inputs"]
]
```

Keeping the type guard in the caller's function is deliberate: it is what leaves `num_patches = [3, 5]` a list of Python ints instead of promoting it to a tensor. `None` passes through, since a micro-batch may mix text-only samples with multimodal ones.

### Join: concatenate by type

`_concat_multimodal_field` merges one field across the samples of a micro-batch. Tensors `cat` on dim 0, sequences concatenate, a tensor meeting a sequence degrades to a list — which happens when sample A's images are all the same size and sample B's are not — and anything else raises rather than being guessed at.

Note that at `--micro-batch-size 1` this join never runs. A VL model can therefore look healthy at mbs 1 and fail the first time someone raises the micro-batch size.

## Adding a VL model

If the model's processor is dynamic-resolution:

- Do not assume `pixel_values` is a tensor anywhere in the forward. Accept the list and iterate it.
- Do not hand the list to a released HF projector's own list branch without checking it: Nemotron 3.5 Super VL's indexes four dimensions (`_, _, H, W = pixel_values.shape`) on elements the processor emits with three, and then concatenates per-image results whose token counts differ.
- Return vision features already flattened to one row per image token, in image order. That is what the placeholder scatter needs, and it makes the stacked-tensor and list cases produce the same shape.

Tests: `tests/test_multimodal_fields.py` covers the traversal contract (no GPU, no torch), and `tests/test_nemotron_35_super_vl.py::test_project_images_handles_a_ragged_list` covers the projection shape contract.
