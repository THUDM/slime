"""Megatron -> HuggingFace name mapping for Nemotron 3.5 Super VL (``nemotron_h_omni``).

The inverse of hf_to_megatron/nemotron_h_vl.py, and a dispatcher for the same
reason: the decoder is the Nemotron-H stack ``nemotron_h.py`` already converts,
and all this file owns is the prefix the omni checkpoint puts in front of it
plus the vision keys.

    Megatron parameter                          emitted HF key
    ----------------------------------------    --------------------------------
    language_model.<anything>                   language_model.<nemotron_h name>
    vision_model.<anything>                     see _VISION_RENAMES
    vision_projector.mlp1.norm.weight           mlp1.0.weight
    vision_projector.mlp1.linear1.weight        mlp1.1.weight
    vision_projector.mlp1.linear2.weight        mlp1.3.weight
    vision_projector.vision_final_layernorm.*   vision_projector.vision_final_layernorm.*

Why the released checkpoint's key set is the target
---------------------------------------------------
GRPO pushes these tensors into the colocated SGLang engine before every rollout.
The engine is ``NemotronH_Omni_Reasoning_V3`` (models/nano_nemotron_vl.py), and
its ``load_weights`` routes purely by prefix:

    language_model.*                  -> the LLM, prefix stripped
    mlp1.*                            -> the projector nn.Sequential, by index
    vision_model.radio_model.*        -> the RADIO tower, `vision_model.` stripped
    vision_projector.vision_final_layernorm.*  -> the Super-only LayerNorm

which is exactly the naming of the released checkpoint, because cold start loads
that checkpoint through this very method. So "emit the checkpoint's own keys" is
not a convention adopted here -- it is the requirement, and it is verifiable
against model.safetensors.index.json rather than inferred.

This cuts both ways, and it is the thing to be careful about. That routing has
**no else branch**: a name matching none of the four prefixes is dropped on the
floor, and the tower's own loader (models/radio.py) likewise skips any key it
cannot find in ``named_parameters()``. Getting a name wrong here therefore does
not raise. It produces an engine running partly stale weights and a quietly
wrong ``train_rollout_logprob_abs_diff``, which is why every line below is
pinned to a checkpoint key or to a named sglang parameter.

Where this is more than a rename -- one rule, and one deliberate absence
------------------------------------------------------------------------
  * **Fused qkv.** RadioModel spells attention as three Linears
    (``attention.attention.{query,key,value}``); the checkpoint and sglang both
    want one ``attn.qkv``. The load direction splits it with ``Chunk(dim=0)``
    (modeling_radio.py:64-72), so the inverse is ``cat([q, k, v], dim=0)`` -- but
    the three arrive as three separate calls, so they are buffered until the
    triple is complete and emitted as one tensor on the third. sglang's
    ``QKVParallelLinear`` weight loader is invoked as ``weight_loader(param, w)``
    with no shard id (models/radio.py:598), i.e. it only accepts the fused form;
    emitting q/k/v separately would hit the silent-drop path above.

  * **LayerScale -- no longer here at all.** ``encoder.layer.{i}.layer_scale{1,2}
    .lambda1`` used to be mapped to sglang's ``...blocks.{i}.ls{1,2}``, because
    the parameter existed on both sides and nothing else would have written it.
    It no longer exists on either side: C-RADIO ViT-H ships no layerscale,
    ``layerscale_value`` is 1.0, and both sides now decline to materialise an
    identity -- ``_load_vision_modules`` replaces the module with ``nn.Identity``
    (slime_plugins/models/nemotron_h_vl.py) and sglang skips the ``nn.Parameter``
    for ``model_type == "radio"`` (models/internvl.py:249-250).

    So there is no rule for it below, and nothing to write: the name cannot
    appear in ``named_parameters()``, which is what drives this converter. The
    two changes are a pair -- exporting without the sglang half would leave the
    engine on its own init value, which is the failure this mapping originally
    existed to prevent.

Buffers -- ``input_conditioner.norm_{mean,std}`` and ``summary_idxs`` -- are not
parameters, and the weight sync only walks ``named_parameters()`` plus
``expert_bias`` (update_weight/common.py:147-190), so they never reach here.
``make_preprocessor_external()`` has replaced the input conditioner with an
nn.Identity anyway (modeling_radio.py:477).
"""

from __future__ import annotations

import re

import torch

from .nemotron_h import convert_nemotron_h_to_hf

# RadioModel's module tree -> the released C-RADIO checkpoint's: every
# WeightRenaming in register_radio_conversion_mapping() (modeling_radio.py:52-75)
# read left-to-right, i.e. the exact transpose of hf_to_megatron's _VISION_RENAMES.
# Applied as an ordered prefix substitution on the name below `vision_model.`.
_VISION_RENAMES = (
    ("embeddings.video_patch_projection", "radio_model.model.patch_generator.video_embedder"),
    ("embeddings.patch_projection", "radio_model.model.patch_generator.embedder"),
    ("embeddings.position_embedding", "radio_model.model.patch_generator.pos_embed"),
    ("embeddings.cls_register_token", "radio_model.model.patch_generator.cls_token.token"),
    ("encoder.layer", "radio_model.model.blocks"),
    ("input_conditioner", "radio_model.input_conditioner"),
)

# Inside a block. norm1/norm2/mlp.fc1/mlp.fc2 are deliberately absent: RadioLayer
# spells them the way timm does, which is why the registered mapping lists no
# rule for them and why they pass through unchanged.
_VISION_BLOCK_RENAMES = (("attention.output.dense", "attn.proj"),)

# nn.Sequential indices in the checkpoint, keyed by the reference projector's
# attribute names (modeling_nemotron_h_omni.py:55-57). Index 2 is the activation.
_MLP1_RENAMES = {"mlp1.norm": "mlp1.0", "mlp1.linear1": "mlp1.1", "mlp1.linear2": "mlp1.3"}

_QKV_ORDER = ("query", "key", "value")

# Partial qkv triples, keyed by (block index, "weight"/"bias"). Entries live only
# between the first component of a triple and its third, which for a given sync
# is a handful of 1280x1280 tensors at most -- the three come from consecutive
# named_parameters() entries, and only a bucket boundary can separate them. The
# same buffering idiom is used for DeepSeek's q_a_proj/kv_a_proj pair in
# __init__.py, and like it this is per-rank state: convert_to_hf runs only on
# pipeline source ranks, so a triple is never split across processes.
_QKV_CACHE: dict[tuple[str, str], dict[str, torch.Tensor]] = {}


def _fuse_qkv(block_idx: str, component: str, suffix: str, param: torch.Tensor):
    """Collect query/key/value; emit the fused attn.qkv once all three are in."""
    slot = _QKV_CACHE.setdefault((block_idx, suffix), {})
    slot[component] = param
    if len(slot) < len(_QKV_ORDER):
        return []

    del _QKV_CACHE[(block_idx, suffix)]
    fused = torch.cat([slot[part] for part in _QKV_ORDER], dim=0).contiguous()
    return [(f"vision_model.radio_model.model.blocks.{block_idx}.attn.qkv.{suffix}", fused)]


def _convert_vision(rest: str, param: torch.Tensor):
    """Map one RadioModel parameter name onto the C-RADIO checkpoint's."""
    for source, target in _VISION_RENAMES:
        if rest == source or rest.startswith(f"{source}."):
            rest = target + rest[len(source) :]
            break

    block_match = re.fullmatch(r"radio_model\.model\.blocks\.(\d+)\.(.+)", rest)
    if block_match:
        block_idx, inner = block_match.groups()

        qkv_match = re.fullmatch(r"attention\.attention\.(query|key|value)\.(weight|bias)", inner)
        if qkv_match:
            component, suffix = qkv_match.groups()
            return _fuse_qkv(block_idx, component, suffix, param)

        for source, target in _VISION_BLOCK_RENAMES:
            if inner == source or inner.startswith(f"{source}."):
                inner = target + inner[len(source) :]
                break
        rest = f"radio_model.model.blocks.{block_idx}.{inner}"

    return [(f"vision_model.{rest}", param)]


def _convert_projector(rest: str, name: str, param: torch.Tensor):
    """Map one vision-projector parameter name onto the checkpoint's."""
    # The Super-only LayerNorm keeps its name: NemotronH_Omni_Reasoning_V3 loads
    # it by the `vision_projector.vision_final_layernorm.` prefix verbatim.
    if rest.startswith("vision_final_layernorm."):
        return [(f"vision_projector.{rest}", param)]

    mlp1_match = re.fullmatch(r"(mlp1\.(?:norm|linear1|linear2))\.(weight|bias)", rest)
    if mlp1_match:
        module, suffix = mlp1_match.groups()
        return [(f"{_MLP1_RENAMES[module]}.{suffix}", param)]

    raise ValueError(f"Unsupported Nemotron 3.5 VL projector parameter {name!r}")


def convert_nemotron_h_vl_to_hf(args, name, param):
    """Convert one Nemotron 3.5 VL Megatron parameter to its HuggingFace form."""
    while name.startswith("module."):
        name = name.removeprefix("module.")

    if name.startswith("vision_model."):
        return _convert_vision(name[len("vision_model.") :], param)

    if name.startswith("vision_projector."):
        return _convert_projector(name[len("vision_projector.") :], name, param)

    # The decoder. convert_nemotron_h_to_hf strips the `language_model.` the VL
    # module puts in front of it and emits plain Nemotron-H names, so the prefix
    # goes back on: sglang routes the LLM on it, and the omni checkpoint's
    # canonical keys carry it (the unprefixed `backbone.*` / `lm_head.weight` in
    # its index are aliases, in model-aliases.safetensors).
    return [
        (f"language_model.{hf_name}", hf_param) for hf_name, hf_param in convert_nemotron_h_to_hf(args, name, param)
    ]
