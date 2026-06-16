"""
Qwen3.5-VL MoE bridge for megatron.bridge.

Registers ``Qwen3_5MoeForConditionalGeneration`` so that
``AutoBridge.from_hf_pretrained`` recognises Qwen3.5-VL MoE checkpoints
and can provide a Megatron-compatible VL model + weight mappings.

Architecture (Qwen3.5-35B-A3B):
  - 40 layers: 30 linear_attention (GDN) + 10 full_attention
  - full_attention_interval=4 (every 4th layer is full attention)
  - 256 experts per layer, 8 active per token + 1 shared expert
  - Expert weights in fused format (gate_up_proj / down_proj)

Architecture (Qwen3.5-397B-A17B):
  - 60 layers: 45 linear_attention (GDN) + 15 full_attention
  - full_attention_interval=4 (every 4th layer is full attention)
  - 512 experts per layer, 10 active per token + 1 shared expert
  - Expert weights in per-expert format (experts.*.gate_proj / up_proj / down_proj)
  - HF vision encoder (Qwen3_5MoeVisionModel, replicated on first PP stage)
  - Megatron GPTModel (MoE language model with M-RoPE)

NOTE on GDN (Gated DeltaNet) layers:
  Qwen3.5-VL MoE uses a hybrid GDN + full-attention architecture.
  The ``linear_attention_freq`` and related GDN parameters are passed
  through to the Megatron TransformerConfig.  Weight mappings for GDN
  layers (conv1d, in_proj, A_log, dt_bias, out_norm) are handled by
  the official ``Qwen35VLMoEBridge`` base class.  GDN *inference*
  support in Megatron requires the ``--experimental-attention-variant
  gated_delta_net`` flag and compatible TransformerEngine.

This bridge inherits from the official
``megatron.bridge.models.qwen_vl.qwen35_vl_bridge.Qwen35VLMoEBridge``
to reuse its proven weight mapping implementations.  We override only:

1. ``mapping_registry()`` — filter out the official Megatron-native vision
   model mappings (which target ``Qwen3VLModel``'s vision encoder paths like
   ``vision_model.decoder.layers.*``) and replace with a simple wildcard
   ``ReplicatedMapping("vision_model.**", "model.visual.**")`` for our HF
   ``Qwen3_5MoeVisionModel`` whose parameters already use HF naming.

2. ``provider_bridge()`` — create ``Qwen35VLMoeVLModelProvider`` instead of
   the official ``Qwen35VLMoEModelProvider``, since we use a hybrid
   architecture (HF vision encoder + Megatron GPTModel) rather than the
   fully Megatron-native ``Qwen3VLModel``.

3. ``maybe_modify_converted_hf_weight()`` — same expert merging logic as the
   parent but with CPU offloading to avoid GPU OOM.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.qwen.qwen_provider import Qwen3MoEModelProvider

# Official Qwen3.5-VL MoE bridge — we inherit from this to reuse its mature
# mapping_registry, maybe_modify_converted_hf_weight, and all GDN/MoE/vision
# mappings.  We only override what differs for our HF-vision-encoder architecture.
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge as _OfficialQwen35VLMoEBridge
from megatron.bridge.utils.common_utils import (
    extract_expert_number_from_param,
    hook_hf_module_setattr_for_tp_grad_sync,
)
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monkey-patch AutoMapping._detect_parallelism_type for robustness
# ---------------------------------------------------------------------------
# AutoMapping._detect_parallelism_type raises ValueError when it encounters
# module types not in its built-in registry (e.g. GatedDeltaNet, Conv1d,
# SharedExpertMLP, nn.Parameter with tensor_model_parallel flag).  This
# crashes the bridge during update_weights (Megatron→HF export).
#
# We replace the method with a version that adds two fallback heuristics
# BEFORE raising:
#   1. If the module has ``tensor_model_parallel = True`` with no
#      ``partition_dim``, assume column-parallel (the most common case
#      for head-dim split parameters like A_log, dt_bias).
#   2. If the module is a plain ``nn.Parameter`` / ``nn.Conv1d`` /
#      ``SharedExpertMLP`` or any other recognisable type, infer from
#      naming conventions or known defaults.
#
# We also register known module types into the AutoMapping registry upfront.
# ---------------------------------------------------------------------------
_Patched_detect_parallelism_type = False  # guard to patch only once


def _infer_parallelism_from_param_name(param_name: str) -> str:
    """Infer parallelism type from the Megatron parameter name.

    Used as a fallback when ``AutoMapping._detect_parallelism_type`` cannot
    be called because ``megatron_module`` is ``None`` (e.g. on non-owning
    PP/EP ranks with pp_size==1).

    Naming conventions in Megatron-LM:
    - Column-parallel: ``linear_qkv``, ``linear_proj`` (output), ``embedding``,
      ``output_layer``, ``linear_fc1`` (gate+up), ``in_proj``, ``A_log``, ``dt_bias``,
      expert weights ``linear_fc1``
    - Row-parallel: ``linear_proj`` (attention output), ``linear_fc2`` (down proj),
      expert weights ``linear_fc2``
    - Replicated: norms (``layernorm``, ``norm``), routers (``router``), biases
    """
    name = param_name.lower()

    # -- Row-parallel patterns (check first, more specific) --
    row_patterns = [
        "linear_proj.weight",  # attention output projection
        "linear_fc2.weight",  # MLP / expert down projection
        "out_proj.weight",  # GDN output projection
        "shared_experts.linear_fc2",  # shared expert down projection
    ]
    for pat in row_patterns:
        if pat in name:
            return "row"

    # -- Column-parallel patterns --
    col_patterns = [
        "linear_qkv",  # QKV projection
        "linear_q_up_proj",  # fused Q+up (some models)
        "linear_kv_up_proj",  # fused KV+up (some models)
        "embedding.word_embeddings",  # vocabulary embedding
        "output_layer",  # output projection
        "linear_fc1.weight",  # MLP / expert gate+up projection
        "in_proj.weight",  # GDN input projection
        "in_proj_qkv",  # GDN QKV part of input projection
        "in_proj_z",  # GDN z gate
        "in_proj_b",  # GDN b gate
        "in_proj_a",  # GDN a gate
        "a_log",  # GDN A_log parameter
        "dt_bias",  # GDT dt_bias parameter
        "conv1d.weight",  # GDN conv1d
        "shared_experts.linear_fc1",  # shared expert gate+up
    ]
    for pat in col_patterns:
        if pat in name:
            return "column"

    # -- Replicated patterns --
    replicated_patterns = [
        "layernorm",  # any layernorm weight/bias
        "layer_norm",  # alternative spelling
        "norm.weight",  # standalone norm
        "norm.bias",  # standalone norm bias
        "router.weight",  # MoE router
        "gate_weight",  # shared expert gate
        "gate.bias",  # gate bias
        "input_layernorm",  # input layernorm
        "pre_mlp_layernorm",  # pre-MLP layernorm
        "q_layernorm",  # Q layernorm
        "k_layernorm",  # K layernorm
        "layer_norm_weight",  # fused TE layernorm weight
        "layer_norm_bias",  # fused TE layernorm bias
    ]
    for pat in replicated_patterns:
        if pat in name:
            return "replicated"

    # Default: column-parallel is the most common case for weight matrices.
    # Bias-free models (like Qwen3.5) have mostly weights, and the majority
    # of weight matrices are column-parallel in Megatron.
    logger.warning(
        f"AutoMapping: could not infer parallelism type from param name "
        f"'{param_name}'. Defaulting to 'column'. If this is incorrect, "
        f"use an explicit mapping type."
    )
    return "column"


def _patch_auto_mapping_for_gdn():
    """Patch AutoMapping._detect_parallelism_type and register GDN module types.

    Safe to call multiple times -- subsequent calls are no-ops.
    """
    global _Patched_detect_parallelism_type
    if _Patched_detect_parallelism_type:
        return
    _Patched_detect_parallelism_type = True

    # --- Register known module types that are missing from the default registry ---
    # GatedDeltaNet: in_proj is ColumnParallelLinear, out_proj is RowParallelLinear,
    # but the GatedDeltaNet module itself acts as column-parallel for its parameters
    # (A_log, dt_bias are head-dim split).
    AutoMapping.register_module_type("GatedDeltaNet", "column")
    # SharedExpertMLP is a container (subclass of MLP); its linear layers are
    # individually Column/RowParallel, but the MLP itself is not a parallel module.
    AutoMapping.register_module_type("SharedExpertMLP", "replicated")
    # Conv1d in GDN has tensor_model_parallel=True but no partition_dim.
    # Treat as column-parallel (split along output dim).
    AutoMapping.register_module_type("Conv1d", "column")

    # --- Monkey-patch _detect_parallelism_type with a graceful fallback ---
    _orig_detect = AutoMapping._detect_parallelism_type

    def _patched_detect_parallelism_type(self, module):
        """Enhanced _detect_parallelism_type with graceful fallback for unknown modules."""
        import torch.nn as nn

        # First, try the original detection (registry + attribute checks + Norm/TELinear)
        try:
            return _orig_detect(self, module)
        except ValueError:
            pass  # Fall through to our heuristics below

        module_type = type(module).__name__

        # Heuristic 1: nn.Parameter with tensor_model_parallel flag
        # Parameters like A_log, dt_bias in GDN have tensor_model_parallel=True
        # but are plain nn.Parameter (not inside a parallel linear).
        if isinstance(module, nn.Parameter):
            if getattr(module, "tensor_model_parallel", False):
                partition_dim = getattr(module, "partition_dim", None)
                if partition_dim == 0:
                    return "column"
                elif partition_dim == 1:
                    return "row"
                # tensor_model_parallel=True with no partition_dim: assume column-parallel
                # (head-dim split, which is the most common case)
                logger.warning(
                    f"AutoMapping: parameter '{self.megatron_param}' is tensor_model_parallel "
                    f"but has no partition_dim. Assuming column-parallel. "
                    f"If this is incorrect, use an explicit mapping type."
                )
                return "column"
            else:
                return "replicated"

        # Heuristic 2: Module has tensor_model_parallel=True but no partition_dim
        # e.g. Conv1d, or custom modules. Column-parallel is the safe default
        # for weight matrices split along the output dimension.
        if hasattr(module, "tensor_model_parallel"):
            if not module.tensor_model_parallel:
                return "replicated"
            partition_dim = getattr(module, "partition_dim", None)
            if partition_dim == 0:
                return "column"
            elif partition_dim == 1:
                return "row"
            # tensor_model_parallel=True with no partition_dim: default to column
            logger.warning(
                f"AutoMapping: module '{module_type}' for param '{self.megatron_param}' "
                f"has tensor_model_parallel=True but no partition_dim. "
                f"Assuming column-parallel. If this is incorrect, use an explicit mapping type."
            )
            return "column"

        # Heuristic 3:_nn.Module submodules that are known to be non-parallel
        # (e.g. TopKRouter is already registered, but add fallbacks for others)
        if isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.Identity)):
            return "replicated"

        # Final fallback: if we truly can't determine, log a warning and assume replicated.
        # This is safer than crashing and allows the pipeline to continue for
        # non-critical parameters.
        logger.warning(
            f"AutoMapping: cannot determine parallelism type for module '{module_type}' "
            f"at weight '{self.megatron_param}'. Assuming replicated. "
            f"If this is incorrect, register the module type with "
            f"AutoMapping.register_module_type('{module_type}', 'column|row|replicated') "
            f"or use an explicit mapping type."
        )
        return "replicated"

    AutoMapping._detect_parallelism_type = _patched_detect_parallelism_type

    # --- Monkey-patch _get_or_create_mapping to handle None parallelism_type ---
    # When megatron_module is None (e.g. on non-owning PP/EP ranks) and pp_size==1,
    # broadcast_obj_from_pp_rank(None, ...) directly returns None, causing
    # _detected_type=None which crashes _get_or_create_mapping.
    # We patch it to infer the parallelism type from the megatron_param name as a fallback.
    _orig_get_or_create_mapping = AutoMapping._get_or_create_mapping

    def _patched_get_or_create_mapping(self, parallelism_type):
        """Enhanced _get_or_create_mapping with fallback for None parallelism_type."""
        if parallelism_type is not None:
            return _orig_get_or_create_mapping(self, parallelism_type)

        # parallelism_type is None — this happens when megatron_module is None
        # (non-owning PP/EP rank) and pp_size==1 so broadcast returns None directly.
        # Infer from the megatron_param name as a heuristic fallback.
        param_name = self.megatron_param or ""
        inferred = _infer_parallelism_from_param_name(param_name)
        logger.warning(
            f"AutoMapping: parallelism_type is None for param '{param_name}'. "
            f"Inferred '{inferred}' from parameter name heuristics. "
            f"This typically occurs when megatron_module is unavailable (e.g. EP split). "
            f"If this is incorrect, use an explicit mapping type."
        )
        return _orig_get_or_create_mapping(self, inferred)

    AutoMapping._get_or_create_mapping = _patched_get_or_create_mapping

    # --- Monkey-patch _add_separate_layernorm_mappings to support non-AutoMapping types ---
    _orig_add_layernorm = MegatronMappingRegistry._add_separate_layernorm_mappings

    def _patched_add_separate_layernorm_mappings(self):
        """Enhanced version that creates correct mapping types for non-AutoMapping mappings."""
        original_mappings = list(self.mappings)
        existing_names = {mapping.megatron_param for mapping in self.mappings}
        extra_mappings = []

        for mapping in original_mappings:
            for old_name, new_name in self._SEPARATE_LAYERNORM_REWRITES:
                if not mapping.megatron_param.endswith(f"*.{old_name}"):
                    continue
                new_megatron_param = mapping.megatron_param[: -len(old_name)] + new_name
                if new_megatron_param in existing_names:
                    break
                # Determine the correct mapping type based on the original mapping
                if isinstance(mapping, AutoMapping):
                    new_mapping = AutoMapping(new_megatron_param, mapping.hf_param, mapping.permute_dims)
                elif isinstance(mapping, ReplicatedMapping):
                    new_mapping = ReplicatedMapping(new_megatron_param, mapping.hf_param)
                elif isinstance(mapping, ColumnParallelMapping):
                    new_mapping = ColumnParallelMapping(new_megatron_param, mapping.hf_param)
                elif isinstance(mapping, RowParallelMapping):
                    new_mapping = RowParallelMapping(new_megatron_param, mapping.hf_param)
                else:
                    # For other mapping types, fall back to AutoMapping (safe default
                    # since layernorm weights are always replicated)
                    new_mapping = ReplicatedMapping(new_megatron_param, mapping.hf_param)
                extra_mappings.append(new_mapping)
                existing_names.add(new_megatron_param)
                break

        if extra_mappings:
            self.mappings.extend(extra_mappings)

    MegatronMappingRegistry._add_separate_layernorm_mappings = _patched_add_separate_layernorm_mappings

    logger.info(
        "Patched AutoMapping._detect_parallelism_type with graceful fallback, "
        "AutoMapping._get_or_create_mapping with None-type fallback, "
        "and MegatronMappingRegistry._add_separate_layernorm_mappings "
        "to support non-AutoMapping types."
    )


# ---------------------------------------------------------------------------
# THD <-> BSHD helpers (same as GLM-4.6V bridge)
# ---------------------------------------------------------------------------
def _thd_to_bshd(packed: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Unpack THD-format [1, T, ...] to BSHD [bs, max_seq, ...] using cu_seqlens."""
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq = seqlens.max().item()
    bs = len(cu_seqlens) - 1
    out = packed.new_zeros(bs, max_seq, *packed.shape[2:])
    for i, sl in enumerate(seqlens):
        out[i, :sl] = packed[0, cu_seqlens[i] : cu_seqlens[i] + sl]
    return out


def _bshd_to_thd(unpacked: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Pack BSHD [bs, max_seq, ...] back to THD [1, T, ...]."""
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    total = cu_seqlens[-1].item()
    out = unpacked.new_zeros(1, total, *unpacked.shape[2:])
    for i, sl in enumerate(seqlens):
        out[0, cu_seqlens[i] : cu_seqlens[i] + sl] = unpacked[i, :sl]
    return out


def _gather_input_ids_from_cp(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct full (global) input_ids from zigzag CP chunks."""
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return input_ids

    gathered = torch.distributed.nn.all_gather(input_ids, group=parallel_state.get_context_parallel_group())

    local_cu_seqlens = cu_seqlens // cp_size
    num_seqs = len(cu_seqlens) - 1
    whole_list = []
    for i in range(num_seqs):
        seqlen = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
        chunk_size = seqlen // 2 // cp_size
        whole_list.extend(
            gathered[cp_rank][0, local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size] for cp_rank in range(cp_size)
        )
        whole_list.extend(
            [
                gathered[cp_rank][0, local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                for cp_rank in range(cp_size)
            ][::-1]
        )
    return torch.cat(whole_list).unsqueeze(0)


def _select_local_image_embeds(
    full_input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    image_token_id: int,
    image_embeds: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Select the subset of *image_embeds* that falls in this CP rank's chunk."""
    device = full_input_ids.device
    full_flat = full_input_ids[0]
    full_mask = full_flat == image_token_id

    T_global = full_flat.shape[0]
    rank_mask = torch.zeros(T_global, dtype=torch.bool, device=device)

    num_seqs = len(cu_seqlens) - 1
    for i in range(num_seqs):
        seq_start = cu_seqlens[i].item()
        seqlen = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
        chunk_size = seqlen // (2 * cp_size)

        first_start = seq_start + cp_rank * chunk_size
        rank_mask[first_start : first_start + chunk_size] = True

        second_end = seq_start + seqlen - cp_rank * chunk_size
        rank_mask[second_end - chunk_size : second_end] = True

    local_image_mask = full_mask & rank_mask
    n_local = local_image_mask.sum().item()

    if n_local == 0:
        return image_embeds[:0]
    if n_local == image_embeds.shape[0]:
        return image_embeds

    image_cumsum = full_mask.long().cumsum(0)
    local_positions = local_image_mask.nonzero(as_tuple=True)[0]
    embed_indices = image_cumsum[local_positions] - 1
    return image_embeds[embed_indices]


# ---------------------------------------------------------------------------
# Megatron VL Model
# ---------------------------------------------------------------------------
class Qwen35VLMoeVLModel(MegatronModule):
    """Qwen3.5-VL MoE vision-language model for Megatron training.

    Wraps an HF vision encoder (only on first PP stage) together with a
    standard Megatron Core GPTModel configured for M-RoPE.

    The vision encoder is frozen (not trained during RL/distillation).
    """

    def __init__(
        self,
        language_transformer_config,
        language_transformer_layer_spec,
        hf_vision_config,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.spatial_merge_size = language_transformer_config.spatial_merge_size

        self.share_embeddings_and_output_weights = False

        # Vision encoder -- only on the first pipeline stage
        self.vision_model = None
        if self.pre_process:
            from transformers import Qwen3_5MoeVisionModel

            self.vision_model = Qwen3_5MoeVisionModel._from_config(hf_vision_config)
            # Freeze vision encoder -- not trained during RL
            self.vision_model.requires_grad_(False)
            self.vision_model.eval()
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)
            if torch.cuda.is_available():
                self.vision_model = self.vision_model.to("cuda")

        # Language model -- standard Megatron GPT with M-RoPE
        self.language_model = MCoreGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
        )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    # -- helpers required by Megatron pipeline engine -----------------------

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    # -- vision helpers -----------------------------------------------------

    def _get_image_features(self, pixel_values, image_grid_thw):
        """Run HF vision encoder and return flat image embeddings.

        The vision model applies a PatchMerger that performs spatial merging
        (2x2→1) and projects from vision hidden_size*4 to out_hidden_size
        (matching the language model's hidden_size).  The merged output is
        stored in ``pooler_output``; ``last_hidden_state`` is the *pre-merge*
        tensor and has the wrong shape.
        """
        pixel_values = pixel_values.to(dtype=self.vision_model.dtype)
        with torch.no_grad():
            output = self.vision_model(pixel_values, grid_thw=image_grid_thw)
            # pooler_output = after PatchMerger: [N_tokens_after_merge, out_hidden_size]
            # last_hidden_state = before PatchMerger: [N_tokens_before_merge, vision_hidden_size]
            if isinstance(output, torch.Tensor):
                return output
            return output.pooler_output

    # -- M-RoPE position IDs -----------------------------------------------

    @staticmethod
    def _get_vision_position_ids(
        start_position: int,
        grid_thw,
        temp_merge_size: int,
        spatial_merge_size: int,
        device,
    ) -> torch.Tensor:
        """Compute 3D positions for one image/video region (ported from HF).

        For Qwen3.5-VL, temp_merge_size is grid_thw[0] (the temporal dimension of grid_thw
        already accounts for temporal_patch_size merging).
        The mRoPE sections are [11, 11, 10] (temporal, height, width).
        """
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size
        n_tokens = llm_grid_h * llm_grid_w * llm_grid_t

        pos_w = torch.arange(start_position, start_position + llm_grid_w, device=device)
        pos_w = pos_w.repeat(llm_grid_h * llm_grid_t)
        pos_h = torch.arange(start_position, start_position + llm_grid_h, device=device)
        pos_h = pos_h.repeat_interleave(llm_grid_w * llm_grid_t)
        pos_t = torch.full((n_tokens,), start_position, device=device, dtype=torch.long)
        return torch.stack([pos_t, pos_h, pos_w], dim=0)  # [3, n_tokens]

    def _compute_mrope_position_ids(
        self,
        input_ids_bshd: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute 3D M-RoPE position IDs from input_ids in [bs, seq] format.

        Image regions are detected by looking for consecutive runs of
        ``image_token_id`` in each sequence.
        """
        bs, seq_len = input_ids_bshd.shape
        device = input_ids_bshd.device
        spatial_merge_size = self.spatial_merge_size

        position_ids = torch.zeros(3, bs, seq_len, dtype=torch.long, device=device)

        if image_grid_thw is None or image_grid_thw.numel() == 0:
            # Text-only: standard 1D positions replicated across 3 dims
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
            position_ids[0] = pos
            position_ids[1] = pos
            position_ids[2] = pos
            return position_ids

        grid_iter = iter(image_grid_thw)

        for b in range(bs):
            ids = input_ids_bshd[b]
            is_image = ids == self.image_token_id

            # Find contiguous groups: text (0) vs image (1)
            token_types = is_image.long()
            groups = []
            for key, group in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
                g = list(group)
                groups.append((key, g[0][0], g[-1][0] + 1))

            current_pos = 0
            pos_list = []
            for modality, start, end in groups:
                if modality == 0:
                    # Text tokens
                    n = end - start
                    pos_list.append(torch.arange(n, device=device).view(1, -1).expand(3, -1) + current_pos)
                    current_pos += n
                else:
                    # Image tokens
                    grid_thw = next(grid_iter)
                    temp_merge_size = grid_thw[0]
                    vis_pos = self._get_vision_position_ids(
                        current_pos,
                        grid_thw,
                        temp_merge_size,
                        spatial_merge_size,
                        device,
                    )
                    pos_list.append(vis_pos)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

            all_pos = torch.cat(pos_list, dim=1)  # [3, seq_for_this_sample]
            position_ids[:, b, : all_pos.shape[1]] = all_pos

        return position_ids

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params=None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        # multimodal kwargs (unpacked from multimodal_train_inputs)
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        # unused VL kwargs that may come through
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        mm_token_type_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        assert pixel_values_videos is None, "Video not supported yet"
        assert inference_params is None, "Inference not supported"

        # -- Extract cu_seqlens and CP info early --
        cu_seqlens = None
        if packed_seq_params is not None:
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
        cp_size = parallel_state.get_context_parallel_world_size()
        full_input_ids = None

        combined_embeddings = None

        if self.pre_process:
            # 1. Text embeddings from language model embedding layer
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()  # [seq, batch, hidden]

            # 2. Vision encoding + masked scatter
            if pixel_values is not None and image_grid_thw is not None:
                image_embeds = self._get_image_features(pixel_values, image_grid_thw)
                image_embeds = image_embeds.to(combined_embeddings.device, combined_embeddings.dtype)

                # With CP > 1, select only the embeddings for this rank's chunk
                if cp_size > 1 and cu_seqlens is not None:
                    full_input_ids = _gather_input_ids_from_cp(input_ids, cu_seqlens)
                    cp_rank = parallel_state.get_context_parallel_rank()
                    image_embeds = _select_local_image_embeds(
                        full_input_ids,
                        cu_seqlens,
                        self.image_token_id,
                        image_embeds,
                        cp_rank,
                        cp_size,
                    )

                image_mask = (input_ids == self.image_token_id).contiguous()
                # Scatter: [seq, bs, hidden] -> [bs, seq, hidden]
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                if image_mask.any():
                    combined_embeddings[image_mask] = image_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            # Scatter to sequence-parallel region if needed
            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()

        # 3. Compute M-RoPE position IDs
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()

        if position_ids is None:
            if self.pre_process:
                if cu_seqlens is not None:
                    if cp_size > 1:
                        if full_input_ids is None:
                            full_input_ids = _gather_input_ids_from_cp(input_ids, cu_seqlens)
                    else:
                        full_input_ids = input_ids
                    input_ids_bshd = _thd_to_bshd(full_input_ids, cu_seqlens)
                    pos_bshd = self._compute_mrope_position_ids(input_ids_bshd, image_grid_thw)
                    pos_packed = _bshd_to_thd(pos_bshd.permute(1, 2, 0), cu_seqlens)
                    position_ids = pos_packed.permute(2, 0, 1).contiguous()  # [3, 1, T_global]
                else:
                    position_ids = self._compute_mrope_position_ids(input_ids, image_grid_thw)
            else:
                # Non-first PP stage: allocate buffer with correct shape
                if cu_seqlens is not None:
                    T = cu_seqlens[-1].item()
                    position_ids = torch.zeros(3, 1, T, dtype=torch.long, device=torch.cuda.current_device())
                else:
                    raise NotImplementedError(
                        "Non-THD position_ids broadcast not yet supported for non-first PP stages"
                    )

            # Broadcast position_ids from first to all PP stages
            if pp_size > 1:
                src = parallel_state.get_pipeline_model_parallel_first_rank()
                torch.distributed.broadcast(
                    position_ids,
                    src=src,
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )

        # 4. Language model forward (pass decoder_input to skip re-embedding)
        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        return output


# ---------------------------------------------------------------------------
# Model Provider (dataclass that doubles as TransformerConfig)
# ---------------------------------------------------------------------------
@dataclass
class Qwen35VLMoeVLModelProvider(Qwen3MoEModelProvider):
    """Provider that creates Qwen35VLMoeVLModel.

    Inherits from Qwen3MoEModelProvider to reuse MoE + TransformerConfig infra.
    Defined at module level (not inside a function) so that the class is
    picklable -- megatron-bridge broadcasts config objects across PP ranks
    via ``torch.distributed.broadcast_object_list`` which requires pickling.
    """

    # Qwen3.5-VL specific config
    image_token_id: int = 248056
    video_token_id: int = 248057
    spatial_merge_size: int = 2

    # Vision config (stored as HF config object)
    hf_vision_config: object = None
    hf_text_config: object = None

    # M-RoPE
    position_embedding_type: str = "mrope"
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])
    scatter_embedding_sequence_parallel: bool = False

    # Language model sequence length
    language_max_sequence_length: int = 262144

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Create a Qwen35VLMoeVLModel instance."""
        from megatron.core import parallel_state as ps

        if pre_process is None:
            pre_process = ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
        if post_process is None:
            post_process = ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)

        # Build per-layer specs respecting moe_layer_freq and experimental attention variant.
        # When experimental_attention_variant is set (e.g. "gated_delta_net"), we must use
        # get_transformer_block_with_experimental_attention_variant_spec instead of
        # get_gpt_decoder_block_spec — the latter asserts that experimental_attention_variant
        # is None and cannot handle hybrid GDN + SDPA architectures.
        if self.experimental_attention_variant is not None:
            from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
                get_transformer_block_with_experimental_attention_variant_spec,
            )

            transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
                config=self,
                vp_stage=vp_stage,
            )
        else:
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=self,
                use_transformer_engine=True,
                vp_stage=vp_stage,
            )

        model = Qwen35VLMoeVLModel(
            language_transformer_config=self,
            language_transformer_layer_spec=transformer_layer_spec,
            hf_vision_config=self.hf_vision_config,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )

        return model


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------
try:
    from transformers import Qwen3_5MoeForConditionalGeneration as _Qwen35MoeHF
except ImportError:
    _Qwen35MoeHF = "Qwen3_5MoeForConditionalGeneration"


@MegatronModelBridge.register_bridge(source=_Qwen35MoeHF, target=Qwen35VLMoeVLModel)
class Qwen35VLMoeBridge(_OfficialQwen35VLMoEBridge):
    """Bridge between HuggingFace Qwen3.5-VL MoE and our custom Megatron VL model.

    Inherits from the official ``Qwen35VLMoEBridge`` to reuse its mature weight
    mappings for the language model (GDN, MoE experts, shared experts, attention,
    MTP, etc.) and its ``maybe_modify_converted_hf_weight`` for expert weight
    merging.

    We override:
    - ``mapping_registry()``: Replace the official vision model mappings (which
      target Megatron-native ``Qwen3VLModel`` vision encoder paths like
      ``vision_model.decoder.layers.*``) with a simple wildcard
      ``ReplicatedMapping("vision_model.**", "model.visual.**")`` because we
      use an HF ``Qwen3_5MoeVisionModel`` whose parameter names already match
      HF conventions.
    - ``provider_bridge()``: Create our ``Qwen35VLMoeVLModelProvider`` instead
      of the official ``Qwen35VLMoEModelProvider``, since we wrap an HF vision
      encoder + Megatron GPTModel instead of the fully Megatron-native
      ``Qwen3VLModel``.
    - ``maybe_modify_converted_hf_weight()``: Same logic as the parent but with
      CPU offloading to avoid GPU OOM when concatenating large expert tensors.
    """

    # ------------------------------------------------------------------
    # Vision model mapping replacement
    # ------------------------------------------------------------------
    # The official bridge maps Megatron-native vision model params like:
    #   vision_model.decoder.layers.*.self_attention.linear_qkv.weight
    #   vision_model.decoder.layers.*.mlp.linear_fc1.weight
    #   vision_model.merger.*.weight
    #   vision_model.patch_embed.proj.*
    #
    # Our ``Qwen35VLMoeVLModel`` uses an HF ``Qwen3_5MoeVisionModel`` directly,
    # so its parameter names follow HF conventions (e.g. ``vision_model.blocks.0.attn.qkv.weight``).
    # We need to strip "vision_model." and replace with "model.visual." — a simple
    # wildcard mapping handles this.
    _VISION_MEGATRON_PREFIX = "vision_model."
    _VISION_HF_PREFIX = "model.visual."

    # ------------------------------------------------------------------
    # Store HF keys before mapping_registry() is called
    # ------------------------------------------------------------------
    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        """Override to store HF config/keys before mapping_registry is called.

        We need access to the HF checkpoint key names to determine the expert
        weight format (fused vs per-expert) at mapping_registry() time.
        """
        self._hf_config = hf_pretrained.config
        self._hf_state_source = hf_pretrained.state.source
        self._hf_keys = list(self._hf_state_source.get_all_keys())
        return super().build_conversion_tasks(hf_pretrained, megatron_model)

    def _uses_fused_experts(self) -> bool:
        """Check whether the HF checkpoint uses fused expert format.

        Qwen3.5 MoE models store expert weights in two possible formats:

        1. **Fused format** (e.g. 35B-A3B with 256 experts):
           - ``model.language_model.layers.*.mlp.experts.gate_up_proj``
             shape: [num_experts, 2*intermediate_size, hidden_size]
           - ``model.language_model.layers.*.mlp.experts.down_proj``
             shape: [num_experts, hidden_size, intermediate_size]

        2. **Per-expert format** (e.g. 397B-A17B with 512 experts):
           - ``model.language_model.layers.*.mlp.experts.*.gate_proj.weight``
           - ``model.language_model.layers.*.mlp.experts.*.up_proj.weight``
           - ``model.language_model.layers.*.mlp.experts.*.down_proj.weight``

        Returns True if the checkpoint uses the fused format.
        """
        hf_keys = getattr(self, "_hf_keys", None)
        if hf_keys:
            if any("mlp.experts.gate_up_proj" in key for key in hf_keys) or any(
                "mlp.experts.down_proj" in key for key in hf_keys
            ):
                return True

        hf_source = getattr(self, "_hf_state_source", None)
        if hf_source is not None:
            return hf_source.has_glob("*mlp.experts.gate_up_proj*") or hf_source.has_glob("*mlp.experts.down_proj*")

        # Default: assume fused format (backward compatible with 35B-A3B)
        return True

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Build weight mappings by reusing the official bridge and replacing vision/expert mappings.

        Calls ``super().mapping_registry()`` to get all language model mappings
        (GDN, MoE, attention, MTP, etc.), then:

        1. Filters out the official vision model mappings and adds our wildcard
           ``ReplicatedMapping`` instead (for our HF vision encoder).

        2. If the HF checkpoint uses **per-expert format** (e.g. 397B-A17B with
           512 experts stores ``experts.*.gate_proj.weight`` instead of
           ``experts.gate_up_proj``), replaces the official ``ExpertMLPGateUpProjMapping``
           and ``ExpertMLPDownProjMapping`` with per-expert style ``GatedMLPMapping``
           and ``AutoMapping``.

        IMPORTANT: We must construct a *new* ``MegatronMappingRegistry`` from the
        filtered mapping list rather than mutating the old registry's ``.mappings``
        attribute, because the registry pre-compiles regex patterns (``_compiled_patterns``,
        ``_reverse_patterns``) during ``__init__`` and does NOT re-compile them when
        ``.mappings`` is replaced.  Mutating would leave stale patterns and cause
        lookup failures (params not matched → ``None`` tasks → crash in
        ``load_weights_hf_to_megatron``).
        """
        registry = super().mapping_registry()

        use_fused = self._uses_fused_experts()

        # Filter out official vision model mappings — they target
        # Megatron-native Qwen3VLModel vision encoder paths which don't
        # match our HF vision encoder's parameter names.
        #
        # Also, if using per-expert format, filter out the official
        # ExpertMLPGateUpProjMapping / ExpertMLPDownProjMapping and replace
        # them with per-expert mappings below.
        _EXPERT_MEGATRON_PREFIXES_FOR_PER_EXPERT = (
            "language_model.decoder.layers.*.mlp.experts.linear_fc1.weight",
            "language_model.decoder.layers.*.mlp.experts.linear_fc2.weight",
            # MTP expert mappings
            "language_model.mtp.layers.*.mtp_model_layer.mlp.experts.linear_fc1.weight",
            "language_model.mtp.layers.*.mtp_model_layer.mlp.experts.linear_fc2.weight",
        )

        filtered_mappings = []
        for m in registry.mappings:
            # Remove vision model mappings
            if m.megatron_param.startswith(self._VISION_MEGATRON_PREFIX):
                continue
            # If per-expert format, remove fused expert mappings for decoder layers
            if not use_fused and m.megatron_param.startswith(_EXPERT_MEGATRON_PREFIXES_FOR_PER_EXPERT):
                continue
            filtered_mappings.append(m)

        # Add our wildcard mapping for the HF vision encoder.
        # This maps ``vision_model.**`` → ``model.visual.**`` one-to-one,
        # which works because our vision encoder is the HF model directly.
        filtered_mappings.append(
            ReplicatedMapping(
                megatron_param="vision_model.**",
                hf_param="model.visual.**",
            )
        )

        # Add per-expert format mappings if the checkpoint doesn't use fused format
        if not use_fused:
            logger.info(
                "Detected per-expert HF weight format (e.g. experts.*.gate_proj.weight). "
                "Using per-expert GatedMLPMapping/AutoMapping instead of "
                "ExpertMLPGateUpProjMapping/ExpertMLPDownProjMapping."
            )
            filtered_mappings.extend(
                [
                    # Per-expert gate+up projection → fused linear_fc1
                    GatedMLPMapping(
                        megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        gate="model.language_model.layers.*.mlp.experts.*.gate_proj.weight",
                        up="model.language_model.layers.*.mlp.experts.*.up_proj.weight",
                    ),
                    # Per-expert down projection → linear_fc2
                    AutoMapping(
                        megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param="model.language_model.layers.*.mlp.experts.*.down_proj.weight",
                    ),
                    # MTP per-expert mappings
                    GatedMLPMapping(
                        megatron_param="language_model.mtp.layers.*.mtp_model_layer.mlp.experts.linear_fc1.weight*",
                        gate="mtp.layers.*.mlp.experts.*.gate_proj.weight",
                        up="mtp.layers.*.mlp.experts.*.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param="language_model.mtp.layers.*.mtp_model_layer.mlp.experts.linear_fc2.weight*",
                        hf_param="mtp.layers.*.mlp.experts.*.down_proj.weight",
                    ),
                ]
            )

        # Construct a NEW registry so that _compiled_patterns and
        # _reverse_patterns are rebuilt with the updated mapping list.
        return MegatronMappingRegistry(*filtered_mappings)

    # ------------------------------------------------------------------
    # Expert weight merging with CPU offloading
    # ------------------------------------------------------------------
    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Mapping,
    ) -> dict[str, torch.Tensor]:
        """Merge per-expert weight exports into a single fused [num_experts, ...] tensor.

        For **fused HF format** (e.g. 35B model, ``experts.gate_up_proj``), all experts
        share a single HF key.  This method caches each expert's contribution and merges
        them when all ``num_experts`` entries are collected.

        For **per-expert HF format** (e.g. 397B model, ``experts.0.gate_proj.weight``,
        ``experts.1.gate_proj.weight``, …), each expert already has a unique HF key
        produced by the mapping.  The merging logic does not apply — we must return the
        converted weights directly, otherwise they are silently dropped (each key only
        ever receives a single entry so the cache-merge path is never triggered).

        The parent class's implementation only supports the fused case.  We detect the
        format via ``_uses_fused_experts()`` and skip merging for per-expert format.
        """
        # Per-expert HF format: each expert has a unique key (e.g.
        # ``experts.0.gate_proj.weight``).  The parent's caching/merging logic
        # assumes all experts share one key, which causes every expert weight to be
        # cached under its own unique key and never merged (the ``len == num_experts``
        # guard is never satisfied), effectively **dropping all expert weights**.
        # Return directly — no merging needed.
        if not self._uses_fused_experts():
            return converted_weights_dict

        # Fused HF format: fall through to the parent's caching/merging logic.
        num_experts = self.hf_config.text_config.num_experts
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        experts_per_rank = num_experts // ep_size

        try:
            local_expert_number = extract_expert_number_from_param(task.param_name) % experts_per_rank
        except ValueError:
            # Not an expert parameter — pass through unchanged.
            return converted_weights_dict

        # Detect if EP gathering was already done by the mapping (e.g. GatedMLPMapping
        # with is_expert=True calls gather_from_ep_ranks internally).
        if ep_size > 1:
            expert_ids_in_dict = set()
            for key in converted_weights_dict:
                try:
                    expert_ids_in_dict.add(extract_expert_number_from_param(key))
                except ValueError:
                    pass
            if len(expert_ids_in_dict) > 1:
                return converted_weights_dict

        result: dict[str, torch.Tensor] = {}
        for key, value in converted_weights_dict.items():
            if key not in self.hf_weights_cache:
                self.hf_weights_cache[key] = {}

            # Move to CPU to avoid GPU OOM when concatenating large expert tensors
            value = value.cpu()

            if ep_size == 1:
                self.hf_weights_cache[key][local_expert_number] = value
            else:
                assert value.shape[0] == ep_size, (
                    f"Expected shape[0]=={ep_size} for EP-gathered expert weight " f"'{key}', got {value.shape}"
                )
                for i, exp_val in enumerate(value):
                    global_expert_number = local_expert_number + (i * experts_per_rank)
                    self.hf_weights_cache[key][global_expert_number] = exp_val

            if len(self.hf_weights_cache[key]) == num_experts:
                logger.debug(f"All {num_experts} experts loaded for {key}")
                merged = torch.cat(
                    [self.hf_weights_cache[key][i].unsqueeze(0) for i in range(num_experts)],
                    dim=0,
                )
                del self.hf_weights_cache[key]
                # Move back to CUDA for downstream processing
                result[key] = merged.cuda()
            else:
                logger.debug(f"{len(self.hf_weights_cache[key])}/{num_experts} experts " f"loaded for {key}")

        return result

    def provider_bridge(self, hf_pretrained):
        """Create a Qwen35VLMoeVLModelProvider from HF config."""
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = deepcopy(hf_config.vision_config)

        model_dtype = self.dtype_from_hf(text_config, default=torch.bfloat16)
        vision_config.torch_dtype = model_dtype

        ProviderClass = Qwen35VLMoeVLModelProvider

        rope_params = getattr(text_config, "rope_parameters", {}) or {}
        mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        rotary_base = rope_params.get("rope_theta", 10000000)
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.25)

        # Determine MoE layer frequency
        # Qwen3.5-VL MoE uses all layers as MoE (no dense layer)
        first_k_dense = getattr(text_config, "first_k_dense_replace", 0)
        num_layers = text_config.num_hidden_layers
        moe_layer_freq_list = [0] * first_k_dense + [1] * (num_layers - first_k_dense)

        # Shared expert intermediate size
        moe_ffn = getattr(text_config, "moe_intermediate_size", 512)
        shared_expert_intermediate = getattr(text_config, "shared_expert_intermediate_size", 512)

        # Read attention bias from config
        add_qkv_bias = getattr(text_config, "attention_bias", False)

        # QK layernorm
        qk_layernorm = True

        # head_dim
        head_dim = getattr(text_config, "head_dim", 256)

        # Qwen3.5 MoE text_config has no intermediate_size; use shared_expert_intermediate_size
        # as the dense-FFN fallback (dense layers don't exist when first_k_dense=0, but the
        # TransformerConfig still requires ffn_hidden_size).
        ffn_hidden_size = getattr(text_config, "intermediate_size", None)
        if ffn_hidden_size is None:
            ffn_hidden_size = shared_expert_intermediate

        provider = ProviderClass(
            # Language model configuration
            num_layers=num_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=head_dim,
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            normalization="RMSNorm",
            layernorm_zero_centered_gamma=True,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=rotary_base,
            rotary_percent=partial_rotary_factor,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=getattr(text_config, "max_position_embeddings", 262144),
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            # MoE configuration
            num_moe_experts=getattr(text_config, "num_experts", 256),
            moe_router_topk=getattr(text_config, "num_experts_per_tok", 8),
            moe_ffn_hidden_size=moe_ffn,
            moe_shared_expert_intermediate_size=shared_expert_intermediate,
            moe_layer_freq=moe_layer_freq_list,
            moe_grouped_gemm=True,
            moe_router_load_balancing_type="global_aux_loss",
            moe_aux_loss_coeff=getattr(text_config, "router_aux_loss_coef", 0.001),
            moe_router_pre_softmax=False,
            moe_router_score_function="softmax",
            moe_router_dtype="fp32",
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            # Attention
            add_qkv_bias=add_qkv_bias,
            add_bias_linear=False,
            qk_layernorm=qk_layernorm,
            # Attention output gate (Qwen3.5 specific)
            attention_output_gate=True,
            # Shared expert gate
            moe_shared_expert_gate=True,
            # GDN (Gated DeltaNet) — attention variant and layer pattern
            # Must set experimental_attention_variant="gated_delta_net" so that
            # Megatron builds GDN layers (GatedDeltaNet modules) instead of
            # standard SDPA attention for the layers marked as linear_attention.
            experimental_attention_variant="gated_delta_net",
            # Qwen3.5-VL MoE uses a hybrid GDN + full-attention architecture.
            # The HF config stores per-layer types in ``layer_types`` list
            # (e.g. ["linear_attention","linear_attention","linear_attention","full_attention", ...])
            # rather than a scalar ``full_attention_interval``.
            # Megatron expects ``linear_attention_freq`` as:
            #   int N  -> one SDPA layer every N layers
            #   list   -> per-layer pattern: 1=linear_attention(GDN), 0=full_attention(SDPA)
            # We convert from HF's ``layer_types`` strings to Megatron's int list.
            linear_attention_freq=(
                [1 if lt == "linear_attention" else 0 for lt in text_config.layer_types]
                if getattr(text_config, "layer_types", None) is not None
                else getattr(text_config, "full_attention_interval", None)
            ),
            linear_conv_kernel_dim=getattr(text_config, "linear_conv_kernel_dim", None),
            linear_key_head_dim=getattr(text_config, "linear_key_head_dim", None),
            linear_value_head_dim=getattr(text_config, "linear_value_head_dim", None),
            linear_num_key_heads=getattr(text_config, "linear_num_key_heads", None),
            linear_num_value_heads=getattr(text_config, "linear_num_value_heads", None),
            # M-RoPE
            mrope_section=mrope_section,
            position_embedding_type="mrope",
            scatter_embedding_sequence_parallel=False,
            # Vision
            hf_vision_config=vision_config,
            hf_text_config=text_config,
            image_token_id=getattr(hf_config, "image_token_id", 248056),
            video_token_id=getattr(hf_config, "video_token_id", 248057),
            spatial_merge_size=getattr(hf_config.vision_config, "spatial_merge_size", 2),
            language_max_sequence_length=getattr(text_config, "max_position_embeddings", 262144),
        )

        return provider


# Apply patches at module load time so that they are active whenever this
# bridge module is imported, regardless of import order.
_patch_auto_mapping_for_gdn()
