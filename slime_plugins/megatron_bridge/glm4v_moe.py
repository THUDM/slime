"""
GLM-4.6V (glm4v_moe) bridge for megatron.bridge.

Registers `Glm4vMoeForConditionalGeneration` so that `AutoBridge.from_hf_pretrained`
recognises GLM-4.6V checkpoints and can provide a Megatron-compatible VL model +
weight mappings.

Architecture:
  HF vision encoder (Glm4vMoeVisionModel, replicated on first PP stage)
  + Megatron GPTModel (MoE language model, standard M-RoPE)
"""

from __future__ import annotations

import itertools
import logging
from copy import deepcopy
from dataclasses import dataclass, field

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping, ReplicatedMapping
from megatron.bridge.models.qwen.qwen_provider import Qwen3MoEModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# THD ↔ BSHD helpers (cf. Qwen3VL bridge)
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
    """Reconstruct full (global) input_ids from zigzag CP chunks.

    With zigzag CP, each CP rank r holds chunks [r] and [2*cp_size-1-r] for
    each sequence.  This function all-gathers across CP ranks and reassembles
    the original token order so that position-ID computation sees the full
    sequence.

    Args:
        input_ids: Local input_ids in THD format [1, T_local].
        cu_seqlens: **Global** cumulative sequence lengths.

    Returns:
        Full input_ids in THD format [1, T_global].
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return input_ids

    # all-gather local input_ids across CP ranks → list of [1, T_local] per rank
    gathered = torch.distributed.nn.all_gather(
        input_ids, group=parallel_state.get_context_parallel_group()
    )  # list of cp_size tensors, each [1, T_local]

    local_cu_seqlens = cu_seqlens // cp_size
    num_seqs = len(cu_seqlens) - 1
    whole_list = []
    for i in range(num_seqs):
        seqlen = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
        chunk_size = seqlen // 2 // cp_size
        # First half: rank 0 chunk, rank 1 chunk, ..., rank cp_size-1 chunk
        whole_list.extend(
            gathered[cp_rank][0, local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size] for cp_rank in range(cp_size)
        )
        # Second half: rank cp_size-1 chunk, ..., rank 0 chunk (reversed)
        whole_list.extend(
            [
                gathered[cp_rank][0, local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                for cp_rank in range(cp_size)
            ][::-1]
        )
    return torch.cat(whole_list).unsqueeze(0)  # [1, T_global]


# ---------------------------------------------------------------------------
# Megatron VL Model
# ---------------------------------------------------------------------------
class Glm4vMoeVLModel(MegatronModule):
    """GLM-4.6V vision-language model for Megatron training.

    Wraps an HF vision encoder (only on first PP stage) together with a
    standard Megatron Core GPTModel configured for M-RoPE.
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
            from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeVisionModel

            self.vision_model = Glm4vMoeVisionModel._from_config(hf_vision_config)
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
        """Run HF vision encoder and return flat image embeddings."""
        pixel_values = pixel_values.to(dtype=self.vision_model.dtype)
        vision_out = self.vision_model(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        return vision_out.pooler_output  # [total_image_tokens, hidden]

    # -- M-RoPE position IDs -----------------------------------------------

    @staticmethod
    def _get_vision_position_ids(
        start_position: int,
        grid_thw,
        temp_merge_size: int,
        spatial_merge_size: int,
        device,
    ) -> torch.Tensor:
        """Compute 3D positions for one image/video (ported from HF)."""
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
        ``image_token_id`` in each sequence — no ``mm_token_type_ids`` needed.
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

                image_mask = (input_ids == self.image_token_id).contiguous()
                # Scatter: [seq, bs, hidden] → [bs, seq, hidden]
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                combined_embeddings[image_mask] = image_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            # Scatter to sequence-parallel region if needed
            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()

        # 3. Compute M-RoPE position IDs
        # position_ids must be available on ALL PP stages for rotary embeddings.
        # On stage 0, compute from input_ids. Then broadcast to other stages.
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()

        if position_ids is None:
            # Determine cu_seqlens for THD unpacking
            cu_seqlens = None
            if packed_seq_params is not None:
                cu_seqlens = (
                    packed_seq_params.cu_seqlens_q_padded
                    if packed_seq_params.cu_seqlens_q_padded is not None
                    else packed_seq_params.cu_seqlens_q
                )

            cp_size = parallel_state.get_context_parallel_world_size()

            if self.pre_process:
                # First PP stage: compute position_ids from input_ids.
                # With CP > 1, input_ids is a local chunk; reconstruct full
                # sequence so that _compute_mrope_position_ids sees all tokens
                # (image token positions affect the M-RoPE IDs).
                if cu_seqlens is not None:
                    if cp_size > 1:
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
class Glm4vMoeVLModelProvider(Qwen3MoEModelProvider):
    """Provider that creates Glm4vMoeVLModel.

    Inherits from Qwen3MoEModelProvider to reuse MoE + TransformerConfig infra.
    Defined at module level (not inside a function) so that the class is
    picklable -- megatron-bridge broadcasts config objects across PP ranks
    via ``torch.distributed.broadcast_object_list`` which requires pickling.
    """

    # GLM-4.6V specific config
    image_token_id: int = 151363
    video_token_id: int = 151364
    spatial_merge_size: int = 2

    # Vision config (stored as HF config object)
    hf_vision_config: object = None
    hf_text_config: object = None

    # M-RoPE
    position_embedding_type: str = "mrope"
    mrope_section: list[int] = field(default_factory=lambda: [8, 12, 12])
    scatter_embedding_sequence_parallel: bool = False

    # Language model sequence length
    language_max_sequence_length: int = 131072

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Create a Glm4vMoeVLModel instance."""

        # Resolve PP stage flags
        if pre_process is None:
            pre_process = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
        if post_process is None:
            post_process = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)

        # Build per-layer specs respecting moe_layer_freq (layer 0 = dense, rest = MoE)
        transformer_layer_spec = get_gpt_decoder_block_spec(
            config=self,
            use_transformer_engine=True,
            vp_stage=vp_stage,
        )

        model = Glm4vMoeVLModel(
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
    from transformers import Glm4vMoeForConditionalGeneration as _Glm4vMoeHF
except ImportError:
    _Glm4vMoeHF = "Glm4vMoeForConditionalGeneration"


@MegatronModelBridge.register_bridge(source=_Glm4vMoeHF, target=Glm4vMoeVLModel)
class Glm4vMoeBridge(MegatronModelBridge):
    """Bridge between HuggingFace GLM-4.6V and the Megatron VL model."""

    def provider_bridge(self, hf_pretrained):
        """Create a Glm4vMoeVLModelProvider from HF config."""
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = deepcopy(hf_config.vision_config)

        model_dtype = self.dtype_from_hf(text_config, default=torch.bfloat16)
        vision_config.torch_dtype = model_dtype

        ProviderClass = Glm4vMoeVLModelProvider

        rope_params = getattr(text_config, "rope_parameters", {}) or {}
        mrope_section = rope_params.get("mrope_section", [8, 12, 12])
        rotary_base = rope_params.get("rope_theta", 500000)
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.5)

        # Determine MoE layer frequency
        first_k_dense = getattr(text_config, "first_k_dense_replace", 1)
        num_layers = text_config.num_hidden_layers
        # Build moe_layer_freq list: first_k_dense dense layers + rest MoE
        moe_layer_freq_list = [0] * first_k_dense + [1] * (num_layers - first_k_dense)

        # Shared expert intermediate size
        n_shared = getattr(text_config, "n_shared_experts", 1)
        moe_ffn = getattr(text_config, "moe_intermediate_size", 1408)
        shared_expert_intermediate = moe_ffn * n_shared

        provider = ProviderClass(
            # Language model configuration
            num_layers=num_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=getattr(text_config, "head_dim", 128),
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=rotary_base,
            rotary_percent=partial_rotary_factor,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            # MoE configuration
            num_moe_experts=getattr(text_config, "n_routed_experts", 128),
            moe_router_topk=getattr(text_config, "num_experts_per_tok", 8),
            moe_ffn_hidden_size=moe_ffn,
            moe_shared_expert_intermediate_size=shared_expert_intermediate,
            moe_layer_freq=moe_layer_freq_list,
            moe_grouped_gemm=True,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=0,
            moe_router_score_function="sigmoid",
            moe_router_pre_softmax=True,
            moe_router_enable_expert_bias=True,
            moe_router_dtype="fp32",
            # Attention
            add_qkv_bias=getattr(text_config, "attention_bias", True),
            qk_layernorm=getattr(text_config, "qk_layernorm", False) or getattr(text_config, "use_qk_norm", False),
            # M-RoPE
            mrope_section=mrope_section,
            position_embedding_type="mrope",
            scatter_embedding_sequence_parallel=False,
            # Vision
            hf_vision_config=vision_config,
            hf_text_config=text_config,
            image_token_id=getattr(hf_config, "image_token_id", 151363),
            video_token_id=getattr(hf_config, "video_token_id", 151364),
            spatial_merge_size=getattr(hf_config.vision_config, "spatial_merge_size", 2),
            language_max_sequence_length=text_config.max_position_embeddings,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Weight mappings from HF GLM-4.6V to Megatron format.

        Follows GLM-4.5 bridge pattern with language_model prefix for VL model.
        Layer 0 is dense, layers 1-45 are MoE. The mapping framework handles
        missing keys gracefully (warnings for non-existent params).
        """
        param_mappings = {
            # Embeddings and output
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            # Attention: input layernorm (fused with TE)
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            # Attention: separate input layernorm (quantization layer spec)
            "language_model.decoder.layers.*.input_layernorm.weight": "model.language_model.layers.*.input_layernorm.weight",
            # Attention output
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            # Post-attention layernorm:
            #   MoE layers → pre_mlp_layernorm, Dense layer → linear_fc1.layer_norm_weight (fused)
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            # Dense MLP output (layer 0)
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
            # MoE router
            "language_model.decoder.layers.*.mlp.router.weight": "model.language_model.layers.*.mlp.gate.weight",
            "language_model.decoder.layers.*.mlp.router.expert_bias": "model.language_model.layers.*.mlp.gate.e_score_correction_bias",
            # MoE expert output (TEGroupedMLP format: weight* suffix)
            "language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.language_model.layers.*.mlp.experts.*.down_proj.weight",
            # MoE shared expert output
            "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.language_model.layers.*.mlp.shared_experts.down_proj.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                # Vision model weights — replicated directly
                ReplicatedMapping(
                    megatron_param="vision_model.**",
                    hf_param="model.visual.**",
                ),
                # QKV weight and bias
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.language_model.layers.*.self_attn.q_proj.bias",
                    k="model.language_model.layers.*.self_attn.k_proj.bias",
                    v="model.language_model.layers.*.self_attn.v_proj.bias",
                ),
                # Dense MLP gate+up (layer 0)
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
                # MoE expert gate+up (TEGroupedMLP format)
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.language_model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                # MoE expert gate+up (SequentialMLP format, for quantization)
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.language_model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # MoE shared expert gate+up
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
