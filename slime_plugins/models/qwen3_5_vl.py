from __future__ import annotations

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.models.common.embeddings.rotary_pos_embedding import MultimodalRotaryEmbedding
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from transformers import AutoConfig

from .qwen3_5 import get_qwen3_5_spec
from .qwen3_5_vl_utils import build_packed_mrope_position_ids, gather_packed_input_ids, get_packed_cp_local_indices


class Qwen3_5MultimodalRotaryEmbedding(MultimodalRotaryEmbedding):
    """Qwen3.5's interleaved temporal/height/width MRoPE layout."""

    def forward(
        self,
        position_ids: torch.Tensor,
        mrope_section: list[int],
        packed_seq: bool = False,
        cp_group=None,
    ) -> torch.Tensor:
        seq = position_ids.to(device=self.inv_freq.device, dtype=torch.float32)
        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        inv_freq = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        freqs = (inv_freq @ seq[:, :, None, :]).transpose(2, 3)

        # Qwen3.5 interleaves T/H/W frequency bands instead of concatenating them.
        mixed_freqs = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            mixed_freqs[..., offset : mrope_section[axis] * 3 : 3] = freqs[
                axis, ..., offset : mrope_section[axis] * 3 : 3
            ]

        emb = torch.cat((mixed_freqs, mixed_freqs), dim=-1)[..., None, :].transpose(0, 1).contiguous()
        cp_group = cp_group or self.cp_group
        if cp_group is not None and cp_group.size() > 1 and not packed_seq:
            from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank

            emb = get_pos_emb_on_this_cp_rank(emb, 0, cp_group)
        return emb


def _load_vision_model(hf_config, dtype: torch.dtype, use_cpu_initialization: bool):
    if hf_config.model_type == "qwen3_5_moe":
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel

        vision_model_cls = Qwen3_5MoeVisionModel
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        vision_model_cls = Qwen3_5VisionModel

    device = torch.device("cpu") if use_cpu_initialization else torch.device("cuda", torch.cuda.current_device())
    with device:
        vision_model = vision_model_cls._from_config(hf_config.vision_config)
    vision_model.to(dtype=dtype)

    # HF modules are replicated across TP ranks. Mark them explicitly so slime's
    # direct weight exporter does not try to tensor-parallel all-gather them.
    for parameter in vision_model.parameters():
        parameter.tensor_model_parallel = False
        parameter.partition_dim = -1
        parameter.partition_stride = 1
    return vision_model


class Qwen3_5VLModel(MegatronModule):
    """Megatron Qwen3.5 language model with a replicated Transformers ViT."""

    def __init__(
        self,
        config,
        language_layer_spec,
        hf_config,
        args,
        *,
        pre_process: bool,
        post_process: bool,
        vp_stage: int | None,
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.image_token_id = hf_config.image_token_id
        self.video_token_id = hf_config.video_token_id
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.spatial_merge_size = hf_config.vision_config.spatial_merge_size

        config.mrope_section = list(hf_config.text_config.rope_parameters["mrope_section"])
        config.apply_rope_fusion = False

        mtp_block_spec = None
        if args.mtp_num_layers:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config,
                language_layer_spec,
                use_transformer_engine=args.transformer_impl == "transformer_engine",
                **({"vp_stage": vp_stage} if vp_stage is not None else {}),
            )

        self.language_model = GPTModel(
            config=config,
            transformer_layer_spec=language_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type="mrope",
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            scatter_embedding_sequence_parallel=False,
            mtp_block_spec=mtp_block_spec,
            **({"vp_stage": vp_stage} if vp_stage is not None else {}),
        )
        self.language_model.rotary_pos_emb = Qwen3_5MultimodalRotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=args.rotary_percent,
            rotary_interleaved=config.rotary_interleaved,
            rotary_base=args.rotary_base,
        )

        self.model = torch.nn.Module()
        self.model.visual = (
            _load_vision_model(hf_config, config.params_dtype, config.use_cpu_initialization) if pre_process else None
        )
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    @property
    def decoder(self):
        return self.language_model.decoder

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor)

    def _inject_vision_embeddings(
        self,
        input_ids: torch.Tensor,
        full_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cp_group,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        embeddings = self.language_model.embedding(input_ids=input_ids, position_ids=None).clone()
        embeddings_bsh = embeddings.transpose(0, 1).contiguous()
        local_indices = get_packed_cp_local_indices(
            cu_seqlens,
            cp_group.size() if cp_group is not None else 1,
            cp_group.rank() if cp_group is not None else 0,
            input_ids.device,
        )

        for values, grids, token_id in (
            (pixel_values, image_grid_thw, self.image_token_id),
            (pixel_values_videos, video_grid_thw, self.video_token_id),
        ):
            if values is None:
                continue
            if grids is None:
                raise ValueError("Qwen3.5-VL pixel values require matching grid_thw")
            vision_output = self.model.visual(values.to(dtype=self.model.visual.dtype), grid_thw=grids)
            vision_embeddings = vision_output.pooler_output.to(device=embeddings.device, dtype=embeddings.dtype)
            full_vision_positions = (full_input_ids[0] == token_id).nonzero(as_tuple=False).flatten()
            if full_vision_positions.numel() != vision_embeddings.shape[0]:
                raise ValueError(
                    f"Qwen3.5-VL token/features mismatch: {full_vision_positions.numel()} tokens, "
                    f"{vision_embeddings.shape[0]} features"
                )

            feature_indices = torch.full(
                (full_input_ids.shape[1],),
                -1,
                dtype=torch.long,
                device=input_ids.device,
            )
            feature_indices[full_vision_positions] = torch.arange(vision_embeddings.shape[0], device=input_ids.device)
            local_feature_indices = feature_indices[local_indices]
            local_vision_mask = local_feature_indices >= 0
            if not torch.equal(local_vision_mask, input_ids[0] == token_id):
                raise ValueError("Qwen3.5-VL CP token layout does not match its full packed sequence")
            embeddings_bsh[0, local_vision_mask] = vision_embeddings[local_feature_indices[local_vision_mask]]

        embeddings = embeddings_bsh.transpose(0, 1).contiguous()
        if self.config.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings).contiguous()
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        loss_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if packed_seq_params is None:
            raise ValueError("Qwen3.5-VL native training currently requires packed sequences")

        cp_group = mpu.get_context_parallel_group()
        full_input_ids = gather_packed_input_ids(input_ids, packed_seq_params.cu_seqlens_q, cp_group)

        decoder_input = None
        if self.pre_process:
            decoder_input = self._inject_vision_embeddings(
                input_ids,
                full_input_ids,
                packed_seq_params.cu_seqlens_q,
                cp_group,
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
            )

        if position_ids is None:
            position_ids = build_packed_mrope_position_ids(
                full_input_ids,
                packed_seq_params.cu_seqlens_q,
                image_grid_thw,
                video_grid_thw,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                spatial_merge_size=self.spatial_merge_size,
            )

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
            **kwargs,
        )


def get_qwen3_5_vl_model_provider(args, config, vp_stage):
    """Return the native Qwen3.5-VL model provider."""

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    if not hasattr(hf_config, "vision_config"):
        raise ValueError(f"{args.hf_checkpoint} is not a Qwen3.5-VL checkpoint")
    language_layer_spec = get_qwen3_5_spec(args, config, vp_stage)

    def model_provider(
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
    ) -> Qwen3_5VLModel:
        return Qwen3_5VLModel(
            config,
            language_layer_spec,
            hf_config,
            args,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

    return model_provider
