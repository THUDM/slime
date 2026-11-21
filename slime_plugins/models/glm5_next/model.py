"""Small VLM wrapper, following slime's native Qwen3.5 integration."""

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule

from slime.utils.model_config import load_hf_config

from .glm5_next import get_glm5_next_spec
from .vision import Glm5NextVisionModel


class Glm5NextModel(MegatronModule):
    def __init__(self, config, spec, hf_config, args, *, pre_process, post_process, vp_stage):
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.image_token_id = hf_config.image_token_id
        self.video_token_id = hf_config.video_token_id
        self.language_model = GPTModel(
            config=config,
            transformer_layer_spec=spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type="none",
            vp_stage=vp_stage,
        )
        self.model = torch.nn.Module()
        if pre_process:
            hf_config.vision_config._attn_implementation = "sdpa"
            self.model.visual = Glm5NextVisionModel(hf_config.vision_config).to(dtype=config.params_dtype)
            for parameter in self.model.visual.parameters():
                parameter.tensor_model_parallel = False
                parameter.partition_dim = -1
                parameter.partition_stride = 1
        self.share_embeddings_and_output_weights = False

    @property
    def decoder(self):
        return self.language_model.decoder

    def set_input_tensor(self, input_tensor):
        self.language_model.set_input_tensor(input_tensor)

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params=None,
        loss_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        if packed_seq_params is None:
            raise ValueError("GLM5-next training requires packed sequences")
        decoder_input = None
        if self.pre_process:
            decoder_input = self.language_model.embedding(input_ids=input_ids, position_ids=None)
            if pixel_values is not None or pixel_values_videos is not None:
                cp_group = self.language_model.pg_collection.cp
                full_input_ids = input_ids
                local_start = 0
                if cp_group is not None and cp_group.size() > 1:
                    full_input_ids = gather_from_sequence_parallel_region(
                        input_ids.t().contiguous(), group=cp_group
                    ).t()
                    local_start = cp_group.rank() * input_ids.numel()
            for values, grids, token in (
                (pixel_values, image_grid_thw, self.image_token_id),
                (pixel_values_videos, video_grid_thw, self.video_token_id),
            ):
                if values is None:
                    continue
                if grids is None:
                    raise ValueError("GLM5-next pixel values require grid_thw")
                features = self.model.visual(values.to(decoder_input.dtype), grid_thw=grids).pooler_output
                mask = input_ids.transpose(0, 1) == token
                full_mask = full_input_ids[0] == token
                if full_mask.sum().item() != features.shape[0]:
                    raise ValueError(
                        f"GLM5-next image token/features mismatch: {full_mask.sum().item()} vs {features.shape[0]}"
                    )
                # Features cover the complete packed stream. Select only the
                # image/video positions owned by this CP rank, in local order.
                feature_indices = full_mask.long().cumsum(0) - 1
                local_features = feature_indices[local_start : local_start + input_ids.numel()][mask[:, 0]]
                decoder_input = decoder_input.clone()
                decoder_input[mask] = features[local_features].to(decoder_input.dtype)
        return self.language_model(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
            **kwargs,
        )


def get_glm5_next_model_provider(args, config, vp_stage):
    hf_config = load_hf_config(args.hf_checkpoint, trust_remote_code=True)
    spec = get_glm5_next_spec(args, config, vp_stage)

    def model_provider(pre_process=True, post_process=True, vp_stage=None):
        return Glm5NextModel(
            config, spec, hf_config, args, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

    return model_provider
