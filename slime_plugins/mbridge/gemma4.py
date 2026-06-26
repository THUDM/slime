import functools
import re

import torch
import torch.nn.functional as F
from mbridge.core import register_model
from mbridge.models import Gemma3Bridge

from slime_plugins.models.gemma4 import get_rope_local_base_freq as _rope_local_base_freq

_gelu_tanh = functools.partial(F.gelu, approximate="tanh")


@register_model(["gemma4", "gemma4_text", "gemma4_unified_text"])
class Gemma4Bridge(Gemma3Bridge):
    """
    Bridge for Gemma4 text dense and MoE variants.

    Megatron-side keys have NO language_model. prefix (text-only model).
    HF-side values have model.language_model. prefix (Gemma4ForConditionalGeneration).
    """

    _ATTENTION_MAPPING = {
        "decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "decoder.layers.{layer_number}.self_attention.linear_proj.weight": [
            "model.language_model.layers.{layer_number}.self_attn.o_proj.weight",
        ],
        "decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight",
        ],
        "decoder.layers.{layer_number}.self_attention.q_layernorm.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_norm.weight",
        ],
        "decoder.layers.{layer_number}.self_attention.k_layernorm.weight": [
            "model.language_model.layers.{layer_number}.self_attn.k_norm.weight",
        ],
    }

    _MLP_MAPPING = {
        "decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "decoder.layers.{layer_number}.mlp.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.down_proj.weight",
        ],
        "decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.pre_feedforward_layernorm.weight",
        ],
        "decoder.layers.{layer_number}.pre_mlp_layernorm.weight": [
            "model.language_model.layers.{layer_number}.pre_feedforward_layernorm.weight",
        ],
        "decoder.layers.{layer_number}.dense_mlp.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "decoder.layers.{layer_number}.dense_mlp.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.down_proj.weight",
        ],
        "decoder.layers.{layer_number}.dense_mlp.linear_fc1.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.pre_feedforward_layernorm.weight",
        ],
        "decoder.layers.{layer_number}.mlp.router.proj.weight": [
            "model.language_model.layers.{layer_number}.router.proj.weight",
        ],
        "decoder.layers.{layer_number}.mlp.router.scale": [
            "model.language_model.layers.{layer_number}.router.scale",
        ],
        "decoder.layers.{layer_number}.mlp.router.per_expert_scale": [
            "model.language_model.layers.{layer_number}.router.per_expert_scale",
        ],
        "decoder.layers.{layer_number}.mlp.pre_feedforward_layernorm_2.weight": [
            "model.language_model.layers.{layer_number}.pre_feedforward_layernorm_2.weight",
        ],
    }

    _OTHER_MAPPING = {
        "decoder.layers.{layer_number}.post_attention_layernorm.weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight",
        ],
        "decoder.layers.{layer_number}.post_feedforward_layernorm.weight": [
            "model.language_model.layers.{layer_number}.post_feedforward_layernorm.weight",
        ],
        "decoder.layers.{layer_number}.layer_scalar": [
            "model.language_model.layers.{layer_number}.layer_scalar",
        ],
        "decoder.layers.{layer_number}.post_feedforward_layernorm_2.weight": [
            "model.language_model.layers.{layer_number}.post_feedforward_layernorm_2.weight",
        ],
        "decoder.layers.{layer_number}.post_feedforward_layernorm_1.weight": [
            "model.language_model.layers.{layer_number}.post_feedforward_layernorm_1.weight",
        ],
    }

    _RE_MOE_EXPERT = re.compile(r"^decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)$")

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "output_layer.weight": "model.language_model.embed_tokens.weight",
    }

    _BUFFER_NAMES = [
        "model.language_model.layers.{layer_number}.layer_scalar",
    ]

    _GLOBAL_ATTN_LAYERS = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hf_text = self.hf_config.text_config if hasattr(self.hf_config, "text_config") else self.hf_config
        layer_types = getattr(hf_text, "layer_types", [])
        self._GLOBAL_ATTN_LAYERS = {i for i, t in enumerate(layer_types) if t == "full_attention"}

    def _attention_shape_for_hf_weights(self, hf_weights: list[torch.Tensor]) -> tuple[int, int]:
        hf_text = self.hf_config.text_config if hasattr(self.hf_config, "text_config") else self.hf_config
        if len(hf_weights) == 2:
            return (
                int(getattr(hf_text, "num_global_key_value_heads", hf_text.num_key_value_heads)),
                int(getattr(hf_text, "global_head_dim", hf_text.head_dim)),
            )
        if len(hf_weights) == 3:
            return (
                int(hf_text.num_key_value_heads),
                int(getattr(hf_text, "head_dim", hf_text.hidden_size // hf_text.num_attention_heads)),
            )
        raise ValueError(f"Gemma4 linear_qkv expects 2 or 3 HF tensors, got {len(hf_weights)}.")

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = int(split_name[2])
        split_name[2] = "{layer_number}"
        key = ".".join(split_name)

        if key == "decoder.layers.{layer_number}.self_attention.linear_qkv.weight":
            if layer_number in self._GLOBAL_ATTN_LAYERS:
                return [
                    f"model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
                    f"model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
                ]

        return [x.format(layer_number=layer_number) for x in self._ATTENTION_MAPPING[key]]

    def _weight_name_mapping_mcore_local_to_global(self, model, consider_ep: bool = True):
        """Restore the GPT-style local->global mapping for text-only Gemma4.

        Gemma3Bridge (our base class) assumes a VLM structure where
        ``model.language_model.decoder.layers`` exists, and only applies the
        PP layer-offset remap when that attribute is present. Our Gemma4
        model provider builds a plain ``GPTModel`` (text-only) with
        ``model.decoder.layers``, so the Gemma3 check fails silently and all
        PP ranks end up mapping their local layer index i -> global index i -
        which means every PP rank loads HF layers ``0..N/PP-1`` into its
        local slots. The result is that, post-conversion, the torch_dist
        checkpoint has layer weights cyclically duplicated with period
        (num_layers / pp_size).

        We override to delegate to ``Bridge._weight_name_mapping_mcore_local_to_global``
        from the top-level mbridge base class, which walks ``model.decoder.layers``
        directly - matching our GPT-style layout.
        """
        from mbridge.core.bridge import Bridge

        return Bridge._weight_name_mapping_mcore_local_to_global(self, model, consider_ep=consider_ep)

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        m = self._RE_MOE_EXPERT.match(name)
        if m:
            layer_number, fc = m.group(1), m.group(2)
            hf_tensor = "gate_up_proj" if fc == "1" else "down_proj"
            return [
                f"model.language_model.layers.{layer_number}.experts.{hf_tensor}",
            ]

        split_name = name.split(".")
        layer_number = split_name[2]
        split_name[2] = "{layer_number}"
        key = ".".join(split_name)
        return [x.format(layer_number=layer_number) for x in self._MLP_MAPPING[key]]

    def _weight_name_mapping_other(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[2]
        split_name[2] = "{layer_number}"
        key = ".".join(split_name)
        return [x.format(layer_number=layer_number) for x in self._OTHER_MAPPING[key]]

    def _weight_to_mcore_format(self, mcore_weights_name, hf_weights):
        m = self._RE_MOE_EXPERT.match(mcore_weights_name)
        if m:
            expert_idx = int(m.group(3))
            assert len(hf_weights) == 1, f"expected exactly one HF tensor for expert weight, got {len(hf_weights)}"
            return hf_weights[0][expert_idx].contiguous()

        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            m = re.search(r"layers\.(\d+)\.", mcore_weights_name)
            layer_num = int(m.group(1)) if m else -1

            hf_text = self.hf_config.text_config if hasattr(self.hf_config, "text_config") else self.hf_config
            num_attention_heads = hf_text.num_attention_heads
            num_kv_heads, head_dim = self._attention_shape_for_hf_weights(hf_weights)

            if len(hf_weights) == 2:
                q, k = hf_weights
                hf_weights = [q, k, k.clone()]
            elif len(hf_weights) != 3:
                raise ValueError(f"Gemma4 linear_qkv expects 2 or 3 HF tensors, got {len(hf_weights)}.")

            q, k, v = hf_weights
            group_dim = head_dim * num_attention_heads // num_kv_heads
            assert q.shape[0] == num_kv_heads * group_dim, (
                f"layer {layer_num}: q_proj rows ({q.shape[0]}) must equal "
                f"num_kv_heads ({num_kv_heads}) * group_dim ({group_dim}); "
                f"check head_dim/num_attention_heads/num_kv_heads consistency"
            )
            assert k.shape[0] == num_kv_heads * head_dim, (
                f"layer {layer_num}: k_proj rows ({k.shape[0]}) must equal "
                f"num_kv_heads ({num_kv_heads}) * head_dim ({head_dim})"
            )
            assert v.shape[0] == num_kv_heads * head_dim, (
                f"layer {layer_num}: v_proj rows ({v.shape[0]}) must equal "
                f"num_kv_heads ({num_kv_heads}) * head_dim ({head_dim})"
            )
            q = q.view(num_kv_heads, group_dim, -1)
            k = k.view(num_kv_heads, head_dim, -1)
            v = v.view(num_kv_heads, head_dim, -1)
            return torch.cat([q, k, v], dim=1).view(-1, hf_text.hidden_size).contiguous()

        if "linear_fc1.weight" in mcore_weights_name:
            assert len(hf_weights) == 2, (
                f"MLP linear_fc1.weight expects [gate_proj, up_proj] from HF " f"(2 tensors); got {len(hf_weights)}"
            )
            gate, up = hf_weights
            return torch.cat([gate, up], dim=0)

        if len(hf_weights) == 1:
            return hf_weights[0]

        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _build_config(self):
        text_config_key = "text_config" if hasattr(self.hf_config, "text_config") else None
        hf_text = self.hf_config.text_config if text_config_key else self.hf_config

        base_kwargs = dict(
            text_config_key=text_config_key,
            use_cpu_initialization=False,
            add_qkv_bias=False,
            qk_layernorm=True,
            layernorm_zero_centered_gamma=False,
            normalization="RMSNorm",
            persist_layer_norm=True,
            activation_func=_gelu_tanh,
            bias_activation_fusion=False,
            bias_dropout_fusion=True,
            rope_local_base_freq=_rope_local_base_freq(hf_text),
        )
        if getattr(hf_text, "enable_moe_block", False):
            base_kwargs.update(
                num_moe_experts=hf_text.num_experts,
                moe_router_topk=hf_text.top_k_experts,
                moe_ffn_hidden_size=hf_text.moe_intermediate_size,
                moe_token_dispatcher_type="alltoall",
                moe_grouped_gemm=True,
                moe_aux_loss_coeff=0.0,
                moe_router_load_balancing_type="none",
                moe_router_score_function="softmax",
                moe_router_topk_scaling_factor=1.0,
                moe_router_pre_softmax=False,
                moe_router_dtype="fp32",
            )

        return self._build_base_config(**base_kwargs)
