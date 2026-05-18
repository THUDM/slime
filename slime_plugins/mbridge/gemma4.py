import functools
import re

import torch
import torch.nn.functional as F
from mbridge.core import register_model
from mbridge.models import Gemma3Bridge

from slime_plugins.models.gemma4 import get_rope_local_base_freq as _rope_local_base_freq

# Gemma uses GeGLU (GELU with tanh approximation + gated linear unit), not SwiGLU.
# See: https://developers.googleblog.com/en/gemma-explained-new-in-gemma-2/
_gelu_tanh = functools.partial(F.gelu, approximate="tanh")


@register_model("gemma4")
class Gemma4Bridge(Gemma3Bridge):
    """
    Bridge for Gemma 4 dense 31B.

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

    # Dense MLP entries. For the 31B dense variant these map the single `.mlp`
    # submodule directly. For the 26B-A4B MoE variant `.mlp` is the MoE block
    # and the dense feed-forward lives at `.dense_mlp` — we map both so state-
    # dict round-trips work regardless of variant.
    _MLP_MAPPING = {
        # 31B dense variant: `.mlp` is the dense MLP.
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
        # 26B-A4B MoE variant: `.dense_mlp` is the parallel dense feed-forward.
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
        # MoE router weights (live under `.mlp.router.*` since self.mlp is the
        # Gemma4MoELayer in the MoE variant).
        "decoder.layers.{layer_number}.mlp.router.proj.weight": [
            "model.language_model.layers.{layer_number}.router.proj.weight",
        ],
        "decoder.layers.{layer_number}.mlp.router.scale": [
            "model.language_model.layers.{layer_number}.router.scale",
        ],
        "decoder.layers.{layer_number}.mlp.router.per_expert_scale": [
            "model.language_model.layers.{layer_number}.router.per_expert_scale",
        ],
        # pre_feedforward_layernorm_2 now owned by Gemma4MoELayer so the Megatron
        # path is `.mlp.pre_feedforward_layernorm_2.weight`. HF still expects it
        # at the decoder-layer level. See slime_plugins/models/gemma4.py — this
        # ownership change aligns slime's router semantic with HF (router sees
        # un-normed residual, experts see pre_ff_norm_2(residual)).
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
        # MoE variant extra layernorms that wrap the dense + MoE paths before
        # summing. `pre_feedforward_layernorm_2` moved to
        # `.mlp.pre_feedforward_layernorm_2.weight` (see `_MLP_MAPPING` above);
        # `post_feedforward_layernorm_1/_2` still live on the layer directly.
        "decoder.layers.{layer_number}.post_feedforward_layernorm_2.weight": [
            "model.language_model.layers.{layer_number}.post_feedforward_layernorm_2.weight",
        ],
        "decoder.layers.{layer_number}.post_feedforward_layernorm_1.weight": [
            "model.language_model.layers.{layer_number}.post_feedforward_layernorm_1.weight",
        ],
    }

    # Matches per-expert linear weights emitted by TEGroupedLinear:
    #   decoder.layers.<L>.mlp.experts.linear_fc1.weight<E>
    #   decoder.layers.<L>.mlp.experts.linear_fc2.weight<E>
    # where <L> is the layer number and <E> is the GLOBAL expert index
    # (after mbridge base's `_weight_name_mapping_mcore_local_to_global`
    # has remapped local→global across EP ranks — its built-in logic
    # handles the `.mlp.experts.linear_fc` pattern automatically).
    _RE_MOE_EXPERT = re.compile(
        r"^decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)$"
    )

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "output_layer.weight": "model.language_model.embed_tokens.weight",  # tied embeddings
    }

    _BUFFER_NAMES = [
        "model.language_model.layers.{layer_number}.layer_scalar",
    ]

    _GLOBAL_ATTN_LAYERS = None  # derived from HF config in __init__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hf_text = self.hf_config.text_config if hasattr(self.hf_config, "text_config") else self.hf_config
        layer_types = getattr(hf_text, "layer_types", [])
        self._GLOBAL_ATTN_LAYERS = {i for i, t in enumerate(layer_types) if t == "full_attention"}

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = int(split_name[2])
        split_name[2] = "{layer_number}"
        key = ".".join(split_name)

        # For global layers with K=V, linear_qkv maps to only [q_proj, k_proj]
        if key == "decoder.layers.{layer_number}.self_attention.linear_qkv.weight":
            if layer_number in self._GLOBAL_ATTN_LAYERS:
                return [
                    f"model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
                    f"model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
                ]

        return [x.format(layer_number=layer_number) for x in self._ATTENTION_MAPPING[key]]

    def _weight_name_mapping_mcore_local_to_global(
        self, model, consider_ep: bool = True
    ):
        """Restore the GPT-style local→global mapping for text-only Gemma4.

        Gemma3Bridge (our base class) assumes a VLM structure where
        ``model.language_model.decoder.layers`` exists, and only applies the
        PP layer-offset remap when that attribute is present. Our Gemma4
        model provider builds a plain ``GPTModel`` (text-only) with
        ``model.decoder.layers``, so the Gemma3 check fails silently and all
        PP ranks end up mapping their local layer index i → global index i —
        which means every PP rank loads HF layers ``0..N/PP-1`` into its
        local slots. The result is that, post-conversion, the torch_dist
        checkpoint has layer weights cyclically duplicated with period
        (num_layers / pp_size).

        We override to delegate to ``Bridge._weight_name_mapping_mcore_local_to_global``
        from the top-level mbridge base class, which walks ``model.decoder.layers``
        directly — matching our GPT-style layout.
        """
        from mbridge.core.bridge import Bridge
        return Bridge._weight_name_mapping_mcore_local_to_global(
            self, model, consider_ep=consider_ep
        )

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        # Per-expert MoE weight: Megatron names the per-expert tensors
        # `mlp.experts.linear_fc{1,2}.weight{E}`. HF stores the 3D stacked
        # tensors `experts.gate_up_proj` / `experts.down_proj`; we slice the
        # expert row out in `_weight_to_mcore_format`.
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
        # Per-expert MoE weight: slice the global 3D HF tensor down to one
        # expert row. The expert index is encoded in the mcore name by
        # `_weight_name_mapping_mcore_local_to_global`, which rewrites local
        # weight{j} → weight{global_expert_idx}.
        m = self._RE_MOE_EXPERT.match(mcore_weights_name)
        if m:
            expert_idx = int(m.group(3))
            assert len(hf_weights) == 1, (
                f"expected exactly one HF tensor for expert weight, got {len(hf_weights)}"
            )
            # HF shape: [num_experts, out_dim, in_dim]. Slice to [out_dim, in_dim].
            return hf_weights[0][expert_idx].contiguous()

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            m = re.search(r"layers\.(\d+)\.", mcore_weights_name)
            layer_num = int(m.group(1)) if m else -1
            is_global = layer_num in self._GLOBAL_ATTN_LAYERS

            hf_text = self.hf_config.text_config if hasattr(self.hf_config, "text_config") else self.hf_config
            num_attention_heads = hf_text.num_attention_heads
            if is_global:
                head_dim = hf_text.global_head_dim
                num_kv_heads = hf_text.num_global_key_value_heads
            else:
                head_dim = hf_text.head_dim
                num_kv_heads = hf_text.num_key_value_heads

            # For K=V global layers the HF checkpoint ships `[q, k]` (no
            # v_proj); reconstruct V by duplicating K so the Mcore linear_qkv
            # weight has the standard `[q, k, v]` layout with v_proj == k_proj.
            if len(hf_weights) == 2:
                assert is_global and getattr(hf_text, "attention_k_eq_v", True), (
                    f"layer {layer_num}: got 2 HF weights ([q, k]) but this is "
                    f"not a K=V global layer (is_global={is_global}, "
                    f"attention_k_eq_v="
                    f"{getattr(hf_text, 'attention_k_eq_v', None)})"
                )
                q, k = hf_weights
                hf_weights = [q, k, k.clone()]

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
                f"MLP linear_fc1.weight expects [gate_proj, up_proj] from HF "
                f"(2 tensors); got {len(hf_weights)}"
            )
            gate, up = hf_weights
            return torch.cat([gate, up], dim=0)

        # Generic 1:1 passthrough for everything else (layernorm weights,
        # single-tensor projections). Placed after the MLP/QKV-specific
        # branches so a malformed 1-tensor input for linear_fc1 / linear_qkv
        # hits its assertion instead of being silently returned rotated.
        if len(hf_weights) == 1:
            return hf_weights[0]

        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _build_config(self):
        text_config_key = "text_config" if hasattr(self.hf_config, "text_config") else None
        hf_text = self.hf_config.text_config if text_config_key else self.hf_config

        return self._build_base_config(
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
