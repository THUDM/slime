import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge


@register_model(["qwen3_5", "qwen3_5_moe"])
class Qwen3_5Bridge(Qwen2MoEBridge):
    """
    Bridge for Qwen3.5 models (both dense and MoE variants).
    Qwen3.5 is a VLM model with weights under model.language_model.layers prefix,
    separate in_proj_qkv + in_proj_z for linear attention, and nested text_config.
    """

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": ["model.language_model.layers.{layer_number}.self_attn.o_proj.weight"],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.q_layernorm.weight": ["model.language_model.layers.{layer_number}.self_attn.q_norm.weight"],
        "self_attention.k_layernorm.weight": ["model.language_model.layers.{layer_number}.self_attn.k_norm.weight"],
        "self_attention.linear_qkv.weight": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.language_model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.language_model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    } | {
        f"self_attention.{weight_name}": ["model.language_model.layers.{layer_number}." + weight_name]
        for weight_name in [
            "input_layernorm.weight",
            # linear attn
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            # gated attn (full attention layers)
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ]
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": ["model.language_model.layers.{layer_number}.mlp.down_proj.weight"],
        # MoE mappings
        "shared_experts.linear_fc1.weight": [
            "model.language_model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
            "model.language_model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
        ],
        "pre_mlp_layernorm": ["model.language_model.layers.{layer_number}.post_attention_layernorm.weight"],
        "shared_experts.linear_fc2.weight": [
            "model.language_model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"
        ],
        "mlp.router.weight": ["model.language_model.layers.{layer_number}.mlp.gate.weight"],
        "shared_experts.gate_weight": ["model.language_model.layers.{layer_number}.mlp.shared_expert_gate.weight"],
        # Fused expert format: single 3D tensor for all experts
        "mlp.experts.linear_fc1": [
            "model.language_model.layers.{layer_number}.mlp.experts.gate_up_proj",
        ],
        "mlp.experts.linear_fc2": ["model.language_model.layers.{layer_number}.mlp.experts.down_proj"],
    }

    # MTP layer uses individual expert format (not fused)
    _MTP_MLP_MAPPING = {
        "mlp.experts.linear_fc1": [
            "mtp.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "mtp.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "mlp.experts.linear_fc2": ["mtp.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"],
    }

    # Override to make ffn_hidden_size optional (Qwen3.5 MoE has no intermediate_size)
    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": ("intermediate_size", None),
        "attention_dropout": "attention_dropout",
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": ("head_dim", None),
    }

    def _get_text_config(self):
        """Get the text config, handling VLM nesting."""
        if hasattr(self.hf_config, "text_config"):
            return self.hf_config.text_config
        return self.hf_config

    def _get_gptmodel_args(self) -> dict:
        """Override to add MTP block spec if needed."""
        ret = super()._get_gptmodel_args()
        text_config = self._get_text_config()
        if getattr(text_config, "mtp_num_hidden_layers", None) is not None:
            transformer_layer_spec = self.config
            mtp_block_spec = get_gpt_mtp_block_spec(self.config, transformer_layer_spec, use_transformer_engine=True)
            ret["mtp_block_spec"] = mtp_block_spec
        return ret

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        """Override to handle fused expert weights.
        For regular layers: experts use fused 3D format (all experts in one tensor).
        For MTP layers: experts use individual format (per-expert tensors).
        """
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    expert_id = name.split("weight")[-1]
                    convert_names.extend(
                        [x.format(layer_number=layer_number, expert_id=expert_id) for x in mapping_names]
                    )
                else:
                    convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """Override to handle MTP layer mappings."""
        if "mtp" in mcore_weights_name:
            return self._convert_mtp_param(mcore_weights_name)
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _convert_mtp_param(self, name: str) -> list[str]:
        """Convert MTP layer parameters from MCore to HF format."""
        if "mtp.layers." not in name:
            raise NotImplementedError(f"Invalid MTP parameter name: {name}")

        parts = name.split(".")
        mtp_layer_idx = parts[2]  # mtp.layers.{idx}

        direct_name_mapping = {
            f"mtp.layers.{mtp_layer_idx}.eh_proj.weight": "mtp.fc.weight",
            f"mtp.layers.{mtp_layer_idx}.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            f"mtp.layers.{mtp_layer_idx}.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            f"mtp.layers.{mtp_layer_idx}.final_layernorm.weight": "mtp.norm.weight",
        }

        if name in direct_name_mapping:
            return [direct_name_mapping[name]]

        if "transformer_layer" in name:
            proxy_name = name.replace(
                f"mtp.layers.{mtp_layer_idx}.transformer_layer",
                f"decoder.layers.{mtp_layer_idx}",
            )

            if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
                convert_names = super()._weight_name_mapping_attention(proxy_name)
            elif "mlp" in proxy_name or "pre_mlp_layernorm" in proxy_name:
                convert_names = super()._weight_name_mapping_mlp(proxy_name)
            else:
                raise NotImplementedError(f"Unsupported transformer component in MTP: {name}")

            # MTP weights use model.language_model prefix in regular layers,
            # but mtp.layers.{idx} directly for MTP layers
            convert_names = [
                cn.replace(f"model.language_model.layers.{mtp_layer_idx}", f"mtp.layers.{mtp_layer_idx}")
                for cn in convert_names
            ]
            return convert_names

        raise NotImplementedError(f"Unsupported MTP parameter name: {name}")

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> tuple[list[str], list[torch.Tensor]]:
        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            # merge qkv
            assert len(hf_weights) == 3
            text_config = self._get_text_config()
            num_key_value_heads = text_config.num_key_value_heads
            hidden_dim = text_config.hidden_size
            num_attention_heads = text_config.num_attention_heads
            num_querys_per_group = num_attention_heads // text_config.num_key_value_heads
            head_dim = getattr(text_config, "head_dim", hidden_dim // num_attention_heads)
            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            # q k v might be tp split
            real_num_key_value_heads = q.shape[0] // (2 * group_dim)
            q = (
                q.view(
                    [
                        real_num_key_value_heads,
                        num_querys_per_group,
                        2,
                        head_dim,
                        -1,
                    ]
                )
                .transpose(1, 2)
                .flatten(1, 3)
            )
            k = k.view([real_num_key_value_heads, head_dim, -1])
            v = v.view([real_num_key_value_heads, head_dim, -1])
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]

            qgkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qgkv

        # Handle fused expert weights: extract single expert from 3D fused tensor
        if "mlp.experts.linear_fc" in mcore_weights_name and len(hf_weights) == 1:
            w = hf_weights[0]
            if w.dim() == 3:
                # Extract expert_id from name like "...linear_fc1.weight42"
                expert_id = int(mcore_weights_name.split("weight")[-1])
                expert_w = w[expert_id]  # (out_features, in_features)
                return expert_w.contiguous()

        weight = super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
        if mcore_weights_name.endswith("eh_proj.weight"):
            first_half, second_half = weight.chunk(2, dim=1)
            weight = torch.cat([second_half, first_half], dim=1)
        return weight

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        if mcore_weights_name.endswith("eh_proj.weight"):
            first_half, second_half = mcore_weights.chunk(2, dim=1)
            mcore_weights = torch.cat([second_half, first_half], dim=1)
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _build_config(self):
        text_config = self._get_text_config()

        mtp_args = {}
        if hasattr(text_config, "mtp_num_hidden_layers"):
            mtp_args["mtp_num_layers"] = text_config.mtp_num_hidden_layers

        base_kwargs = dict(
            text_config_key="text_config" if hasattr(self.hf_config, "text_config") else None,
            use_cpu_initialization=False,
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # Qwen3.5 specific
            moe_router_pre_softmax=False,
            qk_layernorm=True,
            attention_output_gate=True,
            **mtp_args,
        )

        # Handle MoE-specific config
        if hasattr(text_config, "num_experts"):
            base_kwargs.update(
                moe_ffn_hidden_size=text_config.moe_intermediate_size,
                moe_shared_expert_intermediate_size=getattr(text_config, "shared_expert_intermediate_size", None),
                moe_router_bias_update_rate=0.001,
                moe_router_topk=text_config.num_experts_per_tok,
                num_moe_experts=text_config.num_experts,
                moe_aux_loss_coeff=text_config.router_aux_loss_coef,
                moe_router_load_balancing_type="none",
                moe_grouped_gemm=True,
                moe_router_score_function="softmax",
                moe_shared_expert_gate=True,
            )
            # For MoE models without intermediate_size, use shared_expert_intermediate_size
            if not hasattr(text_config, "intermediate_size"):
                base_kwargs["ffn_hidden_size"] = text_config.shared_expert_intermediate_size

        return self._build_base_config(**base_kwargs)
