import inspect

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge


def _get_gdn_dims(config):
    hidden_size = config.hidden_size
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    return hidden_size, qk_head_dim, v_head_dim, num_qk_heads, num_v_heads, qk_dim, v_dim


def _infer_gdn_pack_tp_size(config, qkv_rows: int) -> int:
    _, _, _, _, _, qk_dim, v_dim = _get_gdn_dims(config)
    full_rows = 2 * qk_dim + v_dim
    tp_size = getattr(config, "tensor_model_parallel_size", 1)
    if qkv_rows == full_rows:
        return tp_size
    if tp_size > 1 and qkv_rows == full_rows // tp_size:
        return 1
    raise ValueError(f"Unexpected GDN qkv rows: got {qkv_rows}, expected {full_rows} or {full_rows // tp_size}")


def _merge_gdn_linear_separate(config, qkv, z, b, a):
    hidden_size, qk_head_dim, v_head_dim, _, _, qk_dim, v_dim = _get_gdn_dims(config)
    pack_tp_size = _infer_gdn_pack_tp_size(config, qkv.shape[0])
    input_v_dim = z.shape[0]
    input_qk_dim = (qkv.shape[0] - input_v_dim) // 2
    input_num_qk_heads = input_qk_dim // qk_head_dim
    input_num_v_heads = input_v_dim // v_head_dim
    v_per_group = input_num_v_heads // input_num_qk_heads

    if tuple(qkv.shape) != (2 * input_qk_dim + input_v_dim, hidden_size):
        raise ValueError(f"qkv shape mismatch: got {tuple(qkv.shape)}")
    if tuple(z.shape) != (input_v_dim, hidden_size):
        raise ValueError(f"z shape mismatch: got {tuple(z.shape)}")
    if tuple(b.shape) != (input_num_v_heads, hidden_size):
        raise ValueError(f"b shape mismatch: got {tuple(b.shape)}")
    if tuple(a.shape) != (input_num_v_heads, hidden_size):
        raise ValueError(f"a shape mismatch: got {tuple(a.shape)}")

    q_flat, k_flat, v_flat = torch.split(qkv, [input_qk_dim, input_qk_dim, input_v_dim], dim=0)
    q = q_flat.reshape(input_num_qk_heads, qk_head_dim, hidden_size)
    k = k_flat.reshape(input_num_qk_heads, qk_head_dim, hidden_size)
    v = v_flat.reshape(input_num_qk_heads, v_per_group * v_head_dim, hidden_size)
    z = z.reshape(input_num_qk_heads, v_per_group * v_head_dim, hidden_size)
    b = b.reshape(input_num_qk_heads, v_per_group, hidden_size)
    a = a.reshape(input_num_qk_heads, v_per_group, hidden_size)

    q, k, v, z, b, a = [weight.reshape(pack_tp_size, -1, hidden_size) for weight in [q, k, v, z, b, a]]
    return torch.cat([q, k, v, z, b, a], dim=1).reshape(-1, hidden_size).contiguous()


def _split_gdn_linear_separate(config, in_proj):
    hidden_size, qk_head_dim, v_head_dim, num_qk_heads, num_v_heads, qk_dim, v_dim = _get_gdn_dims(config)
    tp_size = getattr(config, "tensor_model_parallel_size", 1)
    in_proj_dim = 2 * qk_dim + 2 * v_dim + 2 * num_v_heads
    if in_proj.shape[0] == in_proj_dim:
        pack_tp_size = tp_size
    elif tp_size > 1 and in_proj.shape[0] == in_proj_dim // tp_size:
        pack_tp_size = 1
    else:
        raise ValueError(f"Unexpected GDN in_proj rows: got {in_proj.shape[0]}, expected {in_proj_dim}")

    qk_heads_per_block = num_qk_heads // tp_size
    v_heads_per_block = num_v_heads // tp_size
    qk_dim_per_block = qk_head_dim * qk_heads_per_block
    v_dim_per_block = v_head_dim * v_heads_per_block
    output_num_qk_heads = qk_heads_per_block * pack_tp_size

    in_proj = in_proj.reshape(pack_tp_size, -1, hidden_size)
    q, k, v, z, b, a = torch.split(
        in_proj,
        [
            qk_dim_per_block,
            qk_dim_per_block,
            v_dim_per_block,
            v_dim_per_block,
            v_heads_per_block,
            v_heads_per_block,
        ],
        dim=1,
    )
    q, k, v, z, b, a = [
        weight.reshape(output_num_qk_heads, -1, hidden_size) for weight in [q, k, v, z, b, a]
    ]
    qkv = torch.cat([q.reshape(-1, hidden_size), k.reshape(-1, hidden_size), v.reshape(-1, hidden_size)], dim=0)
    return (
        qkv.contiguous(),
        z.reshape(-1, hidden_size).contiguous(),
        b.reshape(-1, hidden_size).contiguous(),
        a.reshape(-1, hidden_size).contiguous(),
    )


def _merge_gdn_conv1d_weight(config, weight):
    _, _, _, _, _, qk_dim, v_dim = _get_gdn_dims(config)
    pack_tp_size = _infer_gdn_pack_tp_size(config, weight.shape[0])
    if weight.shape[0] == 2 * qk_dim + v_dim:
        input_qk_dim = qk_dim
        input_v_dim = v_dim
    else:
        input_qk_dim = qk_dim // getattr(config, "tensor_model_parallel_size", 1)
        input_v_dim = v_dim // getattr(config, "tensor_model_parallel_size", 1)
    q, k, v = torch.split(weight, [input_qk_dim, input_qk_dim, input_v_dim], dim=0)
    q, k, v = [x.reshape(pack_tp_size, -1, *weight.shape[1:]) for x in [q, k, v]]
    return torch.cat([q, k, v], dim=1).reshape(-1, *weight.shape[1:]).contiguous()


def _split_gdn_conv1d_weight(config, weight):
    _, _, _, _, _, qk_dim, v_dim = _get_gdn_dims(config)
    tp_size = getattr(config, "tensor_model_parallel_size", 1)
    conv_dim = 2 * qk_dim + v_dim
    if weight.shape[0] == conv_dim:
        pack_tp_size = tp_size
    elif tp_size > 1 and weight.shape[0] == conv_dim // tp_size:
        pack_tp_size = 1
    else:
        raise ValueError(f"Unexpected GDN conv1d rows: got {weight.shape[0]}, expected {conv_dim}")
    qk_dim_per_block = qk_dim // tp_size
    v_dim_per_block = v_dim // tp_size
    weight = weight.reshape(pack_tp_size, -1, *weight.shape[1:])
    q, k, v = torch.split(weight, [qk_dim_per_block, qk_dim_per_block, v_dim_per_block], dim=1)
    return torch.cat(
        [
            q.reshape(-1, *weight.shape[2:]),
            k.reshape(-1, *weight.shape[2:]),
            v.reshape(-1, *weight.shape[2:]),
        ],
        dim=0,
    ).contiguous()


def _gdn_out_norm_hf_to_mcore(config, weight):
    weight = weight.clone()
    if getattr(config, "layernorm_zero_centered_gamma", False):
        weight = weight - 1
    return weight


def _gdn_out_norm_mcore_to_hf(config, weight):
    if getattr(config, "layernorm_zero_centered_gamma", False):
        weight = weight + 1
    return weight


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
        # linear attention / GDN
        "self_attention.in_proj.layer_norm_weight": [
            "model.language_model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.in_proj.weight": [
            "model.language_model.layers.{layer_number}.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.{layer_number}.linear_attn.in_proj_z.weight",
            "model.language_model.layers.{layer_number}.linear_attn.in_proj_b.weight",
            "model.language_model.layers.{layer_number}.linear_attn.in_proj_a.weight",
        ],
        "self_attention.conv1d.weight": ["model.language_model.layers.{layer_number}.linear_attn.conv1d.weight"],
        "self_attention.A_log": ["model.language_model.layers.{layer_number}.linear_attn.A_log"],
        "self_attention.dt_bias": ["model.language_model.layers.{layer_number}.linear_attn.dt_bias"],
        "self_attention.out_norm.weight": ["model.language_model.layers.{layer_number}.linear_attn.norm.weight"],
        "self_attention.out_proj.weight": ["model.language_model.layers.{layer_number}.linear_attn.out_proj.weight"],
    } | {
        f"self_attention.{weight_name}": ["model.language_model.layers.{layer_number}." + weight_name]
        for weight_name in [
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

    def _supports_transformer_config_kwarg(self, kwarg_name: str) -> bool:
        """Check whether the current TransformerConfig accepts a given kwarg."""
        transformer_config_class = getattr(self, "TransformerConfigClass", None)
        if transformer_config_class is None:
            return True

        dataclass_fields = getattr(transformer_config_class, "__dataclass_fields__", None)
        if dataclass_fields is not None:
            return kwarg_name in dataclass_fields

        try:
            signature = inspect.signature(transformer_config_class)
        except (TypeError, ValueError):
            return True
        return kwarg_name in signature.parameters

    def _get_transformer_layer_spec(self, vp_stage=None):
        transformer_layer_spec = super()._get_transformer_layer_spec(vp_stage)
        self._last_transformer_layer_spec = transformer_layer_spec
        return transformer_layer_spec

    def _get_gptmodel_args(self) -> dict:
        """Override to add MTP block spec if needed."""
        ret = super()._get_gptmodel_args()
        text_config = self._get_text_config()
        if getattr(text_config, "mtp_num_hidden_layers", None) is not None:
            transformer_layer_spec = getattr(self, "_last_transformer_layer_spec", None)
            if transformer_layer_spec is None:
                transformer_layer_spec = self._get_transformer_layer_spec()
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

    def _weight_name_mapping_mtp_mlp(self, name: str) -> list[str]:
        """Handle MTP MLP mappings, keeping per-expert tensors unfused for MoE layers."""
        layer_number = name.split(".")[2]
        mapping = self._MTP_MLP_MAPPING if "mlp.experts.linear_fc" in name else self._MLP_MAPPING
        convert_names = []
        for keyword, mapping_names in mapping.items():
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
                convert_names = self._weight_name_mapping_mtp_mlp(proxy_name)
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
        if "self_attention.in_proj.weight" in mcore_weights_name:
            assert len(hf_weights) == 4
            return _merge_gdn_linear_separate(self.config, *hf_weights)

        if "self_attention.conv1d.weight" in mcore_weights_name:
            assert len(hf_weights) == 1
            return _merge_gdn_conv1d_weight(self.config, hf_weights[0])

        if "self_attention.out_norm.weight" in mcore_weights_name:
            assert len(hf_weights) == 1
            return _gdn_out_norm_hf_to_mcore(self.config, hf_weights[0])

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
                # Extract local expert_id from name like "...linear_fc1.weight42"
                local_expert_id = int(mcore_weights_name.split("weight")[-1])
                # When using Expert Parallelism (EP), the local expert_id is relative
                # to this EP rank. We need to convert to global expert_id to index
                # into the full HF fused tensor [num_experts, ...].
                from megatron.core import mpu

                ep_size = mpu.get_expert_model_parallel_world_size()
                if ep_size > 1:
                    ep_rank = mpu.get_expert_model_parallel_rank()
                    num_local_experts = w.shape[0] // ep_size
                    global_expert_id = ep_rank * num_local_experts + local_expert_id
                else:
                    global_expert_id = local_expert_id
                expert_w = w[global_expert_id]  # (out_features, in_features)
                return expert_w.contiguous()

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        if "self_attention.in_proj.weight" in mcore_weights_name:
            return self._weight_name_mapping_mcore_to_hf(mcore_weights_name), list(
                _split_gdn_linear_separate(self.config, mcore_weights)
            )

        if "self_attention.conv1d.weight" in mcore_weights_name:
            return self._weight_name_mapping_mcore_to_hf(mcore_weights_name), [
                _split_gdn_conv1d_weight(self.config, mcore_weights)
            ]

        if "self_attention.out_norm.weight" in mcore_weights_name:
            return self._weight_name_mapping_mcore_to_hf(mcore_weights_name), [
                _gdn_out_norm_mcore_to_hf(self.config, mcore_weights)
            ]

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
            linear_conv_kernel_dim=getattr(text_config, "linear_conv_kernel_dim", 4),
            linear_key_head_dim=getattr(text_config, "linear_key_head_dim", 128),
            linear_value_head_dim=getattr(text_config, "linear_value_head_dim", 128),
            linear_num_key_heads=getattr(text_config, "linear_num_key_heads", 16),
            linear_num_value_heads=getattr(text_config, "linear_num_value_heads", 32),
            linear_attention_freq=getattr(text_config, "full_attention_interval", 4),
            **mtp_args,
        )

        if self._supports_transformer_config_kwarg("use_gated_attention"):
            base_kwargs["use_gated_attention"] = True

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
