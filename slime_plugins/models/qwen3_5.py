import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers.activations import ACT2FN

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.modules.convolution import causal_conv1d
    from fla.ops.cp import build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    FusedRMSNormGated = None
    ShortConvolution = None
    causal_conv1d = None
    build_cp_context = None
    chunk_gated_delta_rule = None

from .hf_attention import HuggingfaceAttention, _load_hf_config


def _get_text_config(hf_config):
    """Extract text config from a VLM config if needed."""
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


def _chunk_owner_for_layout(chunk_idx: int, cp_size: int, layout: str) -> int:
    if layout == "contiguous":
        return chunk_idx // 2
    if layout == "zigzag":
        return chunk_idx if chunk_idx < cp_size else 2 * cp_size - 1 - chunk_idx
    raise ValueError(f"Unsupported CP layout: {layout}")


def _local_chunk_ids_for_layout(cp_rank: int, cp_size: int, layout: str) -> list[int]:
    if layout == "contiguous":
        return [2 * cp_rank, 2 * cp_rank + 1]
    if layout == "zigzag":
        return [cp_rank, 2 * cp_size - 1 - cp_rank]
    raise ValueError(f"Unsupported CP layout: {layout}")


def _redistribute_sequence_chunks(
    tensor: torch.Tensor,
    *,
    cp_group,
    cp_rank: int,
    cp_size: int,
    source_layout: str,
    target_layout: str,
) -> torch.Tensor:
    if cp_size == 1 or source_layout == target_layout:
        return tensor

    if tensor.size(0) % 2 != 0:
        raise ValueError(f"Hybrid CP expects an even local sequence length, got {tensor.size(0)}")

    local_chunks = [chunk.contiguous() for chunk in tensor.chunk(2, dim=0)]
    chunk_len = local_chunks[0].size(0)
    if local_chunks[1].size(0) != chunk_len:
        raise ValueError("Hybrid CP expects the two local CP chunks to have equal length.")

    local_chunk_ids = _local_chunk_ids_for_layout(cp_rank, cp_size, source_layout)
    send_plan = [
        (_chunk_owner_for_layout(chunk_id, cp_size, target_layout), chunk_id, chunk)
        for chunk_id, chunk in zip(local_chunk_ids, local_chunks, strict=False)
    ]
    send_plan.sort(key=lambda item: item[0])

    input_split_sizes = [0] * cp_size
    send_parts: list[torch.Tensor] = []
    for dest_rank, _, chunk in send_plan:
        input_split_sizes[dest_rank] += chunk.size(0)
        send_parts.append(chunk)
    send_buffer = torch.cat(send_parts, dim=0) if send_parts else tensor.new_empty((0,) + tensor.shape[1:])

    target_chunk_ids = _local_chunk_ids_for_layout(cp_rank, cp_size, target_layout)
    recv_plan = [
        (_chunk_owner_for_layout(chunk_id, cp_size, source_layout), chunk_id)
        for chunk_id in target_chunk_ids
    ]
    output_split_sizes = [0] * cp_size
    for src_rank, _ in recv_plan:
        output_split_sizes[src_rank] += chunk_len

    recv_buffer = tensor.new_empty((sum(output_split_sizes),) + tensor.shape[1:])
    dist.all_to_all_single(
        recv_buffer,
        send_buffer,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=cp_group,
    )

    recv_segments: dict[int, torch.Tensor] = {}
    offset = 0
    for src_rank, split_size in enumerate(output_split_sizes):
        if split_size == 0:
            continue
        recv_segments[src_rank] = recv_buffer[offset : offset + split_size]
        offset += split_size

    reordered = []
    for src_rank, _ in recv_plan:
        recv_chunk = recv_segments[src_rank]
        if recv_chunk.size(0) != chunk_len:
            raise ValueError(
                f"Expected one chunk of length {chunk_len} from rank {src_rank}, got {recv_chunk.size(0)}"
            )
        reordered.append(recv_chunk)
    return torch.cat(reordered, dim=0)


class _CPLayoutRedistribute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, cp_group, source_layout: str, target_layout: str):
        cp_rank = dist.get_rank(group=cp_group)
        cp_size = dist.get_world_size(group=cp_group)
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.source_layout = source_layout
        ctx.target_layout = target_layout
        return _redistribute_sequence_chunks(
            tensor,
            cp_group=cp_group,
            cp_rank=cp_rank,
            cp_size=cp_size,
            source_layout=source_layout,
            target_layout=target_layout,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _redistribute_sequence_chunks(
            grad_output,
            cp_group=ctx.cp_group,
            cp_rank=ctx.cp_rank,
            cp_size=ctx.cp_size,
            source_layout=ctx.target_layout,
            target_layout=ctx.source_layout,
        )
        return grad_input, None, None, None


def _zigzag_to_contiguous_cp(tensor: torch.Tensor, cp_group) -> torch.Tensor:
    return _CPLayoutRedistribute.apply(tensor, cp_group, "zigzag", "contiguous")


def _contiguous_to_zigzag_cp(tensor: torch.Tensor, cp_group) -> torch.Tensor:
    return _CPLayoutRedistribute.apply(tensor, cp_group, "contiguous", "zigzag")


# Adapted from Qwen3NextGatedDeltaNet but with separate in_proj_qkv and in_proj_z
class Qwen3_5GatedDeltaNet(nn.Module):
    """
    Qwen3.5 GatedDeltaNet with varlen support.
    Unlike Qwen3Next which uses a combined in_proj_qkvz, Qwen3.5 uses
    separate in_proj_qkv (for Q,K,V) and in_proj_z (for Z).
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        if FusedRMSNormGated is None or ShortConvolution is None or chunk_gated_delta_rule is None:
            raise ImportError("Qwen3.5 GDN requires flash-linear-attention to be installed.")
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ShortConvolution(
            hidden_size=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
        )

        # Separate projections for QKV and Z (unlike Qwen3Next which combines QKVZ)
        projection_size_qkv = self.key_dim * 2 + self.value_dim
        projection_size_z = self.value_dim
        self.in_proj_qkv = nn.Linear(self.hidden_size, projection_size_qkv, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, projection_size_z, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # time step projection
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=self.activation,
            device=torch.cuda.current_device(),
            dtype=config.dtype if config.dtype is not None else torch.get_current_dtype(),
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.cp_group = None
        self.cp_rank = 0
        self.cp_world_size = 1

    def _build_cp_context(self, seq_len: int, device: torch.device):
        if self.cp_group is None or self.cp_world_size == 1:
            return None
        if build_cp_context is None:
            raise ImportError(
                "Qwen3.5 hybrid CP requires a flash-linear-attention build with CP support (build_cp_context)."
            )
        global_seq_len = seq_len * self.cp_world_size
        return build_cp_context(
            cu_seqlens=torch.tensor([0, global_seq_len], dtype=torch.int32, device=device),
            group=self.cp_group,
            conv1d_kernel_size=self.conv_kernel_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        cp_context = self._build_cp_context(seq_len, hidden_states.device)

        # Projections (flat layout: [Q_all, K_all, V_all])
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Convolution on the flat QKV
        if cp_context is not None:
            if causal_conv1d is None:
                raise ImportError(
                    "Qwen3.5 hybrid CP requires a flash-linear-attention build with causal_conv1d CP support."
                )
            mixed_qkv, _ = causal_conv1d(
                x=mixed_qkv.contiguous(),
                weight=self.conv1d.weight.squeeze(1),
                bias=getattr(self.conv1d, "bias", None),
                activation=self.activation,
                initial_state=None,
                output_final_state=False,
                cp_context=cp_context,
            )
        else:
            mixed_qkv, _ = self.conv1d(
                x=mixed_qkv,
                cu_seqlens=cu_seqlens,
            )

        # Split into Q, K, V (flat split, matching HF layout)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        gated_delta_kwargs = {
            "g": g,
            "beta": beta,
            "initial_state": None,
            "output_final_state": False,
            "use_qk_l2norm_in_kernel": True,
        }
        if cp_context is not None:
            gated_delta_kwargs["cu_seqlens"] = cp_context.cu_seqlens
            gated_delta_kwargs["cp_context"] = cp_context
        elif cu_seqlens is not None:
            gated_delta_kwargs["cu_seqlens"] = cu_seqlens

        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            **gated_delta_kwargs,
        )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


class Attention(HuggingfaceAttention):
    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            pg_collection,
        )
        # Qwen3.5 is a VLM model with nested text_config
        self.hf_config = _get_text_config(self.hf_config)
        self.hf_config._attn_implementation = "flash_attention_2"

        self.linear_attn = Qwen3_5GatedDeltaNet(self.hf_config, self.hf_layer_idx)

        # Use a simple RMSNorm
        try:
            from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

            self.input_layernorm = Qwen3NextRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)
        except ImportError:
            from torch.nn import RMSNorm

            self.input_layernorm = RMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

    def hf_forward(self, hidden_states, packed_seq_params):
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        inference_context=None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params=None,
        sequence_len_offset: int | None = None,
        *,
        inference_params=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep the original THD / packed-sequence path intact. Hybrid CP is only
        # enabled for the documented Qwen3.5 MoE bshd path.
        if packed_seq_params is not None or getattr(self.args, "qkv_format", None) != "bshd":
            return super().forward(
                hidden_states,
                attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )

        if self.args.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=mpu.get_tensor_model_parallel_group(),
            )

        cp_size = mpu.get_context_parallel_world_size()
        cp_group = mpu.get_context_parallel_group() if cp_size > 1 else None
        cp_rank = mpu.get_context_parallel_rank() if cp_size > 1 else 0

        if cp_size > 1:
            if hidden_states.size(1) != 1:
                raise NotImplementedError(
                    "Qwen3.5 MoE hybrid CP currently requires micro-batch-size 1 on the bshd path."
                )
            hidden_states = _zigzag_to_contiguous_cp(hidden_states, cp_group)

        hidden_states = hidden_states.permute(1, 0, 2)  # [batch, seq_len, hidden_dim]
        self.linear_attn.cp_group = cp_group
        self.linear_attn.cp_rank = cp_rank
        self.linear_attn.cp_world_size = cp_size

        hidden_states = self.input_layernorm(hidden_states)
        output = self.linear_attn(hidden_states=hidden_states, cu_seqlens=None)
        output = output.permute(1, 0, 2)  # [seq_len, batch, hidden_dim]

        if cp_size > 1:
            output = _contiguous_to_zigzag_cp(output, cp_group)

        if self.args.sequence_parallel:
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output, group=mpu.get_tensor_model_parallel_group()
            )

        return output, None


def get_qwen3_5_spec(args, config, vp_stage):
    # always use the moe path for MoE models
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = _load_hf_config(args.hf_checkpoint)
    text_config = _get_text_config(hf_config)

    # Compute layer_types if the config class doesn't expose it
    if not hasattr(text_config, "layer_types"):
        interval = getattr(text_config, "full_attention_interval", 4)
        n = text_config.num_hidden_layers
        text_config.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)
        ]

    for layer_id in range(num_layers_to_build):
        if text_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
