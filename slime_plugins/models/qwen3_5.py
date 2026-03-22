import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import mpu, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers.activations import ACT2FN

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    pass

from .hf_attention import HuggingfaceAttention, _load_hf_config


def _get_text_config(hf_config):
    """Extract text config from a VLM config if needed."""
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


# ---------------------------------------------------------------------------
# Context-parallel helpers for linear attention
# ---------------------------------------------------------------------------


def _cp_all_gather_zigzag(local_tensor, cu_seqlens, cp_group, cp_size):
    """All-gather from all CP ranks and reconstruct the full sequence in zigzag order.

    Each CP rank holds two chunks per sequence arranged as [chunk_r, chunk_{2*cp-1-r}].
    This function gathers all ranks' data and reassembles the original sequence order.

    Args:
        local_tensor: [local_total_seq, ...] - this rank's portion
        cu_seqlens: [num_seqs + 1] - global cumulative sequence lengths
        cp_group: the CP process group
        cp_size: number of CP ranks

    Returns:
        full_tensor: [full_total_seq, ...] - reconstructed full sequence
    """
    gathered = [torch.empty_like(local_tensor) for _ in range(cp_size)]
    dist.all_gather(gathered, local_tensor.contiguous(), group=cp_group)

    local_cu_seqlens = cu_seqlens // cp_size
    result = []
    for i in range(len(cu_seqlens) - 1):
        chunk_size = (cu_seqlens[i + 1] - cu_seqlens[i]) // 2 // cp_size
        # First half: forward chunks from rank 0, 1, ..., cp_size-1
        # Second half: backward chunks from rank cp_size-1, ..., 1, 0
        result.extend(
            [gathered[r][local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size] for r in range(cp_size)]
            + [gathered[r][local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]] for r in range(cp_size)][::-1]
        )
    return torch.cat(result, dim=0)


def _cp_slice_zigzag(full_tensor, cu_seqlens, cp_rank, cp_size):
    """Extract this CP rank's chunks from the full sequence in zigzag order.

    Args:
        full_tensor: [full_total_seq, ...] - full sequence
        cu_seqlens: [num_seqs + 1] - global cumulative sequence lengths
        cp_rank: this rank's position in the CP group
        cp_size: number of CP ranks

    Returns:
        local_tensor: [local_total_seq, ...] - this rank's portion
    """
    chunks = []
    for i in range(len(cu_seqlens) - 1):
        seq = full_tensor[cu_seqlens[i] : cu_seqlens[i + 1]]
        seq_chunks = torch.chunk(seq, 2 * cp_size, dim=0)
        chunks.append(seq_chunks[cp_rank])
        chunks.append(seq_chunks[2 * cp_size - 1 - cp_rank])
    return torch.cat(chunks, dim=0)


class CPLinearAttnFunction(torch.autograd.Function):
    """Custom autograd function for context-parallel linear attention.

    Forward: all-gather hidden_states -> compute on full sequence -> slice output.
             Only local hidden_states are saved (full gathered tensor is freed).
    Backward: re-all-gather hidden_states + grad_output -> recompute forward ->
              backward -> slice gradient for this rank.

    This fixes two issues with the previous dist.nn.all_gather approach:
    1. The full all-gathered tensor is no longer kept in the autograd graph (memory savings).
    2. Gradients (including dk, dv) are correctly handled via recomputation + slicing.
    """

    @staticmethod
    def forward(ctx, hidden_states, cu_seqlens, module, packed_seq_params, cp_group, cp_size, cp_rank):
        # All-gather and reconstruct full sequence
        full_hidden = _cp_all_gather_zigzag(hidden_states, cu_seqlens, cp_group, cp_size)

        # Forward without grad (will recompute in backward for activation savings)
        with torch.no_grad():
            full_bhd = full_hidden.permute(1, 0, 2)  # [seq, batch, hidden] -> [batch, seq, hidden]
            output = module.hf_forward(full_bhd, packed_seq_params)
            output_thd = output.permute(1, 0, 2)  # [batch, seq, hidden] -> [seq, batch, hidden]

        # Slice output for this CP rank
        local_output = _cp_slice_zigzag(output_thd, cu_seqlens, cp_rank, cp_size)

        # Save only local data (full_hidden is freed after this scope)
        ctx.save_for_backward(hidden_states, cu_seqlens)
        ctx.module = module
        ctx.packed_seq_params = packed_seq_params
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank

        return local_output

    @staticmethod
    def backward(ctx, d_local_output):
        hidden_states, cu_seqlens = ctx.saved_tensors

        # Re-gather hidden states for recomputation
        full_hidden = _cp_all_gather_zigzag(hidden_states.detach(), cu_seqlens, ctx.cp_group, ctx.cp_size)
        full_hidden = full_hidden.detach().requires_grad_(True)

        # All-gather grad output from all CP ranks to reconstruct full gradient
        d_full_output = _cp_all_gather_zigzag(d_local_output.contiguous(), cu_seqlens, ctx.cp_group, ctx.cp_size)

        # Recompute forward with grad enabled (for parameter gradient computation)
        with torch.enable_grad():
            full_bhd = full_hidden.permute(1, 0, 2)
            output = ctx.module.hf_forward(full_bhd, ctx.packed_seq_params)
            output_thd = output.permute(1, 0, 2)

        # Backward through recomputed graph (accumulates parameter gradients)
        torch.autograd.backward(output_thd, d_full_output)

        # Slice gradient for this CP rank (inverse of all-gather zigzag)
        d_local_hidden = _cp_slice_zigzag(full_hidden.grad, cu_seqlens, ctx.cp_rank, ctx.cp_size)

        return d_local_hidden, None, None, None, None, None, None


# Adapted from Qwen3NextGatedDeltaNet but with separate in_proj_qkv and in_proj_z
class Qwen3_5GatedDeltaNet(nn.Module):
    """
    Qwen3.5 GatedDeltaNet with varlen support.
    Unlike Qwen3Next which uses a combined in_proj_qkvz, Qwen3.5 uses
    separate in_proj_qkv (for Q,K,V) and in_proj_z (for Z).
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        # Projections (flat layout: [Q_all, K_all, V_all])
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Convolution on the flat QKV
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

        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
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
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert packed_seq_params is not None

        if self.args.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=mpu.get_tensor_model_parallel_group()
            )

        cp_size = mpu.get_context_parallel_world_size()

        if cp_size > 1:
            output = CPLinearAttnFunction.apply(
                hidden_states,
                packed_seq_params.cu_seqlens_q,
                self,
                packed_seq_params,
                mpu.get_context_parallel_group(),
                cp_size,
                mpu.get_context_parallel_rank(),
            )
        else:
            hidden_states = hidden_states.permute(1, 0, 2)  # [seq, batch, hidden] -> [batch, seq, hidden]
            output = self.hf_forward(hidden_states, packed_seq_params)
            output = output.permute(1, 0, 2)  # [batch, seq, hidden] -> [seq, batch, hidden]

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
