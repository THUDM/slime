"""GLM-5.3 Flash KDA and no-RoPE MLA with pooled DSA indexing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.modules.conv import causal_conv1d
from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

from slime_plugins.models.glm5.ops.sparse_mla import SparseMLA

from .ops import pooled_indexer_topk


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class KimiDeltaAttention(nn.Module):
    def __init__(self, config, hf_config, pg_collection=None, **kwargs):
        super().__init__()
        self.cp_group = pg_collection.cp if pg_collection is not None else None
        linear = hf_config.linear_attn_config
        self.heads = linear["num_heads"]
        self.head_dim = linear["head_dim"]
        self.lower_bound = linear.get("gate_lower_bound", -5.0 if linear.get("safe_gate") else None)
        width = self.heads * self.head_dim
        hidden = config.hidden_size
        for name in ("q", "k", "v"):
            setattr(self, f"{name}_proj", nn.Linear(hidden, width, bias=False))
            setattr(
                self,
                f"{name}_conv1d",
                ShortConvolution(width, linear["short_conv_kernel_size"], bias=False, activation="silu"),
            )
        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, width, bias=False)
        self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, width, bias=False)
        self.b_proj = nn.Linear(hidden, self.heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(width, dtype=torch.float32))
        self.o_norm = FusedRMSNormGated(self.head_dim, eps=hf_config.rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(width, hidden, bias=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.A_log.data = self.A_log.data.float()
        self.dt_bias.data = self.dt_bias.data.float()
        return self

    def forward(self, hidden_states, packed_seq_params=None, **kwargs):
        x = hidden_states.transpose(0, 1)
        cu = packed_seq_params.cu_seqlens_q
        cp_context = None
        if self.cp_group is not None and self.cp_group.size() > 1:
            # --allgather-cp partitions the packed stream into contiguous
            # intervals. FLA exchanges convolution halos and recurrent states;
            # projections and KDA remain local to each interval.
            cp_context = build_cp_context(cu, self.cp_group, conv1d_kernel_size=self.q_conv1d.kernel_size[0])
        qkv = []
        for name in ("q", "k", "v"):
            projected = getattr(self, f"{name}_proj")(x)
            conv = getattr(self, f"{name}_conv1d")
            if cp_context is None:
                convolved, _ = conv(projected, cu_seqlens=cu)
            else:
                convolved, _ = causal_conv1d(
                    projected, conv.weight.squeeze(1), activation=conv.activation, cp_context=cp_context
                )
            qkv.append(convolved.unflatten(-1, (self.heads, self.head_dim)))
        gate = self.f_b_proj(self.f_a_proj(x)).unflatten(-1, (self.heads, self.head_dim))
        output, _ = chunk_kda(
            *qkv,
            g=gate,
            beta=self.b_proj(x).sigmoid(),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=self.lower_bound is not None,
            lower_bound=self.lower_bound,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            cu_seqlens=cu,
            cp_context=cp_context,
        )
        output_gate = self.g_b_proj(self.g_a_proj(x)).reshape(-1, self.head_dim)
        output = self.o_norm(output.reshape(-1, self.head_dim), output_gate).reshape(
            *x.shape[:-1], self.heads * self.head_dim
        )
        return self.o_proj(output).transpose(0, 1).contiguous(), None


class PooledIndexer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.index_n_heads
        self.dim = config.index_head_dim
        self.topk = config.index_topk
        self.pool_size = config.index_kpool
        self.wq_b = nn.Linear(config.q_lora_rank, self.heads * self.dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.dim, bias=False)
        self.k_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.weights_proj = nn.Linear(config.hidden_size, self.heads, bias=False)
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(self.dim, config.hidden_size))
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.pool_size, self.dim))
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x, q, packed_seq_params, cp_group):
        # MLA-noPE also disables indexer RoPE. The checkpoint already uses
        # SGLang/HF's channel order, so no half-channel permutation is needed.
        index_q = self.wq_b(q).view(-1, self.heads, self.dim)
        index_k = self.k_norm(self.wk(x))
        weights = F.linear(x.float(), self.weights_proj.weight.float()) * (self.heads * self.dim) ** -0.5
        gates = F.linear(x, self.index_kpool_compress_gate)
        cp_size = cp_group.size() if cp_group is not None else 1
        cp_rank = cp_group.rank() if cp_group is not None else 0
        if cp_size > 1:
            pooled_inputs = gather_from_sequence_parallel_region(torch.cat([index_k, gates], dim=-1), group=cp_group)
            index_k, gates = pooled_inputs.chunk(2, dim=-1)
        return pooled_indexer_topk(
            index_q,
            index_k,
            weights,
            gates,
            self.index_kpool_compress_ape,
            packed_seq_params.cu_seqlens_q,
            self.topk,
            self.pool_size,
            query_start=cp_rank * x.shape[0],
        )


class NopeDSAttention(nn.Module):
    def __init__(self, config, hf_config, pg_collection=None, **kwargs):
        super().__init__()
        self.cp_group = pg_collection.cp if pg_collection is not None else None
        self.heads = hf_config.num_attention_heads
        self.qk_dim = hf_config.qk_nope_head_dim
        self.v_dim = hf_config.v_head_dim
        self.kv_rank = hf_config.kv_lora_rank
        self.q_a_proj = nn.Linear(config.hidden_size, hf_config.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(hf_config.q_lora_rank, hf_config.rms_norm_eps)
        self.q_b_proj = nn.Linear(hf_config.q_lora_rank, self.heads * self.qk_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_rank, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_rank, hf_config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.kv_rank, self.heads * (self.qk_dim + self.v_dim), bias=False)
        self.o_proj = nn.Linear(self.heads * self.v_dim, config.hidden_size, bias=False)
        self.indexer = PooledIndexer(hf_config)

    def forward(self, hidden_states, packed_seq_params=None, **kwargs):
        x = hidden_states.squeeze(1)
        q_latent = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_latent).view(-1, self.heads, self.qk_dim)
        kv = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
        wk, wv = self.kv_b_proj.weight.view(self.heads, self.qk_dim + self.v_dim, self.kv_rank).split(
            (self.qk_dim, self.v_dim), dim=1
        )
        q = torch.bmm(q.transpose(0, 1), wk).transpose(0, 1)
        cp_size = self.cp_group.size() if self.cp_group is not None else 1
        if cp_size > 1:
            # Queries stay local; every rank needs all compressed keys/values.
            # The gather's backward reduce-scatters contributions to each KV owner.
            kv = gather_from_sequence_parallel_region(kv, group=self.cp_group)
        indices = self.indexer(x, q_latent, packed_seq_params, self.cp_group)
        # The shared GLM5 sparse kernel uses a 512+64 layout. A zero tail is
        # mathematically exact for no-RoPE MLA and keeps its proven backward.
        out, _ = SparseMLA.apply(F.pad(q, (0, 64)), F.pad(kv[:, None], (0, 64)), indices, self.qk_dim**-0.5)
        out = torch.bmm(out.transpose(0, 1), wv.transpose(1, 2)).transpose(0, 1)
        return self.o_proj(out.flatten(1)).unsqueeze(1), None
