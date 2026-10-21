"""Checkpoint-compatible CogViT using the Transformers GLM-OCR building blocks."""

import torch.nn as nn
import torch.nn.functional as F
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrVisionBlock,
    GlmOcrVisionMlp,
    GlmOcrVisionModel,
    GlmOcrVisionPatchEmbed,
    GlmOcrVisionPatchMerger,
)

from .attention import RMSNorm


class Glm5NextVisionBlock(GlmOcrVisionBlock):
    def __init__(self, config):
        super().__init__(config)
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # SGLang's VisionAttention uses the per-head default epsilon.
        self.attn.q_norm = RMSNorm(config.hidden_size // config.num_heads, 1e-6)
        self.attn.k_norm = RMSNorm(config.hidden_size // config.num_heads, 1e-6)
        self.mlp = Glm5NextVisionMLP(config)

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        attention = self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs
        )
        # Match the fused residual-add/RMSNorm: normalize the FP32 sum,
        # while the residual branch itself is stored in the model dtype.
        residual = hidden_states.float() + attention.float()
        normalized = self.norm2(residual).to(hidden_states.dtype)
        return residual.to(hidden_states.dtype) + self.mlp(normalized)


class Glm5NextVisionMLP(GlmOcrVisionMlp):
    def __init__(self, config):
        super().__init__(config, bias=True)
        self.limit = config.swiglu_limit

    def forward(self, x):
        gate = self.gate_proj(x).float().clamp(max=self.limit)
        up = self.up_proj(x).float().clamp(-self.limit, self.limit)
        return self.down_proj((F.silu(gate) * up).to(x.dtype))


class Glm5NextVisionMerger(GlmOcrVisionPatchMerger):
    def __init__(self, config):
        super().__init__(config.out_hidden_size, config.projection_intermediate_size, config.hidden_act)
        self.limit = config.swiglu_limit

    def forward(self, x):
        x = self.act1(self.post_projection_norm(self.proj(x)))
        gate = self.gate_proj(x).float().clamp(max=self.limit)
        up = self.up_proj(x).float().clamp(-self.limit, self.limit)
        return self.down_proj((F.silu(gate) * up).to(x.dtype))


class Glm5NextPatchEmbed(GlmOcrVisionPatchEmbed):
    def forward(self, hidden_states):
        # Every input is exactly one non-overlapping Conv3d patch. GEMM
        # implements the same contraction without cuDNN's BF16 Conv3d fallback.
        patches = hidden_states.reshape(-1, self.proj.weight[0].numel()).to(self.proj.weight.dtype)
        # SGLang's Conv3dLayer uses this same GEMM and fused bias epilogue.
        return F.linear(patches, self.proj.weight.flatten(1), self.proj.bias)


class Glm5NextVisionModel(GlmOcrVisionModel):
    def __init__(self, config):
        super().__init__(config)
        self.patch_embed = Glm5NextPatchEmbed(config)
        self.merger = Glm5NextVisionMerger(config)
        self.blocks = nn.ModuleList([Glm5NextVisionBlock(config) for _ in range(config.depth)])
        self.post_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def _apply(self, fn, recurse=True):
        # Frequencies are not checkpoint parameters. Preserve their original
        # FP32 values when Megatron casts the model weights to BF16.
        inv_freq = self.rotary_pos_emb.inv_freq
        super()._apply(fn, recurse=recurse)
        self.rotary_pos_emb.inv_freq = inv_freq.to(device=self.rotary_pos_emb.inv_freq.device)
        return self
