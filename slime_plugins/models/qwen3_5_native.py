"""Qwen linear attention with Megatron's tensor/context-parallel GDN."""

import copy

import torch
from megatron.core.ssm.gated_delta_net.gdn import GatedDeltaNet


class Qwen3_5NativeAttention(GatedDeltaNet):
    def _resolve_cu_seqlens(self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size=1):
        cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens_actual
        # Every layer and its recomputation share the same immutable batch
        # offsets. Cache successful validation on that tensor, not the model.
        # Include its version so in-place edits and edits through views recheck.
        if cu_seqlens.is_inference():
            return super()._resolve_cu_seqlens(cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size)
        key = (cu_seqlens._version, total_seq_len, cp_size)
        if getattr(cu_seqlens, "_slime_gdn_validation", None) != key:
            super()._resolve_cu_seqlens(cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size)
            cu_seqlens._slime_gdn_validation = key
        return cu_seqlens

    def __init__(self, config, linear_config, **kwargs):
        config = copy.copy(config)
        for name, value in linear_config.items():
            setattr(config, name, value)
        if not config.layernorm_zero_centered_gamma:
            raise ValueError("Native Qwen GDN requires --apply-layernorm-1p for its input layernorm")
        if config.tensor_model_parallel_size > 1 and not config.sequence_parallel:
            raise ValueError("Native Qwen GDN requires --sequence-parallel to synchronize its output-norm gradients")
        super().__init__(config=config, **kwargs)

        # Qwen's input norm uses (1 + weight), while the gated output norm
        # stores its scale directly. Keep the checkpoint values unchanged.
        self.out_norm.zero_centered_gamma = False
        with torch.no_grad():
            self.out_norm.weight.fill_(1)

        # Each TP rank owns a slice of every projection section. These sizes
        # describe the full tensor, before sharding, for checkpoint conversion
        # and live weight publication to the serving model.
        self.in_proj.weight.partition_sections = tuple(size * self.tp_size for size in self.in_proj_split_sections)
        self.conv1d.weight.partition_sections = (self.qk_dim, self.qk_dim, self.v_dim)
