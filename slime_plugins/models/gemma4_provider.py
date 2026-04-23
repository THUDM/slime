"""Custom model provider for Gemma4.

Installs Gemma4-specific behaviors that sit outside the transformer layer:
- embedding scaling (multiply embeddings by sqrt(hidden_size))
- logit softcapping (`final_logit_softcapping`)
- dual-RoPE (different rope_theta + partial-rotary for global vs sliding layers)
- layer_scalar buffers loaded from the HF checkpoint
"""
import json
import logging
import os

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from slime_plugins.models.gemma4 import _load_hf_text_config

logger = logging.getLogger(__name__)


def _is_rank_zero() -> bool:
    """True on a single-process run, or on rank 0 when distributed."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    args = get_args()
    config = core_transformer_config_from_args(args)

    transformer_layer_spec = import_module(args.spec)
    if callable(transformer_layer_spec):
        transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
    )

    _install_hooks(model, args, config, pre_process, post_process)
    return model


class DualRotaryEmbedding(torch.nn.Module):
    """Wraps a (global, local) pair of RotaryEmbedding modules and emits a
    single concatenated tensor (global part first). ``Gemma4TransformerLayer``
    slices it per-layer based on ``is_sliding``. Concat (not tuple) because
    Megatron's ``SelfAttention.forward`` reads a 2-tuple as
    ``(self_attn, cross_attn)`` RoPE and would misread our pair.
    """

    def __init__(self, local_rope, global_rope, global_dim: int):
        super().__init__()
        self.local_rope = local_rope
        self.global_rope = global_rope
        self.global_dim = global_dim

    def get_rotary_seq_len(self, *args, **kwargs):
        # Both ropes share the same sequence-length logic (they only differ in
        # theta and partial-rotary); delegate to the local one.
        return self.local_rope.get_rotary_seq_len(*args, **kwargs)

    def forward(self, seq_len, **kwargs):
        global_emb = self.global_rope(seq_len, **kwargs)
        local_emb = self.local_rope(seq_len, **kwargs)
        return torch.cat([global_emb, local_emb], dim=-1)


def _install_hooks(model, args, config, pre_process, post_process):
    """Install Gemma4-specific pre/post-process hooks on a built GPTModel.

    We use ``register_forward_hook`` rather than subclassing GPTModel
    because:
      - Two independent behaviors (embed scale, softcap) on two different
        submodules. Subclassing would require overriding
        ``GPTModel.forward`` and branching on pp/vp stage.
      - The hooks are shape- and dtype-preserving, so they compose cleanly
        with PP (only first-stage runs embedding, only last-stage runs
        output_layer) — we gate registration on ``pre_process`` /
        ``post_process`` accordingly.
      - Keeps the diff local to this plugin: we don't need to shadow any
        Megatron-maintained class.
    """
    hf_text = _load_hf_text_config(args.hf_checkpoint)
    hidden_size = config.hidden_size

    inner = model.module if hasattr(model, "module") else model

    # Embedding scaling — HF applies this inside the embedding module.
    # See ``Gemma4TextScaledWordEmbedding``: the scale is stored as an fp32
    # tensor and cast to the embedding weight's dtype at forward time, so
    # the scale-as-applied depends on the current weight dtype (bf16 during
    # training, fp32 during some eval paths). We match that behavior here.
    if pre_process and hasattr(inner, "embedding"):
        embed_scale = torch.tensor(hidden_size ** 0.5)  # fp32

        def _embed_hook(module, inp, output):
            return output * embed_scale.to(output.dtype)

        inner.embedding.register_forward_hook(_embed_hook)

    # Final logit softcapping — HF applies tanh(logits / cap) * cap.
    # Some Megatron output_layer variants (parallel_output paths) return
    # ``(logits, bias)``; we pass the non-logit tail through unchanged.
    softcap = getattr(hf_text, "final_logit_softcapping", None)
    if post_process and softcap and hasattr(inner, "output_layer"):
        def _softcap_hook(module, inp, output):
            if isinstance(output, tuple):
                return (torch.tanh(output[0] / softcap) * softcap,) + output[1:]
            return torch.tanh(output / softcap) * softcap
        inner.output_layer.register_forward_hook(_softcap_hook)

    # Dual RoPE: replace Megatron's single rotary_pos_emb with a wrapper that
    # produces (global, local) RoPE side-by-side. Gemma4 uses partial-rotary
    # on global layers (implemented here by zeroing the tail of inv_freq).
    if hasattr(inner, "rotary_pos_emb"):
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

        rope_params = getattr(hf_text, "rope_parameters", {}) or {}
        full = rope_params.get("full_attention", {}) or {}
        sliding = rope_params.get("sliding_attention", {}) or {}
        global_theta = full.get("rope_theta", 1_000_000.0)
        local_theta = sliding.get("rope_theta", 10_000.0)
        global_head_dim = hf_text.global_head_dim
        global_partial = full.get("partial_rotary_factor", 0.25)

        local_rope = inner.rotary_pos_emb  # already built with args.rotary_base

        global_rope = RotaryEmbedding(
            kv_channels=global_head_dim,
            rotary_percent=1.0,
            rotary_base=global_theta,
        )
        # HF "proportional" RoPE: first (partial * head_dim // 2) inv_freq
        # entries are live, the rest are zero (no rotation on those dims).
        # Writing this to the existing buffer keeps device/dtype correct.
        rope_angles = int(global_partial * global_head_dim // 2)
        half = global_head_dim // 2
        # Guard the RoPE geometry: 0 means "no rotation" (nonsensical here);
        # > half would produce nope<0 and a shape-mismatched copy_. Both
        # should fail loudly rather than silently writing garbage.
        assert 0 < rope_angles <= half, (
            f"global_partial_rotary_factor={global_partial} with "
            f"global_head_dim={global_head_dim} produced rope_angles="
            f"{rope_angles}; must be in (0, {half}]."
        )
        inv_freq_live = 1.0 / (
            global_theta ** (
                torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / global_head_dim
            )
        )
        nope = half - rope_angles
        inv_freq = torch.cat([inv_freq_live, torch.zeros(nope)]) if nope > 0 else inv_freq_live
        assert inv_freq.shape == global_rope.inv_freq.shape, (
            f"inv_freq shape {tuple(inv_freq.shape)} doesn't match "
            f"global_rope.inv_freq shape {tuple(global_rope.inv_freq.shape)}; "
            "Megatron RotaryEmbedding layout may have changed."
        )
        global_rope.inv_freq.copy_(inv_freq.to(global_rope.inv_freq.device))

        inner.rotary_pos_emb = DualRotaryEmbedding(local_rope, global_rope, global_head_dim)
        # Layers split the concatenated tensor by this dim.
        config.dual_rope_global_dim = global_head_dim
        if _is_rank_zero():
            logger.info(
                "DualRotaryEmbedding: local_theta=%s global_theta=%s "
                "global_dim=%s rope_angles=%d (nope=%d)",
                local_theta, global_theta, global_head_dim, rope_angles, nope,
            )

    # Load layer scalars from the HF checkpoint. These are buffers, not
    # parameters, and are applied once per layer after the MoE/MLP block.
    if hasattr(inner, "decoder") and args.hf_checkpoint:
        _load_layer_scalars(inner, args.hf_checkpoint, config)


def _read_layer_scalars_from_safetensors(hf_checkpoint: str) -> dict[int, float] | None:
    """Read all ``layer_scalar`` values from the HF safetensors checkpoint.

    Returns ``{global_layer_idx: scalar}`` or ``None`` if the checkpoint has
    no safetensors index (older HF layouts) or no layer_scalar weights. Only
    called on rank 0 — results are broadcast to the other ranks.
    """
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        logger.warning("No safetensors index at %s; skipping layer scalars", index_path)
        return None

    from safetensors import safe_open

    with open(index_path) as f:
        index = json.load(f)

    scalars: dict[int, float] = {}
    for key, filename in index["weight_map"].items():
        if "layer_scalar" not in key:
            continue
        layer_idx = int(key.split(".layers.")[1].split(".")[0])
        with safe_open(os.path.join(hf_checkpoint, filename), framework="pt", device="cpu") as sf:
            scalars[layer_idx] = sf.get_tensor(key).item()

    if not scalars:
        logger.warning("No layer_scalar weights found in checkpoint %s", hf_checkpoint)
        return None
    return scalars


def _broadcast_layer_scalars(scalars: dict[int, float] | None) -> dict[int, float] | None:
    """Broadcast the rank-0-read ``scalars`` dict to every rank.

    safetensors reads on every rank cause an O(world_size) fan-out of tiny
    reads on the shared filesystem; the dict itself is a few kilobytes. If
    ``torch.distributed`` isn't initialized (single-process run), we simply
    return the input dict.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return scalars
    obj = [scalars] if torch.distributed.get_rank() == 0 else [None]
    torch.distributed.broadcast_object_list(obj, src=0)
    return obj[0]


def _load_layer_scalars(inner, hf_checkpoint, config):
    # Wrong layer_scalars materially change activations vs HF (they're per-
    # layer multiplicative gains on the residual stream, not decorative), so
    # by default we fail hard if the load breaks. Set
    # GEMMA4_ALLOW_MISSING_LAYER_SCALARS=1 to downgrade to a warning and
    # train with the default value of 1.0 — only useful for debug runs
    # against a checkpoint that genuinely lacks these buffers.
    allow_missing = os.environ.get("GEMMA4_ALLOW_MISSING_LAYER_SCALARS") == "1"
    try:
        scalars = _read_layer_scalars_from_safetensors(hf_checkpoint) if _is_rank_zero() else None
        scalars = _broadcast_layer_scalars(scalars)
        if not scalars:
            if allow_missing:
                return
            raise RuntimeError(
                "No layer_scalar weights found in checkpoint; set "
                "GEMMA4_ALLOW_MISSING_LAYER_SCALARS=1 to proceed with "
                "default values (not numerically equivalent to HF)."
            )

        # Under pipeline-parallelism, inner.decoder.layers holds only this
        # rank's local subset. Translate the local index back to the global
        # (HF 0-indexed) layer index so we apply the right scalar per layer.
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
        pp_offset = get_transformer_layer_offset(config)

        loaded = 0
        for i, layer in enumerate(inner.decoder.layers):
            if hasattr(layer, "layer_scalar"):
                global_idx = i + pp_offset
                if global_idx not in scalars:
                    if allow_missing:
                        logger.warning(
                            "layer_scalar for global layer %d missing; using default 1.0",
                            global_idx,
                        )
                    else:
                        raise KeyError(
                            f"layer_scalar for global layer {global_idx} "
                            f"missing in checkpoint (have: {sorted(scalars)[:10]}...); "
                            "checkpoint may be truncated."
                        )
                layer.layer_scalar.fill_(scalars.get(global_idx, 1.0))
                loaded += 1
        if _is_rank_zero():
            logger.info(
                "Applied %d/%d layer scalars (pp_offset=%d, range=%.4f..%.4f)",
                loaded, len(inner.decoder.layers), pp_offset,
                min(scalars.values()), max(scalars.values()),
            )
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # These are recoverable: older HF layouts have no safetensors index,
        # or json is malformed. Warn and fall back.
        if allow_missing:
            logger.warning("layer scalars unavailable (%s: %s); using default 1.0",
                           type(e).__name__, e)
            return
        raise
