"""NVIDIA Nemotron-H (`nemotron_h`) support for slime's Megatron backend.

Nemotron 3 Super is an 88-layer hybrid: Mamba2, MoE and attention layers
interleaved by ``hybrid_override_pattern``. That is not a GPTModel shape, so
this module does not return a layer spec the way the other plugins here do --
it returns a *model provider*, which slime's model_provider.py calls directly
(backends/megatron_utils/model_provider.py:146-153, the branch that checks for
a ``pre_process`` parameter). The model built is MCore's own
``HybridModel``; nothing about the architecture is reimplemented here.

That is possible because MCore 0.20 already has every piece:

    Symbols.MAMBA 'M' / Symbols.MOE 'E' / Symbols.ATTENTION '*'
        models/hybrid/layers/utils.py:16-33 -- exactly the three symbols
        Nemotron 3's pattern uses.
    moe_latent_size
        transformer/transformer_config.py:1001. Nemotron's experts run in a
        1024-wide latent space (fc1_latent_proj 4096->1024, expert
        1024->2688->1024, fc2_latent_proj 1024->4096), and MoELayer names those
        projections fc1_latent_proj / fc2_latent_proj -- the same names the HF
        checkpoint uses.
    squared_relu experts
        activations.py. Nemotron's `mlp_hidden_act: relu2` experts are
        non-gated, so gated_linear_unit is off and activation_func is
        squared_relu.

So the whole job here is config translation plus picking the stack spec. The
shapes themselves come from scripts/models/nemotron3-super-120B-A12B.sh, which
is what --spec is used alongside.

Usage (see that model config file):

    --spec slime_plugins.models.nemotron_h get_nemotron_h_model_provider
"""

from __future__ import annotations

import dataclasses

from megatron.core import mpu
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams


def mark_ungated_linear_fc1_sections(model) -> int:
    """Make slime shard Nemotron's ``linear_fc1`` as one contiguous block.

    slime's shared HF->Megatron sharder (hf_to_megatron/common.py) treats every
    ``linear_fc1`` as ``cat(gate, up)`` and interleaves the two halves per TP
    rank. Nemotron's ``relu2`` experts are not gated: ``linear_fc1`` is
    ``up_proj`` alone, so the interleave yields tensors of the right shape with
    the wrong rows. It is invisible for routed experts at ETP=1 (nothing is
    sharded) but real for the shared expert, which runs at TP>1.

    The sharder honours a per-parameter ``partition_sections`` ahead of that
    name-based rule (the same mechanism qwen3_5_native.py uses). One section
    spanning the full width is exactly a plain contiguous TP split, so tagging
    the parameter fixes the load path without touching shared code.

    The size is the *unsharded* width: local rows times the TP group the
    parameter is actually sharded over (ETP for experts, TP otherwise).
    Returns the number of parameters tagged.
    """
    tagged = 0
    for name, param in model.named_parameters():
        if "linear_fc1.weight" not in name and "linear_fc1.bias" not in name:
            continue
        if not getattr(param, "tensor_model_parallel", False):
            continue
        if ".experts." in name:
            parallel_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            parallel_size = mpu.get_tensor_model_parallel_world_size()
        if parallel_size == 1:
            continue
        if getattr(param, "partition_stride", 1) != 1:
            raise ValueError(
                f"{name}: partition_stride={param.partition_stride}, expected 1 for an ungated linear_fc1"
            )
        dim = getattr(param, "partition_dim", 0)
        param.partition_sections = (param.shape[dim] * parallel_size,)
        tagged += 1
    return tagged


def with_total_tokens(packed_seq_params: PackedSeqParams) -> PackedSeqParams:
    """Return packed_seq_params carrying ``total_tokens``, so MCore derives ``seq_idx``.

    Mamba2's conv1d and SSM scan reset their state at packed-sequence
    boundaries only when given ``seq_idx``; MCore computes it in
    ``PackedSeqParams.__post_init__`` from ``total_tokens``. slime's get_batch
    builds the params without it. The CPU copy of the boundaries is used when
    present, so this does not synchronise with the GPU.
    """
    if packed_seq_params.total_tokens is not None:
        return packed_seq_params
    cpu_offsets = getattr(packed_seq_params, "cu_seqlens_q_cpu", None)
    total = int(cpu_offsets[-1]) if cpu_offsets is not None else int(packed_seq_params.cu_seqlens_q[-1])
    return dataclasses.replace(packed_seq_params, total_tokens=total)


def get_nemotron_h_model_provider(args, config, vp_stage=None):
    """Return a model provider that builds MCore's HybridModel for Nemotron-H.

    slime calls this with (args, config, vp_stage) and then calls the returned
    function with pre_process / post_process / vp_stage.
    """

    # The pattern is the architecture. It has to come from the run's arguments
    # rather than be hardcoded, because it is what distinguishes the sizes in
    # this family from each other, and because MCore validates it against
    # num_layers (models/hybrid/hybrid_layer_allocation.py).
    hybrid_override_pattern = getattr(args, "hybrid_override_pattern", None)
    if not hybrid_override_pattern:
        raise ValueError(
            "Nemotron-H needs --hybrid-override-pattern. It is the 88-character "
            "M/E/* string from the HF config; scripts/models/"
            "nemotron3-super-120B-A12B.sh passes it."
        )

    expected = len(hybrid_override_pattern)
    if expected != args.num_layers:
        raise ValueError(
            f"--hybrid-override-pattern is {expected} characters but --num-layers "
            f"is {args.num_layers}; every layer needs a symbol."
        )

    def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None):
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            hybrid_override_pattern=hybrid_override_pattern,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            # Only the 8 attention layers use position embeddings at all; the
            # Mamba and MoE layers ignore this. HybridModel defaults to 'none'
            # because a pure-Mamba stack needs no positions, so it has to be
            # passed explicitly for a hybrid that does have attention.
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            vp_stage=vp_stage,
        )
        mark_ungated_linear_fc1_sections(model)
        return model

    return model_provider
