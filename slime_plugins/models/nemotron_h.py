"""Nemotron-3 (HF ``model_type: nemotron_h``) as a ``--spec`` target.

Used from ``scripts/models/nemotron-3-super-120b-a12b.sh``::

    --spec "slime_plugins.models.nemotron_h" "get_nemotron_h_spec"

Two tokens, not a dotted path: ``--spec`` is ``nargs='*'`` and
``spec_utils.import_module()`` unpacks ``base_path, name = module_path``.

WHY THIS IS A MODEL PROVIDER AND NOT A LAYER SPEC
-------------------------------------------------
``slime/backends/megatron_utils/model_provider.py`` calls the ``--spec`` target
as ``spec(args, config, vp_stage)`` and then inspects the result: if it is itself
callable with a ``pre_process`` parameter, slime delegates model construction to
it and never reaches its ``GPTModel`` branch. ``qwen3_next`` cannot use that
path -- megatron.core has no ``Qwen3NextModel``, so it swaps a ``ModuleSpec``
*inside* ``GPTModel`` -- but Nemotron-3 is a hybrid Mamba-2 / attention /
latent-MoE stack for which megatron.core has a native model class, so returning
a provider is both shorter and more faithful.

There is no Nemotron model class in megatron.core either, and there does not
need to be: every Nemotron generation is the *generic* hybrid stack configured
by arguments. ``MambaStack`` allocates one of {``MambaLayer``,
``TransformerLayer``, ``MLPLayer``, ``MoETransformerLayer``} per character of
``--hybrid-override-pattern``, and megatron.core 0.16 is the first release where
``E`` (MoE) is a legal character
(``ssm/mamba_hybrid_layer_allocation.py::Symbols``). Everything Nemotron-3 Super
needs on top of that -- latent MoE, sigmoid routing with an expert bias, a
shared expert, squared-ReLU experts -- is ordinary ``TransformerConfig``, which
is why this file has no modules in it.

Verified against megatron.core 0.16.0rc0, NVIDIA/Megatron-LM ``1dcf0daf``.
"""

from __future__ import annotations

from megatron.core.models.mamba import MambaModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec


class NemotronHModel(MambaModel):
    """``MambaModel`` that tolerates the one keyword slime always sends.

    slime builds its forward kwargs unconditionally
    (``slime/backends/megatron_utils/model.py``) and ``loss_mask`` is always
    among them. ``GPTModel.forward`` declares it keyword-only;
    ``MambaModel.forward`` does not, and the string ``loss_mask`` does not occur
    anywhere in ``mamba_model.py``. Without this subclass the first microbatch
    dies with::

        TypeError: forward() got an unexpected keyword argument 'loss_mask'

    Dropping it loses nothing for this stack. In ``GPTModel`` the argument feeds
    exactly one consumer -- the multi-token-prediction block, entered alongside
    ``mtp_in_postprocess=self.mtp_process`` -- and a Mamba stack has no MTP block
    at all. If MTP is ever configured, silently discarding the mask would be
    wrong, so that case raises instead.

    The cleaner fix belongs upstream of here: slime could pass ``loss_mask`` only
    to models that declare it, which is one ``inspect.signature`` call at the
    call site and would let every non-GPT provider drop this subclass.
    """

    def forward(self, *args, loss_mask=None, **kwargs):
        if loss_mask is not None and getattr(self.config, "mtp_num_layers", None):
            raise ValueError(
                "nemotron_h: loss_mask was passed with mtp_num_layers set, but a Mamba stack "
                "has no MTP block to consume it."
            )
        return super().forward(*args, **kwargs)


def get_nemotron_h_spec(args, config, vp_stage: int | None = None):
    """Return a model provider for the Nemotron-3 hybrid stack.

    ``config`` arrives already built by slime, so unlike a standalone provider
    this does not call ``core_transformer_config_from_args`` itself -- that is
    the call that turns ``--squared-relu`` into ``config.activation_func``,
    ``--num-experts`` into ``config.num_moe_experts`` and the four ``--mamba-*``
    flags into the fields ``MambaMixer`` reads, and slime has already made it.
    """
    _assert_supported(args, config, vp_stage)

    def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None):
        _assert_no_virtual_pipeline(vp_stage)
        return NemotronHModel(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            # Ratios are the *other* way of describing the stack, for runs that
            # let Megatron place the layers. An explicit pattern wins, and
            # passing both is only legal when they agree, so these stay at their
            # argument defaults (0.0) and the pattern decides.
            hybrid_attention_ratio=args.hybrid_attention_ratio,
            hybrid_mlp_ratio=args.hybrid_mlp_ratio,
            hybrid_override_pattern=args.hybrid_override_pattern,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            # 'none' for Nemotron-3; the two rotary arguments are then unused and
            # are passed only so a RoPE variant of this file stays a flag change.
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
        )

    return model_provider


def _assert_no_virtual_pipeline(vp_stage: int | None) -> None:
    # MambaStack takes no vp_stage and megatron's own mamba_builder never passes
    # one, so a virtual-pipeline schedule would build the wrong number of layers
    # per chunk and surface only as a shape mismatch deep in the checkpoint load.
    # Refuse it where the message can still say why.
    if vp_stage is not None:
        raise ValueError(
            "nemotron_h: virtual pipeline parallelism is not supported by MambaStack; "
            "drop --num-layers-per-virtual-pipeline-stage."
        )


def _assert_supported(args, config, vp_stage: int | None) -> None:
    _assert_no_virtual_pipeline(vp_stage)

    # `is_hybrid_model` gates the correct activation-memory and layer-offset
    # arithmetic in megatron.training. The pattern being non-empty and the right
    # length is what makes this a Nemotron rather than a pure Mamba stack, and
    # getting either wrong produces a model that builds and trains and is simply
    # not the model the checkpoint holds.
    if not config.is_hybrid_model:
        raise ValueError("nemotron_h: --is-hybrid-model is required")
    if not args.hybrid_override_pattern:
        raise ValueError("nemotron_h: --hybrid-override-pattern is required")
    if len(args.hybrid_override_pattern) != args.num_layers:
        raise ValueError(
            f"nemotron_h: --hybrid-override-pattern has {len(args.hybrid_override_pattern)} "
            f"layers but --num-layers is {args.num_layers}"
        )
