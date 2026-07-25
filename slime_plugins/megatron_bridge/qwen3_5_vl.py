"""Qwen3.5-VL support on top of the official Megatron Bridge providers.

Megatron Bridge ships ``Qwen35VLBridge`` / ``Qwen35VLMoEBridge`` (since v0.4.0)
and registers them on import, so no registration shim is needed here. What is
missing is the interaction with slime's data layout: the providers build their
Gated DeltaNet (GDN) layers from megatron-core's ``experimental_attention_variant
= "gated_delta_net"``, and megatron-core's ``GatedDeltaNet.forward`` rejects
packed sequences outright::

    if packed_seq_params is not None:
        raise NotImplementedError("GDN does not support packed sequence for now.")

Since slime removed BSHD support, every microbatch is packed THD, so the first
GDN layer raises and the bridge path is unusable for Qwen3.5-VL.

With ``--micro-batch-size 1`` and no dynamic batching (already required for GDN,
see examples/geo3k_vlm/README.md) a THD microbatch holds a single sequence
optionally followed by right padding, and the hidden states are ``[T, 1, H]`` --
exactly the layout the unpacked GDN path expects. GDN is causal, so trailing
padding cannot affect the outputs of the real tokens. We therefore subclass the
megatron-core module and drop the packed metadata after checking that we really
are in that regime.

The subclass adds no parameters and renames nothing, so every GDN weight mapping
in the official bridge (in_proj / conv1d / A_log / dt_bias / out_norm / out_proj)
keeps working for checkpoint load, ``--save-hf`` and weight sync to SGLang.

Not supported, and rejected loudly rather than silently miscomputed:
  * context parallel -- slime hands each rank a zigzag slice while cu_seqlens
    stays global, and GDN's recurrence cannot be split that way;
  * micro batch size > 1 and dynamic batching -- several real sequences in one
    microbatch would be fused into a single recurrent stream.

Lifting the last restriction means forwarding cu_seqlens into the varlen kernels
instead of dropping it, which is left to a follow-up.

Checked on Qwen3.5-2B with Megatron Bridge 0.5.0 and megatron-core 0.16.0rc0:
without this module the first GDN layer raises, with it a packed forward runs and
its logits agree with HuggingFace on 47/48 argmax positions (bf16, TP=1), with
every Megatron top-1 inside the HuggingFace top-5. Exporting back to HuggingFace
yields 621/632 tensors -- the 11 absent ones are MTP layers, which this
configuration does not build -- including all 162 GDN tensors, with a maximum
weight difference of 0.0039, i.e. one bf16 ulp.

Gradients were checked separately at TP=2 with sequence length 512: finite
throughout, including all 90 GDN parameters. That check needs the example's
``--attention-backend flash``; letting Megatron choose picks Transformer
Engine's cuDNN fused attention, whose backward goes non-finite for this model
under bf16 with packed sequences and TP above 1.
"""

import contextlib
import functools

from megatron.bridge.models import gpt_provider
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.qwen_vl import qwen35_vl_provider
from megatron.core import mpu
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.training import get_args


class SinglePackedSequenceGatedDeltaNet(GatedDeltaNet):
    """GDN that accepts slime's THD microbatches holding one real sequence."""

    def forward(self, hidden_states, attention_mask=None, *args, packed_seq_params=None, **kwargs):
        if packed_seq_params is None:
            return super().forward(hidden_states, attention_mask, *args, **kwargs)

        if mpu.get_context_parallel_world_size() != 1:
            raise NotImplementedError(
                "Qwen3.5-VL GDN does not support context parallel; use --context-parallel-size 1."
            )
        if packed_seq_params.qkv_format != "thd":
            raise NotImplementedError(f"Qwen3.5-VL GDN expects thd packing, got {packed_seq_params.qkv_format}.")
        if getattr(packed_seq_params, "cu_seqlens_q_padded", None) is not None:
            raise NotImplementedError("Qwen3.5-VL GDN does not support pre-padded cu_seqlens.")
        if hidden_states.shape[1] != 1:
            raise NotImplementedError(f"Qwen3.5-VL GDN expects a batch dimension of 1, got {hidden_states.shape[1]}.")

        total_tokens = hidden_states.shape[0] * self.sp_size
        if int(packed_seq_params.cu_seqlens_q[-1]) != total_tokens:
            raise RuntimeError(
                f"cu_seqlens[-1]={int(packed_seq_params.cu_seqlens_q[-1])} does not match "
                f"{total_tokens} tokens; the microbatch is not a single packed sequence."
            )

        return super().forward(hidden_states, attention_mask, *args, **kwargs)


def _patch_gated_delta_net_specs(block_spec) -> None:
    """Swap GDN modules in a block spec, mirroring the bridge's own spec patching.

    Standard attention layers are left alone; only layers whose self_attention is
    a megatron-core GatedDeltaNet are replaced.
    """
    if block_spec is None:
        return

    layer_specs = getattr(block_spec, "layer_specs", None)
    if layer_specs is not None:
        for layer_spec in layer_specs:
            _patch_gated_delta_net_specs(layer_spec)
        return

    submodules = getattr(block_spec, "submodules", None)
    if submodules is None:
        return

    if hasattr(submodules, "mtp_model_layer"):
        _patch_gated_delta_net_specs(submodules.mtp_model_layer)

    attention_spec = getattr(submodules, "self_attention", None)
    if attention_spec is None:
        return
    module = getattr(attention_spec, "module", None)
    if isinstance(module, type) and issubclass(module, GatedDeltaNet):
        attention_spec.module = SinglePackedSequenceGatedDeltaNet


def _check_single_sequence_microbatches() -> None:
    """Reject configurations that put more than one real sequence per microbatch.

    Unlike the checks in ``forward``, this cannot be detected from the packed
    metadata: a microbatch holding one sequence plus right padding and one
    holding two sequences both expose two cu_seqlens segments.
    """
    args = get_args()
    if getattr(args, "use_dynamic_batch_size", False):
        raise NotImplementedError(
            "Qwen3.5-VL GDN packs several sequences per microbatch under "
            "--use-dynamic-batch-size, which its recurrence cannot separate; drop the flag."
        )
    if getattr(args, "micro_batch_size", 1) != 1:
        raise NotImplementedError(f"Qwen3.5-VL GDN requires --micro-batch-size 1, got {args.micro_batch_size}.")


def _wrap_spec_builder(builder):
    """Wrap the provider's block spec builder so the GDN layers it emits get swapped."""
    if getattr(builder, "_slime_patches_gdn", False):
        return builder

    @functools.wraps(builder)
    def build_spec(*args, **kwargs):
        spec = builder(*args, **kwargs)
        _patch_gated_delta_net_specs(spec)
        return spec

    build_spec._slime_patches_gdn = True
    return build_spec


@contextlib.contextmanager
def _gdn_spec_patching():
    """Wrap the spec builders for the duration of one provide() call.

    The Qwen3.5-VL providers call the block spec builders as module globals rather
    than through ``self.transformer_layer_spec``, so they are wrapped in place and
    restored afterwards to keep other models untouched.
    """
    targets = []
    for module, name in (
        (qwen35_vl_provider, "get_transformer_block_with_experimental_attention_variant_spec"),
        (gpt_provider, "mtp_block_spec"),
    ):
        builder = getattr(module, name, None)
        if callable(builder):
            targets.append((module, name, builder))
            setattr(module, name, _wrap_spec_builder(builder))
    try:
        yield
    finally:
        for module, name, builder in targets:
            setattr(module, name, builder)


def _wrap_provide(provider_cls, method_name: str) -> None:
    original = getattr(provider_cls, method_name)

    def provide(self, *args, _original=original, **kwargs):
        _check_single_sequence_microbatches()
        # The language-model-only path goes through GPTModelProvider.provide, which
        # does read the builder off the instance.
        builder = self.transformer_layer_spec
        if callable(builder):
            self.transformer_layer_spec = _wrap_spec_builder(builder)
        else:
            _patch_gated_delta_net_specs(builder)
        with _gdn_spec_patching():
            return _original(self, *args, **kwargs)

    setattr(provider_cls, method_name, provide)


# AutoMapping dispatches on the exact module class name, so the subclass has to be
# registered the same way megatron-bridge registers GatedDeltaNet itself.
AutoMapping.register_module_type(SinglePackedSequenceGatedDeltaNet.__name__, "column")

for _provider_cls in (qwen35_vl_provider.Qwen35VLModelProvider, qwen35_vl_provider.Qwen35VLMoEModelProvider):
    for _method_name in ("provide", "provide_language_model"):
        _wrap_provide(_provider_cls, _method_name)
