# Adapt from https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/pretrain_gpt.py
import argparse
import inspect
import re
from contextlib import nullcontext
from typing import Literal

import torch
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args

from slime.backends.megatron_utils.compact_logits import CompactLogitsGPTModel, can_compact_actor_logits
from slime.utils.misc import load_function

_INDEXER_DIRECT_SUBMODULE_NAMES = frozenset(
    {
        "wq_b",
        "wk",
        "k_norm",
        "weights_proj",
        "index_kpool_compress_ape",
        "index_kpool_compress_gate",
    }
)


def _is_indexer_parameter(name: str) -> bool:
    """Return whether *name* belongs to a DSA indexer.

    The GLM plugin exposes indexer projections directly under
    ``self_attention``. Megatron's upstream DSA implementation instead nests
    them under ``self_attention.core_attention.indexer``. Keep the ownership
    check structural so similarly named non-indexer projections stay trainable.
    """

    parts = name.split(".")
    for index, part in enumerate(parts):
        if part != "self_attention" or index + 1 >= len(parts):
            continue
        attention_parts = parts[index + 1 :]
        if attention_parts[0] in _INDEXER_DIRECT_SUBMODULE_NAMES:
            return True
        if "indexer" in attention_parts[:-1]:
            return True
    return False


# Adapt from https://github.com/volcengine/verl/blob/c3b20575d2bc815fcccd84bddb4c0401fc4b632b/verl/models/llama/megatron/layers/parallel_linear.py#L82
class LinearForLastLayer(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True
            if bias:
                self.bias.sequence_parallel = True

        init_method_std = getattr(config, "init_method_std", None)
        if init_method_std is None:
            init_method_std = 0.02
        self.weight.data.normal_(mean=0.0, std=init_method_std)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


def _get_model_provider_func(
    args: argparse.Namespace,
    role: Literal["actor", "critic"] = "actor",
):
    # Support custom model provider path (similar to --custom-rm-path for reward models)
    if getattr(args, "custom_model_provider_path", None):

        def wrapped_model_provider(
            pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None
        ) -> GPTModel:
            custom_model_provider = load_function(args.custom_model_provider_path)
            # Check if the custom provider supports vp_stage parameter
            has_vp_stage = "vp_stage" in inspect.signature(custom_model_provider).parameters
            if has_vp_stage:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            else:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process)
            # Apply critic output layer if needed
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(
                    input_size=model.config.hidden_size, output_size=1, config=model.config
                )
            return model

        return wrapped_model_provider

    def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None) -> GPTModel:
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        use_te = args.transformer_impl == "transformer_engine"

        # Experimental loading arguments from yaml
        config: TransformerConfig = core_transformer_config_from_args(args)
        # Older GLM Megatron forks consumed this flag from TransformerConfig.
        # Preserve that contract for custom specs, while freeze_model_params()
        # below provides the concrete implementation on current Megatron.
        config.freeze_indexer = getattr(args, "freeze_indexer", False)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
            # Allow the spec to be a function so that user can use customized Megatron easier.
            if callable(transformer_layer_spec):
                result = transformer_layer_spec(args, config, vp_stage)
                # If the result is itself a model provider (callable with pre_process param),
                # delegate model construction to it directly (e.g. glm-omni VL model).
                if callable(result) and "pre_process" in inspect.signature(result).parameters:
                    model = result(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
                    if post_process and role == "critic":
                        model.output_layer = LinearForLastLayer(
                            input_size=config.hidden_size, output_size=1, config=config
                        )
                    return model
                transformer_layer_spec = result
        else:
            if args.num_experts:
                # Define the decoder block spec
                kwargs = {
                    "use_transformer_engine": use_te,
                }
                if vp_stage is not None:
                    kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except Exception as e:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found."
                ) from e

        kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
        }

        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage

        if args.mtp_num_layers:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

            mtp_kwargs = {
                "use_transformer_engine": use_te,
            }
            if vp_stage is not None:
                mtp_kwargs["vp_stage"] = vp_stage

            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
            kwargs["mtp_block_spec"] = mtp_block_spec

        model_cls = (
            CompactLogitsGPTModel if role == "actor" and post_process and can_compact_actor_logits(args) else GPTModel
        )
        with build_model_context(**build_model_context_args):
            model = model_cls(**kwargs)

        if post_process and role == "critic":
            model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)

        return model

    return model_provider


def wrap_model_provider_with_freeze(original_provider, args):
    def wrapped_provider(
        pre_process=True,
        post_process=True,
        **kwargs,
    ):
        sig = inspect.signature(original_provider)
        provider_kwargs = {
            "pre_process": pre_process,
            "post_process": post_process,
        }
        for key in ["vp_stage", "config", "pg_collection"]:
            if key in sig.parameters:
                provider_kwargs[key] = kwargs.get(key, None)

        model = original_provider(**provider_kwargs)
        freeze_model_params(model, args)

        return model

    return wrapped_provider


def get_model_provider_func(args, role="actor"):
    return wrap_model_provider_with_freeze(_get_model_provider_func(args, role), args)


def freeze_model_params(model: GPTModel, args: argparse.Namespace):
    if getattr(args, "only_train_params_name_list", None):
        for name, param in model.named_parameters():
            param.requires_grad = False
            for pattern in args.only_train_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = True
                    break

    if getattr(args, "freeze_params_name_list", None):
        for name, param in model.named_parameters():
            for pattern in args.freeze_params_name_list:
                if re.search(pattern, name):
                    param.requires_grad = False
                    break

    if getattr(args, "freeze_indexer", False):
        frozen_indexer_params = []
        has_self_attention_params = False
        for name, param in model.named_parameters():
            has_self_attention_params |= "self_attention" in name.split(".")
            if _is_indexer_parameter(name):
                param.requires_grad = False
                frozen_indexer_params.append(name)

        if has_self_attention_params and not frozen_indexer_params:
            raise RuntimeError(
                "--freeze-indexer was requested, but this model chunk has self-attention "
                "parameters and no recognized DSA indexer parameters."
            )

        # Some pipeline stages may legitimately own no indexer weights, so an
        # empty local tuple is not itself an error.
        model._slime_frozen_indexer_param_names = tuple(frozen_indexer_params)
