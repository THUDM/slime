import inspect
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.tensor_parallel.layers import ColumnParallelLinear

_COMPACT_ACTOR_LOGITS = ContextVar("slime_compact_actor_logits", default=False)

try:
    _BASE_POSTPROCESS_SIGNATURE = inspect.signature(GPTModel._postprocess)
except (TypeError, ValueError):
    _BASE_POSTPROCESS_SIGNATURE = None

_CONFIG_LOGGER_ENABLED = getattr(GPTModel._postprocess, "__globals__", {}).get("has_config_logger_enabled")


def can_compact_actor_logits(args) -> bool:
    """Return whether compact logits preserve the selected built-in actor path."""
    if not getattr(args, "compact_actor_logits", False):
        return False
    if getattr(args, "loss_type", None) not in {"policy_loss", "sft_loss"}:
        return False

    unsupported_values = (
        getattr(args, "enable_mtp_training", False),
        getattr(args, "custom_model_provider_path", None),
        getattr(args, "custom_megatron_init_path", None),
        getattr(args, "custom_megatron_before_log_prob_hook_path", None),
        getattr(args, "custom_megatron_before_train_step_hook_path", None),
        getattr(args, "custom_advantage_function_path", None),
        getattr(args, "rollout_data_postprocess_path", None),
        getattr(args, "custom_pg_loss_reducer_function_path", None),
        getattr(args, "use_tis", False),
        getattr(args, "get_mismatch_metrics", False),
        getattr(args, "use_rollout_entropy", False),
        getattr(args, "save_debug_train_data", None),
        getattr(args, "use_rollout_logprobs", False),
    )
    if any(unsupported_values):
        return False

    return not (getattr(args, "advantage_estimator", None) == "ppo" and getattr(args, "kl_coef", 0) != 0)


@contextmanager
def compact_actor_logits(enabled: bool):
    """Enable compact actor logits for the current pipeline schedule call."""
    token = _COMPACT_ACTOR_LOGITS.set(bool(enabled))
    try:
        yield
    finally:
        _COMPACT_ACTOR_LOGITS.reset(token)


class CompactLogitsGPTModel(GPTModel):
    """Built-in actor model that can skip vocabulary projection for masked rows."""

    def _postprocess(self, *args, **kwargs):
        if not _COMPACT_ACTOR_LOGITS.get() or _BASE_POSTPROCESS_SIGNATURE is None:
            return super()._postprocess(*args, **kwargs)

        try:
            bound = _BASE_POSTPROCESS_SIGNATURE.bind(self, *args, **kwargs)
        except TypeError:
            return super()._postprocess(*args, **kwargs)
        bound.apply_defaults()
        call_args = bound.arguments

        if not getattr(self, "post_process", False):
            return super()._postprocess(*args, **kwargs)

        hidden_states = call_args.get("hidden_states")
        labels = call_args.get("labels")
        loss_mask = call_args.get("loss_mask")
        inference_params = call_args.get("inference_params")
        inference_context = call_args.get("inference_context")
        mtp_in_postprocess = call_args.get("mtp_in_postprocess")
        runtime_gather_output = call_args.get("runtime_gather_output")
        mtp_kwargs = call_args.get("mtp_kwargs")

        config = getattr(self, "config", None)
        output_layer = getattr(self, "output_layer", None)
        if config is None or not isinstance(output_layer, ColumnParallelLinear):
            return super()._postprocess(*args, **kwargs)
        if labels is not None or not isinstance(loss_mask, torch.Tensor):
            return super()._postprocess(*args, **kwargs)
        if inference_params is not None or inference_context is not None:
            return super()._postprocess(*args, **kwargs)
        if (
            mtp_in_postprocess
            or getattr(self, "mtp_process", False)
            or getattr(config, "mtp_num_layers", None)
            or mtp_kwargs
        ):
            return super()._postprocess(*args, **kwargs)
        if not callable(_CONFIG_LOGGER_ENABLED):
            return super()._postprocess(*args, **kwargs)
        try:
            if _CONFIG_LOGGER_ENABLED(config):
                return super()._postprocess(*args, **kwargs)
        except (AttributeError, TypeError):
            return super()._postprocess(*args, **kwargs)
        if getattr(config, "defer_embedding_wgrad_compute", False):
            return super()._postprocess(*args, **kwargs)
        if getattr(config, "cuda_graph_impl", "none") != "none":
            return super()._postprocess(*args, **kwargs)
        if not getattr(self, "parallel_output", False) or getattr(output_layer, "gather_output", True):
            return super()._postprocess(*args, **kwargs)
        if runtime_gather_output is not None and runtime_gather_output is not False:
            return super()._postprocess(*args, **kwargs)

        sequence_parallel = getattr(output_layer, "sequence_parallel", None)
        if not isinstance(sequence_parallel, bool):
            return super()._postprocess(*args, **kwargs)
        if sequence_parallel != bool(getattr(config, "sequence_parallel", False)):
            return super()._postprocess(*args, **kwargs)
        if sequence_parallel and getattr(output_layer, "allreduce_dgrad", False):
            return super()._postprocess(*args, **kwargs)
        if sequence_parallel and getattr(output_layer, "disable_grad_reduce", False):
            return super()._postprocess(*args, **kwargs)
        if sequence_parallel and getattr(output_layer, "explicit_expert_comm", False):
            return super()._postprocess(*args, **kwargs)

        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            return super()._postprocess(*args, **kwargs)
        if hidden_states.size(1) != 1 or loss_mask.ndim != 2 or loss_mask.size(0) != 1:
            return super()._postprocess(*args, **kwargs)
        if hidden_states.device != loss_mask.device:
            return super()._postprocess(*args, **kwargs)

        tp_group = getattr(output_layer, "tp_group", None)
        if tp_group is None or not callable(getattr(tp_group, "size", None)):
            return super()._postprocess(*args, **kwargs)
        tp_size = tp_group.size()
        if not isinstance(tp_size, int) or tp_size < 1:
            return super()._postprocess(*args, **kwargs)
        expected_sequence_length = hidden_states.size(0) * (tp_size if sequence_parallel else 1)
        if loss_mask.size(1) != expected_sequence_length:
            return super()._postprocess(*args, **kwargs)

        vocab_size_per_partition = getattr(output_layer, "output_size_per_partition", None)
        if not isinstance(vocab_size_per_partition, int) or vocab_size_per_partition < 1:
            return super()._postprocess(*args, **kwargs)

        output_weight = None
        if getattr(self, "share_embeddings_and_output_weights", False):
            shared_weight = getattr(self, "shared_embedding_or_output_weight", None)
            if not callable(shared_weight):
                return super()._postprocess(*args, **kwargs)
            output_weight = shared_weight()
        projection_weight = output_weight if output_weight is not None else getattr(output_layer, "weight", None)
        if not isinstance(projection_weight, torch.Tensor) or projection_weight.ndim != 2:
            return super()._postprocess(*args, **kwargs)
        if projection_weight.shape != (vocab_size_per_partition, hidden_states.size(2)):
            return super()._postprocess(*args, **kwargs)
        if projection_weight.device != hidden_states.device:
            return super()._postprocess(*args, **kwargs)

        if sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=tp_group,
            )

        selected_rows = loss_mask[0].to(dtype=torch.bool)
        compact_hidden_states = hidden_states[:, 0, :][selected_rows].unsqueeze(1)

        original_sequence_parallel = output_layer.sequence_parallel
        try:
            # The explicit gather above owns SP. The output layer's TP copy
            # still performs the required dgrad all-reduce.
            output_layer.sequence_parallel = False
            if compact_hidden_states.size(0) == 0:
                anchor = hidden_states.reshape(-1)[:1].sum() * 0
                anchor = anchor + projection_weight.reshape(-1)[:1].sum() * 0
                logits = hidden_states.new_empty((0, 1, vocab_size_per_partition)) + anchor
            else:
                logits, _ = output_layer(
                    compact_hidden_states,
                    weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                )
        finally:
            output_layer.sequence_parallel = original_sequence_parallel

        return logits.transpose(0, 1).contiguous()
