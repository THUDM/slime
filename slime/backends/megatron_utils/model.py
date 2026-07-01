import dataclasses
import gc
import logging
import math
import os
from argparse import Namespace
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model
from tqdm import tqdm

try:
    from megatron.core.pipeline_parallel.utils import unwrap_model
except ImportError:
    from megatron.core.utils import unwrap_model
from slime.utils import logging_utils
from slime.utils.memory_utils import clear_memory

from .checkpoint import load_checkpoint, save_checkpoint
from .cp_utils import reduce_train_step_metrics
from .data import (
    DataIterator,
    get_batch,
    has_multimodal_train_inputs,
    qwen_vl_text_fastpath_requires_unsplit_input,
    qwen_vl_unsplit_only_with_mm,
)
from .initialize import get_use_gloo_process_groups
from .loss import _build_shifted_tokens, loss_function, set_deferred_lm_head_weight, sft_precomputed_loss_function
from .model_provider import get_model_provider_func

logger = logging.getLogger(__name__)


_DEFER_ACTIVE = {"on": False}
_QWENVL_FORWARD_DEBUG_COUNT = 0


def _qwen_vl_external_sft_loss() -> bool:
    return os.environ.get("SLIME_QWENVL_EXTERNAL_SFT_LOSS", "0").lower() in {"1", "true", "yes", "on"}


def _qwen_vl_text_language_fastpath() -> bool:
    return os.environ.get("SLIME_QWENVL_TEXT_LANGUAGE_FASTPATH", "0").lower() in {"1", "true", "yes", "on"}


def _qwen_vl_text_language_fastpath_safe() -> bool:
    return _qwen_vl_text_language_fastpath() and not qwen_vl_text_fastpath_requires_unsplit_input()


def _qwen_vl_text_fastpath_local_mrope() -> bool:
    return os.environ.get("SLIME_QWENVL_TEXT_FASTPATH_LOCAL_MROPE", "1").lower() in {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, os.environ.get(name.lower(), "0")).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, os.environ.get(name.lower(), str(default))))
    except ValueError:
        return default


def _distributed_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _debug_tensor_summary(value):
    if torch.is_tensor(value):
        return {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
            "numel": value.numel(),
        }
    return None


def _debug_packed_seq_params(value):
    if type(value).__name__ != "PackedSeqParams":
        return None
    summary = {
        "qkv_format": getattr(value, "qkv_format", None),
        "max_seqlen_q": getattr(value, "max_seqlen_q", None),
        "max_seqlen_kv": getattr(value, "max_seqlen_kv", None),
    }
    for name in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
        tensor = getattr(value, name, None)
        if not torch.is_tensor(tensor):
            summary[name] = None
            continue
        vals = tensor.detach().cpu()
        diffs = vals[1:] - vals[:-1] if vals.numel() > 1 else vals.new_empty((0,))
        summary[name] = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "device": str(tensor.device),
            "head": vals[:8].tolist(),
            "tail": vals[-8:].tolist(),
            "diff_min": int(diffs.min().item()) if diffs.numel() else None,
            "diff_max": int(diffs.max().item()) if diffs.numel() else None,
        }
    return summary


def _debug_value_summary(value, depth: int = 0):
    if depth > 2:
        return type(value).__name__
    tensor_summary = _debug_tensor_summary(value)
    if tensor_summary is not None:
        return tensor_summary
    packed_summary = _debug_packed_seq_params(value)
    if packed_summary is not None:
        return {"type": "PackedSeqParams", **packed_summary}
    if isinstance(value, dict):
        return {str(key): _debug_value_summary(val, depth + 1) for key, val in list(value.items())[:12]}
    if isinstance(value, (list, tuple)):
        return [_debug_value_summary(item, depth + 1) for item in value[:4]]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


def _maybe_log_qwenvl_forward_debug(
    args: Namespace,
    batch: dict,
    forward_kwargs: dict,
    *,
    has_mm_inputs: bool,
    needs_unsplit: bool,
    use_unsplit: bool,
    use_precomputed_sft_loss: bool = False,
) -> None:
    global _QWENVL_FORWARD_DEBUG_COUNT

    if not _env_flag("SLIME_QWENVL_FORWARD_SHAPE_DEBUG"):
        return

    rank_filter = _env_int("SLIME_QWENVL_FORWARD_SHAPE_DEBUG_RANK", 0)
    rank = _distributed_rank()
    if rank_filter >= 0 and rank != rank_filter:
        return

    limit = _env_int("SLIME_QWENVL_FORWARD_SHAPE_DEBUG_LIMIT", 16)
    if _QWENVL_FORWARD_DEBUG_COUNT >= limit:
        return
    _QWENVL_FORWARD_DEBUG_COUNT += 1

    try:
        logger.info(
            "QwenVL forward shape debug: sample=%s rank=%s has_mm_inputs=%s "
            "uses_unsplit_forward=%s qwen_vl_unsplit_only_with_mm=%s needs_unsplit=%s "
            "use_unsplit=%s use_precomputed_sft_loss=%s qkv_format=%s allgather_cp=%s "
            "input_ids=%s position_ids=%s attention_mask=%s loss_mask=%s labels=%s "
            "packed_seq_params=%s batch_packed_seq_params=%s vlm_packed_seq_params=%s "
            "tokens=%s unsplit_tokens=%s full_loss_masks=%s multimodal_train_inputs=%s "
            "max_seq_lens=%s padded_total_lengths=%s",
            _QWENVL_FORWARD_DEBUG_COUNT,
            rank,
            has_mm_inputs,
            getattr(args, "uses_unsplit_forward", False),
            qwen_vl_unsplit_only_with_mm(),
            needs_unsplit,
            use_unsplit,
            use_precomputed_sft_loss,
            getattr(args, "qkv_format", None),
            getattr(args, "allgather_cp", None),
            _debug_value_summary(forward_kwargs.get("input_ids")),
            _debug_value_summary(forward_kwargs.get("position_ids")),
            _debug_value_summary(forward_kwargs.get("attention_mask")),
            _debug_value_summary(forward_kwargs.get("loss_mask")),
            _debug_value_summary(forward_kwargs.get("labels")),
            _debug_value_summary(forward_kwargs.get("packed_seq_params")),
            _debug_value_summary(batch.get("packed_seq_params")),
            _debug_value_summary(batch.get("vlm_packed_seq_params")),
            _debug_value_summary(batch.get("tokens")),
            _debug_value_summary(batch.get("unsplit_tokens")),
            _debug_value_summary(batch.get("full_loss_masks")),
            _debug_value_summary(batch.get("multimodal_train_inputs")),
            _debug_value_summary(batch.get("max_seq_lens")),
            _debug_value_summary(batch.get("padded_total_lengths")),
        )
    except Exception:
        logger.exception("Failed to log QwenVL forward shape debug")


def _unwrap_to_core_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _ensure_deferred_output_layer(model) -> bool:
    import types

    core = _unwrap_to_core_model(model)
    output_layer = getattr(core, "output_layer", None)
    if output_layer is None or getattr(output_layer, "weight", None) is None:
        return False

    set_deferred_lm_head_weight(output_layer.weight)
    if not getattr(output_layer, "_slime_deferred", False):
        output_layer._slime_orig_forward = output_layer.forward

        def _deferred_forward(self, input_, *args, **kwargs):
            if _DEFER_ACTIVE["on"]:
                return input_, None
            return self._slime_orig_forward(input_, *args, **kwargs)

        output_layer.forward = types.MethodType(_deferred_forward, output_layer)
        output_layer._slime_deferred = True
    return True


def _disable_tqdm_for_non_main_rank() -> bool:
    return not (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
    )


def _should_update_microbatch_pbar(model) -> bool:
    if _disable_tqdm_for_non_main_rank():
        return False

    while hasattr(model, "module"):
        model = model.module
    vp_stage = getattr(model, "vp_stage", None)
    if mpu.get_virtual_pipeline_model_parallel_world_size() is not None and vp_stage is not None:
        return mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
    return mpu.is_pipeline_last_stage(ignore_virtual=True)


def _wrap_forward_step_with_microbatch_pbar(forward_step_func, pbar):
    if pbar is None:
        return forward_step_func

    def wrapped_forward_step(*args, **kwargs):
        result = forward_step_func(*args, **kwargs)
        model = args[1] if len(args) > 1 else kwargs.get("model")
        if model is not None and _should_update_microbatch_pbar(model):
            pbar.update(1)
        return result

    return wrapped_forward_step


def _iter_critic_output_layers(model: Sequence[DDP]):
    for chunk_id, module in enumerate(unwrap_model(model)):
        output_layer = getattr(module, "output_layer", None)
        if output_layer is not None:
            yield chunk_id, output_layer


def _critic_output_layer_needs_reinit(args: Namespace, model: Sequence[DDP], role: str) -> bool:
    if role != "critic" or args.load is None:
        return False

    from megatron.core.dist_checkpointing.serialization import load_tensors_metadata
    from megatron.training.checkpointing import get_load_checkpoint_path_by_args

    checkpoint_path = Path(get_load_checkpoint_path_by_args(args))
    if not (checkpoint_path / ".metadata").is_file():
        return False

    checkpoint_metadata = load_tensors_metadata(str(checkpoint_path))
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        for name in ("weight", "bias"):
            param = getattr(output_layer, name, None)
            if param is None:
                continue

            param_name = f"output_layer.{name}"
            ckpt_tensor_metadata = next(
                (
                    tensor_metadata
                    for key, tensor_metadata in checkpoint_metadata.items()
                    if key == param_name or key.endswith(f".{param_name}")
                ),
                None,
            )
            expected_shape = tuple(param.shape)
            checkpoint_shape = tuple(ckpt_tensor_metadata.global_shape) if ckpt_tensor_metadata is not None else None
            if checkpoint_shape == expected_shape:
                continue

            reason = (
                "missing from checkpoint metadata"
                if checkpoint_shape is None
                else f"shape mismatch checkpoint={checkpoint_shape} runtime={expected_shape}"
            )
            logger.warning(
                "Will reinitialize critic %s after checkpoint load because it is %s",
                param_name,
                reason,
            )
            return True

    return False


@torch.no_grad()
def _reinitialize_critic_output_layer(model: Sequence[DDP]) -> None:
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        output_layer.weight.data.normal_(mean=0.0, std=0.02)
        if output_layer.bias is not None:
            output_layer.bias.data.zero_()


def get_optimizer_param_scheduler(args: Namespace, optimizer: MegatronOptimizer) -> OptimizerParamScheduler:
    """Create and configure the optimizer learning-rate/weight-decay scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        optimizer (MegatronOptimizer): Megatron optimizer bound to the model.

    Returns:
        OptimizerParamScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training. ``train_iters`` is an estimate of the total
    # number of training steps — it's only used to size Megatron's LR decay
    # schedule (and ``lr_decay_iters`` defaults to it). With variable per-rollout
    # sample counts (dynamic sampling / filtering / custom step splitter) the
    # *actual* total can drift; the schedule still tracks the true progress via
    # ``opt_param_scheduler.num_steps`` (samples consumed, also persisted across
    # resume), so the worst case is the cosine/linear schedule reaches its
    # plateau slightly early or late. Pass ``--lr-decay-iters`` explicitly if you
    # need exact decay control.
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def setup_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
    """Build model(s), wrap with DDP, and construct optimizer and scheduler.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        role (str): Logical role of the model (e.g., "actor", "critic").
        no_wd_decay_cond (Callable[..., bool] | None): Predicate to exclude
            parameters from weight decay.
        scale_lr_cond (Callable[..., bool] | None): Predicate to scale LR for
            selected parameter groups.
        lr_mult (float): Global learning-rate multiplier for the optimizer.

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
            - List of model chunks wrapped by ``DDP``.
            - The constructed ``MegatronOptimizer`` instance.
            - The learning-rate/weight-decay scheduler tied to the optimizer.
    """
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)

    # Optimizer
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None

    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=get_use_gloo_process_groups(args),
    )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return model, optimizer, opt_param_scheduler


def enable_forward_pre_hook(model_chunks: Sequence[DDP]) -> None:
    """Enable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to enable hooks on.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks: Sequence[DDP], param_sync: bool = True) -> None:
    """Disable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to disable hooks on.
        param_sync (bool): Whether to synchronize parameters when disabling.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


@torch.no_grad()
def forward_only(
    f: Callable[..., dict[str, list[torch.Tensor]]],
    args: Namespace,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    store_prefix: str = "",
) -> dict[str, list[torch.Tensor]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs).

    The model is put into evaluation mode, a forward-only pipeline pass is
    executed, and relevant outputs are aggregated and returned.

    Args:
        f (Callable[..., dict[str, list[torch.Tensor]]]): Post-forward callback used to
            compute and package outputs to collect. This should accept a logits
            tensor as its first positional argument and additional keyword-only
            arguments; see ``get_log_probs_and_entropy``/``get_values`` in
            ``megatron_utils.loss`` for examples. It will be partially applied
            so that the callable returned from the internal forward step only
            requires the logits tensor.
        args (Namespace): Runtime arguments.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding batches for inference.
        num_microbatches (Sequence[int]): Number of microbatches per rollout step.
        store_prefix (str): Prefix to prepend to stored output keys.

    Returns:
        dict[str, list[torch.Tensor]]: Aggregated outputs keyed by ``store_prefix + key``.
    """

    # reset data iterator
    for iterator in data_iterator:
        iterator.reset()

    config = get_model_config(model[0])

    def forward_step(
        data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
        """Forward step used by Megatron's pipeline engine.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
            Output tensor(s) and a callable that computes and packages results
            to be collected by the engine.
        """

        assert not return_schedule_plan, "forward_only step should never return schedule plan"

        # Get the batch.
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "loss_masks",
                "multimodal_train_inputs",
                "total_lengths",
                "response_lengths",
                "max_seq_lens",
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = batch["packed_seq_params"]
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]

        has_mm_inputs = has_multimodal_train_inputs(batch.get("multimodal_train_inputs", None))
        uses_unsplit_forward = getattr(args, "uses_unsplit_forward", False)
        needs_unsplit = has_mm_inputs or (
            uses_unsplit_forward
            and (not qwen_vl_unsplit_only_with_mm() or qwen_vl_text_fastpath_requires_unsplit_input())
        )
        text_language_fastpath = (
            uses_unsplit_forward
            and not needs_unsplit
            and _qwen_vl_text_language_fastpath_safe()
            and batch.get("packed_seq_params") is not None
        )
        mm_kwargs = batch["multimodal_train_inputs"] if has_mm_inputs else {}
        use_unsplit = needs_unsplit and "unsplit_tokens" in batch

        position_ids = None
        if getattr(args, "position_embedding_type", "rope") == "mrope" and not use_unsplit:
            from .mrope_utils import build_mrope_position_ids

            position_ids = build_mrope_position_ids(
                batch,
                local_thd_cp=text_language_fastpath and _qwen_vl_text_fastpath_local_mrope(),
            )

        forward_kwargs = {
            "input_ids": batch["unsplit_tokens"] if use_unsplit else tokens,
            "position_ids": position_ids,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": None if use_unsplit else packed_seq_params,
            "loss_mask": batch["full_loss_masks"],
        }

        if needs_unsplit and "vlm_packed_seq_params" in batch:
            forward_kwargs["attention_mask"] = batch["unsplit_attention_mask"]
            forward_kwargs["packed_seq_params"] = batch["vlm_packed_seq_params"]
            forward_kwargs["loss_mask"] = None

        if has_mm_inputs:
            forward_kwargs.update(mm_kwargs)

        _maybe_log_qwenvl_forward_debug(
            args,
            batch,
            forward_kwargs,
            has_mm_inputs=has_mm_inputs,
            needs_unsplit=needs_unsplit,
            use_unsplit=use_unsplit,
        )
        output_tensor = model(**forward_kwargs)

        return output_tensor, partial(
            f,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
            padded_total_lengths=batch.get("padded_total_lengths", None),
        )

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.custom_megatron_before_log_prob_hook_path:
        from slime.utils.misc import load_function

        custom_before_log_prob_hook = load_function(args.custom_megatron_before_log_prob_hook_path)
        custom_before_log_prob_hook(args, model, store_prefix)

    forward_backward_func = get_forward_backward_func()
    # Don't care about timing during evaluation
    config.timers = None
    forward_data_store = []
    num_steps_per_rollout = len(num_microbatches)
    microbatch_pbar = tqdm(
        total=sum(num_microbatches),
        desc=f"{(store_prefix or getattr(model[0], 'role', 'actor')).rstrip('_')} forward",
        unit="microbatch",
        dynamic_ncols=True,
        leave=False,
        disable=_disable_tqdm_for_non_main_rank(),
    )
    forward_step_with_progress = _wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar)
    for step_id in range(num_steps_per_rollout):
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step_with_progress,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
        )
    microbatch_pbar.close()

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    rollout_data = {}
    # Store the results on the last stage
    if mpu.is_pipeline_last_stage():
        keys = forward_data_store[0].keys()
        for key in keys:
            values = []
            for value in forward_data_store:
                assert isinstance(value[key], list)
                values += value[key]

            if args.use_dynamic_batch_size:
                # TODO: This is ugly... Find a better way to make the data have the same order.
                # TODO: move this out of the loop.
                origin_values = [None] * len(values)
                origin_indices = sum(data_iterator[0].micro_batch_indices, [])
                for value, origin_index in zip(values, origin_indices, strict=False):
                    origin_values[origin_index] = value
                values = origin_values
            rollout_data[f"{store_prefix}{key}"] = values
    return rollout_data


def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_microbatches: int,
    step_global_batch_size: int,
    microbatch_pbar=None,
) -> tuple[dict[str, float], float]:
    """Execute a single pipeline-parallel training step.

    Runs forward/backward over ``num_microbatches``, applies optimizer step and
    one scheduler step when gradients are valid.

    Args:
        args (Namespace): Runtime arguments.
        rollout_id (int): Rollout identifier.
        step_id (int): Step index within the current rollout.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        num_microbatches (int): Number of microbatches to process.
        step_global_batch_size (int): Rollout count for this training step
            (total across DP; one "rollout" = one execution of one of the
            ``n_samples_per_prompt`` rollouts, which may emit >1 training
            sample under compact / subagent). Used both as the loss
            normalizer inside the closure and as the LR scheduler
            ``increment``. In the common case (1 rollout = 1 sample) this
            equals the per-step sample count, so behavior is unchanged.

    Returns:
        tuple[dict[str, float], float]: Reduced loss dictionary (last stage only)
        and gradient norm for logging.
    """
    args = get_args()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from slime.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    def forward_step(data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False) -> tuple[
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]],
    ]:
        """Forward step used by Megatron's pipeline engine during training.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]]]:
            Output tensor(s) and the loss function, which returns
            (loss, num_elems, {"keys": list[str], "values": torch.Tensor}).
        """

        # Get the batch.
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "multimodal_train_inputs",
                "packed_seq_params",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "log_probs",
                "ref_log_probs",
                "values",
                "advantages",
                "returns",
                "rollout_log_probs",
                "max_seq_lens",
                "teacher_log_probs",
                "rollout_mask_sums",
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            old_stage = os.environ["ROUTING_REPLAY_STAGE"]
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

        use_precomputed_sft_loss = False

        if return_schedule_plan:
            assert not args.enable_mtp_training, "MTP training should not be enabled when using combined 1f1b"
            position_ids = None
            if getattr(args, "position_embedding_type", "rope") == "mrope":
                from .mrope_utils import build_mrope_position_ids

                position_ids = build_mrope_position_ids(batch)

            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=position_ids,
                attention_mask=None,
                labels=None,
                packed_seq_params=batch["packed_seq_params"],
                loss_mask=batch["full_loss_masks"],
            )
        else:
            has_mm_inputs = has_multimodal_train_inputs(batch.get("multimodal_train_inputs", None))
            uses_unsplit_forward = getattr(args, "uses_unsplit_forward", False)
            needs_unsplit = has_mm_inputs or (
                uses_unsplit_forward
                and (not qwen_vl_unsplit_only_with_mm() or qwen_vl_text_fastpath_requires_unsplit_input())
            )
            text_language_fastpath = (
                uses_unsplit_forward
                and not needs_unsplit
                and _qwen_vl_text_language_fastpath_safe()
                and batch.get("packed_seq_params") is not None
            )
            use_unsplit = needs_unsplit and "unsplit_tokens" in batch
            use_precomputed_sft_loss = (
                args.loss_type == "sft_loss"
                and (needs_unsplit or text_language_fastpath)
                and getattr(args, "calculate_per_token_loss", False)
                and not _qwen_vl_external_sft_loss()
            )

            position_ids = None
            if getattr(args, "position_embedding_type", "rope") == "mrope" and not use_unsplit:
                from .mrope_utils import build_mrope_position_ids

                position_ids = build_mrope_position_ids(
                    batch,
                    local_thd_cp=text_language_fastpath and _qwen_vl_text_fastpath_local_mrope(),
                )

            forward_kwargs = {
                "input_ids": batch["unsplit_tokens"] if use_unsplit else batch["tokens"],
                "position_ids": position_ids,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": None if use_unsplit else batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
            }

            if needs_unsplit and "vlm_packed_seq_params" in batch:
                forward_kwargs["attention_mask"] = batch["unsplit_attention_mask"]
                forward_kwargs["packed_seq_params"] = batch["vlm_packed_seq_params"]
                if not use_precomputed_sft_loss:
                    forward_kwargs["loss_mask"] = None

            if use_precomputed_sft_loss:
                shifted_labels = _build_shifted_tokens(
                    batch["tokens"].numel(),
                    batch["tokens"].device,
                    batch["unconcat_tokens"],
                    batch["total_lengths"],
                    batch["response_lengths"],
                    args.qkv_format,
                    batch.get("max_seq_lens", None),
                    args.allgather_cp,
                    batch.get("padded_total_lengths", None),
                ).view_as(batch["tokens"])
                forward_kwargs["labels"] = shifted_labels.masked_fill(
                    batch["full_loss_masks"].view_as(shifted_labels).le(0),
                    -100,
                )
                forward_kwargs["loss_mask"] = batch["full_loss_masks"]

            if args.enable_mtp_training:
                forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}

            if has_mm_inputs:
                forward_kwargs.update(batch["multimodal_train_inputs"])

            _maybe_log_qwenvl_forward_debug(
                args,
                batch,
                forward_kwargs,
                has_mm_inputs=has_mm_inputs,
                needs_unsplit=needs_unsplit,
                use_unsplit=use_unsplit,
                use_precomputed_sft_loss=use_precomputed_sft_loss,
            )
            if (
                os.environ.get("CHUNKED_LM_HEAD", "0") == "1"
                and args.loss_type == "sft_loss"
                and not use_precomputed_sft_loss
                and _ensure_deferred_output_layer(model)
            ):
                _DEFER_ACTIVE["on"] = True
                try:
                    output_tensor = model(**forward_kwargs)
                finally:
                    _DEFER_ACTIVE["on"] = False
            else:
                output_tensor = model(**forward_kwargs)

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            os.environ["ROUTING_REPLAY_STAGE"] = old_stage

        if not return_schedule_plan and use_precomputed_sft_loss:
            loss_func = sft_precomputed_loss_function
        else:
            loss_func = loss_function
        return output_tensor, partial(loss_func, args, batch, num_microbatches, step_global_batch_size)

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=_wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar),
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    valid_step = True
    grad_norm = float("nan")
    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    # CI check: verify only MTP parameters have non-zero gradients when truncation happens
    # This check must happen before optimizer.step() as gradients may be modified during step
    if args.ci_test and args.enable_mtp_training:
        from slime.backends.megatron_utils.ci_utils import check_mtp_only_grad

        check_mtp_only_grad(model, step_id)

    if valid_step:
        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        # Update learning rate. Use the per-step global_batch_size when dynamic
        # batching is on so the scheduler's samples-seen counter tracks reality.
        assert update_successful
        opt_param_scheduler.step(increment=step_global_batch_size)

    # release grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        loss_reduced = reduce_train_step_metrics(
            losses_reduced,
            calculate_per_token_loss=args.calculate_per_token_loss,
            step_global_batch_size=step_global_batch_size,
            cp_size=mpu.get_context_parallel_world_size(),
            dp_with_cp_group=mpu.get_data_parallel_group(with_context_parallel=True),
        )
        return loss_reduced, grad_norm
    return {}, grad_norm


def should_disable_forward_pre_hook(args: Namespace) -> bool:
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    global_batch_sizes: Sequence[int],
) -> None:
    """Run training over a rollout consisting of multiple steps.

    The model is switched to train mode, training hooks are configured, and
    ``train_one_step`` is invoked for each step in the rollout.

    Args:
        rollout_id (int): Rollout identifier.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        num_microbatches (Sequence[int]): Microbatches per step in the rollout.
        global_batch_sizes (Sequence[int]): Rollout count per step (total
            across DP; one "rollout" = one execution of one of the
            ``n_samples_per_prompt`` rollouts of a prompt). Same length as
            ``num_microbatches``; consumed by ``train_one_step`` for loss
            scaling and LR scheduler increments. Equals per-step sample count
            in the common case (1 rollout = 1 sample).
    """
    args = get_args()

    assert len(num_microbatches) == len(global_batch_sizes), (
        f"num_microbatches and global_batch_sizes must have the same length, "
        f"got {len(num_microbatches)} vs {len(global_batch_sizes)}"
    )

    for iterator in data_iterator:
        iterator.reset()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Setup some training config params.
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    pre_hook_enabled = False

    if args.reset_optimizer_states:
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            print("Reset optimizer states")
        for chained_optimizer in optimizer.chained_optimizers:
            for group in chained_optimizer.optimizer.param_groups:
                if "step" in group:
                    group["step"] = 0
            for state in chained_optimizer.optimizer.state.values():
                if "step" in state:
                    if isinstance(state["step"], torch.Tensor):
                        state["step"].zero_()
                    else:
                        state["step"] = 0
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    num_steps_per_rollout = len(num_microbatches)
    microbatch_pbar = tqdm(
        total=sum(num_microbatches),
        desc=f"{getattr(model[0], 'role', 'actor')} train",
        unit="microbatch",
        dynamic_ncols=True,
        leave=False,
        disable=_disable_tqdm_for_non_main_rank(),
    )

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):

        # Run training step.
        loss_dict, grad_norm = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
            global_batch_sizes[step_id],
            microbatch_pbar=microbatch_pbar,
        )

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        if args.enable_mtp_training:
            from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

            mtp_loss_scale = 1 / num_microbatches[step_id]
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                values = tracker["values"]
                if tracker.get("reduce_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
                if tracker.get("avg_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG)
                # here we assume only one mtp layer
                mtp_losses = (tracker["values"] * mtp_loss_scale).item()
                MTPLossLoggingHelper.clean_loss_in_tracker()

                # CI check: verify MTP loss is within expected bounds
                if args.ci_test:
                    from slime.backends.megatron_utils.ci_utils import check_mtp_loss

                    check_mtp_loss(mtp_losses)

        # per train step log.
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            role = getattr(model[0], "role", "actor")
            role_tag = "" if role == "actor" else f"{role}-"
            log_dict = {
                f"train/{role_tag}{key}": val.mean().item() if isinstance(val, torch.Tensor) else val
                for key, val in loss_dict.items()
            }
            log_dict[f"train/{role_tag}grad_norm"] = grad_norm
            if args.enable_mtp_training:
                log_dict[f"train/{role_tag}mtp_loss"] = mtp_losses

            for param_group_id, param_group in enumerate(optimizer.param_groups):
                log_dict[f"train/{role_tag}lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

            # Per-step gbs — uneven step sizes are easy to miss without this.
            log_dict[f"train/{role_tag}global_batch_size"] = global_batch_sizes[step_id]
            log_dict["train/step"] = accumulated_step_id
            logging_utils.log(args, log_dict, step_key="train/step")

            if args.ci_test and "train/train_rollout_logprob_abs_diff" in log_dict:
                assert log_dict["train/train_rollout_logprob_abs_diff"] <= 0.1, f"{log_dict=}"

            if args.ci_test and not args.ci_disable_kl_checker:
                if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
                    # TODO: figure out why KL is not exactly zero when using PPO loss with KL clipping, and whether this is expected behavior or a bug.
                    assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
                # R3 replays rollout routing for the actor path, while ref
                # log-probs are computed with normal routing. The initial
                # actor/ref KL is therefore not expected to be exactly zero.
                if (
                    accumulated_step_id == 0
                    and not getattr(args, "use_rollout_routing_replay", False)
                    and "train/kl_loss" in log_dict
                ):
                    assert log_dict["train/kl_loss"] < 1e-8, f"{log_dict=}"

            logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")

            if args.ci_save_grad_norm is not None:
                ci_save_grad_norm_path = args.ci_save_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                torch.save(grad_norm, ci_save_grad_norm_path)
            elif args.ci_load_grad_norm is not None:
                ci_load_grad_norm_path = args.ci_load_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                expected_grad_norm = torch.load(ci_load_grad_norm_path)
                assert math.isclose(
                    grad_norm,
                    expected_grad_norm,
                    rel_tol=0.01,
                    abs_tol=0.01,
                ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"
    microbatch_pbar.close()
    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)


def save(
    iteration: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
) -> None:
    """Persist a training checkpoint safely with forward hooks disabled.

    Args:
        iteration (int): Current global iteration number.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
    """
    args = get_args()
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)
    save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far=0,
        checkpointing_context=None,
        train_data_iterator=None,
        preprocess_common_state_dict_fn=None,
    )
    if should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)


def save_hf_model(args, rollout_id: int, model: Sequence[DDP]) -> None:
    """Save Megatron model in HuggingFace format.

    Args:
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        rollout_id (int): Rollout ID for path formatting.
    """
    if args.megatron_to_hf_mode != "bridge":
        try:
            from slime.backends.megatron_utils.hf_checkpoint_saver import save_hf_model_direct

            save_hf_model_direct(args, rollout_id, model)
        except Exception as e:
            if (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
            ):
                logger.error(f"Failed to save HuggingFace format: {e}")
        return

    should_log = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )

    try:
        from megatron.bridge import AutoBridge

        from slime.utils.megatron_bridge_utils import patch_auto_bridge_hf_config, patch_megatron_model

        path = Path(args.save_hf.format(rollout_id=rollout_id))

        if should_log:
            logger.info(f"Saving model in HuggingFace format to {path}")

        bridge = patch_auto_bridge_hf_config(AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True))

        path.mkdir(parents=True, exist_ok=True)

        with patch_megatron_model(model):
            bridge.save_hf_pretrained(
                model,
                path=path,
            )

        if should_log:
            logger.info(f"Successfully saved HuggingFace model to {path}")
    except Exception as e:
        if should_log:
            logger.error(f"Failed to save HuggingFace format: {e}")


def initialize_model_and_optimizer(
    args: Namespace, role: str = "actor"
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
    """Initialize model(s), optimizer, scheduler, and load from checkpoint.

    Args:
        args (Namespace): Runtime arguments.
        role (str): Logical role of the model (e.g., "actor", "critic").

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
            DDP-wrapped model chunks, optimizer, scheduler, and iteration index.
    """

    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    model[0].role = role
    reinit_critic_output_layer = _critic_output_layer_needs_reinit(args, model, role)
    clear_memory()
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    if reinit_critic_output_layer:
        _reinitialize_critic_output_layer(model)
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
    clear_memory()

    return model, optimizer, opt_param_scheduler, iteration
