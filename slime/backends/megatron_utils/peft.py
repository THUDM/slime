from __future__ import annotations

import inspect
import logging
from argparse import Namespace
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)


class LoRAConfigurationError(RuntimeError):
    """Raised when the runtime cannot support the requested LoRA configuration."""


def lora_enabled(args: Namespace) -> bool:
    return bool(getattr(args, "enable_lora", False))


def validate_lora_args(args: Namespace) -> None:
    """Validate the first supported LoRA GRPO path.

    The initial implementation intentionally supports only the narrow path that is
    safe to reason about: Megatron GRPO actor LoRA through Megatron-Bridge with
    colocated SGLang weight updates.  Broader combinations should be enabled only
    after they have parity tests.
    """

    if not lora_enabled(args):
        return

    errors = []
    if getattr(args, "train_backend", "megatron") != "megatron":
        errors.append("--enable-lora requires --train-backend megatron")
    if getattr(args, "megatron_to_hf_mode", None) != "bridge":
        errors.append("--enable-lora requires --megatron-to-hf-mode bridge")
    if getattr(args, "advantage_estimator", None) != "grpo":
        errors.append("--enable-lora currently supports only --advantage-estimator grpo")
    if not getattr(args, "colocate", False) and not getattr(args, "debug_train_only", False):
        errors.append("--enable-lora currently requires --colocate outside debug-train-only runs")
    if getattr(args, "custom_model_provider_path", None) is not None:
        errors.append("--enable-lora does not yet support --custom-model-provider-path")
    if getattr(args, "only_train_params_name_list", None):
        errors.append("--enable-lora cannot be combined with --only-train-params-name-list")
    if getattr(args, "freeze_params_name_list", None):
        errors.append("--enable-lora cannot be combined with --freeze-params-name-list")
    if not getattr(args, "enable_weights_backuper", True):
        errors.append("--enable-lora cannot be combined with --disable-weights-backuper")
    if getattr(args, "use_opd", False):
        errors.append("--enable-lora does not yet support on-policy distillation")
    if getattr(args, "num_experts", None):
        errors.append("--enable-lora does not yet support MoE models")
    if getattr(args, "ref_update_interval", None) is not None:
        errors.append("--enable-lora does not yet support --ref-update-interval")
    lora_rank = getattr(args, "lora_rank", None)
    if lora_rank is None or lora_rank <= 0:
        errors.append("--lora-rank must be positive")
    lora_alpha = getattr(args, "lora_alpha", None)
    if lora_alpha is None or lora_alpha <= 0:
        errors.append("--lora-alpha must be positive")
    lora_dropout = getattr(args, "lora_dropout", None)
    if lora_dropout is None or not 0.0 <= lora_dropout < 1.0:
        errors.append("--lora-dropout must be in [0.0, 1.0)")

    if errors:
        raise ValueError("; ".join(errors))

    ensure_lora_runtime_available()


def ensure_lora_runtime_available() -> None:
    """Fail early if the installed Megatron-Bridge does not expose LoRA PEFT."""

    _get_lora_cls()


def build_lora_config(args: Namespace):
    LoRA = _get_lora_cls()
    signature = inspect.signature(LoRA)
    parameters = signature.parameters

    for required in ("dim", "alpha", "dropout"):
        if required not in parameters:
            raise LoRAConfigurationError(
                f"Installed megatron.bridge.peft.lora.LoRA does not accept the required '{required}' argument."
            )

    kwargs = {
        "dim": getattr(args, "lora_rank"),
        "alpha": getattr(args, "lora_alpha"),
        "dropout": getattr(args, "lora_dropout"),
    }
    target_modules = getattr(args, "lora_target_modules", None)
    if target_modules:
        if "target_modules" not in parameters:
            raise LoRAConfigurationError(
                "Installed megatron.bridge.peft.lora.LoRA does not accept target_modules, "
                "but --lora-target-modules was set."
            )
        kwargs["target_modules"] = list(target_modules)

    return LoRA(**kwargs)


def maybe_apply_lora(model, args: Namespace, role: str):
    if not lora_enabled(args) or role != "actor":
        return model

    lora_config = build_lora_config(args)
    model = lora_config(model, training=True)
    setattr(model, "_slime_lora_config", lora_config)
    setattr(model, "_slime_lora_enabled", True)
    return model


def count_parameters(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in _iter_parameters(model):
        numel = param.numel()
        total += numel
        if getattr(param, "requires_grad", False):
            trainable += numel
    return total, trainable


def log_lora_parameter_summary(model) -> None:
    total, trainable = count_parameters(model)
    pct = 0.0 if total == 0 else trainable / total * 100
    logger.info("LoRA local parameter summary: trainable=%s total=%s trainable_pct=%.6f", trainable, total, pct)


def merge_lora_weights_for_export(model) -> None:
    """Merge LoRA adapter weights into base weights for a temporary export.

    Megatron-Bridge's public PEFT entrypoint freezes models when called directly,
    which is not appropriate during weight sync.  Calling LoRAMerge.transform on
    each module performs only the merge operation and leaves the training wrapper
    structure in place; callers must restore the unmerged weights afterwards.
    """

    LoRAMerge = _get_lora_merge_cls()
    merger = LoRAMerge()
    for module in _iter_modules(model):
        merger.transform(module)


def _get_lora_cls():
    try:
        from megatron.bridge.peft.lora import LoRA
    except Exception as exc:
        raise LoRAConfigurationError(
            "--enable-lora requires Megatron-Bridge PEFT LoRA support "
            "(megatron.bridge.peft.lora.LoRA). The installed Megatron-Bridge runtime does not expose it."
        ) from exc
    return LoRA


def _get_lora_merge_cls():
    try:
        from megatron.bridge.peft.lora import LoRAMerge
    except Exception as exc:
        raise LoRAConfigurationError(
            "LoRA weight sync requires megatron.bridge.peft.lora.LoRAMerge, "
            "but the installed Megatron-Bridge runtime does not expose it."
        ) from exc
    return LoRAMerge


def _iter_parameters(model) -> Iterable:
    if isinstance(model, (list, tuple)):
        for model_chunk in model:
            yield from model_chunk.parameters()
    else:
        yield from model.parameters()


def _iter_modules(model) -> Iterable:
    if isinstance(model, (list, tuple)):
        for model_chunk in model:
            yield from model_chunk.modules()
    else:
        yield from model.modules()


@torch.no_grad()
def restore_model_from_named_tensors(model, named_tensors):
    """Restore TensorBackuper-style tensors before or after temporary LoRA merge export."""

    missing_names = []
    for name, tensor in _iter_model_named_tensors(model):
        if name not in named_tensors:
            missing_names.append(name)
            continue
        tensor.copy_(named_tensors[name].to(device=tensor.device, non_blocking=True), non_blocking=True)
    if missing_names:
        preview = ", ".join(missing_names[:5])
        suffix = "" if len(missing_names) <= 5 else f", ... ({len(missing_names)} total)"
        raise KeyError(f"LoRA weight export backup is missing model tensors: {preview}{suffix}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _iter_model_named_tensors(model):
    model_chunks = model if isinstance(model, (list, tuple)) else [model]
    for vp_stage, model_module in enumerate(model_chunks):

        def _compute_fqn(name, vp_stage=vp_stage):
            return f"vp_stages.{vp_stage}.{_strip_param_name_prefix(name)}"

        for name, param in model_module.named_parameters():
            yield _compute_fqn(name), param

        for name, buffer in model_module.named_buffers():
            if "expert_bias" not in name:
                continue
            yield _compute_fqn(name), buffer


def _strip_param_name_prefix(name: str):
    prefix = "module."
    while name.startswith(prefix):
        name = name[len(prefix) :]
    return name
