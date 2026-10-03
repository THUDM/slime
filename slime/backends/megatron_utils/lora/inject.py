"""LoRA injection into an unwrapped Megatron model.

Injection happens inside the model provider, i.e. *before* Megatron wraps the
model in ``DistributedDataParallel`` and before ``get_megatron_optimizer`` builds
parameter groups. That ordering is what makes the base parameters cost neither a
gradient buffer nor Adam moments.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch

from slime.utils.lora_config import LORA_A_NAME, LORA_B_NAME, LoRAConfig, LoRAConfigError, is_lora_param_name

from .layers import (
    _BACKWARD_CALLS_ATTR,
    _INPUT_GRAD_PROMOTIONS_ATTR,
    LoRAInjectionError,
    LoRAParallelSpec,
    _register_megatron_main_grad_bridge,
    attach_lora_adapter,
    describe_lora_target,
    is_lora_module,
    parallel_rank_context,
)

logger = logging.getLogger(__name__)


@dataclass
class LoRAInjectionReport:
    config: LoRAConfig
    specs: list[LoRAParallelSpec] = field(default_factory=list)
    embedding_modules_hooked: int = 0
    trainable_parameters: int = 0
    frozen_parameters: int = 0

    @property
    def matched_module_count(self) -> int:
        return len(self.specs)

    @property
    def total_parameters(self) -> int:
        return self.trainable_parameters + self.frozen_parameters

    @property
    def trainable_ratio(self) -> float:
        total = self.total_parameters
        return self.trainable_parameters / total if total else 0.0

    def expected_trainable_parameters(self) -> int:
        return sum(spec.rank * spec.lora_in_features + spec.lora_out_features * spec.rank_local for spec in self.specs)

    def numeric_metrics(self) -> dict[str, float]:
        """Only scalars — never modules, tensors or nested dicts."""
        return {
            "lora/enabled": 1.0,
            "lora/rank": float(self.config.rank),
            "lora/alpha": float(self.config.alpha),
            "lora/scale": float(self.config.scale),
            "lora/matched_module_count": float(self.matched_module_count),
            "lora/trainable_parameter_count": float(self.trainable_parameters),
            "lora/frozen_parameter_count": float(self.frozen_parameters),
            "lora/trainable_ratio": float(self.trainable_ratio),
        }


@dataclass(frozen=True)
class LoRAGradientStats:
    """Backward-time adapter gradient statistics for one model-parallel rank."""

    parameter_count: int
    parameter_count_with_grad: int
    parameter_count_seen_in_backward: int
    element_count: int
    gradient_norm: float
    a_gradient_norm: float
    b_gradient_norm: float

    def numeric_metrics(self) -> dict[str, float]:
        return {
            "lora_gradient_norm": self.gradient_norm,
            "lora_a_gradient_norm": self.a_gradient_norm,
            "lora_b_gradient_norm": self.b_gradient_norm,
            "lora_gradient_parameter_fraction": (
                self.parameter_count_with_grad / self.parameter_count if self.parameter_count else 0.0
            ),
            "lora_backward_parameter_fraction": (
                self.parameter_count_seen_in_backward / self.parameter_count if self.parameter_count else 0.0
            ),
        }


def match_target_modules(model: torch.nn.Module, config: LoRAConfig) -> list[tuple[str, torch.nn.Module]]:
    """Return ``(name, module)`` pairs matched by the target regexes minus the excludes."""
    targets = config.compile_targets()
    excludes = config.compile_excludes()

    matched: list[tuple[str, torch.nn.Module]] = []
    excluded: list[str] = []
    for name, module in model.named_modules():
        if not name or not any(pattern.search(name) for pattern in targets):
            continue
        if any(pattern.search(name) for pattern in excludes):
            excluded.append(name)
            continue
        matched.append((name, module))

    if not matched:
        raise LoRAConfigError(
            "LoRA target selection matched no module.\n"
            f"  targets  : {list(config.target_modules)}\n"
            f"  excludes : {list(config.exclude_modules)}\n"
            f"  excluded by the exclude list: {excluded[:20]}\n"
            "Inspect the model's named_modules() and fix --lora-target-modules / --lora-target-preset. "
            "slime refuses to continue training with zero adapters."
        )

    seen: dict[int, str] = {}
    for name, module in matched:
        previous = seen.get(id(module))
        if previous is not None:
            raise LoRAConfigError(
                f"Module {name!r} was matched twice (also as {previous!r}); overlapping LoRA target regexes "
                f"would inject two adapters into one module: {list(config.target_modules)}"
            )
        seen[id(module)] = name

    return matched


def inject_lora(
    model: torch.nn.Module,
    config: LoRAConfig,
    *,
    generator: torch.Generator | None = None,
) -> LoRAInjectionReport:
    """Freeze the base model and attach LoRA adapters to every matched module."""
    if any(is_lora_module(module) for module in model.modules()):
        raise LoRAInjectionError("inject_lora() called twice on the same model")

    matched = match_target_modules(model, config)

    report = LoRAInjectionReport(config=config)
    for name, module in matched:
        spec = describe_lora_target(name, module, config)
        attach_lora_adapter(module, spec, config, generator=generator)
        report.specs.append(spec)

    # Freeze everything, then re-enable only the adapters. Doing it in this order means a
    # target that failed to receive an adapter can never stay silently trainable.
    for param in model.parameters():
        param.requires_grad = False
    for _, module in matched:
        getattr(module, LORA_A_NAME).requires_grad = True
        getattr(module, LORA_B_NAME).requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            report.trainable_parameters += param.numel()
        else:
            report.frozen_parameters += param.numel()
        if param.requires_grad and not is_lora_param_name(name):
            raise LoRAInjectionError(
                f"Non-LoRA parameter {name!r} is still trainable after LoRA injection; refusing to fall back "
                "to full fine-tuning"
            )

    # Must run after freezing: whether the decoder input needs promoting is a
    # direct consequence of the base model being frozen.
    report.embedding_modules_hooked = enable_gradient_checkpointing_compat(model)

    expected = report.expected_trainable_parameters()
    if report.trainable_parameters != expected:
        raise LoRAInjectionError(
            f"Trainable parameter count mismatch after LoRA injection: got {report.trainable_parameters}, "
            f"expected {expected} (matched {report.matched_module_count} modules, "
            f"ranks={parallel_rank_context()})"
        )

    return report


_EMBEDDING_MARKER_ATTRIBUTES: tuple[str, ...] = ("word_embeddings", "position_embeddings")


def _promote_decoder_input_gradients(module: torch.nn.Module, args: tuple, output):
    """Make the activation entering the decoder differentiable.

    ``torch.is_grad_enabled()`` is respected so the ``forward_only`` log-prob
    passes stay allocation-free.
    """
    if not torch.is_grad_enabled():
        return output

    def promote(tensor):
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and not tensor.requires_grad:
            # The tensor is a graph leaf precisely because every producing
            # parameter is frozen, so ``requires_grad_`` is legal here. If the
            # embedding ever becomes trainable the tensor already requires grad
            # and this branch is skipped.
            tensor.requires_grad_(True)
            setattr(module, _INPUT_GRAD_PROMOTIONS_ATTR, getattr(module, _INPUT_GRAD_PROMOTIONS_ATTR, 0) + 1)
        return tensor

    if isinstance(output, tuple):
        return tuple(promote(item) for item in output)
    return promote(output)


def _find_embedding_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name.split(".")[-1] == "embedding" and any(hasattr(module, a) for a in _EMBEDDING_MARKER_ATTRIBUTES)
    ]


def enable_gradient_checkpointing_compat(model: torch.nn.Module) -> int:
    """Reconnect the autograd graph across Megatron's activation-recomputation boundary.

    ``CheckpointFunction`` only takes activations as inputs, not parameters. With the
    embedding frozen the decoder input does not require grad, so every checkpointed
    layer returns a graph-less tensor and backward never runs -- no adapter would
    receive a gradient. Promoting the decoder input restores the boundary, like PEFT's
    ``enable_input_require_grads``.

    Returns the number of embedding modules hooked.
    """
    embeddings = _find_embedding_modules(model)
    if embeddings:
        for _, module in embeddings:
            setattr(module, _INPUT_GRAD_PROMOTIONS_ATTR, 0)
            module.register_forward_hook(_promote_decoder_input_gradients)
        return len(embeddings)

    from megatron.core import mpu

    # pre_process belongs to this model chunk; the physical PP rank alone cannot
    # distinguish the first virtual chunk from later chunks on the same rank.
    pre_process = getattr(model, "pre_process", None)
    if pre_process is None:
        pre_process = mpu.is_pipeline_first_stage()
    if pre_process:
        raise LoRAInjectionError(
            "Cannot find the language-model embedding module on the first pipeline stage, so the LoRA "
            "autograd graph cannot be reconnected across activation recomputation. Inspect the model "
            f"layout (ranks={parallel_rank_context()}); refusing to train adapters that would silently "
            "receive no gradient."
        )
    # Later pipeline stages receive their input from p2p tensors that Megatron
    # already creates with ``requires_grad=True``.
    return 0


def log_injection_report(report: LoRAInjectionReport, *, log_modules: bool = True) -> None:
    """Log the full target list exactly once, at initialisation."""
    lines = ["Matched LoRA modules:"]
    if log_modules:
        for index, spec in enumerate(report.specs):
            lines.append(
                f"  {index}: {spec.module_name} "
                f"[{spec.parallel_mode}, {spec.module_type}, base={spec.base_shape}, "
                f"A={(spec.rank, spec.lora_in_features)}, B={(spec.lora_out_features, spec.rank_local)}"
                f"{', fused-LN' if spec.fused_layernorm else ''}"
                f"{', seq-parallel' if spec.sequence_parallel else ''}]"
            )
    lines.extend(
        [
            f"Total matched modules:        {report.matched_module_count}",
            f"Trainable LoRA parameters:    {report.trainable_parameters}",
            f"Frozen base parameters:       {report.frozen_parameters}",
            f"Total parameters:             {report.total_parameters}",
            f"Trainable percentage:         {report.trainable_ratio * 100:.4f}%",
            f"Expected LoRA parameters:     {report.expected_trainable_parameters()}",
            f"LoRA rank / alpha / scale:    {report.config.rank} / {report.config.alpha} / {report.config.scale}",
            f"LoRA dropout / bias:          {report.config.dropout} / {report.config.bias}",
            f"LoRA rollout sync mode:       {report.config.rollout_sync_mode}",
            f"Embedding grad-bridge hooks:  {report.embedding_modules_hooked}",
            f"Parallel ranks:               {parallel_rank_context()}",
        ]
    )
    logger.info("\n".join(lines))


def iter_lora_modules(model: torch.nn.Module | list[torch.nn.Module]):
    """Yield ``(module_name, module)`` for every LoRA-injected module of one or more chunks."""
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for name, module in chunk.named_modules():
            if is_lora_module(module):
                yield name, module


def prepare_lora_backward(model: torch.nn.Module | list[torch.nn.Module]) -> int:
    """Install hooks on runtime adapter Parameters and reset step counters.

    Megatron may replace Parameter objects while moving, wrapping or loading
    the model after LoRA injection. Hooks registered on the original objects do
    not follow such replacements, so this lifecycle check runs after the final
    runtime model has been constructed and is idempotent on later steps.

    Returns the number of hooks newly installed on the current runtime model.
    """
    installed = 0
    for _, module in iter_lora_modules(model):
        for param in (module.lora_A, module.lora_B):
            installed += int(_register_megatron_main_grad_bridge(param))
            setattr(param, _BACKWARD_CALLS_ATTR, 0)
    return installed


def assert_lora_backward(model: torch.nn.Module | list[torch.nn.Module]) -> None:
    """Detect disconnected adapters without scanning tensor values or synchronizing CUDA."""
    missing = [
        f"{name}.{factor}"
        for name, module in iter_lora_modules(model)
        for factor in (LORA_A_NAME, LORA_B_NAME)
        if getattr(getattr(module, factor), _BACKWARD_CALLS_ATTR, 0) == 0
    ]
    if missing:
        from .layers import format_lora_forward_diagnostics, lora_forward_diagnostics

        raise LoRAInjectionError(
            f"LoRA backward did not reach adapter parameters: {missing[:10]}\n"
            + format_lora_forward_diagnostics(lora_forward_diagnostics(model))
        )


def assert_lora_gradients(
    model: torch.nn.Module | list[torch.nn.Module],
    *,
    require_nonzero: bool = False,
) -> LoRAGradientStats:
    """Validate adapter gradients immediately after backward and return scalar statistics.

    ``lora_B`` is zero-initialised, so ``lora_A`` legitimately has a zero
    gradient on the first update.  The aggregate adapter gradient must still be
    non-zero because ``lora_B`` receives gradient through the random A factor.
    """
    chunks = model if isinstance(model, (list, tuple)) else [model]
    missing: list[str] = []
    unexpected: list[str] = []
    parameter_count = 0
    parameter_count_with_grad = 0
    parameter_count_seen_in_backward = 0
    element_count = 0
    squared_norm = None
    a_squared_norm = None
    b_squared_norm = None
    for chunk in chunks:
        for name, param in chunk.named_parameters():
            # Megatron DDP accumulates into ``main_grad``; ``param.grad`` may
            # be absent or a non-authoritative placeholder when gradient
            # accumulation fusion is active. Plain PyTorch modules fall back
            # to the ordinary ``grad`` field.
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if is_lora_param_name(name):
                parameter_count += 1
                if getattr(param, _BACKWARD_CALLS_ATTR, 0) > 0:
                    parameter_count_seen_in_backward += 1
                if grad is None:
                    missing.append(name)
                    continue
                parameter_count_with_grad += 1
                element_count += grad.numel()
                # Megatron's DDP main_grad is fp32 when
                # --accumulate-allreduce-grads-in-fp32 is enabled.  The explicit
                # cast also keeps this diagnostic correct for ordinary bf16
                # PyTorch modules used by unit tests.
                grad_norm = torch.linalg.vector_norm(grad.detach().float())
                grad_squared = grad_norm.square()
                squared_norm = grad_squared if squared_norm is None else squared_norm + grad_squared
                if name.endswith(f".{LORA_A_NAME}"):
                    a_squared_norm = grad_squared if a_squared_norm is None else a_squared_norm + grad_squared
                else:
                    b_squared_norm = grad_squared if b_squared_norm is None else b_squared_norm + grad_squared
            elif grad is not None and param.requires_grad:
                unexpected.append(name)
    if missing or unexpected:
        raise LoRAInjectionError(
            "LoRA gradient sanity check failed.\n"
            f"  LoRA parameters without gradient : {missing[:10]}\n"
            f"  frozen parameters with gradient  : {unexpected[:10]}\n"
            f"  ranks: {parallel_rank_context()}"
        )

    if squared_norm is None:
        raise LoRAInjectionError(
            "LoRA gradient sanity check found no adapter parameters.\n" f"  ranks: {parallel_rank_context()}"
        )

    gradient_norm = float(squared_norm.sqrt().item())
    a_gradient_norm = float(a_squared_norm.sqrt().item()) if a_squared_norm is not None else 0.0
    b_gradient_norm = float(b_squared_norm.sqrt().item()) if b_squared_norm is not None else 0.0
    if not math.isfinite(gradient_norm):
        raise LoRAInjectionError(
            "LoRA backward produced a non-finite adapter gradient.\n"
            f"  gradient norm: {gradient_norm}\n"
            f"  ranks: {parallel_rank_context()}"
        )
    if require_nonzero and gradient_norm == 0.0:
        if parameter_count_seen_in_backward == 0:
            diagnosis = "the injected LoRA forward is disconnected from the policy loss"
        elif parameter_count_seen_in_backward < parameter_count:
            diagnosis = "only part of the injected LoRA graph was reached by backward"
        else:
            diagnosis = "backward reached every adapter, but the effective training signal is zero"
        raise LoRAInjectionError(
            "LoRA backward produced an all-zero adapter gradient; refusing to run an optimizer step that "
            f"cannot update the policy; diagnosis: {diagnosis}.\n"
            f"  LoRA A gradient norm: {a_gradient_norm}\n"
            f"  LoRA B gradient norm: {b_gradient_norm}\n"
            f"  adapter parameters with gradient: {parameter_count_with_grad}/{parameter_count}\n"
            f"  adapter parameters reached by backward: {parameter_count_seen_in_backward}/{parameter_count}\n"
            f"  ranks: {parallel_rank_context()}"
        )

    return LoRAGradientStats(
        parameter_count=parameter_count,
        parameter_count_with_grad=parameter_count_with_grad,
        parameter_count_seen_in_backward=parameter_count_seen_in_backward,
        element_count=element_count,
        gradient_norm=gradient_norm,
        a_gradient_norm=a_gradient_norm,
        b_gradient_norm=b_gradient_norm,
    )


def assert_optimizer_holds_only_lora(optimizer, model: torch.nn.Module | list[torch.nn.Module]) -> int:
    """Verify that the model and optimizer expose only LoRA parameters for training."""
    chunks = model if isinstance(model, (list, tuple)) else [model]
    named_parameters = [(name, param) for chunk in chunks for name, param in chunk.named_parameters()]
    parameter_names = {id(param): name for name, param in named_parameters}
    lora_ids = {id(param) for name, param in named_parameters if is_lora_param_name(name)}
    trainable_ids = {id(param) for _, param in named_parameters if param.requires_grad}

    if trainable_ids != lora_ids:
        non_lora_trainable = [parameter_names[param_id] for param_id in trainable_ids - lora_ids]
        frozen_lora = [parameter_names[param_id] for param_id in lora_ids - trainable_ids]
        raise LoRAInjectionError(
            "Model trainability does not match the injected LoRA adapters.\n"
            f"  trainable non-LoRA parameters : {non_lora_trainable[:10]}\n"
            f"  frozen LoRA parameters        : {frozen_lora[:10]}\n"
            f"  ranks: {parallel_rank_context()}"
        )

    optimizers = getattr(optimizer, "chained_optimizers", None) or [optimizer]
    optimizer_param_ids: set[int] = set()
    optimizer_model_param_ids: set[int] = set()
    has_distributed_optimizer = False
    for chained in optimizers:
        inner = getattr(chained, "optimizer", chained)
        inner_param_ids: set[int] = set()
        for group in getattr(inner, "param_groups", []):
            for param in group.get("params", []):
                inner_param_ids.add(id(param))
        optimizer_param_ids.update(inner_param_ids)

        # DistributedOptimizer replaces model parameters with DP-local FP32
        # shards in the inner optimizer. Its model_param_gbuf_map retains the
        # identity-preserving mapping to the original model parameters.
        model_param_map = getattr(chained, "model_param_gbuf_map", None)
        if model_param_map is not None:
            has_distributed_optimizer = True
            optimizer_model_param_ids.update(id(param) for param in model_param_map)
        elif hasattr(chained, "float16_groups"):
            # Float16OptimizerWithFloat16Params also owns FP32 master copies,
            # even when the distributed optimizer is disabled.
            main_to_model = {}
            for model_group, main_group in zip(chained.float16_groups, chained.fp32_from_float16_groups, strict=True):
                for model_param, main_param in zip(model_group, main_group, strict=True):
                    main_to_model[id(main_param)] = id(model_param)
            optimizer_model_param_ids.update(main_to_model.get(param_id, param_id) for param_id in inner_param_ids)
        else:
            optimizer_model_param_ids.update(inner_param_ids)

    unexpected_ids = optimizer_model_param_ids - lora_ids
    if unexpected_ids:
        unexpected_names = [
            parameter_names.get(param_id, "<unmapped optimizer parameter>") for param_id in unexpected_ids
        ]
        raise LoRAInjectionError(
            "Frozen base parameters must never enter the optimizer.\n"
            f"  unexpected optimizer parameters : {unexpected_names[:10]}\n"
            f"  ranks: {parallel_rank_context()}"
        )

    # An ordinary optimizer owns full model tensors and must contain every
    # adapter. DistributedOptimizer instead owns only the shards intersecting
    # this DP rank, so its local tensor count is intentionally smaller.
    if not has_distributed_optimizer and optimizer_model_param_ids != lora_ids:
        missing_names = [parameter_names[param_id] for param_id in lora_ids - optimizer_model_param_ids]
        raise LoRAInjectionError(
            "Optimizer is missing LoRA parameters.\n"
            f"  missing parameters : {missing_names[:10]}\n"
            f"  ranks: {parallel_rank_context()}"
        )

    return len(optimizer_param_ids)
