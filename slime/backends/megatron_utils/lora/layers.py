"""LoRA adapters for Megatron linear layers.

Design notes
------------
* **Base parameter names are never renamed.** The adapter is attached *in place*:
  ``module.weight`` keeps its canonical name and ``lora_A`` / ``lora_B`` are added
  as sibling parameters. A forward hook adds the adapter contribution, so every name-based rule in
  ``megatron_to_hf`` / ``hf_to_megatron`` / ``all_gather_param`` keeps working.

* **Column/row LoRA factors are tensor-parallel sharded.** Explicitly allow-listed
  replicated modules instead keep identical adapters on each TP rank.
  The low-rank intermediate is combined with a *symmetric* all-reduce
  (all-reduce in both forward and backward), which makes each rank's ``lora_A`` /
  ``lora_B`` gradient exact on its own shard.

  ``column`` targets (``linear_qkv``, ``linear_fc1``): base weight is
  ``[out_local, in_full]``; ``lora_A`` is ``[r, in_local]`` (sharded over the
  contracted dim) and ``lora_B`` is ``[out_local, r]`` (sharded like the base
  output).

  ``row`` targets (``linear_proj``, ``linear_fc2``): base weight is
  ``[out_full, in_local]``; ``lora_A`` is ``[r, in_local]`` and ``lora_B`` is
  ``[out_full, r_local]`` (sharded over ``r``), so the LoRA contribution joins the
  same TP reduction as the base output.

* **Merging is always local**, i.e. ``W_local += scale * B @ A`` on the local base
  shard, which makes it oblivious to fused-QKV interleaving and GLU chunk layout.

* **Routed MoE experts** (``mlp.experts.linear_fc1`` / ``linear_fc2``, i.e. TE's
  ``GroupedLinear``) get a *single* adapter with both factors shared by every
  expert. Per-expert adapters would multiply the
  trainable parameters by ``num_experts`` and make the merge depend on the
  expert-parallel placement; one shared adapter keeps both bounded. See
  ``_EXPERT_PARALLEL_MODES`` for why these layers need their own collectives.
"""

from __future__ import annotations

import logging
import math
import re
import weakref
from dataclasses import dataclass
from types import MethodType
from typing import Any, Final, Literal

import torch
import torch.nn.functional as F

from slime.utils.lora_config import LORA_A_NAME, LORA_B_NAME, LoRAConfig

logger = logging.getLogger(__name__)

LoRAParallelMode = Literal["column", "row", "replicated", "expert_column", "expert_row"]

#: Routed-expert modes. These differ from the dense modes in three ways, all of
#: which follow from ``TEGroupedLinear`` setting ``explicit_expert_comm`` and
#: therefore disabling TE's own tensor parallelism:
#:
#: 1. The MoE token dispatcher gathers/reduce-scatters along the *token* axis, not
#:    the hidden axis, so every expert-TP rank already sees the full hidden width.
#:    The adapter must not add a second gather or reduce.
#: 2. Sharding uses the *expert* tensor-parallel group, not the ordinary TP group.
#: 3. Weights are packed per local expert as ``weight0``, ``weight1``, ... rather
#:    than a single ``weight``.
_EXPERT_PARALLEL_MODES: Final[frozenset[str]] = frozenset({"expert_column", "expert_row"})

_SPEC_ATTR: Final[str] = "_slime_lora_spec"
_CONFIG_ATTR: Final[str] = "_slime_lora_config"
_BACKWARD_CALLS_ATTR: Final[str] = "_slime_lora_backward_calls"
_INPUT_GRAD_PROMOTIONS_ATTR: Final[str] = "_slime_lora_input_grad_promotions"
_GRAD_HOOKED_PARAMETERS: weakref.WeakValueDictionary[int, torch.nn.Parameter] = weakref.WeakValueDictionary()

# ``mlp.experts`` are the routed experts; ``mlp.shared_experts`` are ordinary dense
# linears and must not be caught by this pattern.
_ROUTED_EXPERT_MODULE: Final[re.Pattern[str]] = re.compile(r"(?:^|\.)experts(?:\.|$)")


class LoRAUnsupportedModuleError(RuntimeError):
    """Raised when a matched module cannot receive a LoRA adapter."""


class LoRAInjectionError(RuntimeError):
    """Raised for structural problems during injection (duplicate injection, ...)."""


@dataclass(frozen=True)
class LoRAParallelSpec:
    """Everything the LoRA forward/merge paths need to know about a base module."""

    module_name: str
    module_type: str
    parallel_mode: LoRAParallelMode
    rank: int
    #: second dimension of ``lora_B`` on this rank (``rank`` unless the rank is TP-sharded)
    rank_local: int
    #: second dimension of ``lora_A`` on this rank, i.e. the contracted width it consumes
    lora_in_features: int
    #: first dimension of ``lora_B``, always equal to the local base output width
    lora_out_features: int
    #: global (unsharded) input width of the base GEMM, used only for initialisation scale
    global_in_features: int
    tp_size: int
    tp_rank: int
    sequence_parallel: bool
    fused_layernorm: bool
    #: shape of the local base weight shard; every merge must reproduce it exactly
    base_shape: tuple[int, int]
    #: number of local experts packed into this module (1 for every dense target).
    #: Routed experts store ``weight0 .. weight{num_local_experts-1}`` instead of
    #: a single ``weight``, and one shared adapter is merged into each of them.
    num_local_experts: int = 1

    @property
    def is_expert(self) -> bool:
        return self.parallel_mode in _EXPERT_PARALLEL_MODES

    @property
    def base_weight_names(self) -> tuple[str, ...]:
        """Names of the base weights this adapter contributes to."""
        if not self.is_expert:
            return ("weight",)
        return tuple(f"weight{index}" for index in range(self.num_local_experts))

    @property
    def a_is_tensor_parallel(self) -> bool:
        """Whether ``lora_A`` is sharded over this spec's tensor-parallel group.

        The shared routed-expert adapter for a column-parallel layer consumes the
        full hidden width, so its A factor is replicated rather than sharded.
        """
        return self.parallel_mode != "expert_column"

    @property
    def replica_groups_a(self) -> tuple:
        if not self.is_expert:
            return ()
        return _expert_replica_groups(self, replicated_over_tensor_parallel=not self.a_is_tensor_parallel)

    @property
    def replica_groups_b(self) -> tuple:
        if not self.is_expert:
            return ()
        # B is sharded over expert-TP in both expert modes (output width for
        # expert_column, the rank dimension for expert_row).
        return _expert_replica_groups(self, replicated_over_tensor_parallel=False)


# --------------------------------------------------------------------------- #
# distributed helpers (Megatron is imported lazily so CPU tests can run)
# --------------------------------------------------------------------------- #


def tensor_model_parallel_world_size() -> int:
    try:
        from megatron.core import mpu
    except ImportError:
        return 1
    try:
        return mpu.get_tensor_model_parallel_world_size()
    except (AssertionError, AttributeError):
        return 1


def tensor_model_parallel_rank() -> int:
    try:
        from megatron.core import mpu
    except ImportError:
        return 0
    try:
        return mpu.get_tensor_model_parallel_rank()
    except (AssertionError, AttributeError):
        return 0


def tensor_model_parallel_group():
    from megatron.core import mpu

    return mpu.get_tensor_model_parallel_group()


def expert_tensor_parallel_world_size() -> int:
    try:
        from megatron.core import mpu
    except ImportError:
        return 1
    try:
        return mpu.get_expert_tensor_parallel_world_size()
    except (AssertionError, AttributeError):
        return 1


def expert_tensor_parallel_rank() -> int:
    try:
        from megatron.core import mpu
    except ImportError:
        return 0
    try:
        return mpu.get_expert_tensor_parallel_rank()
    except (AssertionError, AttributeError):
        return 0


def expert_tensor_parallel_group():
    from megatron.core import mpu

    return mpu.get_expert_tensor_parallel_group()


def expert_model_parallel_world_size() -> int:
    try:
        from megatron.core import mpu
    except ImportError:
        return 1
    try:
        return mpu.get_expert_model_parallel_world_size()
    except (AssertionError, AttributeError):
        return 1


def expert_model_parallel_group():
    from megatron.core import mpu

    return mpu.get_expert_model_parallel_group()


def spec_parallel_world_size(spec: LoRAParallelSpec) -> int:
    """TP world size of the group this spec's factors are sharded over."""
    return expert_tensor_parallel_world_size() if spec.is_expert else tensor_model_parallel_world_size()


def spec_parallel_group(spec: LoRAParallelSpec):
    """The TP group this spec's factors are sharded over."""
    return expert_tensor_parallel_group() if spec.is_expert else tensor_model_parallel_group()


def _broadcast_parameter(tensor: torch.Tensor, *, src: int, group) -> None:
    """Broadcast a parameter through a process group that may be NCCL-only.

    Megatron invokes the model provider while parameters are still on CPU, but
    its tensor-parallel group uses NCCL. NCCL cannot operate on a CPU tensor, so
    use a temporary tensor on the current CUDA device and preserve the original
    parameter placement for Megatron's subsequent model-device transfer.
    """
    import torch.distributed as dist

    backend = str(dist.get_backend(group)).lower()
    if tensor.device.type == "cpu" and backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot broadcast a CPU LoRA parameter through NCCL because CUDA is unavailable")
        broadcast_tensor = tensor.detach().to(torch.device("cuda", torch.cuda.current_device()))
        dist.broadcast(broadcast_tensor, src=src, group=group)
        tensor.copy_(broadcast_tensor.to(tensor.device))
        return

    dist.broadcast(tensor, src=src, group=group)


def parallel_rank_context() -> dict[str, Any]:
    """Best-effort TP/PP/EP rank description used in error messages."""
    try:
        from megatron.core import mpu
    except ImportError:
        return {"tp": "n/a", "pp": "n/a", "ep": "n/a"}
    try:
        return {
            "tp": f"{mpu.get_tensor_model_parallel_rank()}/{mpu.get_tensor_model_parallel_world_size()}",
            "pp": f"{mpu.get_pipeline_model_parallel_rank()}/{mpu.get_pipeline_model_parallel_world_size()}",
            "ep": f"{mpu.get_expert_model_parallel_rank()}/{mpu.get_expert_model_parallel_world_size()}",
        }
    except (AssertionError, AttributeError):
        return {"tp": "n/a", "pp": "n/a", "ep": "n/a"}


class _SymmetricAllReduce(torch.autograd.Function):
    """All-reduce in forward *and* backward.

    Used for the low-rank intermediate so that both LoRA factors can stay
    TP-sharded while their gradients remain exact without any extra sync.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group) -> torch.Tensor:  # type: ignore[override]
        import torch.distributed as dist

        ctx.group = group
        tensor = tensor.contiguous().clone()
        dist.all_reduce(tensor, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        import torch.distributed as dist

        grad_output = grad_output.contiguous().clone()
        dist.all_reduce(grad_output, group=ctx.group)
        return grad_output, None


def symmetric_all_reduce(tensor: torch.Tensor, group) -> torch.Tensor:
    return _SymmetricAllReduce.apply(tensor, group)


class _ReplicaGradientAllReduce(torch.autograd.Function):
    """Identity in forward; all-reduce the gradient across replica groups.

    Applied to a LoRA factor that is *replicated* over a process group whose ranks
    each see only part of the training signal. Summing the per-rank gradients is
    both the mathematically correct total gradient and what keeps the replicas
    bit-identical over time.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, groups: tuple) -> torch.Tensor:  # type: ignore[override]
        ctx.groups = groups
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        import torch.distributed as dist

        grad_output = grad_output.contiguous().clone()
        for group in ctx.groups:
            dist.all_reduce(grad_output, group=group)
        return grad_output, None


def _expert_replica_groups(spec: LoRAParallelSpec, *, replicated_over_tensor_parallel: bool) -> tuple:
    """Groups over which a routed-expert factor is replicated and must sum gradients.

    One adapter is shared by every routed expert, but each expert-parallel rank only
    routes tokens to its *local* experts, so the total gradient is the sum over the
    expert-parallel group. A factor that is additionally not sharded over the expert
    tensor-parallel group must sum over that group too.
    """
    groups = []
    if expert_model_parallel_world_size() > 1:
        groups.append(expert_model_parallel_group())
    if replicated_over_tensor_parallel and expert_tensor_parallel_world_size() > 1:
        groups.append(expert_tensor_parallel_group())
    return tuple(groups)


def _sum_gradient_over_replicas(tensor: torch.Tensor, groups: tuple) -> torch.Tensor:
    return _ReplicaGradientAllReduce.apply(tensor, groups) if groups else tensor


def _gather_along_sequence(tensor: torch.Tensor) -> torch.Tensor:
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    # The down-projection consumes different hidden slices on each TP rank.
    # Backward must sum those contributions before returning the local tokens.
    return gather_from_sequence_parallel_region(tensor, tensor_parallel_output_grad=True)


def _reduce_scatter_along_sequence(tensor: torch.Tensor) -> torch.Tensor:
    from megatron.core.tensor_parallel.mappings import reduce_scatter_to_sequence_parallel_region

    return reduce_scatter_to_sequence_parallel_region(tensor)


def _reduce_across_tensor_parallel(tensor: torch.Tensor) -> torch.Tensor:
    from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region

    return reduce_from_tensor_model_parallel_region(tensor)


def _set_tp_attributes(param: torch.nn.Parameter, *, is_parallel: bool, dim: int, stride: int) -> None:
    try:
        from megatron.core.tensor_parallel import set_tensor_model_parallel_attributes
    except ImportError:
        param.tensor_model_parallel = is_parallel
        param.partition_dim = dim
        param.partition_stride = stride
        return
    set_tensor_model_parallel_attributes(param, is_parallel, dim, stride)


def _register_megatron_main_grad_bridge(param: torch.nn.Parameter) -> bool:
    """Route an ordinary PyTorch LoRA gradient into Megatron's DDP buffer.

    The adapter GEMMs use ``F.linear``, but Megatron's optimizer consumes
    ``param.main_grad``, not ``param.grad``; some Megatron/TE combinations leave the
    adapter's ``main_grad`` allocated but zero, making the optimizer step a silent
    no-op. This tensor hook runs before Megatron's ``AccumulateGrad`` post-hook and
    sets ``grad_added_to_main_grad`` so DDP does not add the gradient twice. Outside
    Megatron there is no ``main_grad`` and autograd is unchanged.
    """
    # Track the exact Parameter object rather than putting a boolean marker on
    # it. Some model-conversion paths copy Python attributes to a replacement
    # Parameter without copying its autograd hooks.
    if _GRAD_HOOKED_PARAMETERS.get(id(param)) is param:
        return False

    setattr(param, _BACKWARD_CALLS_ATTR, 0)
    parameter_ref = weakref.ref(param)

    def bridge(grad: torch.Tensor) -> torch.Tensor:
        parameter = parameter_ref()
        if parameter is None:
            return grad

        setattr(parameter, _BACKWARD_CALLS_ATTR, getattr(parameter, _BACKWARD_CALLS_ATTR, 0) + 1)
        main_grad = getattr(parameter, "main_grad", None)
        if main_grad is not None:
            with torch.no_grad():
                main_grad.add_(grad.detach())
            # Megatron's DDP hook checks this flag before copying param.grad to
            # main_grad.  The bridge has already done that copy for this
            # microbatch, so a second addition would double the gradient.
            parameter.grad_added_to_main_grad = True
        return grad

    param.register_hook(bridge)
    _GRAD_HOOKED_PARAMETERS[id(param)] = param
    return True


# --------------------------------------------------------------------------- #
# target-module classification
# --------------------------------------------------------------------------- #


def _megatron_parallel_linear_classes() -> tuple[tuple[type, ...], tuple[type, ...]]:
    column: list[type] = []
    row: list[type] = []
    try:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

        column.append(ColumnParallelLinear)
        row.append(RowParallelLinear)
    except ImportError:
        pass
    return tuple(column), tuple(row)


def _infer_parallel_mode(module: torch.nn.Module) -> LoRAParallelMode | None:
    declared = getattr(module, "parallel_mode", None)
    if declared in ("column", "row"):
        return declared
    if declared == "duplicated":
        return "replicated"

    column_classes, row_classes = _megatron_parallel_linear_classes()
    if column_classes and isinstance(module, column_classes):
        return "column"
    if row_classes and isinstance(module, row_classes):
        return "row"
    if isinstance(module, torch.nn.Linear):
        return "replicated"
    return None


def grouped_expert_weights(module: torch.nn.Module) -> list[torch.Tensor]:
    """Local per-expert weights of a grouped-GEMM module, ordered by expert index.

    TE's ``GroupedLinear`` stores ``weight0 .. weight{num_gemms-1}`` instead of a
    single ``weight``, so the generic ``module.weight`` lookup finds nothing.
    """
    weights: list[torch.Tensor] = []
    index = 0
    while True:
        weight = getattr(module, f"weight{index}", None)
        if not isinstance(weight, torch.Tensor):
            break
        weights.append(weight)
        index += 1
    return weights


def _infer_expert_parallel_mode(module: torch.nn.Module, module_name: str) -> LoRAParallelMode | None:
    """Classify a routed-expert grouped linear as expert-column or expert-row.

    ``TEGroupedLinear`` sets ``parallel_mode=None`` on itself (it performs the
    expert communication explicitly, so TE must stay parallelism-agnostic), which
    means the declared attribute cannot be used. Fall back to the class name and
    finally to the fc1/fc2 naming convention shared by every mcore MoE layer.
    """
    for base in type(module).__mro__:
        if base.__name__ == "TEColumnParallelGroupedLinear":
            return "expert_column"
        if base.__name__ == "TERowParallelGroupedLinear":
            return "expert_row"
    if module_name.endswith("linear_fc1"):
        return "expert_column"
    if module_name.endswith("linear_fc2"):
        return "expert_row"
    return None


def _describe_routed_expert_target(
    module_name: str,
    module: torch.nn.Module,
    config: LoRAConfig,
    fail,
) -> LoRAParallelSpec:
    """Classify a routed-expert grouped linear (one adapter shared by all experts)."""
    weights = grouped_expert_weights(module)
    if not weights:
        raise fail(
            "it is a routed MoE expert layer but exposes no `weight0` parameter, so slime cannot determine "
            "its grouped-GEMM layout. Only Transformer Engine's GroupedLinear (--moe-grouped-gemm) is supported"
        )
    shapes = {tuple(weight.shape) for weight in weights}
    if len(shapes) != 1:
        raise fail(f"its local experts have differing weight shapes {sorted(shapes)}")
    if weights[0].dim() != 2:
        raise fail(f"its expert weights are not 2-D (got shape {tuple(weights[0].shape)})")

    parallel_mode = _infer_expert_parallel_mode(module, module_name)
    if parallel_mode is None:
        raise fail(
            "slime cannot tell whether this routed-expert layer is column- or row-parallel; expected a "
            "TE*ParallelGroupedLinear subclass or a module named `linear_fc1` / `linear_fc2`"
        )

    tp_size = expert_tensor_parallel_world_size()
    tp_rank = expert_tensor_parallel_rank()
    # A TP checkpoint coordinate must identify exactly one expert-TP shard.
    # Several TP coordinates may store replicas of the same expert shard.
    if tensor_model_parallel_world_size() % tp_size != 0 or tp_rank != tensor_model_parallel_rank() % tp_size:
        raise fail(
            "routed-expert LoRA requires expert-TP to divide TP and expert-TP rank to equal TP rank modulo expert-TP; "
            "adapter checkpoints are indexed by TP/PP only. Align the parallel layout or exclude routed experts"
        )
    base_out, base_in = int(weights[0].shape[0]), int(weights[0].shape[1])

    # The MoE token dispatcher gathers along the token axis, so the activation
    # entering the experts already carries the full hidden width on every
    # expert-TP rank. Only the expert-local FFN dimension is sharded.
    rank_local = config.rank
    if parallel_mode == "expert_column":
        # weight is [ffn_local (x2 for GLU), hidden_full]: A consumes the full
        # hidden width and is therefore replicated; B is sharded like the output.
        lora_in_features = base_in
        global_in_features = base_in
    else:
        # weight is [hidden_full, ffn_local]: A consumes the local FFN shard, and
        # B is sharded over the rank so the outputs sum to the full contraction.
        if config.rank % tp_size != 0:
            raise fail(
                f"row-parallel routed-expert targets shard the LoRA rank across expert tensor parallelism, "
                f"so --lora-rank ({config.rank}) must be divisible by the expert tensor-parallel size ({tp_size})"
            )
        lora_in_features = base_in
        global_in_features = base_in * tp_size
        rank_local = config.rank // tp_size

    if getattr(module, "layer_norm_weight", None) is not None:
        raise fail("routed-expert layers with a fused normalisation are not supported")

    return LoRAParallelSpec(
        module_name=module_name,
        module_type=f"{type(module).__module__}.{type(module).__name__}",
        parallel_mode=parallel_mode,
        rank=config.rank,
        rank_local=rank_local,
        lora_in_features=lora_in_features,
        lora_out_features=base_out,
        global_in_features=global_in_features,
        tp_size=tp_size,
        tp_rank=tp_rank,
        # The dispatcher owns all sequence-parallel communication for MoE layers.
        sequence_parallel=False,
        fused_layernorm=False,
        base_shape=(base_out, base_in),
        num_local_experts=len(weights),
    )


def describe_lora_target(module_name: str, module: torch.nn.Module, config: LoRAConfig) -> LoRAParallelSpec:
    """Classify a matched module or raise ``LoRAUnsupportedModuleError``."""

    def fail(reason: str) -> LoRAUnsupportedModuleError:
        weight = getattr(module, "weight", None)
        if weight is None:
            # Grouped-GEMM modules pack their experts as weight0, weight1, ...
            weight = getattr(module, "weight0", None)
        return LoRAUnsupportedModuleError(
            f"Cannot inject LoRA into module {module_name!r}: {reason}\n"
            f"  module type   : {type(module).__module__}.{type(module).__name__}\n"
            f"  base weight   : {tuple(getattr(weight, 'shape', ()) or ())}\n"
            f"  parallel rank : {parallel_rank_context()}\n"
            f"  lora config   : rank={config.rank} alpha={config.alpha} bias={config.bias} "
            f"targets={list(config.target_modules)}"
        )

    if hasattr(module, _SPEC_ATTR):
        raise LoRAInjectionError(
            f"Module {module_name!r} already has a LoRA adapter; check for overlapping "
            f"--lora-target-modules regexes ({list(config.target_modules)})"
        )

    # Routed experts first: they are grouped-GEMM modules with no plain ``weight``
    # and need the expert-parallel collectives. ``shared_experts`` are ordinary
    # dense TP linears and must not be caught here.
    if _ROUTED_EXPERT_MODULE.search(module_name):
        return _describe_routed_expert_target(module_name, module, config, fail)

    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        raise fail("it has no 2-D `weight` parameter")

    parallel_mode = _infer_parallel_mode(module)
    if parallel_mode is None:
        raise fail("its class is not a recognised Megatron/TE/torch linear layer")

    tp_size = tensor_model_parallel_world_size()
    tp_rank = tensor_model_parallel_rank()
    base_out, base_in = int(weight.shape[0]), int(weight.shape[1])

    duplicated_allowed = parallel_mode == "replicated" and any(
        pattern.search(module_name) for pattern in config.compile_allow_replicated()
    )
    if parallel_mode == "replicated" and tp_size > 1 and not duplicated_allowed:
        raise fail(
            "it is replicated across tensor-parallel ranks. LoRA only injects into TP-sharded linear layers "
            "or modules explicitly allow-listed via --lora-allow-replicated-modules (for layers that "
            "internally gather the full sequence and therefore see identical inputs/gradients on every TP "
            "rank), so that no unaccounted cross-TP gradient synchronisation is required (this is why vision "
            "LoRA is unsupported by default)."
        )

    rank_local = config.rank
    if parallel_mode == "column":
        if getattr(module, "gather_output", False) and tp_size > 1:
            raise fail("column-parallel layers with gather_output=True are not supported")
        # base weight is [out_local, in_full]; lora_A consumes a 1/tp slice of the contracted dim.
        if base_in % tp_size != 0:
            raise fail(
                f"its input width ({base_in}) is not divisible by the tensor-parallel size ({tp_size}), "
                "so the LoRA down-projection cannot be sharded over the contracted dimension"
            )
        lora_in_features = base_in // tp_size
        global_in_features = base_in
    elif parallel_mode == "row":
        if tp_size > 1 and not getattr(module, "input_is_parallel", True):
            raise fail("row-parallel layers with input_is_parallel=False are not supported")
        if config.rank % tp_size != 0:
            raise fail(
                f"row-parallel targets shard the LoRA rank across tensor parallelism, so --lora-rank "
                f"({config.rank}) must be divisible by the tensor-parallel size ({tp_size})"
            )
        # base weight is [out_full, in_local]; lora_A already consumes the local contracted shard.
        lora_in_features = base_in
        global_in_features = base_in * tp_size
        rank_local = config.rank // tp_size
    else:
        lora_in_features = base_in
        global_in_features = base_in

    fused_layernorm = getattr(module, "layer_norm_weight", None) is not None
    if fused_layernorm:
        normalization = getattr(module, "normalization", None)
        if normalization not in ("LayerNorm", "RMSNorm"):
            raise fail(
                f"it fuses a normalisation of unknown kind {normalization!r}; the LoRA input must be the "
                "normalised activation, which slime cannot reconstruct for this module"
            )
        if getattr(module, "eps", None) is None:
            raise fail("it fuses a normalisation but exposes no `eps`, so the normalised input cannot be recomputed")

    return LoRAParallelSpec(
        module_name=module_name,
        module_type=f"{type(module).__module__}.{type(module).__name__}",
        parallel_mode=parallel_mode,
        rank=config.rank,
        rank_local=rank_local,
        lora_in_features=lora_in_features,
        lora_out_features=base_out,
        global_in_features=global_in_features,
        tp_size=tp_size,
        tp_rank=tp_rank,
        sequence_parallel=bool(getattr(module, "sequence_parallel", False)),
        fused_layernorm=fused_layernorm,
        base_shape=(base_out, base_in),
    )


# --------------------------------------------------------------------------- #
# the adapter itself
# --------------------------------------------------------------------------- #


def _resolve_forward_input(
    module: torch.nn.Module,
    args: tuple,
    kwargs: dict,
) -> torch.Tensor:
    """Find the activation the base GEMM consumed.

    Linear implementations name their first argument differently: Megatron uses
    ``input_``, Transformer Engine uses ``inp``, plain ``torch.nn.Linear`` uses
    ``input`` and the Hugging Face gated-delta-net modules are called with
    ``hidden_states=``. Failing to locate it must raise: silently skipping the
    adapter would train nothing while looking healthy.
    """
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    for key in ("input_", "inp", "input", "hidden_states", "x"):
        candidate = kwargs.get(key)
        if isinstance(candidate, torch.Tensor):
            return candidate
    spec: LoRAParallelSpec = getattr(module, _SPEC_ATTR)
    raise LoRAInjectionError(
        f"Cannot locate the input activation of LoRA target {spec.module_name!r} "
        f"({spec.module_type}); positional args={len(args)}, keyword args={sorted(kwargs)}. "
        "Refusing to run a forward pass with a silently disabled adapter."
    )


def _normalized_lora_input(module: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
    """Recompute the fused normalisation so LoRA sees the same input as the base GEMM."""
    weight = module.layer_norm_weight
    if getattr(module, "zero_centered_gamma", False):
        weight = weight + 1.0
    eps = float(module.eps)
    if module.normalization == "RMSNorm":
        variance = input_.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = input_.float() * torch.rsqrt(variance + eps)
        return (normalized * weight.float()).to(input_.dtype)
    bias = getattr(module, "layer_norm_bias", None)
    return F.layer_norm(input_, (input_.shape[-1],), weight=weight, bias=bias, eps=eps)


def compute_lora_delta(module: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
    """``scale * B @ (A @ x)`` in the local shard layout of the base GEMM's output."""
    spec: LoRAParallelSpec = getattr(module, _SPEC_ATTR)
    config: LoRAConfig = getattr(module, _CONFIG_ATTR)
    lora_a: torch.Tensor = module.lora_A
    lora_b: torch.Tensor = module.lora_B

    x = _normalized_lora_input(module, input_) if spec.fused_layernorm else input_
    dropout = getattr(module, "_slime_lora_dropout", None)
    if dropout is not None:
        x = dropout(x)
    x = x.to(lora_a.dtype)

    if spec.parallel_mode == "expert_column":
        # The token dispatcher already gathered along the token axis, so ``x``
        # carries the full hidden width on every expert-TP rank and A is
        # replicated: no input communication, and no output reduction because
        # the local FFN shard is exactly what the base GEMM produces.
        hidden = F.linear(x, _sum_gradient_over_replicas(lora_a, spec.replica_groups_a))
        delta = F.linear(hidden, _sum_gradient_over_replicas(lora_b, spec.replica_groups_b))
    elif spec.parallel_mode == "expert_row":
        # ``x`` is the local FFN shard. Summing the low-rank intermediate across
        # expert-TP reproduces the full contraction; the dispatcher owns the
        # output reduce-scatter, so it must not be duplicated here and each rank
        # keeps only the slice of the rank dimension its own B factor owns.
        hidden = F.linear(x, _sum_gradient_over_replicas(lora_a, spec.replica_groups_a))
        if spec.tp_size > 1:
            hidden = symmetric_all_reduce(hidden, expert_tensor_parallel_group())
            start = spec.tp_rank * spec.rank_local
            hidden = hidden[..., start : start + spec.rank_local]
        delta = F.linear(hidden, _sum_gradient_over_replicas(lora_b, spec.replica_groups_b))
    elif spec.parallel_mode == "column":
        # ColumnParallelLinear gathers a sequence-parallel input before its GEMM.
        # Do the same before slicing the contracted hidden dimension: reducing
        # local sequence shards first would mix different tokens across TP ranks.
        if spec.tp_size > 1 and spec.sequence_parallel:
            x = _gather_along_sequence(x)
        elif spec.tp_size > 1:
            from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region

            x = copy_to_tensor_model_parallel_region(x)
        if spec.tp_size > 1:
            start = spec.tp_rank * spec.lora_in_features
            x = x[..., start : start + spec.lora_in_features]
        hidden = F.linear(x, lora_a)
        if spec.tp_size > 1:
            hidden = symmetric_all_reduce(hidden, tensor_model_parallel_group())
        delta = F.linear(hidden, lora_b)
    elif spec.parallel_mode == "row":
        hidden = F.linear(x, lora_a)
        if spec.tp_size > 1:
            hidden = symmetric_all_reduce(hidden, tensor_model_parallel_group())
            start = spec.tp_rank * spec.rank_local
            hidden = hidden[..., start : start + spec.rank_local]
        delta = F.linear(hidden, lora_b)
        if spec.tp_size > 1:
            delta = (
                _reduce_scatter_along_sequence(delta)
                if spec.sequence_parallel
                else _reduce_across_tensor_parallel(delta)
            )
    else:
        delta = F.linear(F.linear(x, lora_a), lora_b)

    return delta * config.scale


def lora_forward_hook(module: torch.nn.Module, args: tuple, kwargs: dict, output):
    """Add the adapter contribution to the base module's output.

    A hook rather than a ``forward`` override: an *instance* attribute named
    ``forward`` -- installed by TE wrappers, CUDA-graph capture and some HF modules --
    shadows any class-level override and would silently disable the adapter. Hooks are
    invoked by ``_call_impl`` itself and survive ``Float16Module`` wrapping and
    ``__class__`` reassignment.
    """
    input_ = _resolve_forward_input(module, args, kwargs)
    delta = compute_lora_delta(module, input_)

    # Cheap probes so a disconnected adapter graph is diagnosable directly
    # instead of only surfacing as an all-zero gradient.
    module._slime_lora_forward_calls = getattr(module, "_slime_lora_forward_calls", 0) + 1
    module._slime_lora_last_grad_enabled = torch.is_grad_enabled()
    module._slime_lora_last_delta_requires_grad = bool(delta.requires_grad)

    if isinstance(output, tuple):
        return (output[0] + delta, *output[1:])
    return output + delta


def reset_lora_forward_counters(model: torch.nn.Module | list[torch.nn.Module]) -> None:
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for module in chunk.modules():
            if hasattr(module, _SPEC_ATTR):
                module._slime_lora_forward_calls = 0
                module._slime_lora_last_grad_enabled = None
                module._slime_lora_last_delta_requires_grad = None


def lora_forward_diagnostics(model: torch.nn.Module | list[torch.nn.Module]) -> dict[str, Any]:
    """Forward-side probes explaining why a backward may not have reached the adapters."""
    chunks = model if isinstance(model, (list, tuple)) else [model]
    total = 0
    executed = 0
    forward_calls = 0
    grad_disabled: list[str] = []
    delta_detached: list[str] = []
    never_executed: list[str] = []
    hook_missing: list[str] = []
    shadowed_forward: list[str] = []
    for chunk in chunks:
        for name, module in chunk.named_modules():
            if not hasattr(module, _SPEC_ATTR):
                continue
            total += 1
            calls = getattr(module, "_slime_lora_forward_calls", 0)
            forward_calls += calls
            if calls:
                executed += 1
                if getattr(module, "_slime_lora_last_grad_enabled", None) is False:
                    grad_disabled.append(name)
                if getattr(module, "_slime_lora_last_delta_requires_grad", None) is False:
                    delta_detached.append(name)
            else:
                never_executed.append(name)
            if not any(hook is lora_forward_hook for hook in module._forward_hooks.values()):
                hook_missing.append(name)
            if "forward" in vars(module):
                shadowed_forward.append(name)
    return {
        "module_count": total,
        "modules_executed": executed,
        "forward_calls": forward_calls,
        "modules_never_executed": never_executed,
        "modules_with_grad_disabled": grad_disabled,
        "modules_with_detached_delta": delta_detached,
        "modules_without_hook": hook_missing,
        "modules_with_instance_forward": shadowed_forward,
    }


def format_lora_forward_diagnostics(diagnostics: dict[str, Any]) -> str:
    def sample(key: str) -> str:
        values = diagnostics[key]
        return f"{len(values)}" + (f", e.g. {values[:3]}" if values else "")

    return (
        f"  injected modules             : {diagnostics['module_count']}\n"
        f"  modules that ran forward     : {diagnostics['modules_executed']}\n"
        f"  total adapter forward calls  : {diagnostics['forward_calls']}\n"
        f"  modules never executed       : {sample('modules_never_executed')}\n"
        f"  modules missing the hook     : {sample('modules_without_hook')}\n"
        f"  modules with instance forward: {sample('modules_with_instance_forward')}\n"
        f"  forwards under no_grad       : {sample('modules_with_grad_disabled')}\n"
        f"  deltas without requires_grad : {sample('modules_with_detached_delta')}"
    )


@torch.no_grad()
def attach_lora_adapter(
    module: torch.nn.Module,
    spec: LoRAParallelSpec,
    config: LoRAConfig,
    *,
    generator: torch.Generator | None = None,
) -> None:
    """Attach ``lora_A`` / ``lora_B`` to ``module`` and enable the LoRA forward path."""
    weight = module.weight if not spec.is_expert else grouped_expert_weights(module)[0]
    factory = {"dtype": weight.dtype, "device": weight.device}

    lora_a = torch.nn.Parameter(torch.empty(spec.rank, spec.lora_in_features, **factory))
    lora_b = torch.nn.Parameter(torch.zeros(spec.lora_out_features, spec.rank_local, **factory))

    # Kaiming-uniform on A (a=sqrt(5) => bound = 1/sqrt(fan_in)), zeros on B. B @ A == 0, so the
    # initial LoRA policy reproduces the base model exactly. fan_in uses the *global* input width so
    # the initialisation scale does not depend on the tensor-parallel size.
    bound = 1.0 / math.sqrt(spec.global_in_features)
    init = torch.empty(lora_a.shape, dtype=torch.float32, device=lora_a.device)
    init.uniform_(-bound, bound, generator=generator)
    lora_a.copy_(init.to(lora_a.dtype))

    is_parallel = spec.tp_size > 1 or spec.parallel_mode != "replicated"
    if spec.parallel_mode == "column":
        _set_tp_attributes(lora_a, is_parallel=is_parallel, dim=1, stride=1)
        _set_tp_attributes(lora_b, is_parallel=is_parallel, dim=0, stride=1)
    elif spec.parallel_mode == "row":
        _set_tp_attributes(lora_a, is_parallel=is_parallel, dim=1, stride=1)
        _set_tp_attributes(lora_b, is_parallel=is_parallel, dim=1, stride=1)
    elif spec.parallel_mode == "expert_column":
        # A consumes the full hidden width and is replicated over expert-TP.
        _set_tp_attributes(lora_a, is_parallel=False, dim=0, stride=1)
        _set_tp_attributes(lora_b, is_parallel=spec.tp_size > 1, dim=0, stride=1)
    elif spec.parallel_mode == "expert_row":
        _set_tp_attributes(lora_a, is_parallel=spec.tp_size > 1, dim=1, stride=1)
        _set_tp_attributes(lora_b, is_parallel=spec.tp_size > 1, dim=1, stride=1)
    else:
        _set_tp_attributes(lora_a, is_parallel=False, dim=0, stride=1)
        _set_tp_attributes(lora_b, is_parallel=False, dim=0, stride=1)

    if spec.is_expert:
        # Megatron's DDP routes parameters with allreduce=False into the separate
        # expert-parallel gradient buffer reduced over the expert-DP group, which
        # is exactly the grouping TE gives the base expert weights. Keeping the
        # adapters in the same bucket keeps their gradient scaling consistent.
        use_expert_groups = (
            expert_model_parallel_world_size() > 1 or spec.tp_size != tensor_model_parallel_world_size()
        )
        # Preserve additional expert-topology requirements encoded by the base
        # layer, while also handling older TE versions that only checked EP.
        for param in (lora_a, lora_b):
            param.allreduce = getattr(weight, "allreduce", True) and not use_expert_groups

    module.register_parameter(LORA_A_NAME, lora_a)
    module.register_parameter(LORA_B_NAME, lora_b)
    module._slime_lora_original_sharded_state_dict = getattr(module, "sharded_state_dict", None)
    module.sharded_state_dict = MethodType(lora_sharded_state_dict, module)

    _broadcast_replicated_factors(lora_a, lora_b, spec)

    setattr(module, _SPEC_ATTR, spec)
    setattr(module, _CONFIG_ATTR, config)
    module._slime_lora_dropout = torch.nn.Dropout(config.dropout) if config.dropout > 0.0 else None

    # ``register_forward_hook`` instead of a ``forward`` override: see
    # ``lora_forward_hook`` for why class-level overrides are not reliable here.
    handle = module.register_forward_hook(lora_forward_hook, with_kwargs=True)
    module._slime_lora_hook_handle = handle
    module._slime_lora_forward_calls = 0
    module._slime_lora_last_grad_enabled = None
    module._slime_lora_last_delta_requires_grad = None


def is_lora_module(module: torch.nn.Module) -> bool:
    return hasattr(module, _SPEC_ATTR)


@torch.no_grad()
def _broadcast_replicated_factors(
    lora_a: torch.nn.Parameter, lora_b: torch.nn.Parameter, spec: LoRAParallelSpec
) -> None:
    """Make every replica of a factor start from bit-identical values.

    Ambient RNG state is not guaranteed to match across ranks, and a factor that
    is replicated rather than sharded must agree everywhere for the merged weight
    to be consistent. The gradients are summed over the same groups in backward,
    so the replicas stay in lockstep afterwards.
    """
    import torch.distributed as dist

    groups = []
    if spec.parallel_mode == "replicated" and spec.tp_size > 1:
        # Explicitly allow-listed replicated linears share their factors across TP.
        groups.append((tensor_model_parallel_group(), (lora_a, lora_b)))
    elif spec.is_expert:
        # One adapter is shared by every routed expert, so it is replicated over
        # expert parallelism, and A is additionally replicated over expert-TP in
        # the column case.
        if expert_model_parallel_world_size() > 1:
            groups.append((expert_model_parallel_group(), (lora_a, lora_b)))
        if not spec.a_is_tensor_parallel and spec.tp_size > 1:
            groups.append((expert_tensor_parallel_group(), (lora_a,)))

    for group, parameters in groups:
        source_rank = dist.get_global_rank(group, 0)
        for parameter in parameters:
            _broadcast_parameter(parameter, src=source_rank, group=group)


def lora_sharded_state_dict(module, prefix="", sharded_offsets=(), metadata=None):
    """Preserve the base checkpoint contract and shard the adapter factors explicitly.

    Megatron linear layers use an explicit TP axis map, not Parameter.partition_dim.
    Without these entries, their default state dict treats the adapters as TP replicas.
    """
    from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint

    metadata = ensure_metadata_has_dp_cp_group(metadata)
    original = module._slime_lora_original_sharded_state_dict
    spec = getattr(module, _SPEC_ATTR)
    tp_group = getattr(module, "_tp_group", None)
    if tp_group is None:
        tp_group = getattr(module, "tp_group", None)
    if tp_group is None:
        tp_group = spec_parallel_group(spec)
    axes = {}
    if spec.parallel_mode == "column":
        axes = {LORA_A_NAME: 1, LORA_B_NAME: 0}
    elif spec.parallel_mode == "row":
        axes = {LORA_A_NAME: 1, LORA_B_NAME: 1}
    elif spec.parallel_mode == "expert_column":
        # A is replicated over expert-TP, so it gets no axis entry.
        axes = {LORA_B_NAME: 0}
    elif spec.parallel_mode == "expert_row":
        axes = {LORA_A_NAME: 1, LORA_B_NAME: 1}

    if original is None:
        # Plain torch linears are only accepted as replicated modules.
        state = module.state_dict(prefix="", keep_vars=True)
        result = {}
    else:
        result = original(prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata)
        state = {LORA_A_NAME: module.lora_A, LORA_B_NAME: module.lora_B}
    adapters = make_sharded_tensors_for_checkpoint(
        state, prefix, axes, sharded_offsets, tp_group=tp_group, dp_cp_group=metadata["dp_cp_group"]
    )
    if spec.is_expert:
        # The shared adapter is replicated over expert parallelism, so the DP
        # coordinate of its replica id must distinguish those ranks. Without this
        # every EP rank claims to be the primary writer of the same key and the
        # distributed checkpoint save fails its uniqueness check.
        _mark_expert_adapter_replicas(adapters)
    result.update(adapters)
    return result


def _mark_expert_adapter_replicas(sharded_tensors: dict) -> None:
    """Give each expert-parallel replica of a shared adapter a distinct replica id.

    Unlike the base expert weights, whose keys embed the global expert index, the
    shared adapter has one key that every expert-parallel rank holds a copy of.
    Megatron picks the ``replica_id == 0`` rank as the writer, so the replica id
    must encode both the expert-parallel and the expert-data-parallel coordinate.
    """
    from megatron.core import mpu

    try:
        expert_rank = mpu.get_expert_model_parallel_rank()
        expert_dp_rank = mpu.get_expert_data_parallel_group().rank()
        expert_size = mpu.get_expert_model_parallel_world_size()
    except (AssertionError, AttributeError):
        return

    replica_index = expert_dp_rank * expert_size + expert_rank
    for key, sharded_tensor in sharded_tensors.items():
        replica_id = sharded_tensor.replica_id
        if not isinstance(replica_id, tuple) or len(replica_id) != 3:
            raise LoRAInjectionError(
                f"Expected replica_id for {key} to be in (PP, TP, DP) format, got: {replica_id}"
            )
        sharded_tensor.replica_id = (*replica_id[:2], replica_index)


@torch.no_grad()
def lora_local_delta(
    module: torch.nn.Module,
    *,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """``scale * B @ A`` for the *local* base shard, computed in fp32.

    Requires a small TP all-gather of the sharded factor; never touches the base
    weight and never mutates the adapter. ``lora_a`` / ``lora_b`` allow callers
    to merge a coherent host-side policy snapshot instead of the live module.

    For routed experts this returns the delta of the *shared* adapter, which the
    caller adds to every local expert weight.
    """
    spec: LoRAParallelSpec = getattr(module, _SPEC_ATTR)
    config: LoRAConfig = getattr(module, _CONFIG_ATTR)
    adapter_device = module.lora_A.device
    lora_a = (module.lora_A if lora_a is None else lora_a).detach().to(device=adapter_device, dtype=torch.float32)
    lora_b = (module.lora_B if lora_b is None else lora_b).detach().to(device=adapter_device, dtype=torch.float32)

    if spec.tp_size > 1:
        group = spec_parallel_group(spec)
        if spec.parallel_mode == "column":
            lora_a = _all_gather_concat(lora_a, dim=1, world_size=spec.tp_size, group=group)
        elif spec.parallel_mode in ("row", "expert_row"):
            # Both row modes shard B over the rank dimension.
            lora_b = _all_gather_concat(lora_b, dim=1, world_size=spec.tp_size, group=group)
        # expert_column needs no gather: A is already replicated and B is
        # sharded exactly like the local base output.

    delta = config.scale * (lora_b @ lora_a)
    if tuple(delta.shape) != spec.base_shape:
        raise LoRAInjectionError(
            f"LoRA merge shape mismatch for {spec.module_name!r}: delta {tuple(delta.shape)} "
            f"!= base {spec.base_shape} (parallel_mode={spec.parallel_mode}, tp={spec.tp_size}, "
            f"ranks={parallel_rank_context()})"
        )
    return delta


def _all_gather_concat(tensor: torch.Tensor, *, dim: int, world_size: int, group=None) -> torch.Tensor:
    import torch.distributed as dist

    tensor = tensor.contiguous()
    shards = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(shards, tensor, group=group if group is not None else tensor_model_parallel_group())
    return torch.cat(shards, dim=dim)
