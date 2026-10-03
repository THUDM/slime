"""Non-destructive materialisation of ``W_effective = W_base + (alpha / r) * B @ A``.

Everything in this module exists for exactly two consumers: rollout weight
synchronisation and merged HF export. Training, adapter checkpointing and the
optimizer keep using the raw parameter iterator, so the frozen base weights are
never modified in place and the delta is never accumulated twice.
"""

from __future__ import annotations

import logging
import re
from argparse import Namespace
from collections.abc import Iterator, Mapping, Sequence

import torch

from slime.utils.lora_config import LORA_A_NAME, LORA_B_NAME, is_lora_param_name
from .inject import iter_lora_modules
from .layers import _SPEC_ATTR, LoRAInjectionError, lora_local_delta

logger = logging.getLogger(__name__)

#: ``weight`` for dense targets, ``weight0``/``weight1``/... for grouped experts.
_BASE_WEIGHT_SUFFIX = re.compile(r"weight\d*")

_DISTRIBUTED_TENSOR_ATTRIBUTES = (
    "tensor_model_parallel",
    "partition_dim",
    "partition_stride",
    "parallel_mode",
    "sequence_parallel",
    "allreduce",
)


def build_merge_plan(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    *,
    convert_to_global_name: bool = True,
) -> dict[str, torch.nn.Module]:
    """Map the canonical name of every adapted base weight to its LoRA module.

    A routed-expert module contributes several entries -- one per local expert --
    that all point at the same shared adapter, so the expert count rather than the
    module count is what the completeness check compares against.
    """
    from ..update_weight.common import named_params_and_buffers

    modules_by_weight_id: dict[int, torch.nn.Module] = {}
    expected_entries = 0
    for _, module in iter_lora_modules(list(model)):
        spec = getattr(module, _SPEC_ATTR)
        for weight_name in spec.base_weight_names:
            weight = getattr(module, weight_name, None)
            if weight is None:
                raise LoRAInjectionError(
                    f"LoRA module {spec.module_name!r} is missing its base weight {weight_name!r}"
                )
            modules_by_weight_id[id(weight)] = module
            expected_entries += 1

    plan: dict[str, torch.nn.Module] = {}
    for name, tensor in named_params_and_buffers(args, model, convert_to_global_name=convert_to_global_name):
        module = modules_by_weight_id.get(id(tensor))
        if module is not None:
            plan[name] = module

    if len(plan) != expected_entries:
        raise LoRAInjectionError(
            f"Could not resolve canonical names for all LoRA base weights: resolved {len(plan)} of "
            f"{expected_entries}. The parameter iterator and the injected modules disagree."
        )
    return plan


@torch.no_grad()
def merge_into(
    base: torch.Tensor,
    module: torch.nn.Module,
    *,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a fresh tensor ``base + scale * B @ A``; ``base`` is left untouched."""
    delta = lora_local_delta(module, lora_a=lora_a, lora_b=lora_b)
    if tuple(delta.shape) != tuple(base.shape):
        raise LoRAInjectionError(
            f"LoRA delta shape {tuple(delta.shape)} does not match base weight shape {tuple(base.shape)}"
        )
    merged = base.to(dtype=torch.float32) + delta.to(device=base.device, dtype=torch.float32)
    result = merged.to(dtype=base.dtype)
    # ``all_gather_param`` and Transformer Engine inspect these Python-side
    # attributes. Arithmetic creates a fresh Tensor and drops them, so preserve
    # the base shard's distribution contract on the effective weight.
    for attr in _DISTRIBUTED_TENSOR_ATTRIBUTES:
        if hasattr(base, attr):
            setattr(result, attr, getattr(base, attr))
    del merged, delta
    return result


def merged_named_params(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """``named_params_and_buffers`` with the adapters folded in and ``lora_A/lora_B`` dropped.

    Tensors are merged one at a time and released as soon as the consumer is done,
    so no second full copy of the model is kept alive.
    """
    from ..update_weight.common import named_params_and_buffers

    plan = build_merge_plan(args, model, convert_to_global_name=convert_to_global_name)
    for name, tensor in named_params_and_buffers(args, model, convert_to_global_name=convert_to_global_name):
        if is_lora_param_name(name):
            continue
        module = plan.get(name)
        yield name, (tensor if module is None else merge_into(tensor, module))


class EffectiveWeightMapping(Mapping):
    """Lazy ``name -> effective tensor`` view.

    Used where the weight-sync code expects a mapping instead of an iterator
    (colocated tensor/IPC sync and ``--save-hf``). Merging happens inside
    ``__getitem__`` so only the tensors of the current bucket are materialised.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        backing: Mapping[str, torch.Tensor],
        *,
        convert_to_global_name: bool = True,
    ) -> None:
        self._backing = backing
        self._plan = build_merge_plan(args, model, convert_to_global_name=convert_to_global_name)
        self._names = [name for name in backing if not is_lora_param_name(name)]

    def __getitem__(self, name: str) -> torch.Tensor:
        base = self._backing[name]
        module = self._plan.get(name)
        if module is None:
            return base
        prefix, separator, suffix = name.rpartition(".")
        # Routed experts are packed as weight0, weight1, ... and share one adapter.
        if separator != "." or not _BASE_WEIGHT_SUFFIX.fullmatch(suffix):
            raise LoRAInjectionError(f"Adapted base tensor has unexpected canonical name {name!r}")
        lora_a_name = f"{prefix}.{LORA_A_NAME}"
        lora_b_name = f"{prefix}.{LORA_B_NAME}"
        try:
            lora_a = self._backing[lora_a_name]
            lora_b = self._backing[lora_b_name]
        except KeyError as exc:
            raise LoRAInjectionError(
                f"Policy snapshot for {name!r} is missing adapter tensor {exc.args[0]!r}"
            ) from exc
        return merge_into(base, module, lora_a=lora_a, lora_b=lora_b)

    def __iter__(self):
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)
