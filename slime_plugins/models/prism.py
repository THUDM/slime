# Copyright (c) 2025, OpenMOSS. All rights reserved.
"""Prism's native Megatron model and selected-action scoring adapter.

A chronological row is a text token or a complete RVQ frame. The Tensor
returned by ``forward`` is [1, T, Q + 1]: Q selected code log probabilities,
then the selected binary stop log probability. There is no extra token shift.
An optional second Q + 1 columns contain per-head entropy. All distributions
use temperature one; policy masks describe real sampled actions separately.

The pure topology, configuration and carrier helpers can be used without
Megatron installed. Constructing the training model requires Megatron Core.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from enum import IntEnum
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_MEGATRON_IMPORT_ERROR = None
try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.models.common.language_module.language_module import LanguageModule
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import ModelType
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.spec_utils import ModuleSpec, build_module
    from megatron.core.transformer.transformer_block import (
        LayerNormImpl,
        TransformerBlock,
        TransformerBlockSubmodules,
    )
    from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
    from megatron.core.utils import init_method_normal
except ImportError as error:
    _MEGATRON_IMPORT_ERROR = error
    LanguageModule = LanguageModelEmbedding = TransformerLayer = nn.Module
    TransformerConfig = object
    tensor_parallel = SimpleNamespace(ColumnParallelLinear=nn.Module)


def _require_megatron():
    if _MEGATRON_IMPORT_ERROR is not None:
        raise RuntimeError("Prism training requires a working Megatron Core installation") from _MEGATRON_IMPORT_ERROR


class PrismItemKind(IntEnum):
    TEXT = 0
    AUDIO = 1


class PrismAudioRole(IntEnum):
    NONE = 0
    REFERENCE = 1
    TARGET = 2


PrismSiteRecord: TypeAlias = tuple[Literal["ordinary"], int] | tuple[Literal["loop"], int, int]


@dataclass(frozen=True, slots=True)
class PrismExecutionSite:
    """Identity of one prediction site in physical execution order.

    Ordinary physical-layer indices are zero-based. Loop block and application
    indices are one-based, matching the architecture notation ``C(b, r)``.
    """

    kind: Literal["ordinary", "loop"]
    index: int
    application: int | None = None

    def canonical_record(self) -> PrismSiteRecord:
        if self.kind == "ordinary":
            return ("ordinary", self.index)
        if self.application is None:  # pragma: no cover - construction invariant
            raise RuntimeError("loop execution site is missing its application")
        return ("loop", self.index, self.application)


@dataclass(frozen=True, slots=True)
class PrismExecutionUnit:
    """One ordinary layer or one complete loop-range application."""

    site: PrismExecutionSite
    layer_start: int
    layer_stop: int


@dataclass(frozen=True, slots=True)
class ResolvedPrismTopology:
    """Immutable, canonical Prism construction and execution topology."""

    n_vq: int
    num_layers: int
    prism_mlp_only_layers: tuple[int, ...]
    prism_loop_block_ranges: tuple[tuple[int, int], ...]
    prism_loop_block_repetitions: tuple[int, ...]
    prism_rvq_heads_per_execution_unit: tuple[int, ...]
    reapply_rvq_conditioning_in_loop: bool
    execution_units: tuple[PrismExecutionUnit, ...]
    sigma: tuple[PrismExecutionSite, ...]
    co_site_groups: tuple[tuple[int, ...], ...]
    occupied_execution_unit_indices: tuple[int, ...]
    prediction_norm_index_by_execution_unit: tuple[int | None, ...]

    @property
    def num_loop_blocks(self) -> int:
        return len(self.prism_loop_block_ranges)

    @property
    def num_execution_units(self) -> int:
        return len(self.execution_units)

    @property
    def num_prediction_norms(self) -> int:
        return len(self.occupied_execution_unit_indices)

    @property
    def num_effective_layer_calls(self) -> int:
        return sum(unit.layer_stop - unit.layer_start for unit in self.execution_units)

    @property
    def execution_sites(self) -> tuple[PrismExecutionSite, ...]:
        return tuple(unit.site for unit in self.execution_units)

    @property
    def final_site(self) -> PrismExecutionSite:
        return self.execution_units[-1].site

    @property
    def site_records(self) -> tuple[PrismSiteRecord, ...]:
        return tuple(site.canonical_record() for site in self.execution_sites)

    @property
    def sigma_records(self) -> tuple[PrismSiteRecord, ...]:
        return tuple(site.canonical_record() for site in self.sigma)

    @property
    def loop_block_prefix_counts(self) -> tuple[int, ...]:
        """Return leading unoccupied applications before each block's first head.

        The compact head-site vector already contains this information.  A
        prefix count is therefore a runtime scheduling property rather than a
        second topology field.  Every valid block has an occupied final
        application, so a first occupied application always exists.
        """

        spans: list[tuple[int, int, int]] = []
        index = 0
        while index < len(self.execution_units):
            unit = self.execution_units[index]
            if unit.site.kind != "loop":
                index += 1
                continue
            block = unit.site.index
            start = index
            while index < len(self.execution_units):
                candidate = self.execution_units[index]
                if candidate.site.kind != "loop" or candidate.site.index != block:
                    break
                index += 1
            spans.append((block, start, index))

        if len(spans) != self.num_loop_blocks:  # pragma: no cover - resolver invariant
            raise RuntimeError("Prism topology did not contain every loop block")
        result: list[int] = []
        for block, block_start, block_stop in spans:
            block_groups = self.co_site_groups[block_start:block_stop]
            first_head = next((offset for offset, group in enumerate(block_groups) if group), None)
            if first_head is None:  # pragma: no cover - resolver invariant
                raise RuntimeError(f"Prism loop block {block} has no RVQ head")
            result.append(first_head)
        return tuple(result)

    def execution_unit_indices_for_prefix_counts(self, prefix_counts: Sequence[int] | None = None) -> tuple[int, ...]:
        """Select execution units for one forward's loop prefix schedule.

        ``prefix_counts[b]`` is the number of unoccupied applications retained
        immediately before block ``b``'s first occupied site.  The omitted
        applications are only the leading zero-head applications; all
        occupied sites and all units after the first occupied site remain in
        their canonical order.  ``None`` selects the complete static schedule.
        """

        maximums = self.loop_block_prefix_counts
        if prefix_counts is None:
            return tuple(range(len(self.execution_units)))
        if isinstance(prefix_counts, (str, bytes, bytearray)) or not isinstance(prefix_counts, Sequence):
            raise TypeError("prefix_counts must be an ordered integer vector")
        if len(prefix_counts) != self.num_loop_blocks:
            raise ValueError("prefix_counts must contain exactly " f"{self.num_loop_blocks} entries")
        normalized: list[int] = []
        for block, (value, maximum) in enumerate(zip(prefix_counts, maximums, strict=True), start=1):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"prefix_counts[{block - 1}] must be a non-Boolean integer")
            if not 0 <= value <= maximum:
                raise ValueError(f"prefix_counts[{block - 1}] must be in [0, {maximum}]")
            normalized.append(value)

        selected: list[int] = []
        block = 0
        index = 0
        while index < len(self.execution_units):
            unit = self.execution_units[index]
            if unit.site.kind != "loop":
                selected.append(index)
                index += 1
                continue
            block_index = unit.site.index - 1
            expected_block = block_index + 1
            block_start = index
            while index < len(self.execution_units):
                candidate = self.execution_units[index]
                if candidate.site.kind != "loop" or candidate.site.index != expected_block:
                    break
                index += 1
            leading = maximums[block_index]
            retained_prefix = normalized[block_index]
            selected.extend(range(block_start, block_start + retained_prefix))
            selected.extend(range(block_start + leading, index))
            block += 1
        if block != self.num_loop_blocks:  # pragma: no cover - resolver invariant
            raise RuntimeError("Prism schedule did not visit every loop block")
        return tuple(selected)

    def canonical_dict(self) -> dict[str, Any]:
        """Return the alias-free, JSON-compatible topology representation."""

        return {
            "n_vq": self.n_vq,
            "num_layers": self.num_layers,
            "prism_mlp_only_layers": list(self.prism_mlp_only_layers),
            "prism_loop_block_ranges": [[start, stop] for start, stop in self.prism_loop_block_ranges],
            "prism_loop_block_repetitions": list(self.prism_loop_block_repetitions),
            "prism_rvq_heads_per_execution_unit": list(self.prism_rvq_heads_per_execution_unit),
            "sigma": [list(record) for record in self.sigma_records],
            "reapply_rvq_conditioning_in_loop": (self.reapply_rvq_conditioning_in_loop),
        }


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive non-Boolean integer")
    return value


def normalize_prism_rvq_label_smoothing(value: object, *, n_vq: int) -> tuple[float, ...]:
    """Return one finite label-smoothing value for every RVQ channel.

    ``None`` is the intentional omitted-option spelling and means that
    smoothing is disabled for every audio head.  Once supplied, the vector is
    deliberately exact-length: scalar broadcasting and an empty explicit
    vector are both rejected.
    """

    n_vq = _positive_integer(n_vq, "n_vq")
    if value is None:
        return (0.0,) * n_vq
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("rvq_label_smoothing must be an ordered vector")
    if len(value) != n_vq:
        raise ValueError("rvq_label_smoothing must contain exactly " f"n_vq={n_vq} entries")
    result: list[float] = []
    for index, entry in enumerate(value):
        if isinstance(entry, bool) or not isinstance(entry, Real):
            raise ValueError(f"rvq_label_smoothing[{index}] must be a real non-Boolean number")
        number = float(entry)
        if not math.isfinite(number) or not 0.0 <= number < 1.0:
            raise ValueError(f"rvq_label_smoothing[{index}] must be finite and in [0, 1)")
        result.append(number)
    return tuple(result)


def normalize_prism_input_rvq_channels(value: object, *, n_vq: int) -> tuple[int, ...]:
    """Return the canonical 1-based RVQ channels used at backbone input."""

    n_vq = _positive_integer(n_vq, "n_vq")
    if value is None:
        return tuple(range(1, n_vq + 1))
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("prism_input_rvq_channels must be an ordered vector")
    if not value:
        raise ValueError("prism_input_rvq_channels must not be empty")
    result: list[int] = []
    previous = 0
    for index, entry in enumerate(value):
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ValueError(f"prism_input_rvq_channels[{index}] must be a positive " "non-Boolean integer")
        if not 1 <= entry <= n_vq:
            raise ValueError(f"prism_input_rvq_channels[{index}] must be in [1, {n_vq}]")
        if entry <= previous:
            raise ValueError("prism_input_rvq_channels must be strictly increasing without " "duplicates")
        result.append(entry)
        previous = entry
    return tuple(result)


def _ordered_collection(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an ordered collection")
    return value


def _canonical_ranges(value: object) -> tuple[tuple[int, int], ...]:
    ranges = _ordered_collection(value, "prism_loop_block_ranges")
    result: list[tuple[int, int]] = []
    previous_stop = 0
    for block_offset, raw_range in enumerate(ranges):
        name = f"prism_loop_block_ranges[{block_offset}]"
        pair = _ordered_collection(raw_range, name)
        if len(pair) != 2:
            raise ValueError(f"{name} must contain exactly [start, stop]")
        start, stop = pair
        if any(isinstance(endpoint, bool) or not isinstance(endpoint, int) for endpoint in pair):
            raise ValueError(f"{name} endpoints must be non-Boolean integers")
        if not 0 <= start < stop:
            raise ValueError(f"{name} must satisfy 0 <= start < stop")
        if result and previous_stop > start:
            raise ValueError("Prism loop block ranges must be ordered and nonoverlapping")
        result.append((start, stop))
        previous_stop = stop
    return tuple(result)


def _canonical_integer_vector(value: object, *, name: str, positive: bool) -> tuple[int, ...]:
    sequence = _ordered_collection(value, name)
    result: list[int] = []
    for index, entry in enumerate(sequence):
        if isinstance(entry, bool) or not isinstance(entry, int):
            qualifier = "positive " if positive else "nonnegative "
            raise ValueError(f"{name}[{index}] must be a {qualifier}non-Boolean integer")
        if entry < (1 if positive else 0):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"{name}[{index}] must be {qualifier}")
        result.append(entry)
    return tuple(result)


def _execution_units(
    *,
    num_layers: int,
    ranges: tuple[tuple[int, int], ...],
    repetitions: tuple[int, ...],
) -> tuple[PrismExecutionUnit, ...]:
    result: list[PrismExecutionUnit] = []
    layer = 0
    for block_offset, ((start, stop), repeat) in enumerate(zip(ranges, repetitions, strict=True)):
        while layer < start:
            site = PrismExecutionSite("ordinary", layer)
            result.append(PrismExecutionUnit(site, layer, layer + 1))
            layer += 1
        block = block_offset + 1
        for application in range(1, repeat + 1):
            site = PrismExecutionSite("loop", block, application)
            result.append(PrismExecutionUnit(site, start, stop))
        layer = stop
    while layer < num_layers:
        site = PrismExecutionSite("ordinary", layer)
        result.append(PrismExecutionUnit(site, layer, layer + 1))
        layer += 1
    return tuple(result)


def resolve_prism_topology(values: object) -> ResolvedPrismTopology:
    """Resolve exact Prism config/argument attributes without mutating ``values``."""

    n_vq = _positive_integer(values.n_vq, "n_vq")
    ranges = _canonical_ranges(values.prism_loop_block_ranges)
    repetitions = _canonical_integer_vector(
        values.prism_loop_block_repetitions,
        name="prism_loop_block_repetitions",
        positive=True,
    )
    if len(repetitions) != len(ranges):
        raise ValueError("prism_loop_block_repetitions must have one entry per loop block range")
    if len(ranges) > n_vq:
        raise ValueError("the number of Prism loop blocks cannot exceed n_vq")

    counts = _canonical_integer_vector(
        values.prism_rvq_heads_per_execution_unit,
        name="prism_rvq_heads_per_execution_unit",
        positive=False,
    )
    if not counts:
        raise ValueError("prism_rvq_heads_per_execution_unit must contain at least one " "execution unit")
    num_layers = len(counts) + sum(
        (stop - start) - repeat for (start, stop), repeat in zip(ranges, repetitions, strict=True)
    )
    if num_layers <= 0:
        raise ValueError("the concrete Prism topology must infer a positive physical layer count")
    mlp_only_layers = _canonical_integer_vector(
        getattr(values, "prism_mlp_only_layers", ()),
        name="prism_mlp_only_layers",
        positive=False,
    )
    if mlp_only_layers != tuple(sorted(set(mlp_only_layers))):
        raise ValueError("prism_mlp_only_layers must be strictly increasing without duplicates")
    if any(index >= num_layers for index in mlp_only_layers):
        raise ValueError(f"prism_mlp_only_layers entries must be in [0, {num_layers})")
    for block_offset, (_start, stop) in enumerate(ranges):
        if stop > num_layers:
            raise ValueError(
                f"prism_loop_block_ranges[{block_offset}] must stop at or before "
                f"the inferred physical layer count ({num_layers})"
            )

    reapply = getattr(values, "reapply_rvq_conditioning_in_loop", False)
    if not isinstance(reapply, bool):
        raise ValueError("reapply_rvq_conditioning_in_loop must be exactly Boolean")

    units = _execution_units(num_layers=num_layers, ranges=ranges, repetitions=repetitions)
    if len(counts) != len(units):
        raise RuntimeError("inferred Prism physical depth did not reproduce the declared " "execution-unit count")
    if sum(counts) != n_vq:
        raise ValueError("prism_rvq_heads_per_execution_unit must sum exactly to n_vq")
    if counts[-1] == 0:
        raise ValueError("the final Prism execution site must own at least one RVQ head")

    co_site_groups: list[tuple[int, ...]] = []
    sigma: list[PrismExecutionSite] = []
    next_q = 1
    for unit, count in zip(units, counts, strict=True):
        group = tuple(range(next_q, next_q + count))
        co_site_groups.append(group)
        sigma.extend((unit.site,) * count)
        next_q += count

    occupied_execution_unit_indices = tuple(index for index, group in enumerate(co_site_groups) if group)
    prediction_norm_index_by_execution_unit: list[int | None] = []
    next_prediction_norm = 0
    for group in co_site_groups:
        prediction_norm_index_by_execution_unit.append(next_prediction_norm if group else None)
        if group:
            next_prediction_norm += 1

    count_by_site = {unit.site: count for unit, count in zip(units, counts, strict=True)}
    for block, repeat in enumerate(repetitions, start=1):
        final_block_site = PrismExecutionSite("loop", block, repeat)
        if count_by_site[final_block_site] == 0:
            raise ValueError(f"Prism loop block {block} final application must own an RVQ head")

    if sigma[-1] != units[-1].site:  # defensive expression of sigma(n_vq)=v_M
        raise ValueError("RVQ n_vq must be assigned to the final execution site")

    return ResolvedPrismTopology(
        n_vq=n_vq,
        num_layers=num_layers,
        prism_mlp_only_layers=mlp_only_layers,
        prism_loop_block_ranges=ranges,
        prism_loop_block_repetitions=repetitions,
        prism_rvq_heads_per_execution_unit=counts,
        reapply_rvq_conditioning_in_loop=reapply,
        execution_units=units,
        sigma=tuple(sigma),
        co_site_groups=tuple(co_site_groups),
        occupied_execution_unit_indices=occupied_execution_unit_indices,
        prediction_norm_index_by_execution_unit=tuple(prediction_norm_index_by_execution_unit),
    )


def validate_prism_stop_gradient_layer(value: object, topology: ResolvedPrismTopology) -> int | None:
    """Validate one physical-layer gradient boundary without changing topology.

    The boundary is expressed in physical coordinates: ``K`` means that the
    first ``K`` layers have run and the first post-boundary unit receives
    detached copies of both the crossing residual and incoming
    successor-conditioning delta.  A boundary strictly inside a loop range has
    no single place in the expanded recurrence, so it is rejected.  Ordinary
    boundaries and range endpoints are valid; an endpoint is handled by the
    schedule as one block transition.  Validation does not change topology.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("prism_stop_gradient_layer must be None or a non-Boolean integer")
    if not 1 <= value < topology.num_layers:
        raise ValueError(
            "prism_stop_gradient_layer must satisfy " f"1 <= K < physical_num_layers={topology.num_layers}"
        )
    for start, stop in topology.prism_loop_block_ranges:
        if start < value < stop:
            raise ValueError(
                "prism_stop_gradient_layer cannot be inside loop block " f"[{start}, {stop}); got K={value}"
            )
    return value


def validate_prism_recompute_policy(values: object, *, effective_config: bool = False) -> bool:
    """Validate Prism's one supported Megatron recomputation tuple.

    ``effective_config`` accounts for ``TransformerConfig`` replacing an
    absent ``recompute_modules`` value with its inert ``["core_attn"]``
    default. Argument validation still rejects every explicit module choice.
    The return value says whether full-layer recomputation is enabled.
    """

    modules = getattr(values, "recompute_modules", None)
    if effective_config:
        if modules not in (None, ["core_attn"], ("core_attn",)):
            raise ValueError(
                "MOSS-TTS Prism does not support an effective --recompute-modules " f"selection; got {modules!r}"
            )
    elif modules is not None:
        raise ValueError(
            "MOSS-TTS Prism does not support explicit --recompute-modules, " "including an empty selection"
        )

    granularity = getattr(values, "recompute_granularity", None)
    method = getattr(values, "recompute_method", None)
    count = getattr(values, "recompute_num_layers", None)
    distributed = getattr(values, "distribute_saved_activations", False)
    tensor_parallel_size = getattr(values, "tensor_model_parallel_size", 1)
    distributed_disabled = distributed is False or (effective_config and distributed is None)

    if granularity is None and method is None and count is None and not distributed:
        return False

    supported = (
        granularity == "full"
        and method == "uniform"
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count == 1
        and distributed_disabled
        and isinstance(tensor_parallel_size, int)
        and not isinstance(tensor_parallel_size, bool)
        and tensor_parallel_size == 1
    )
    if not supported:
        raise ValueError(
            "MOSS-TTS Prism activation recomputation supports only "
            "--recompute-granularity full --recompute-method uniform "
            "--recompute-num-layers 1 at --tensor-model-parallel-size 1, "
            "without --distribute-saved-activations; got "
            f"--recompute-granularity={granularity!r}, "
            f"--recompute-method={method!r}, "
            f"--recompute-num-layers={count!r}, "
            f"--tensor-model-parallel-size={tensor_parallel_size!r}, "
            f"--distribute-saved-activations={distributed!r}"
        )
    return True


@dataclass
class MossTTSPrismConfig(TransformerConfig):
    """Transformer and immutable execution-topology config for Prism."""

    _cli_hidden_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "prism_loop_block_ranges",
            "prism_loop_block_repetitions",
            "prism_rvq_heads_per_execution_unit",
            "prism_mlp_only_layers",
            # These options have custom vector/scalar parsers in train.py.
            "rvq_label_smoothing",
            "prism_stop_gradient_layer",
            "prism_input_rvq_channels",
            "prism_randomize_loop_prefix",
        }
    )

    transformer_impl: str = "transformer_engine"
    max_position_embeddings: int = 128000
    pad_token_id: int = 151667
    text_pad_idx: int = 151643
    audio_start_token_id: int = 151652

    n_vq: int = 8
    speech_vocab_size: int = 1024
    speech_embedding_init_method_std: float = 0.0
    speech_head_init_method_std: float = 0.02
    speech_pad_index: int = field(init=False)

    prism_loop_block_ranges: tuple[tuple[int, int], ...] = ()
    prism_mlp_only_layers: tuple[int, ...] = ()
    prism_loop_block_repetitions: tuple[int, ...] = ()
    prism_rvq_heads_per_execution_unit: tuple[int, ...] = ()
    reapply_rvq_conditioning_in_loop: bool = False
    rvq_label_smoothing: tuple[float, ...] | None = None
    prism_stop_gradient_layer: int | None = None
    prism_input_rvq_channels: tuple[int, ...] | None = None
    prism_topology: ResolvedPrismTopology = field(init=False, repr=False)

    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    kv_channels: int = 128

    def __post_init__(self) -> None:
        _require_megatron()
        if self.normalization != "RMSNorm":
            raise ValueError("MOSS-TTS Prism requires RMSNorm")
        topology = resolve_prism_topology(self)
        if self.num_layers not in (0, topology.num_layers):
            raise ValueError(
                "MOSS-TTS Prism physical layer count is derived from its concrete "
                f"topology ({topology.num_layers}), not num_layers={self.num_layers}"
            )
        self.num_layers = topology.num_layers
        # Shared layers run backward in reverse call order. TE's first-call
        # flag would overwrite fused main_grad after later calls accumulated it.
        # Passing None through the existing cache setting keeps accumulation on.
        self.disable_parameter_transpose_cache = True
        super().__post_init__()
        if isinstance(self.text_pad_idx, bool) or not isinstance(self.text_pad_idx, int) or self.text_pad_idx < 0:
            raise ValueError("text_pad_idx must be a nonnegative non-Boolean integer")
        if (
            isinstance(self.speech_vocab_size, bool)
            or not isinstance(self.speech_vocab_size, int)
            or self.speech_vocab_size <= 0
        ):
            raise ValueError("speech_vocab_size must be a positive integer")
        for name in ("speech_embedding_init_method_std", "speech_head_init_method_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.speech_pad_index = self.speech_vocab_size

        self.prism_loop_block_ranges = topology.prism_loop_block_ranges
        self.prism_mlp_only_layers = topology.prism_mlp_only_layers
        self.prism_loop_block_repetitions = topology.prism_loop_block_repetitions
        self.prism_rvq_heads_per_execution_unit = topology.prism_rvq_heads_per_execution_unit
        self.reapply_rvq_conditioning_in_loop = topology.reapply_rvq_conditioning_in_loop
        self.rvq_label_smoothing = normalize_prism_rvq_label_smoothing(self.rvq_label_smoothing, n_vq=self.n_vq)
        self.prism_stop_gradient_layer = validate_prism_stop_gradient_layer(self.prism_stop_gradient_layer, topology)
        self.prism_input_rvq_channels = normalize_prism_input_rvq_channels(
            self.prism_input_rvq_channels, n_vq=self.n_vq
        )
        self.prism_topology = topology


def language_model_embedding_with_weight(
    embedding: LanguageModelEmbedding,
    input_ids: Tensor,
    position_ids: Tensor | None,
    weight: Tensor,
    *,
    pad_token_id: int | None = None,
) -> Tensor:
    """Run a MOSS-TTS embedding with an externally supplied TP-sharded weight."""

    if embedding.config.sequence_parallel:
        raise ValueError("external-weight MOSS-TTS embeddings reject sequence parallelism")
    if embedding.tokentype_embeddings is not None:
        raise ValueError("external-weight MOSS-TTS embeddings reject token types")

    parallel_embedding = embedding.word_embeddings
    expected_shape = (
        parallel_embedding.num_embeddings_per_partition,
        parallel_embedding.embedding_dim,
    )
    if tuple(weight.shape) != expected_shape:
        raise ValueError(f"external embedding weight has shape {tuple(weight.shape)}; " f"expected {expected_shape}")

    pad_mask = None
    lookup_ids = input_ids
    if pad_token_id is not None:
        pad_mask = input_ids.eq(pad_token_id)
        lookup_ids = input_ids.masked_fill(pad_mask, parallel_embedding.vocab_start_index)

    if parallel_embedding.tp_group.size() > 1:
        input_mask = (lookup_ids < parallel_embedding.vocab_start_index) | (
            lookup_ids >= parallel_embedding.vocab_end_index
        )
        local_ids = lookup_ids.clone() - parallel_embedding.vocab_start_index
        local_ids.masked_fill_(input_mask, 0)
    else:
        input_mask = None
        local_ids = lookup_ids

    if parallel_embedding.deterministic_mode:
        output_parallel = weight[local_ids]
    else:
        output_parallel = F.embedding(local_ids, weight)
    if input_mask is not None:
        output_parallel = output_parallel.masked_fill(input_mask.unsqueeze(-1), 0)
    word_embeddings = tensor_parallel.reduce_from_tensor_model_parallel_region(
        output_parallel, group=parallel_embedding.tp_group
    )

    if embedding.add_position_embedding:
        if position_ids is None:
            raise ValueError("position_ids are required for learned position embeddings")
        word_embeddings = word_embeddings + embedding.position_embeddings(position_ids)
    if pad_mask is not None:
        word_embeddings = word_embeddings.masked_fill(pad_mask.unsqueeze(-1), 0)

    output = word_embeddings.transpose(0, 1).contiguous()
    if embedding.config.fp32_residual_connection:
        output = output.float()
    return embedding.embedding_dropout(output)


class LanguageModelEmbeddingWithPad(LanguageModelEmbedding):
    """Language model embedding that masks out pad tokens."""

    def __init__(
        self,
        pad_token_id: int,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal["learned_absolute", "rope", "none"] = ("learned_absolute"),
        num_tokentypes: int = 0,
        scatter_to_sequence_parallel: bool = True,
        tp_group: torch.distributed.ProcessGroup | None = None,
        *,
        word_embedding_init_method: Callable[[Tensor], object] | None = None,
    ) -> None:
        if word_embedding_init_method is None:
            super().__init__(
                config=config,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                position_embedding_type=position_embedding_type,
                num_tokentypes=num_tokentypes,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=tp_group,
            )
        else:
            allocation_config = copy(config)
            allocation_config.perform_initialization = False
            super().__init__(
                config=allocation_config,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                position_embedding_type=position_embedding_type,
                num_tokentypes=num_tokentypes,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=tp_group,
            )
            self.config = config
            self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
                num_embeddings=self.vocab_size,
                embedding_dim=config.hidden_size,
                init_method=word_embedding_init_method,
                reduce_scatter_embeddings=self.reduce_scatter_embeddings,
                config=config,
                tp_group=self.tp_group,
            )
            if config.perform_initialization:
                if self.add_position_embedding:
                    config.embedding_init_method(self.position_embeddings.weight)
                if self.num_tokentypes > 0:
                    config.embedding_init_method(self.tokentype_embeddings.weight)
        self.pad_token_id = pad_token_id

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None,
        tokentype_ids: int | None = None,
    ) -> Tensor:
        """
        Inputs:
            input_ids: [b, s]
            position_ids: [b, s] or None
            tokentype_ids: int or None
        Outputs:
            output: [s, b, h]
        """
        b, s = input_ids.shape
        h = self.config.hidden_size
        input_mask = input_ids != self.pad_token_id  # Binary mask of shape [b, s]
        input_ids_masked = input_ids[input_mask].unsqueeze(0)  # [1, C]
        position_ids_masked = position_ids[input_mask].unsqueeze(0) if position_ids is not None else None  # [1, C]
        masked_output = super().forward(input_ids_masked, position_ids_masked, tokentype_ids)  # [C, 1, H]
        output = torch.zeros(s, b, h, device=input_ids.device, dtype=self.word_embeddings.weight.dtype)
        output[input_mask.transpose(0, 1)] = masked_output.squeeze(1)
        return output  # [s, b, h]


class FP32StopHead(nn.Linear):
    """Binary projection in FP32; initialization belongs to the model."""

    def reset_parameters(self) -> None:
        pass

    def forward(self, hidden: Tensor) -> Tensor:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return F.linear(hidden.float(), self.weight.float())


class EmbeddingHead(tensor_parallel.ColumnParallelLinear):
    """Enter the weight owner's forward hooks for both projection and lookup."""

    def forward(self, input_, *, embedding=None, position_ids=None, pad_token_id=None, **kwargs):
        if embedding is not None:
            return language_model_embedding_with_weight(
                embedding, input_, position_ids, self.weight, pad_token_id=pad_token_id
            )
        return super().forward(input_, **kwargs)


class PrismMLPOnlyLayer(TransformerLayer):
    """Remove attention while retaining the native pre-norm MLP and residual."""

    def _forward_attention(self, hidden_states, *, context=None, **_kwargs):
        return hidden_states, context


def prism_block_spec(
    layer_spec: ModuleSpec, *, num_layers: int, mlp_only_layers: tuple[int, ...]
) -> TransformerBlockSubmodules:
    submodules = layer_spec.submodules
    mlp_spec = replace(
        layer_spec,
        module=PrismMLPOnlyLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=submodules.pre_mlp_layernorm,
            mlp=submodules.mlp,
            mlp_bda=submodules.mlp_bda,
            sharded_state_dict_keys_map=submodules.sharded_state_dict_keys_map,
        ),
    )
    return TransformerBlockSubmodules(
        layer_specs=[mlp_spec if index in mlp_only_layers else layer_spec for index in range(num_layers)],
        layer_norm=LayerNormImpl,
    )


_ApplyRange = Callable[[int, int, Tensor], Tensor]
_ReadSite = Callable[[int, PrismExecutionUnit, tuple[int, ...], Tensor, bool], Tensor | None]


def _normalize_loop_block_prefix_counts(
    value: Tensor | Sequence[int] | None, topology: ResolvedPrismTopology
) -> tuple[int, ...] | None:
    """Normalize a per-forward prefix vector before schedule construction."""

    if value is None:
        return None
    if torch.is_tensor(value):
        if value.dtype != torch.int64 or value.ndim != 1:
            raise ValueError("loop_block_prefix_counts must be a one-dimensional int64 tensor")
        value = value.detach().to(device="cpu").tolist()
    topology.execution_unit_indices_for_prefix_counts(value)
    return tuple(value)


def _add_delta(base: Tensor, delta: Tensor | None) -> Tensor:
    return base if delta is None else base + delta


def _accumulate_delta(accumulator: Tensor | None, delta: Tensor | None) -> Tensor | None:
    if accumulator is None:
        return delta
    return accumulator if delta is None else accumulator + delta


def _detach_optional(value: Tensor | None) -> Tensor | None:
    """Detach an optional conditioning path without changing its forward value."""

    return None if value is None else value.detach()


def _validate_delta(raw: Tensor, delta: Tensor | None) -> Tensor | None:
    if delta is not None and (not torch.is_tensor(delta) or tuple(delta.shape) != tuple(raw.shape)):
        raise RuntimeError("Prism conditioning delta must match the raw site state")
    return delta


def _execute_prism_schedule(
    topology: ResolvedPrismTopology,
    pre_layer_residual: Tensor,
    *,
    apply_range: _ApplyRange,
    read_site: _ReadSite,
    stop_gradient_layer: int | None = None,
    loop_block_prefix_counts: Sequence[int] | None = None,
) -> Tensor:
    """Execute the recurrence, optionally cutting one physical-layer boundary.

    ``stop_gradient_layer`` is a physical boundary rather than an expanded
    execution-unit index.  The backbone residual and incoming conditioning
    delta are detached together when they enter the first unit after that
    boundary.  This keeps the boundary site's prediction graph intact and
    naturally gives a repeated loop block one detach event, even when the
    block has several applications. Conditioning deltas produced after the
    boundary remain differentiable.
    """

    selected_indices = topology.execution_unit_indices_for_prefix_counts(loop_block_prefix_counts)
    units = tuple(topology.execution_units[index] for index in selected_indices)
    groups = tuple(topology.co_site_groups[index] for index in selected_indices)
    if not units or len(groups) != len(units):
        raise RuntimeError("resolved Prism topology has inconsistent execution units")

    previous_raw: Tensor | None = None
    incoming_delta: Tensor | None = None
    unit_index = 0
    while unit_index < len(units):
        unit = units[unit_index]
        if unit.site.kind == "ordinary":
            unit_base = pre_layer_residual if previous_raw is None else previous_raw
            if stop_gradient_layer == unit.layer_start:
                # Cut every path crossing the boundary. The values are still
                # added below, so detach changes only the backward graph.
                unit_base = unit_base.detach()
                incoming_delta = _detach_optional(incoming_delta)
            unit_input = unit_base if incoming_delta is None else unit_base + incoming_delta
            raw = apply_range(unit.layer_start, unit.layer_stop, unit_input)
            if not torch.is_tensor(raw) or tuple(raw.shape) != tuple(unit_input.shape):
                raise RuntimeError("a registered Prism layer changed the residual shape")
            incoming_delta = _validate_delta(
                raw,
                read_site(
                    selected_indices[unit_index],
                    unit,
                    groups[unit_index],
                    raw,
                    unit_index == len(units) - 1,
                ),
            )
            previous_raw = raw
            unit_index += 1
            continue

        block = unit.site.index
        block_start = unit_index
        block_stop = block_start
        while (
            block_stop < len(units) and units[block_stop].site.kind == "loop" and units[block_stop].site.index == block
        ):
            block_stop += 1

        pre_block_residual = pre_layer_residual if previous_raw is None else previous_raw
        if stop_gradient_layer == unit.layer_start:
            # The whole loop range is one schedule unit at this boundary.
            # Detach the fixed anchor and incoming accumulator once before
            # application 1; post-boundary recurrence and deltas stay live.
            pre_block_residual = pre_block_residual.detach()
            incoming_delta = _detach_optional(incoming_delta)
        accumulator = incoming_delta
        for application_index in range(block_start, block_stop):
            application = units[application_index]
            coupled_base = (
                pre_block_residual if application_index == block_start else previous_raw + pre_block_residual
            )
            unit_input = _add_delta(coupled_base, accumulator)
            raw = apply_range(application.layer_start, application.layer_stop, unit_input)
            if not torch.is_tensor(raw) or tuple(raw.shape) != tuple(unit_input.shape):
                raise RuntimeError("a registered Prism loop range changed the residual shape")
            delta = _validate_delta(
                raw,
                read_site(
                    selected_indices[application_index],
                    application,
                    groups[application_index],
                    raw,
                    application_index == len(units) - 1,
                ),
            )
            previous_raw = raw
            if application_index + 1 < block_stop:
                accumulator = (
                    _accumulate_delta(accumulator, delta) if topology.reapply_rvq_conditioning_in_loop else delta
                )
            else:
                # Block-local history never escapes. Only the final site's own
                # delta can reach the immediate successor.
                incoming_delta = delta
        unit_index = block_stop

    if previous_raw is None:  # pragma: no cover - topology rejects this state
        raise RuntimeError("Prism execution produced no raw state")
    return previous_raw


class MossTTSPrismModel(LanguageModule):
    """Dense stop/RVQ training model with a unified recurrent layer schedule."""

    config: MossTTSPrismConfig

    def __init__(
        self,
        config: MossTTSPrismConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["learned_absolute", "rope", "none"] = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 1000000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        seq_len_interpolation_factor: float | None = None,
    ) -> None:
        _require_megatron()
        super().__init__(config=config)
        self._validate_constructor_envelope(
            config,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=parallel_output,
            position_embedding_type=position_embedding_type,
        )

        self.model_type = ModelType.encoder_or_decoder
        self.vocab_size = int(vocab_size)
        self.max_sequence_length = int(max_sequence_length)
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent
        self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling
        self.prism_topology = config.prism_topology
        self._full_activation_recompute = validate_prism_recompute_policy(config, effective_config=True)

        speech_embedding_init_method = init_method_normal(config.speech_embedding_init_method_std)
        speech_head_init_method = init_method_normal(config.speech_head_init_method_std)

        self.text_embedding = LanguageModelEmbeddingWithPad(
            pad_token_id=config.text_pad_idx,
            config=config,
            vocab_size=self.vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type=position_embedding_type,
        )
        self.audio_embeddings = nn.ModuleList(
            [
                LanguageModelEmbeddingWithPad(
                    pad_token_id=config.speech_pad_index,
                    config=config,
                    vocab_size=config.speech_vocab_size,
                    max_sequence_length=max_sequence_length,
                    position_embedding_type=position_embedding_type,
                    word_embedding_init_method=speech_embedding_init_method,
                )
                for _ in range(config.n_vq)
            ]
        )

        self.rotary_pos_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=config.rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            use_cpu_initialization=config.use_cpu_initialization,
        )
        self.decoder = TransformerBlock(
            config=config,
            spec=prism_block_spec(
                transformer_layer_spec,
                num_layers=config.num_layers,
                mlp_only_layers=self.prism_topology.prism_mlp_only_layers,
            ),
            post_layer_norm=False,
            pre_process=True,
            post_process=True,
        )
        if len(self.decoder.layers) != config.num_layers:
            raise RuntimeError("MOSS-TTS Prism must register exactly num_layers physical layers")
        if self.decoder.final_layernorm is not None:
            raise RuntimeError("MOSS-TTS Prism must not register a global final norm")
        self.prediction_norms = nn.ModuleList(
            [
                build_module(
                    self.decoder.submodules.layer_norm,
                    config=config,
                    hidden_size=config.hidden_size,
                    eps=config.layernorm_epsilon,
                )
                for _ in range(self.prism_topology.num_prediction_norms)
            ]
        )

        self.stop_head = FP32StopHead(config.hidden_size, 2, bias=False)
        if config.perform_initialization:
            nn.init.normal_(self.stop_head.weight, mean=0.0, std=config.init_method_std)
        self.stop_head.weight.average_gradients_across_tp_domain = True
        self.audio_lm_heads = nn.ModuleList(
            [
                EmbeddingHead(
                    config.hidden_size,
                    config.speech_vocab_size,
                    config=config,
                    init_method=speech_head_init_method,
                    bias=False,
                    gather_output=False,
                    skip_weight_param_allocation=False,
                )
                for _ in range(config.n_vq)
            ]
        )
        self.setup_embeddings_and_output_layer()

    @staticmethod
    def _validate_constructor_envelope(
        config: MossTTSPrismConfig,
        *,
        pre_process: bool,
        post_process: bool,
        parallel_output: bool,
        position_embedding_type: str,
    ) -> None:
        for attribute, label in (
            ("pipeline_model_parallel_size", "pipeline parallel"),
            ("context_parallel_size", "context parallel"),
        ):
            if getattr(config, attribute, 1) != 1:
                raise ValueError(f"MOSS-TTS Prism requires {label} size one")
        if not pre_process or not post_process:
            raise ValueError("MOSS-TTS Prism requires pre_process and post_process")
        if not parallel_output:
            raise ValueError("MOSS-TTS Prism requires tensor-parallel output shards")
        if position_embedding_type != "rope":
            raise ValueError("MOSS-TTS Prism requires RoPE")
        if getattr(config, "multi_latent_attention", False):
            raise ValueError("MOSS-TTS Prism does not support multi-latent attention")
        if getattr(config, "sequence_parallel", False):
            raise ValueError("MOSS-TTS Prism does not support sequence parallelism")
        validate_prism_recompute_policy(config, effective_config=True)
        unsupported = []
        if getattr(config, "virtual_pipeline_model_parallel_size", None) is not None:
            unsupported.append("virtual pipeline parallelism")
        if getattr(config, "fp8", None) is not None:
            unsupported.append("FP8")
        for attribute, label in (
            ("cpu_offloading", "CPU offload"),
            ("enable_cuda_graph", "CUDA graphs"),
            ("flash_decode", "flash decoding"),
        ):
            if getattr(config, attribute, False):
                unsupported.append(label)
        if unsupported:
            raise ValueError("MOSS-TTS Prism does not support " + ", ".join(unsupported))
        if config.attention_dropout != 0.0 or config.hidden_dropout != 0.0:
            raise ValueError("MOSS-TTS Prism requires zero attention and hidden dropout")

    def setup_embeddings_and_output_layer(self) -> None:
        """Mark canonical parameters and freeze redundant tied embeddings."""

        text_weight = self.text_embedding.word_embeddings.weight
        text_weight.is_embedding_or_output_parameter = True
        self.stop_head.weight.is_embedding_or_output_parameter = True
        for embedding in self.audio_embeddings:
            embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        for head in self.audio_lm_heads:
            head.weight.is_embedding_or_output_parameter = True
        if self.share_embeddings_and_output_weights:
            for embedding in self.audio_embeddings:
                embedding.word_embeddings.weight.requires_grad_(False)
            for head in self.audio_lm_heads:
                head.weight.zero_out_wgrad = True

    def _audio_embedding(self, channel_index: int, input_ids: Tensor) -> Tensor:
        embedding = self.audio_embeddings[channel_index]
        if not self.share_embeddings_and_output_weights:
            return embedding(input_ids, None)
        return self.audio_lm_heads[channel_index](
            input_ids, embedding=embedding, pad_token_id=self.config.speech_pad_index
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Preserve TP metadata for every RVQ embedding and head."""

        if self.prism_topology.prism_mlp_only_layers:
            # Mixed layers cannot stack attention tensors across physical depth.
            metadata = {**(metadata or {}), "non_homogeneous_layers": True}
        sharded_state = MegatronModule.sharded_state_dict(
            self, prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )
        module_list_prefixes = (
            f"{prefix}audio_embeddings.",
            f"{prefix}audio_lm_heads.",
        )
        for key in list(sharded_state):
            if key.startswith(module_list_prefixes):
                del sharded_state[key]
        for index, embedding in enumerate(self.audio_embeddings):
            sharded_state.update(
                embedding.sharded_state_dict(
                    prefix=f"{prefix}audio_embeddings.{index}.",
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
            )
        for index, head in enumerate(self.audio_lm_heads):
            sharded_state.update(
                head.sharded_state_dict(
                    prefix=f"{prefix}audio_lm_heads.{index}.",
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
            )
        return sharded_state

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        if isinstance(input_tensor, list):
            if len(input_tensor) != 1:
                raise ValueError("expected one pipeline input tensor")
            input_tensor = input_tensor[0]
        self.decoder.set_input_tensor(input_tensor)

    @staticmethod
    def _tensor(
        value: object,
        *,
        name: str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> Tensor:
        if not torch.is_tensor(value) or value.dtype != dtype or tuple(value.shape) != shape or value.device != device:
            raise ValueError(f"{name} must be a {dtype} tensor on {device} with shape {shape}")
        return value

    def _prepare_inputs(
        self,
        input_ids: Tensor,
        item_kind: Tensor,
        audio_role: Tensor,
        target_history_retention_mask: Tensor,
    ) -> Tensor:
        text = item_kind == int(PrismItemKind.TEXT)
        target = audio_role == int(PrismAudioRole.TARGET)
        selected_channels = getattr(
            self.config,
            "prism_input_rvq_channels",
            tuple(range(1, self.config.n_vq + 1)),
        )
        text_ids = input_ids[..., 0].masked_fill(~text, self.config.text_pad_idx)
        hidden_states = self.text_embedding(text_ids, None)
        for channel_index in range(self.config.n_vq):
            if channel_index + 1 in selected_channels:
                retained = ~target | target_history_retention_mask[..., channel_index]
                active = ~text & retained
            else:
                # The channel selection belongs to the target/output side.
                # Reference audio is user input and always exposes its full
                # RVQ frame to the backbone.
                active = ~text & ~target
            audio_ids = input_ids[..., channel_index + 1].masked_fill(~active, self.config.speech_pad_index)
            hidden_states = hidden_states + self._audio_embedding(channel_index, audio_ids)
        return hidden_states

    def _successor_conditioning_delta(
        self,
        successor_audio_codes: Tensor,
        successor_audio_mask: Tensor,
        group: tuple[int, ...],
    ) -> Tensor | None:
        delta = None
        for q in group:
            channel_index = q - 1
            successor_ids = successor_audio_codes[..., channel_index].masked_fill(
                ~successor_audio_mask, self.config.speech_pad_index
            )
            contribution = self._audio_embedding(channel_index, successor_ids)
            delta = contribution if delta is None else delta + contribution
        return delta

    def _prediction_read(self, unit_index: int, raw: Tensor) -> Tensor:
        """Apply the normalization owned by one occupied execution unit."""

        norm_index = self.prism_topology.prediction_norm_index_by_execution_unit[unit_index]
        if norm_index is None:
            raise RuntimeError("an unoccupied Prism execution unit has no prediction norm")
        return self.prediction_norms[norm_index](raw)

    def _apply_registered_range(
        self,
        start: int,
        stop: int,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        context: Tensor | None,
        context_mask: Tensor | None,
        rotary_pos_emb: Tensor | None,
        packed_seq_params: PackedSeqParams,
    ) -> tuple[Tensor, Tensor | None]:
        def bind_layer(layer):
            def layer_forward(
                value: Tensor,
                layer_attention_mask: Tensor | None,
                layer_context: Tensor | None,
                layer_context_mask: Tensor | None,
                layer_rotary_pos_emb: Tensor | None,
            ) -> tuple[Tensor, Tensor | None]:
                return layer(
                    hidden_states=value,
                    attention_mask=layer_attention_mask,
                    context=layer_context,
                    context_mask=layer_context_mask,
                    rotary_pos_emb=layer_rotary_pos_emb,
                    rotary_pos_cos=None,
                    rotary_pos_sin=None,
                    attention_bias=None,
                    inference_context=None,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=None,
                )

            return layer_forward

        checkpoint_layers = self.training and getattr(self, "_full_activation_recompute", False)
        for layer_index in range(start, stop):
            layer = self.decoder.layers[layer_index]
            layer_forward = bind_layer(layer)
            if checkpoint_layers:
                if not hidden_states.requires_grad:
                    # Reentrant checkpoint needs a differentiable input to build
                    # parameter gradients after a stop-gradient boundary.
                    hidden_states = hidden_states.detach().requires_grad_(True)
                hidden_states, context = tensor_parallel.checkpoint(
                    layer_forward,
                    False,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
            else:
                hidden_states, context = layer_forward(
                    hidden_states, attention_mask, context, context_mask, rotary_pos_emb
                )
        return hidden_states, context

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        *,
        packed_seq_params=None,
        labels=None,
        loss_mask=None,
        audio_codes: Tensor,
        item_kind: Tensor,
        audio_role: Tensor,
        successor_audio_codes: Tensor,
        successor_audio_mask: Tensor,
        target_history_retention_mask: Tensor | None = None,
        loop_block_prefix_counts: Tensor | Sequence[int] | None = None,
        policy_code_mask: Tensor,
        policy_stop_mask: Tensor,
        policy_stop_actions: Tensor,
        policy_return_entropy: bool = False,
        **policy_fields,
    ) -> Tensor:
        """Score the supplied trajectory, preserving the native recurrence.

        Auxiliary ``policy_*`` tensors are consumed by slime's objective. They
        never change the backbone input or the successor-conditioning edge.
        """
        unexpected = [key for key in policy_fields if not key.startswith("policy_")]
        if unexpected:
            raise TypeError(f"unexpected Prism inputs: {unexpected}")
        if position_ids is not None or attention_mask is not None:
            raise ValueError("Prism uses THD packed RoPE with position_ids=attention_mask=None")
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError("Prism requires THD PackedSeqParams")
        batch = prepare_policy_inputs(
            input_ids,
            audio_codes=audio_codes,
            item_kind=item_kind,
            audio_role=audio_role,
            successor_audio_codes=successor_audio_codes,
            successor_audio_mask=successor_audio_mask,
            target_history_retention_mask=target_history_retention_mask,
            policy_code_mask=policy_code_mask,
            policy_stop_mask=policy_stop_mask,
            policy_stop_actions=policy_stop_actions,
            n_vq=self.config.n_vq,
            speech_vocab_size=self.config.speech_vocab_size,
            text_vocab_size=self.vocab_size,
            text_pad_idx=self.config.text_pad_idx,
            audio_start_token_id=self.config.audio_start_token_id,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        prefix_counts = _normalize_loop_block_prefix_counts(loop_block_prefix_counts, self.prism_topology)
        hidden = self._prepare_inputs(batch["tokens"], batch["item_kind"], batch["audio_role"], batch["retention"])
        rotary_length = self.rotary_pos_emb.get_rotary_seq_len(
            None, self.decoder, hidden, self.config, packed_seq_params
        )
        rotary = self.rotary_pos_emb(rotary_length, packed_seq=True)
        code_scores = [None] * self.config.n_vq
        code_entropies = [None] * self.config.n_vq
        stop_score = stop_entropy = None
        context = None

        def apply_range(start, stop, value):
            nonlocal context
            value, context = self._apply_registered_range(
                start,
                stop,
                value,
                attention_mask=None,
                context=context,
                context_mask=None,
                rotary_pos_emb=rotary,
                packed_seq_params=packed_seq_params,
            )
            return value

        def read_site(unit_index, _unit, group, raw, is_global_final):
            nonlocal stop_score, stop_entropy
            if not group:
                return None
            normalized = self._prediction_read(unit_index, raw)
            if 1 in group:
                stop_logits = self.stop_head(normalized).transpose(0, 1)
                stop_score, stop_entropy = selected_logprob_and_entropy(
                    stop_logits,
                    batch["stop_actions"],
                    return_entropy=policy_return_entropy,
                )
            # Each co-site head reads this same state. No selected-code
            # feedback occurs until every head in the group has been scored.
            for q in group:
                logits, _ = self.audio_lm_heads[q - 1](normalized)
                logits = logits.transpose(0, 1)
                targets = batch["successor_codes"][..., q - 1]
                if policy_return_entropy:
                    code_entropies[q - 1] = vocab_parallel_entropy(logits)
                # MCore CE exponentiates its FP32 argument in place. Keep the
                # head output intact for other differentiable statistics.
                code_scores[q - 1] = -tensor_parallel.vocab_parallel_cross_entropy(logits.float().clone(), targets)
            if is_global_final:
                return None
            return self._successor_conditioning_delta(batch["successor_codes"], batch["successor_mask"], group)

        _execute_prism_schedule(
            self.prism_topology,
            hidden,
            apply_range=apply_range,
            read_site=read_site,
            loop_block_prefix_counts=prefix_counts,
        )
        if stop_score is None or any(value is None for value in code_scores):
            raise RuntimeError("Prism schedule did not score every action head")
        scores = torch.cat((torch.stack(code_scores, dim=-1), stop_score.unsqueeze(-1)), dim=-1)
        if policy_return_entropy:
            entropies = torch.cat((torch.stack(code_entropies, dim=-1), stop_entropy.unsqueeze(-1)), dim=-1)
            scores = torch.cat((scores, entropies), dim=-1)
        return scores


def selected_logprob_and_entropy(logits: Tensor, targets: Tensor, *, return_entropy=False):
    """FP32 temperature-one categorical scores, including the binary stop."""
    logprobs = logits.float().log_softmax(-1)
    selected = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(logprobs.exp() * logprobs).sum(-1) if return_entropy else None
    return selected, entropy


def vocab_parallel_entropy(logits: Tensor) -> Tensor:
    """Exact entropy over the complete TP vocabulary with differentiable reductions."""
    group = parallel_state.get_tensor_model_parallel_group()
    if torch.distributed.get_world_size(group) == 1:
        logprobs = logits.float().log_softmax(-1)
        return -(logprobs.exp() * logprobs).sum(-1)
    # The maximum is a numerical offset and cancels from the entropy.
    maximum = logits.detach().float().amax(-1)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=group)
    shifted = logits.float() - maximum.unsqueeze(-1)
    exps = shifted.exp()
    # TP replicas consume this same scalar. The region reduction performs a
    # SUM in forward and passes the local gradient through in backward.
    denominator = tensor_parallel.reduce_from_tensor_model_parallel_region(exps.sum(-1), group=group)
    weighted = tensor_parallel.reduce_from_tensor_model_parallel_region((exps * shifted).sum(-1), group=group)
    return denominator.log() - weighted / denominator


def _carrier_tensor(value, name, shape, dtype, device, *, pad_value=0):
    if torch.is_tensor(value) and value.ndim == len(shape) - 1:
        value = value.unsqueeze(0)
    if (
        torch.is_tensor(value)
        and value.ndim == len(shape)
        and value.shape[0] == 1
        and tuple(value.shape[2:]) == shape[2:]
        and value.shape[1] < shape[1]
    ):
        padding = torch.full(
            (1, shape[1] - value.shape[1], *shape[2:]), pad_value, dtype=value.dtype, device=value.device
        )
        value = torch.cat((value, padding), dim=1)
    if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype or value.device != device:
        raise ValueError(f"{name} must be {dtype} {shape} on {device}")
    return value


def prepare_policy_inputs(
    input_ids: Tensor,
    *,
    audio_codes,
    item_kind,
    audio_role,
    successor_audio_codes,
    successor_audio_mask,
    target_history_retention_mask,
    policy_code_mask,
    policy_stop_mask,
    policy_stop_actions,
    n_vq,
    speech_vocab_size,
    text_vocab_size,
    text_pad_idx,
    audio_start_token_id=151652,
    cu_seqlens=None,
) -> dict[str, Tensor]:
    """Validate a chronological carrier without imposing SFT terminal labels.

    Reference and prompt-prefix successor edges are real conditioning even
    when their action masks are zero. A zero-frame sampled stop may live on
    the TEXT/NONE audio_start row; a length cap has no invented stop action.
    """
    if (
        not torch.is_tensor(input_ids)
        or input_ids.dtype != torch.int64
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
    ):
        raise ValueError("Prism input_ids must be int64 [1,T] text/control IDs")
    shape = tuple(input_ids.shape)
    rvq_shape = (*shape, n_vq)
    device = input_ids.device
    audio = _carrier_tensor(audio_codes, "audio_codes", rvq_shape, torch.int64, device)
    kinds = _carrier_tensor(item_kind, "item_kind", shape, torch.int64, device)
    roles = _carrier_tensor(audio_role, "audio_role", shape, torch.int64, device)
    codes = _carrier_tensor(successor_audio_codes, "successor_audio_codes", rvq_shape, torch.int64, device)
    edges = _carrier_tensor(successor_audio_mask, "successor_audio_mask", shape, torch.bool, device)
    code_mask = _carrier_tensor(policy_code_mask, "policy_code_mask", rvq_shape, torch.bool, device)
    stop_mask = _carrier_tensor(policy_stop_mask, "policy_stop_mask", shape, torch.bool, device)
    stops = _carrier_tensor(policy_stop_actions, "policy_stop_actions", shape, torch.int64, device)
    retention = (
        torch.ones(rvq_shape, dtype=torch.bool, device=device)
        if target_history_retention_mask is None
        else _carrier_tensor(
            target_history_retention_mask,
            "target_history_retention_mask",
            rvq_shape,
            torch.bool,
            device,
            pad_value=True,
        )
    )
    text = kinds.eq(PrismItemKind.TEXT)
    target = roles.eq(PrismAudioRole.TARGET)
    frame_actions = code_mask.any(-1)
    checks = (
        ((kinds < 0) | (kinds > 1), "invalid item_kind"),
        ((roles < 0) | (roles > 2) | text.ne(roles.eq(PrismAudioRole.NONE)), "invalid item/audio role"),
        ((input_ids < 0) | (input_ids >= text_vocab_size), "text ID outside vocabulary"),
        (~text & input_ids.ne(text_pad_idx), "audio rows require the text-pad filler"),
        (text.unsqueeze(-1) & audio.ne(0), "text rows require zero audio fillers"),
        (
            (audio < 0) | (audio >= speech_vocab_size) | (codes < 0) | (codes >= speech_vocab_size),
            "audio code outside vocabulary",
        ),
        (~edges.unsqueeze(-1) & codes.ne(0), "inactive successor payload must be zero"),
        (~target.unsqueeze(-1) & ~retention, "text/reference input must retain all channels"),
        (code_mask.ne(code_mask[..., :1]), "the policy must sample complete RVQ frames"),
        (
            frame_actions & (~edges | ~stop_mask | stops.ne(0)),
            "frame actions require a real edge and sampled continue",
        ),
        (stop_mask & stops.eq(0) & ~frame_actions, "a sampled continue must produce a complete frame"),
        (stop_mask & stops.eq(1) & edges, "a sampled stop cannot condition on an emitted successor"),
        ((stops < 0) | (stops > 1), "stop actions must be binary"),
        (
            stop_mask & ~(target | (text & input_ids.eq(audio_start_token_id))),
            "stop actions require target history or audio_start",
        ),
    )
    for invalid, message in checks:
        if bool(invalid.any()):
            raise ValueError(message)
    next_kind = torch.cat((kinds[:, 1:], torch.zeros_like(kinds[:, :1])), dim=1)
    next_role = torch.cat((roles[:, 1:], torch.zeros_like(roles[:, :1])), dim=1)
    next_codes = torch.cat((audio[:, 1:], torch.zeros_like(audio[:, :1])), dim=1)
    if bool((edges & (next_kind.ne(PrismItemKind.AUDIO) | codes.ne(next_codes).any(-1))).any()):
        raise ValueError("successor conditioning must contain the actual next audio frame")
    if bool((edges & ((~text & roles.ne(next_role)) | (text & input_ids.ne(audio_start_token_id)))).any()):
        raise ValueError("successor audio must remain within its reference/target span")
    if bool((frame_actions & next_role.ne(PrismAudioRole.TARGET)).any()):
        raise ValueError("policy frame actions cannot supervise a reference successor")
    if cu_seqlens is not None:
        boundaries = cu_seqlens.detach().cpu().tolist()
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != shape[1]
            or any(a >= b for a, b in zip(boundaries, boundaries[1:], strict=False))
        ):
            raise ValueError("packed boundaries must partition all chronological rows")
        ends = torch.tensor([position - 1 for position in boundaries[1:]], device=device)
        if bool(edges[0, ends].any()):
            raise ValueError("successor conditioning cannot cross a packed sequence boundary")
    return {
        "tokens": torch.cat((input_ids.unsqueeze(-1), audio), dim=-1),
        "item_kind": kinds,
        "audio_role": roles,
        "retention": retention,
        "successor_codes": codes,
        "successor_mask": edges,
        "code_mask": code_mask,
        "stop_mask": stop_mask,
        "stop_actions": stops,
    }


@dataclass(frozen=True)
class PrismModelSpec:
    """Architecture values owned by the inference artifact, independent of runtime TP."""

    topology: ResolvedPrismTopology
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    text_vocab_size: int
    speech_vocab_size: int
    max_position_embeddings: int
    norm_epsilon: float
    rotary_base: float
    text_pad_idx: int
    audio_start_token_id: int
    input_rvq_channels: tuple[int, ...]
    embedding_head_usage: str
    dtype: str

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any]):
        if config.get("model_type") != "moss_tts_prism":
            raise ValueError("expected a moss_tts_prism Hugging Face configuration")
        language = config.get("language_config")
        if not isinstance(language, Mapping) or language.get("model_type") != "qwen3":
            raise ValueError("Prism requires language_config describing Qwen3")
        for name, expected in (("hidden_act", "silu"), ("attention_bias", False)):
            if language.get(name) != expected:
                raise ValueError(f"unsupported Prism language_config.{name}")
        if language.get("attention_dropout", 0.0) != 0.0 or language.get("use_sliding_window", False):
            raise ValueError("Prism requires zero dropout and full causal attention")
        fields = {
            "n_vq": config["n_vq"],
            "prism_loop_block_ranges": config["prism_loop_block_ranges"],
            "prism_loop_block_repetitions": config["prism_loop_block_repetitions"],
            "prism_rvq_heads_per_execution_unit": config["prism_rvq_heads_per_execution_unit"],
            "prism_mlp_only_layers": config.get("prism_mlp_only_layers", []),
            "reapply_rvq_conditioning_in_loop": config.get("reapply_rvq_conditioning_in_loop", False),
        }
        topology = resolve_prism_topology(SimpleNamespace(**fields))
        if language["num_hidden_layers"] != topology.num_layers:
            raise ValueError("Prism language depth disagrees with physical topology")
        if any(kind != "full_attention" for kind in language.get("layer_types", [])):
            raise ValueError("Prism requires full_attention layer types")
        dimensions = {
            "hidden_size": language["hidden_size"],
            "ffn_hidden_size": language["intermediate_size"],
            "num_attention_heads": language["num_attention_heads"],
            "num_query_groups": language["num_key_value_heads"],
            "kv_channels": language["head_dim"],
            "text_vocab_size": language["vocab_size"],
            "speech_vocab_size": config["speech_vocab_size"],
            "max_position_embeddings": language["max_position_embeddings"],
        }
        for name, value in dimensions.items():
            _positive_integer(value, name)
        if dimensions["num_attention_heads"] % dimensions["num_query_groups"] or dimensions["kv_channels"] % 2:
            raise ValueError("Prism requires valid GQA and even full-head RoPE dimensions")
        rope = language.get("rope_parameters")
        if rope is None:
            if language.get("rope_scaling") is not None:
                raise ValueError("Prism requires unscaled RoPE")
            theta = language["rope_theta"]
        else:
            if rope.get("rope_type") not in ("default", "rope") or set(rope) != {"rope_type", "rope_theta"}:
                raise ValueError("Prism requires the default, unscaled RoPE descriptor")
            theta = rope["rope_theta"]
        eps = language["rms_norm_eps"]
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0
            for value in (theta, eps)
        ):
            raise ValueError("Prism norm epsilon and rotary base must be finite and positive")
        usage = config.get("embedding_head_usage")
        if usage not in ("untied", "tied_head_authoritative"):
            raise ValueError("Prism requires an exact embedding_head_usage")
        audio_start = config["audio_start_token_id"]
        if type(audio_start) is not int or audio_start != 151652:
            raise ValueError("Prism's moss_tts_v2 protocol requires audio_start_token_id=151652")
        dtype = config.get("dtype", config.get("torch_dtype"))
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("Prism artifact must declare its trainable dtype")
        if language.get("dtype", language.get("torch_dtype", dtype)) != dtype:
            raise ValueError("Prism language and root dtype disagree")
        return cls(
            topology=topology,
            **dimensions,
            norm_epsilon=float(eps),
            rotary_base=float(theta),
            text_pad_idx=config["text_pad_idx"],
            audio_start_token_id=audio_start,
            embedding_head_usage=usage,
            dtype=dtype,
            input_rvq_channels=normalize_prism_input_rvq_channels(
                config["prism_input_rvq_channels"], n_vq=topology.n_vq
            ),
        )

    def check_native_identity(self, identity: Mapping[str, Any]):
        if identity.get("architecture") != "MOSS-TTS-Prism" or identity.get("schema_version") != 3:
            raise ValueError("Prism initialization requires a native schema-3 identity")
        tensors, semantics = identity["tensor_schema"], identity["forward_semantics"]
        expected_tensors = {
            "hidden_size": self.hidden_size,
            "ffn_hidden_size": self.ffn_hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_query_groups": self.num_query_groups,
            "kv_channels": self.kv_channels,
            "text_vocab_size": self.text_vocab_size,
            "speech_vocab_size": self.speech_vocab_size,
            "n_vq": self.topology.n_vq,
            "num_layers": self.topology.num_layers,
            "qk_layernorm": True,
            "add_bias_linear": False,
            "add_qkv_bias": False,
            "activation": "swiglu",
            "stop_head_classes": 2,
            "embedding_head_storage": "split_v1",
        }
        for key, expected in expected_tensors.items():
            if type(tensors.get(key)) is not type(expected) or tensors[key] != expected:
                raise ValueError(f"native Prism tensor identity mismatch: {key}")
        topology = dict(tensors["prism_topology"])
        topology.setdefault("prism_mlp_only_layers", [])
        if topology != self.topology.canonical_dict():
            raise ValueError("native Prism recurrence/topology does not match the inference artifact")
        expected_semantics = {
            "embedding_head_usage": self.embedding_head_usage,
            "input_rvq_channels": list(self.input_rvq_channels),
            "input_rvq_channels_scope": "target_current_only_reference_full_v1",
            "conditioning": "golden_atomic_co_site_any_role_successor_audio_v2",
            "prediction_read_normalization": "per_occupied_execution_unit_rmsnorm_v1",
            "recurrence": "coupled_loop_residual_configurable_delta_replay",
            "stop_head_semantics": "stop_before_next_target_frame_binary",
            "input_representation": "exclusive_typed_text_or_complete_audio_frame_v1",
            "final_site_rule": "last_execution_unit_owns_rvq_n_vq",
            "max_position_embeddings": self.max_position_embeddings,
            "text_pad_idx": self.text_pad_idx,
            "norm_epsilon": self.norm_epsilon,
            "normalization": "RMSNorm",
            "rotary_base": self.rotary_base,
            "rotary_percent": 1.0,
            "rotary_interleaved": False,
            "position_embedding_type": "rope",
            "apply_query_key_layer_scaling": False,
            "rope_scaling_factor": None,
            "rotary_seq_len_interpolation_factor": None,
            "projection_revision": 3,
            "item_protocol": "moss_tts_v2_to_prism_v2",
            "source_protocol": "moss_tts_v2",
        }
        for key, expected in expected_semantics.items():
            if semantics.get(key) != expected:
                raise ValueError(f"native Prism forward identity mismatch: {key}")


def read_prism_spec(hf_checkpoint: str | Path) -> PrismModelSpec:
    path = Path(hf_checkpoint)
    config = path if path.is_file() else path / "config.json"
    return PrismModelSpec.from_hf_config(json.loads(config.read_text()))


def prism_megatron_cli_args(hf_checkpoint: str | Path) -> list[str]:
    """Generate ordinary Megatron architecture flags before its parser runs.

    The provider fills Prism's private topology/RVQ fields after normal CLI
    validation. Attention backend, parallelism and optimizer choices remain
    caller-owned. Standard text/output tying is absent from Prism; audio
    embedding/head use is independently described by embedding_head_usage.
    """
    spec = read_prism_spec(hf_checkpoint)
    if not spec.rotary_base.is_integer():
        raise ValueError("Megatron's --rotary-base requires an integer Prism RoPE base")
    values = (
        ("--num-layers", spec.topology.num_layers),
        ("--hidden-size", spec.hidden_size),
        ("--ffn-hidden-size", spec.ffn_hidden_size),
        ("--num-attention-heads", spec.num_attention_heads),
        ("--num-query-groups", spec.num_query_groups),
        ("--kv-channels", spec.kv_channels),
        ("--vocab-size", spec.text_vocab_size),
        ("--padded-vocab-size", spec.text_vocab_size),
        ("--max-position-embeddings", spec.max_position_embeddings),
        ("--normalization", "RMSNorm"),
        ("--layernorm-epsilon", spec.norm_epsilon),
        ("--position-embedding-type", "rope"),
        ("--rotary-base", int(spec.rotary_base)),
        ("--rotary-percent", 1.0),
        ("--attention-dropout", 0.0),
        ("--hidden-dropout", 0.0),
        ("--transformer-impl", "transformer_engine"),
    )
    result = [token for flag, field_value in values for token in (flag, str(field_value))]
    result.extend(("--swiglu", "--qk-layernorm", "--group-query-attention", "--disable-bias-linear"))
    result.append("--untie-embeddings-and-output-weights")
    if spec.dtype in ("bfloat16", "float16"):
        result.append("--bf16" if spec.dtype == "bfloat16" else "--fp16")
    return result


def configure_prism_args(args) -> PrismModelSpec:
    """Populate provider-owned Prism construction values from its HF artifact.

    Normal entry points pass prism_megatron_cli_args before CLI validation;
    this provider helper fills the topology, RVQ and forward-semantics fields.
    """
    spec = read_prism_spec(args.hf_checkpoint)
    topology = spec.topology
    values = {
        "num_layers": topology.num_layers,
        "hidden_size": spec.hidden_size,
        "ffn_hidden_size": spec.ffn_hidden_size,
        "num_attention_heads": spec.num_attention_heads,
        "num_query_groups": spec.num_query_groups,
        "kv_channels": spec.kv_channels,
        "padded_vocab_size": spec.text_vocab_size,
        "max_position_embeddings": spec.max_position_embeddings,
        "n_vq": topology.n_vq,
        "speech_vocab_size": spec.speech_vocab_size,
        "text_pad_idx": spec.text_pad_idx,
        "audio_start_token_id": spec.audio_start_token_id,
        "prism_loop_block_ranges": topology.prism_loop_block_ranges,
        "prism_loop_block_repetitions": topology.prism_loop_block_repetitions,
        "prism_rvq_heads_per_execution_unit": topology.prism_rvq_heads_per_execution_unit,
        "prism_mlp_only_layers": topology.prism_mlp_only_layers,
        "reapply_rvq_conditioning_in_loop": topology.reapply_rvq_conditioning_in_loop,
        "prism_input_rvq_channels": spec.input_rvq_channels,
        "rvq_label_smoothing": None,
        "prism_stop_gradient_layer": None,
        "qk_layernorm": True,
        "normalization": "RMSNorm",
        "layernorm_epsilon": spec.norm_epsilon,
        "position_embedding_type": "rope",
        "rotary_base": spec.rotary_base,
        "rotary_percent": 1.0,
        "rotary_interleaved": False,
        "rotary_seq_len_interpolation_factor": None,
        "use_rope_scaling": False,
        "swiglu": True,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "add_bias_linear": False,
        "add_qkv_bias": False,
        "apply_query_key_layer_scaling": False,
        "group_query_attention": True,
        "untie_embeddings_and_output_weights": True,
    }
    for key, value in values.items():
        setattr(args, key, value)
    return spec


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Build Prism through slime's existing custom-model-provider entry point."""
    _require_megatron()
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    if vp_stage is not None:
        raise ValueError("Prism does not support virtual pipeline stages")
    args = get_args()
    spec = configure_prism_args(args)
    if args.transformer_impl != "transformer_engine":
        raise ValueError("Prism uses Transformer Engine parameter names to preserve its native checkpoint contract")
    if getattr(args, "fp8_param_gather", False):
        raise ValueError("Prism does not support FP8 parameter gather")
    config = core_transformer_config_from_args(args, config_class=MossTTSPrismConfig)
    if config.params_dtype != getattr(torch, spec.dtype):
        raise ValueError("Prism Megatron precision must match the inference artifact dtype")
    # MCore will cast/wrap parameters after construction. Native import runs
    # at slime's checkpoint-load boundary, after DDP/optimizer allocation.
    layer_spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
    model = MossTTSPrismModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=spec.text_vocab_size,
        max_sequence_length=spec.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=spec.embedding_head_usage == "tied_head_authoritative",
        rotary_base=spec.rotary_base,
    )
    model.prism_spec = spec
    return model


def select_native_checkpoint(path: str | Path, step: int | None = None):
    """Select a native model component, accepting a RUN, checkpoint root or iteration."""
    root = Path(path)
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError("checkpoint step must be a nonnegative integer")
    if (root / "checkpoints").is_dir():
        root = root / "checkpoints"
    if (root / "checkpoint.json").is_file():
        selected = root
    else:
        if step is None:
            marker = (root / "latest_checkpointed_iteration.txt").read_text().strip()
            if marker == "release":
                selected = root / "release"
            else:
                if not marker.isdecimal() or int(marker) <= 0:
                    raise ValueError("invalid native checkpoint tracker")
                selected = root / f"iter_{int(marker):07d}"
        else:
            selected = root / "release" if step == 0 else root / f"iter_{step:07d}"
    manifest = json.loads((selected / "checkpoint.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Prism requires the native logical checkpoint manifest schema 2")
    expected_iteration = 0 if selected.name == "release" else int(selected.name.removeprefix("iter_"))
    if manifest.get("iteration") != expected_iteration:
        raise ValueError("native selected directory and manifest iteration disagree")
    if step is not None and step != expected_iteration:
        raise ValueError("--ckpt-step disagrees with the explicit native checkpoint directory")
    component = selected / "model"
    if not (component / ".metadata").is_file():
        raise ValueError("native checkpoint model DCP is missing")
    return component, manifest


def _align_native_shards(value, tensor_metadata, num_layers):
    """Reconcile MCore's per-layer template with older layer-stacked DCP keys.

    Keep factory merge functions intact so SwiGLU gate/up TP placement remains
    correct. Only logical metadata gains an axis; local parameter shapes do not.
    """
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

    if isinstance(value, dict):
        return {key: _align_native_shards(item, tensor_metadata, num_layers) for key, item in value.items()}
    if isinstance(value, list):
        return [_align_native_shards(item, tensor_metadata, num_layers) for item in value]
    if isinstance(value, ShardedTensorFactory):
        original_build = value.build_fn

        def build(key, tensor, replica_id, flattened_range):
            return _align_native_shards(
                original_build(key, tensor, replica_id, flattened_range), tensor_metadata, num_layers
            )

        return replace(value, build_fn=build)
    if not isinstance(value, ShardedTensor):
        return value
    if value.key in tensor_metadata:
        expected = tuple(tensor_metadata[value.key].global_shape)
        if tuple(value.global_shape) != expected:
            raise ValueError(f"native shape mismatch for {value.key}: {value.global_shape} versus {expected}")
        return value
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", value.key)
    if match:
        logical = f"decoder.layers.{match[2]}"
        if logical in tensor_metadata:
            expected = (num_layers, *value.global_shape)
            if tuple(tensor_metadata[logical].global_shape) != expected:
                raise ValueError(f"native stacked shape mismatch for {logical}")
            return replace(
                value,
                key=logical,
                global_shape=expected,
                global_offset=(int(match[1]), *value.global_offset),
                axis_fragmentations=(num_layers, *value.axis_fragmentations),
                prepend_axis_num=value.prepend_axis_num + 1,
            )
    raise ValueError(f"native Prism weight is missing: {value.key}")


def load_prism_pretrained(models, args) -> None:
    """Import only the SFT model component; slime owns RL optimizer/resume state.

    Use --custom-megatron-model-initializer-path with --pretrained-checkpoint
    and optional --ckpt-step. Slime dispatches this only without --load, owns
    master-parameter refresh, and keeps all RL progress/optimizer state fresh.
    """
    _require_megatron()
    from megatron.core import dist_checkpointing

    if len(models) != 1:
        raise ValueError("Prism imports exactly one PP=1 model")
    model = models[0]
    while hasattr(model, "module"):
        model = model.module
    spec = model.prism_spec
    source = getattr(args, "pretrained_checkpoint", None)
    if source is None:
        raise ValueError("Prism's model initializer requires --pretrained-checkpoint")
    component, manifest = select_native_checkpoint(source, getattr(args, "ckpt_step", None))
    spec.check_native_identity(manifest["model_identity"])
    common = dist_checkpointing.load_common_state_dict(str(component))
    if common.get("iteration") != manifest["iteration"] or common.get("checkpoint_version") != 3.0:
        raise ValueError("native model common state disagrees with selected manifest")
    saved_metadata = dist_checkpointing.load_tensors_metadata(str(component))
    expected_dtype = getattr(torch, spec.dtype)
    if any(metadata.dtype != expected_dtype for key, metadata in saved_metadata.items() if key.endswith(".weight")):
        raise ValueError("native trainable dtype disagrees with the inference artifact")
    template = model.sharded_state_dict(metadata={"non_homogeneous_layers": True})
    # TE/FA auxiliary states are parameter-free and excluded from the model
    # importer. The target engine initializes its own auxiliary kernel state.
    template = {key: value for key, value in template.items() if "_extra_state" not in key}
    template = _align_native_shards(template, saved_metadata, spec.topology.num_layers)
    loaded = dist_checkpointing.load({"model": template}, str(component))
    incompatible = model.load_state_dict(loaded["model"], strict=False)
    missing = [key for key in incompatible.missing_keys if "_extra_state" not in key]
    if missing or incompatible.unexpected_keys:
        raise ValueError(
            f"native Prism tensor load mismatch: missing={missing}, unexpected={incompatible.unexpected_keys}"
        )
    return None


def extract_policy_log_probs(args, batch, output: Tensor) -> dict[str, list[Tensor]]:
    """Slice packed scores without shifting rows or dropping stop-only actions.

    ``log_probs`` is the legacy RVQ-only frame summary; the objective uses the two
    row-aligned action arrays so a zero-frame terminal action remains visible.
    Slime's forward-only callback owns its ``(empty_tensor, result)`` wrapper.
    """
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError("Prism policy output must have shape [1,T,C]")
    mm = batch["multimodal_train_inputs"]
    code_mask = mm["policy_code_mask"]
    if code_mask.ndim == 3:
        code_mask = code_mask.squeeze(0)
    n_vq = code_mask.shape[-1]
    width = n_vq + 1
    with_entropy = getattr(args, "use_rollout_entropy", False) or getattr(args, "entropy_coef", 0.0) > 0
    if output.shape[-1] not in (width, 2 * width):
        raise ValueError("Prism policy output has an unexpected action dimension")
    if with_entropy and output.shape[-1] != 2 * width:
        raise ValueError("Prism entropy was requested but the model did not return entropy columns")
    results = {"log_probs": [], "policy_code_logprobs": [], "policy_stop_logprobs": []}
    if output.shape[-1] == 2 * width:
        results.update(entropy=[], policy_code_entropy=[], policy_stop_entropy=[])
    offset = 0
    for length in batch["total_lengths"]:
        length = int(length)
        end = offset + length
        codes = output[0, offset:end, :n_vq]
        stops = output[0, offset:end, n_vq]
        selected_rows = code_mask[offset:end].any(-1)
        results["policy_code_logprobs"].append(codes)
        results["policy_stop_logprobs"].append(stops)
        results["log_probs"].append(codes.sum(-1)[selected_rows])
        if output.shape[-1] == 2 * width:
            code_entropy = output[0, offset:end, width : width + n_vq]
            stop_entropy = output[0, offset:end, width + n_vq]
            results["policy_code_entropy"].append(code_entropy)
            results["policy_stop_entropy"].append(stop_entropy)
            results["entropy"].append(code_entropy.sum(-1)[selected_rows])
        offset = end
    if offset > output.shape[1] or offset != code_mask.shape[0]:
        raise ValueError("packed policy masks and total_lengths disagree")
    return results
