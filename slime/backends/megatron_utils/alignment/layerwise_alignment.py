"""Opt-in Megatron layer-output dumps for train/rollout alignment gates."""

from __future__ import annotations

import logging
import os
import re
from functools import partial
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"^(?:.*\.)?decoder\.layers\.(\d+)$")


def _global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class _MegatronLayerwiseDumper:
    def __init__(self, dump_dir: str, selected_layers: set[int], store_prefix: str):
        self.dump_dir = Path(dump_dir) / f"rank{_global_rank():05d}"
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.selected_layers = selected_layers
        self.store_prefix = store_prefix.rstrip("_") or "actor"
        self.module_suffixes = tuple(
            suffix.strip()
            for suffix in os.getenv("SLIME_LAYERWISE_ALIGNMENT_MODULE_SUFFIXES", "").split(",")
            if suffix.strip()
        )
        self.pass_id = 0
        self.current: dict[str, Any] = {}

    def pre_forward(self, module, args, kwargs):
        del module, args
        self.current = {
            "store_prefix": self.store_prefix,
            "layers": {},
            "modules": {},
        }
        input_ids = kwargs.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            self.current["input_ids"] = input_ids.detach().cpu()
        packed_seq_params = kwargs.get("packed_seq_params")
        cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
        if isinstance(cu_seqlens, torch.Tensor):
            self.current["cu_seqlens"] = cu_seqlens.detach().cpu()

    def record_layer(self, layer_id: int, module, args, output):
        del module, args
        tensor = _first_tensor(output)
        if tensor is not None:
            self.current.setdefault("layers", {})[layer_id] = tensor.detach().cpu()

    def record_module(self, module_name: str, module, args, output):
        del module, args
        tensor = _first_tensor(output)
        if tensor is not None:
            self.current.setdefault("modules", {})[module_name] = tensor.detach().cpu()

    def post_forward(self, module, args, output):
        del module, args, output
        if "input_ids" not in self.current:
            raise RuntimeError("Megatron layerwise dump did not observe model input_ids")
        observed_layers = set(self.current.get("layers", {}))
        missing_layers = self.selected_layers - observed_layers
        if missing_layers:
            raise RuntimeError("Megatron layerwise dump missed selected layers: " f"{sorted(missing_layers)}")
        output_path = self.dump_dir / f"{self.store_prefix}_Pass{self.pass_id:05d}.pt"
        torch.save(self.current, output_path)
        logger.info("Dumped Megatron layer outputs to %s", output_path)
        self.pass_id += 1
        self.current = {}

    def register(self, model_chunk) -> int:
        model_chunk.register_forward_pre_hook(self.pre_forward, with_kwargs=True)
        model_chunk.register_forward_hook(self.post_forward)
        registered_layers = 0
        for module_name, module in model_chunk.named_modules():
            match = _LAYER_RE.match(module_name)
            if match is None:
                continue
            layer_number = getattr(module, "layer_number", None)
            layer_id = int(layer_number) - 1 if layer_number is not None else int(match.group(1))
            if layer_id not in self.selected_layers:
                continue
            module.register_forward_hook(partial(self.record_layer, layer_id))
            registered_layers += 1
        for module_name, module in model_chunk.named_modules():
            if any(module_name.endswith(suffix) for suffix in self.module_suffixes):
                module.register_forward_hook(partial(self.record_module, module_name))
        return registered_layers


def enable_megatron_layerwise_dump(args, model, store_prefix: str) -> None:
    """Register one dump hook per selected layer on every model rank."""

    dump_dir = os.getenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR")
    if not dump_dir:
        return

    selected_layers = set(getattr(args, "megatron_deepgemm_forward_layers", []) or [])
    if not selected_layers:
        raise RuntimeError("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR requires " "--megatron-deepgemm-forward-layers")

    registered_layers = 0
    for model_chunk in model:
        if getattr(model_chunk, "_slime_layerwise_dump_registered", False):
            continue
        dumper = _MegatronLayerwiseDumper(dump_dir, selected_layers, store_prefix)
        registered_layers += dumper.register(model_chunk)
        model_chunk._slime_layerwise_dump_registered = True
        model_chunk._slime_layerwise_dumper = dumper

    if registered_layers == 0 and not any(
        getattr(model_chunk, "_slime_layerwise_dump_registered", False) for model_chunk in model
    ):
        raise RuntimeError("Could not find selected Megatron decoder layers to dump")
    if registered_layers:
        logger.info(
            "Enabled Megatron layerwise alignment dump for layers %s at %s",
            sorted(selected_layers),
            dump_dir,
        )
