"""Default-off timing hooks for Megatron-Bridge Qwen-VL modules."""

from __future__ import annotations

import functools
import inspect
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

_PATCH_PREFIX = "megatron.bridge.models.qwen_vl.modelling_qwen3_vl"
_CORE_PATCH_PREFIXES = (
    "megatron.core.transformer",
    "megatron.core.ssm",
)
_PATCHED_CLASSES: list[str] = []
_PATCHED_FUNCTIONS: list[str] = []


def _env_value(name: str, default: str) -> str:
    return os.environ.get(name, os.environ.get(name.lower(), default))


def _env_flag(name: str) -> bool:
    return _env_value(name, "0").lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_value(name, str(default)))
    except ValueError:
        return default


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _rank_enabled() -> bool:
    rank_filter = _env_int("SLIME_QWENVL_BRIDGE_TIMING_RANK", 0)
    return rank_filter < 0 or _rank() == rank_filter


def _shape_summary(value, depth: int = 0):
    if depth > 2:
        return type(value).__name__
    if torch.is_tensor(value):
        return {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
    if type(value).__name__ == "PackedSeqParams":
        summary = {
            "type": "PackedSeqParams",
            "qkv_format": getattr(value, "qkv_format", None),
            "max_seqlen_q": getattr(value, "max_seqlen_q", None),
            "max_seqlen_kv": getattr(value, "max_seqlen_kv", None),
        }
        for name in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
            tensor = getattr(value, name, None)
            if torch.is_tensor(tensor):
                vals = tensor.detach().cpu().tolist()
                summary[name] = {
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "device": str(tensor.device),
                    "head": vals[:8],
                    "tail": vals[-8:],
                }
            else:
                summary[name] = None
        return summary
    if isinstance(value, (list, tuple)):
        return [_shape_summary(item, depth + 1) for item in value[:3]]
    if isinstance(value, dict):
        return {str(key): _shape_summary(val, depth + 1) for key, val in list(value.items())[:6]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


def _safe_shape_summary(value):
    try:
        return _shape_summary(value)
    except Exception as exc:
        return {"summary_error": repr(exc), "type": type(value).__name__}


def _cuda_elapsed_ms(fn):
    if not torch.cuda.is_available():
        start = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - start) * 1000.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def _make_function_timing(module_name: str, fn_name: str, original_fn):
    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        if (
            not _env_flag("SLIME_QWENVL_BRIDGE_TIMING")
            or not _env_flag("SLIME_QWENVL_BRIDGE_TIMING_DETAIL")
            or not _rank_enabled()
        ):
            return original_fn(*args, **kwargs)

        count_attr = f"_slime_bridge_timing_detail_count_{fn_name}"
        count = getattr(wrapper, count_attr, 0) + 1
        setattr(wrapper, count_attr, count)
        limit = _env_int("SLIME_QWENVL_BRIDGE_TIMING_DETAIL_LIMIT", _env_int("SLIME_QWENVL_BRIDGE_TIMING_LIMIT", 32))
        interval = max(
            1,
            _env_int("SLIME_QWENVL_BRIDGE_TIMING_DETAIL_INTERVAL", _env_int("SLIME_QWENVL_BRIDGE_TIMING_INTERVAL", 1)),
        )
        should_sample = count <= limit and (count == 1 or count % interval == 0)
        if not should_sample:
            return original_fn(*args, **kwargs)

        result, elapsed_ms = _cuda_elapsed_ms(lambda: original_fn(*args, **kwargs))
        logger.info(
            "Qwen VL Bridge detail timing: function=%s.%s sample=%s rank=%s elapsed_ms=%.3f args=%s kwargs=%s output=%s",
            module_name,
            fn_name,
            count,
            _rank(),
            elapsed_ms,
            _safe_shape_summary(args),
            _safe_shape_summary(kwargs),
            _safe_shape_summary(result),
        )
        return result

    wrapper._slime_bridge_detail_timing_patched = True
    return wrapper


def _patch_model_detail_functions(model_module) -> None:
    if not _env_flag("SLIME_QWENVL_BRIDGE_TIMING_DETAIL"):
        return

    function_names = (
        "reorganize_inputs",
        "preprocess_packed_seqs",
        "get_rope_index",
        "split_deepstack_embs",
        "split_data_cp_rank",
        "qwen3vl_cp_split",
        "get_vision_cp_data",
        "collapse_thw",
    )
    module_name = getattr(model_module, "__name__", type(model_module).__name__)
    for fn_name in function_names:
        original_fn = getattr(model_module, fn_name, None)
        if original_fn is None or getattr(original_fn, "_slime_bridge_detail_timing_patched", False):
            continue
        setattr(model_module, fn_name, _make_function_timing(module_name, fn_name, original_fn))
        _PATCHED_FUNCTIONS.append(f"{module_name}.{fn_name}")


def _class_selected(cls: type) -> bool:
    module = getattr(cls, "__module__", "")
    name = cls.__name__
    skipped_terms = ("RotaryEmbedding",)
    if module.startswith(_PATCH_PREFIX):
        selected_terms = (
            "Model",
            "Layer",
            "Attention",
            "Gated",
            "Delta",
            "MLP",
            "Vision",
            "Patch",
            "Merger",
        )
        return any(term in name for term in selected_terms) and not any(term in name for term in skipped_terms)

    if not _env_flag("SLIME_QWENVL_BRIDGE_TIMING_CORE"):
        return False
    if not module.startswith(_CORE_PATCH_PREFIXES):
        return False

    selected_names = {
        "TransformerLayer",
        "TransformerBlock",
        "SelfAttention",
        "DotProductAttention",
        "MLP",
        "MoELayer",
        "GroupedMLP",
        "SequentialMLP",
        "GatedDeltaNet",
    }
    return name in selected_names and not any(term in name for term in skipped_terms)


def _make_forward(cls: type, original_forward):
    @functools.wraps(original_forward)
    def wrapper(self, *args, **kwargs):
        if not _env_flag("SLIME_QWENVL_BRIDGE_TIMING") or not _rank_enabled():
            return original_forward(self, *args, **kwargs)

        count = getattr(self, "_slime_bridge_timing_count", 0) + 1
        self._slime_bridge_timing_count = count
        limit = _env_int("SLIME_QWENVL_BRIDGE_TIMING_LIMIT", 128)
        interval = max(1, _env_int("SLIME_QWENVL_BRIDGE_TIMING_INTERVAL", 32))
        should_sample = count <= limit and (count == 1 or count % interval == 0)
        if not should_sample:
            return original_forward(self, *args, **kwargs)

        logger.info(
            "Qwen VL Bridge timing pre-call: class=%s module=%s sample=%s rank=%s args=%s kwargs=%s",
            cls.__name__,
            cls.__module__,
            count,
            _rank(),
            _safe_shape_summary(args),
            _safe_shape_summary(kwargs),
        )
        result, elapsed_ms = _cuda_elapsed_ms(lambda: original_forward(self, *args, **kwargs))
        logger.info(
            "Qwen VL Bridge timing: class=%s module=%s sample=%s rank=%s elapsed_ms=%.3f "
            "args=%s kwargs=%s output=%s",
            cls.__name__,
            cls.__module__,
            count,
            _rank(),
            elapsed_ms,
            _safe_shape_summary(args),
            _safe_shape_summary(kwargs),
            _safe_shape_summary(result),
        )
        return result

    return wrapper


def _patch_module(module) -> None:
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if not _class_selected(cls):
            continue
        original_forward = getattr(cls, "forward", None)
        if original_forward is None or getattr(original_forward, "_slime_bridge_timing_patched", False):
            continue
        patched = _make_forward(cls, original_forward)
        patched._slime_bridge_timing_patched = True
        cls.forward = patched
        _PATCHED_CLASSES.append(f"{cls.__module__}.{cls.__name__}")


def apply_qwen_vl_bridge_timing_patch() -> None:
    if not _env_flag("SLIME_QWENVL_BRIDGE_TIMING"):
        return

    module_names = [
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model",
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model",
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model",
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention",
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block",
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils",
    ]
    if _env_flag("SLIME_QWENVL_BRIDGE_TIMING_CORE"):
        module_names.extend(
            [
                "megatron.core.transformer.transformer_block",
                "megatron.core.transformer.transformer_layer",
                "megatron.core.transformer.attention",
                "megatron.core.transformer.dot_product_attention",
                "megatron.core.transformer.mlp",
                "megatron.core.transformer.moe.moe_layer",
                "megatron.core.transformer.moe.experts",
                "megatron.core.ssm.gated_delta_net",
            ]
        )
    imported = []
    import_errors = {}
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=["*"])
        except ImportError as exc:
            import_errors[module_name] = repr(exc)
            continue
        imported.append(module_name)
        _patch_module(module)
        if module_name == "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model":
            _patch_model_detail_functions(module)

    logger.info(
        "Qwen VL Bridge timing patch enabled: core=%s detail=%s imported=%s import_errors=%s "
        "patched_count=%s patched=%s detail_patched=%s",
        _env_flag("SLIME_QWENVL_BRIDGE_TIMING_CORE"),
        _env_flag("SLIME_QWENVL_BRIDGE_TIMING_DETAIL"),
        imported,
        import_errors,
        len(_PATCHED_CLASSES),
        _PATCHED_CLASSES[:64],
        _PATCHED_FUNCTIONS,
    )


apply_qwen_vl_bridge_timing_patch()
