"""Default-off QwenVL recompute tail compatibility patch."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_LOGGED_ADJUSTMENTS: set[tuple] = set()
_LOGGED_PROBES: set[tuple] = set()
_INSPECTED_MODEL_IDS: set[int] = set()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, os.environ.get(name.lower(), "0")).lower() in {"1", "true", "yes", "on"}


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _largest_divisor_at_most(value: int, upper: int) -> int:
    for candidate in range(min(value, upper), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _layer_count(block) -> tuple[int, str]:
    layers = getattr(block, "layers", None)
    if layers is not None:
        try:
            return len(layers), "len(layers)"
        except TypeError:
            pass

    count = _safe_int(getattr(block, "num_layers_per_pipeline_rank", 0))
    if count > 0:
        return count, "num_layers_per_pipeline_rank"

    count = _safe_int(getattr(block, "num_layers", 0))
    if count > 0:
        return count, "num_layers"

    return 0, "unknown"


def _run_checkpointed_forward_with_safe_tail(block, original_call, source: str, *args, **kwargs):
    config = getattr(block, "config", None)
    layer_count, layer_count_source = _layer_count(block)
    recompute_method = getattr(config, "recompute_method", None)
    recompute_num_layers = _safe_int(getattr(config, "recompute_num_layers", 0))
    block_cls = block.__class__
    class_name = getattr(block_cls, "__name__", "<unknown>")
    class_path = f"{getattr(block_cls, '__module__', '<unknown>')}.{class_name}"

    probe_key = (
        source,
        class_path,
        str(recompute_method),
        layer_count,
        recompute_num_layers,
        _safe_int(getattr(block, "num_layers_per_pipeline_rank", 0)),
        layer_count_source,
    )
    if _rank() == 0 and probe_key not in _LOGGED_PROBES:
        _LOGGED_PROBES.add(probe_key)
        logger.info(
            "QwenVL recompute tail patch probe: pid=%s rank=%s source=%s class=%s layer_count=%s "
            "layer_source=%s num_layers_per_pipeline_rank=%s recompute_method=%r "
            "recompute_num_layers=%s layers_type=%s",
            os.getpid(),
            _rank(),
            source,
            class_path,
            layer_count,
            layer_count_source,
            getattr(block, "num_layers_per_pipeline_rank", None),
            recompute_method,
            recompute_num_layers,
            type(getattr(block, "layers", None)).__name__,
        )

    if (
        recompute_method == "uniform"
        and layer_count > 0
        and recompute_num_layers > 1
        and layer_count % recompute_num_layers != 0
    ):
        safe_recompute_num_layers = _largest_divisor_at_most(layer_count, recompute_num_layers)
        if safe_recompute_num_layers != recompute_num_layers:
            key = (
                source,
                class_path,
                layer_count,
                recompute_num_layers,
                safe_recompute_num_layers,
            )
            if _rank() == 0 and key not in _LOGGED_ADJUSTMENTS:
                _LOGGED_ADJUSTMENTS.add(key)
                logger.info(
                    "QwenVL recompute tail patch: pid=%s rank=%s source=%s class=%s layer_count=%s "
                    "layer_source=%s recompute_num_layers=%s -> %s",
                    os.getpid(),
                    _rank(),
                    source,
                    class_path,
                    layer_count,
                    layer_count_source,
                    recompute_num_layers,
                    safe_recompute_num_layers,
                )
            setattr(config, "recompute_num_layers", safe_recompute_num_layers)
            try:
                return original_call(*args, **kwargs)
            finally:
                setattr(config, "recompute_num_layers", recompute_num_layers)

    return original_call(*args, **kwargs)


def _patch_transformer_block_class(block_cls, source: str) -> bool:
    if getattr(block_cls, "_slime_recompute_tail_patched", False):
        return False

    original_checkpointed_forward = block_cls._checkpointed_forward

    def patched_checkpointed_forward(self, *args, **kwargs):
        return _run_checkpointed_forward_with_safe_tail(
            self,
            lambda *call_args, **call_kwargs: original_checkpointed_forward(self, *call_args, **call_kwargs),
            f"class/{source}",
            *args,
            **kwargs,
        )

    patched_checkpointed_forward._slime_recompute_tail_wrapper = True
    block_cls._checkpointed_forward = patched_checkpointed_forward
    block_cls._slime_recompute_tail_patched = True
    logger.info(
        "QwenVL recompute tail patch enabled: pid=%s rank=%s source=%s class=%s.%s",
        os.getpid(),
        _rank(),
        source,
        getattr(block_cls, "__module__", "<unknown>"),
        getattr(block_cls, "__name__", "<unknown>"),
    )
    return True


def _is_checkpointed_block(module) -> bool:
    if not callable(getattr(module, "_checkpointed_forward", None)):
        return False
    if not callable(getattr(module, "_get_layer", None)):
        return False

    layer_count, _ = _layer_count(module)
    return layer_count > 0


def _patch_transformer_block_instance(block, source: str) -> bool:
    if getattr(block, "_slime_recompute_tail_instance_patched", False):
        return False

    original_checkpointed_forward = getattr(block, "_checkpointed_forward", None)
    if not callable(original_checkpointed_forward):
        return False

    def patched_checkpointed_forward(*args, **kwargs):
        return _run_checkpointed_forward_with_safe_tail(
            block,
            original_checkpointed_forward,
            f"instance/{source}",
            *args,
            **kwargs,
        )

    patched_checkpointed_forward._slime_recompute_tail_wrapper = True
    object.__setattr__(block, "_checkpointed_forward", patched_checkpointed_forward)
    object.__setattr__(block, "_slime_recompute_tail_instance_patched", True)

    if _rank() == 0:
        layer_count, layer_count_source = _layer_count(block)
        block_cls = block.__class__
        logger.info(
            "QwenVL recompute tail instance patch enabled: pid=%s rank=%s source=%s class=%s.%s "
            "layer_count=%s layer_source=%s recompute_method=%r recompute_num_layers=%s",
            os.getpid(),
            _rank(),
            source,
            getattr(block_cls, "__module__", "<unknown>"),
            getattr(block_cls, "__name__", "<unknown>"),
            layer_count,
            layer_count_source,
            getattr(getattr(block, "config", None), "recompute_method", None),
            getattr(getattr(block, "config", None), "recompute_num_layers", None),
        )
    return True


def apply_qwen_vl_recompute_tail_patch() -> None:
    if not _env_flag("SLIME_QWENVL_RECOMPUTE_TAIL_PATCH"):
        return

    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import (
            Qwen3VLTransformerBlock,
        )
    except ImportError as exc:
        logger.warning("QwenVL recompute tail patch requested but Qwen3VLTransformerBlock import failed: %r", exc)
        return

    _patch_transformer_block_class(Qwen3VLTransformerBlock, "import")


def apply_qwen_vl_recompute_tail_patch_to_model(model) -> None:
    if not _env_flag("SLIME_QWENVL_RECOMPUTE_TAIL_PATCH") or model is None:
        return

    inspect_model = _env_flag("SLIME_QWENVL_RECOMPUTE_TAIL_INSPECT")
    inspect_rows: list[str] = []
    patched_instances = 0
    for name, module in model.named_modules():
        block_cls = type(module)
        module_name = getattr(block_cls, "__module__", "")
        class_name = getattr(block_cls, "__name__", "")
        has_checkpoint = hasattr(module, "_checkpointed_forward")
        layer_count, layer_count_source = _layer_count(module)

        if inspect_model and has_checkpoint and (layer_count > 0 or "qwen" in module_name.lower()):
            inspect_rows.append(
                "path={path} class={module}.{cls} layer_count={layer_count} "
                "layer_source={source} num_layers_per_pipeline_rank={num_layers} "
                "recompute_method={method!r} recompute_num_layers={num_recompute} "
                "class_patched={class_patched} instance_wrapper={instance_wrapper}".format(
                    path=name or "<root>",
                    module=module_name,
                    cls=class_name,
                    layer_count=layer_count,
                    source=layer_count_source,
                    num_layers=getattr(module, "num_layers_per_pipeline_rank", None),
                    method=getattr(getattr(module, "config", None), "recompute_method", None),
                    num_recompute=getattr(getattr(module, "config", None), "recompute_num_layers", None),
                    class_patched=getattr(block_cls, "_slime_recompute_tail_patched", False),
                    instance_wrapper=getattr(
                        getattr(module, "_checkpointed_forward", None), "_slime_recompute_tail_wrapper", False
                    ),
                )
            )

        if (
            class_name == "Qwen3VLTransformerBlock"
            and "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block" in module_name
            and hasattr(block_cls, "_checkpointed_forward")
        ):
            _patch_transformer_block_class(block_cls, "model")

        if _is_checkpointed_block(module):
            if _patch_transformer_block_instance(module, name or "<root>"):
                patched_instances += 1

    model_id = id(model)
    if _rank() == 0 and model_id not in _INSPECTED_MODEL_IDS:
        _INSPECTED_MODEL_IDS.add(model_id)
        logger.info(
            "QwenVL recompute tail model scan: pid=%s rank=%s model=%s.%s patched_instances=%s "
            "inspect=%s candidates=%s\n%s",
            os.getpid(),
            _rank(),
            model.__class__.__module__,
            model.__class__.__name__,
            patched_instances,
            inspect_model,
            len(inspect_rows),
            "\n".join(inspect_rows[:80]) if inspect_model and inspect_rows else "<inspect disabled>",
        )


apply_qwen_vl_recompute_tail_patch()
