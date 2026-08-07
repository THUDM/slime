from __future__ import annotations

import inspect
from typing import Any

from slime.utils.misc import load_function
from slime.utils.types import Sample

_current_rollout_id: int | None = None


def set_current_rollout_id(rollout_id: int | None) -> None:
    global _current_rollout_id
    _current_rollout_id = rollout_id


def _accepted_kwargs(function, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(function)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


async def _apply_to_sample(args, sample: Sample, paths: list[str], **kwargs) -> Sample:
    for path in paths:
        hook = load_function(path)
        result = hook(args, sample, **_accepted_kwargs(hook, kwargs))
        if inspect.isawaitable(result):
            result = await result
        if result is not None:
            if not isinstance(result, Sample):
                raise TypeError(
                    f"Rollout sample hook {path!r} returned {type(result).__name__}, expected Sample or None."
                )
            sample = result
    return sample


async def apply_rollout_sample_hooks(args, value, **kwargs):
    """Apply configured hooks to every Sample leaf while preserving list shape."""

    paths = getattr(args, "rollout_sample_hook_path", None) or []
    if not paths:
        return value
    kwargs.setdefault("rollout_id", _current_rollout_id)
    if isinstance(value, Sample):
        return await _apply_to_sample(args, value, paths, **kwargs)
    if isinstance(value, list):
        return [await apply_rollout_sample_hooks(args, item, **kwargs) for item in value]
    raise TypeError(f"Rollout sample hooks expected Sample or list, got {type(value).__name__}.")
