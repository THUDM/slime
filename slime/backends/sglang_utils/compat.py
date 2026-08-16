"""Compatibility helpers for SGLang modules moved between supported releases."""

from importlib import import_module
from types import ModuleType


def import_sglang_module(current_path: str, legacy_path: str) -> ModuleType:
    """Import an SGLang 0.5.17 module, falling back to its 0.5.15 path."""
    try:
        return import_module(current_path)
    except ModuleNotFoundError as error:
        missing_path = error.name
        if missing_path is None or not (current_path == missing_path or current_path.startswith(f"{missing_path}.")):
            raise
        return import_module(legacy_path)
