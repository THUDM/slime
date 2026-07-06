"""Default-off compatibility shims for Megatron-Bridge runtime experiments."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, os.environ.get(name.lower(), "0")).lower() in {"1", "true", "yes", "on"}


def apply_megatron_bridge_compat_shims() -> None:
    """Patch harmless import-time gaps between tested Bridge and image MCore.

    Newer Megatron-Bridge eagerly imports the Mamba provider from
    ``megatron.bridge.models.__init__``.  The QwenVL smoke does not use Mamba,
    but the import can fail on older Megatron-Core pins before Qwen bridges are
    registered.  Keep the shim opt-in and make accidental Mamba use fail loudly.
    """

    if not _env_flag("SLIME_MBRIDGE_MAMBA_IMPORT_SHIM"):
        return

    try:
        import megatron.core.ssm.mamba_hybrid_layer_allocation as allocation
    except Exception as exc:
        logger.warning("Cannot import MCore Mamba allocation module for Bridge shim: %r", exc)
        return

    if hasattr(allocation, "parse_hybrid_pattern"):
        return

    def _parse_hybrid_pattern_unavailable(*args, **kwargs):
        raise RuntimeError(
            "SLIME_MBRIDGE_MAMBA_IMPORT_SHIM only bypasses Megatron-Bridge eager imports for non-Mamba models; "
            "parse_hybrid_pattern is unavailable in this Megatron-Core image."
        )

    allocation.parse_hybrid_pattern = _parse_hybrid_pattern_unavailable
    logger.warning(
        "Installed SLIME_MBRIDGE_MAMBA_IMPORT_SHIM: added placeholder "
        "megatron.core.ssm.mamba_hybrid_layer_allocation.parse_hybrid_pattern for non-Mamba Bridge imports."
    )
