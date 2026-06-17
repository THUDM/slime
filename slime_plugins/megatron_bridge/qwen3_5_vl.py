"""Qwen3.5 Vision-Language bridges (dense + MoE).

This module is a *thin* registration shim: it imports the official Qwen3.5-VL
bridges from `megatron.bridge` so that their ``@MegatronModelBridge.register_bridge``
decorators run, registering both the dense and MoE Qwen3.5-VL HF architectures
with `AutoBridge.from_hf_pretrained`:

  - ``Qwen3_5ForConditionalGeneration``     -> Qwen35VLBridge      (e.g. Qwen3.5-9B / 27B)
  - ``Qwen3_5MoeForConditionalGeneration``  -> Qwen35VLMoEBridge   (e.g. Qwen3.5-35B-A3B / 397B-A17B)

The bridges, providers, mapping registries and hybrid (GDN + Gated Attention)
layer specs all live in NVIDIA's package; we explicitly do NOT reimplement
them on the slime side. See:
  https://docs.nvidia.com/nemo/megatron-bridge/0.4.0/apidocs/bridge/bridge.models.qwen_vl.qwen35_vl_bridge.html

Requires `megatron-bridge >= 0.4.0` and a `transformers` version that exposes
``Qwen3_5ForConditionalGeneration`` (and, for the MoE path,
``Qwen3_5MoeForConditionalGeneration``). The import is wrapped in
``try/except`` so older environments that don't yet ship Qwen3.5-VL still
load the plugin without errors — just without the Qwen3.5-VL bridge
registered.
"""

from __future__ import annotations

import copy
import logging

logger = logging.getLogger(__name__)


def _add_legacy_mtp_aliases(registry):
    """Duplicate every ``mtp.*.mtp_model_layer.*`` mapping with the legacy
    Megatron-LM name ``transformer_layer``.

    The bridge's ``mapping_registry()`` returns a fresh registry on every call
    and ``MegatronMappingRegistry.__init__`` *pre-compiles* the patterns into
    ``_compiled_patterns`` / ``_reverse_patterns`` — so we cannot just append
    to ``registry.mappings``: the new entries would never be matched at
    lookup time. Instead we build a brand-new registry from the augmented
    mapping list, which lets ``__init__`` re-compile everything.
    """
    if registry is None:
        return registry
    original = list(registry.mappings)
    extra = []
    for mapping in original:
        m_param = getattr(mapping, "megatron_param", None)
        if isinstance(m_param, str) and ".mtp_model_layer." in m_param:
            alias = copy.copy(mapping)
            alias.megatron_param = m_param.replace(".mtp_model_layer.", ".transformer_layer.")
            extra.append(alias)
    if not extra:
        return registry

    cls = registry.__class__
    new_registry = cls(*original, *extra)
    return new_registry


def _patch_bridge_mapping_registry(bridge_cls):
    """Wrap ``bridge_cls.mapping_registry`` so callers see the alias-augmented
    registry.  Idempotent — no-op if already wrapped."""
    original = bridge_cls.mapping_registry
    if getattr(original, "_slime_mtp_alias_patched", False):
        return

    def patched(self, *args, **kwargs):
        registry = original(self, *args, **kwargs)
        return _add_legacy_mtp_aliases(registry)

    patched._slime_mtp_alias_patched = True  # type: ignore[attr-defined]
    bridge_cls.mapping_registry = patched


try:
    # Importing these triggers @MegatronModelBridge.register_bridge(...) at
    # module-import time, which is the entire purpose of this file.
    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLBridge, Qwen35VLMoEBridge  # noqa: F401

    _patch_bridge_mapping_registry(Qwen35VLBridge)
    _patch_bridge_mapping_registry(Qwen35VLMoEBridge)
except ImportError as exc:  # pragma: no cover - environment dependent
    logger.info(
        "Qwen3.5-VL bridges not registered (megatron-bridge >= 0.4.0 with a "
        "compatible transformers version is required): %s",
        exc,
    )
