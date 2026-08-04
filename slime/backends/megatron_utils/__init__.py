import logging

import torch

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        tms_impl = torch_memory_saver._impl
        if tms_impl is None:
            return old_init(self, *args, **kwargs)

        cdll = tms_impl._binary_wrapper.cdll
        original_interesting_region = cdll.tms_get_interesting_region()
        cdll.tms_set_interesting_region(False)
        try:
            old_init(self, *args, **kwargs)
            # DeepEP owns persistent buffers and may initialize them on its
            # internal streams. Make their lifetime independent of the TMS
            # disabled region before restoring allocation tracking.
            torch.cuda.synchronize()
        finally:
            cdll.tms_set_interesting_region(original_interesting_region)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
        Qwen3VLMoETextRotaryEmbedding,
        Qwen3VLTextRotaryEmbedding,
    )

    def patch_rotary_embedding(cls):
        _original_forward = cls.forward

        def _patched_forward(self, *args, packed_seq_params=None, **kwargs):
            return _original_forward(self, *args, **kwargs)

        cls.forward = _patched_forward

    patch_rotary_embedding(Qwen3VLTextRotaryEmbedding)
    patch_rotary_embedding(Qwen3VLMoETextRotaryEmbedding)
except ImportError:
    pass

logging.getLogger("megatron").setLevel(logging.WARNING)

from . import megatron_patch  # noqa: F401, E402
