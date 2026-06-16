import logging

import torch

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

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

try:
    # Patch Qwen3VLModel.forward to accept loss_mask kwarg, which slime
    # passes through multimodal_train_inputs.  The Megatron-LM version
    # does not declare loss_mask, so we intercept and strip it.
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel

    _qwen3vl_orig_forward = Qwen3VLModel.forward

    def _qwen3vl_patched_forward(self, *args, loss_mask=None, **kwargs):
        return _qwen3vl_orig_forward(self, *args, **kwargs)

    Qwen3VLModel.forward = _qwen3vl_patched_forward
except ImportError:
    pass

logging.getLogger("megatron").setLevel(logging.WARNING)

from . import megatron_patch  # noqa: F401, E402
