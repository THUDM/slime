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
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLTextRotaryEmbedding

    _original_forward = Qwen3VLTextRotaryEmbedding.forward

    def _patched_forward(self, *args, packed_seq_params=None, **kwargs):
        return _original_forward(self, *args, **kwargs)

    Qwen3VLTextRotaryEmbedding.forward = _patched_forward
except ImportError:
    pass

logging.getLogger().setLevel(logging.WARNING)
