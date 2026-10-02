import logging

import torch

from slime.utils import accelerator

accelerator.initialize_accelerator()

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
            # CPU-only imports intentionally have no selected device; explicit
            # accelerator requests still fail fast in initialize_accelerator().
            selected_accelerator = accelerator.initialize_accelerator()
            if selected_accelerator is not None:
                selected_accelerator.synchronize()
            else:
                # Keep the historical CUDA hook observable for CPU test
                # doubles, while ignoring the expected no-CUDA runtime error.
                try:
                    torch.cuda.synchronize()
                except RuntimeError:
                    pass
        finally:
            cdll.tms_set_interesting_region(original_interesting_region)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

logging.getLogger("megatron").setLevel(logging.WARNING)
