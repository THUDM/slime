import logging

try:
    from torch.distributed.fsdp import fully_shard  # noqa: F401

    _FSDP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"FSDP backend dependencies not available: {e}")
    _FSDP_AVAILABLE = False

if _FSDP_AVAILABLE:
    from .actor import FSDPTrainRayActor
    from .arguments import load_fsdp_args
else:

    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            "FSDP backend is not available. "
            "Please ensure PyTorch with FSDP2 support is installed. "
            "For installation instructions, refer to: https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html"
        )

    FSDPTrainRayActor = _raise_import_error
    load_fsdp_args = _raise_import_error

__all__ = ["load_fsdp_args", "FSDPTrainRayActor"]

logging.getLogger().setLevel(logging.WARNING)
